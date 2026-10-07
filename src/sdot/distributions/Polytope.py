import numpy

from loom.tensor import Axis, CtShapeVar, RealTensor, ShapeVar
from loom.util import ComputedAttribute

from .Distribution import Distribution


def _volume_of( directions, offsets ):
    """The volume of the convex polytope `{ x : directions . x <= offsets }`, on the host. Raises if it is empty or unbounded."""
    d = directions.shape[ 1 ]
    if d == 1:
        up = offsets[ directions[ :, 0 ] > 0 ] / directions[ directions[ :, 0 ] > 0, 0 ]
        lo = offsets[ directions[ :, 0 ] < 0 ] / directions[ directions[ :, 0 ] < 0, 0 ]
        if not len( up ) or not len( lo ):
            raise ValueError( "Polytope : unbounded" )
        vol = float( up.min() - lo.max() )
        if not vol > 0:
            raise ValueError( "Polytope : empty or flat" )
        return vol

    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull, HalfspaceIntersection
    # an interior point: the center of the largest ball ( directions are unit vectors )
    res = linprog( c = numpy.r_[ numpy.zeros( d ), -1.0 ], A_ub = numpy.c_[ directions, numpy.ones( len( offsets ) ) ], b_ub = offsets,
                   bounds = [ ( None, None ) ] * d + [ ( 0, None ) ] )
    if res.status == 3 or ( res.status == 0 and res.x[ d ] <= 0 ):
        raise ValueError( "Polytope : unbounded or empty" if res.status == 3 else "Polytope : empty or flat" )
    if res.status != 0:
        raise ValueError( "Polytope : empty or flat" )
    hs = HalfspaceIntersection( numpy.c_[ directions, -offsets ], res.x[ :d ] )
    return float( ConvexHull( hs.intersections ).volume )


class Polytope( Distribution ):
    """A CONSTANT density on a convex polytope, zero elsewhere.

        rho( x ) = density      if  directions( c ) . x <= offsets( c )  for every c
                 = 0            otherwise

    `directions` ( `[ nb_cuts, d ]` ) are normalized to unit vectors, so that `offsets( c )` is the
    distance from the origin to the cut. The geometry is a constant of the problem, read on the host ( it
    must not be traced ); `density` may be traced and is differentiable. As for any distribution,
    `normalized_version` rescales it so that the total mass is `target_mass`: the VALUE of `density` is
    meaningless on its own until sums of densities give it a weight.

    The support it declares is the polytope itself ( `bounding_half_spaces` ), which is what makes
    it a domain for `OtProblem`. The cells of a power diagram then START FROM the polytope: on the card, a small simple
    polygon or polyhedron is laid down as it is ( `StartCell.py`: `start_vertices` ), anything else cuts the box of its vertices.
    No blurring is possible with it ( `Convolved` is absent: it is as smooth as a density gets ). See also `Polygon` ( the polytope of the convex hull of points ),
    and `Box` ( the parallelepiped, which is an `Image` of one pixel ).
    """

    nb_cuts          : ShapeVar
    nb_dims          : CtShapeVar
    nb_densities     : ShapeVar

    num_cut          : Axis[ "nb_cuts" ]
    dim              : Axis[ "nb_dims" ]
    num_density      : Axis[ "nb_densities" ]

    directions       : RealTensor[ "num_cut", "dim" ]
    offsets          : RealTensor[ "num_cut" ]
    density          : RealTensor[ "num_density" ]         # one value ( a tensor of size 1: it has to be indexed in the kernel )

    current_mass     : ComputedAttribute[ RealTensor, ( "density", ) ]

    # the cuts are cut by the cell itself: the integrator reserves a spare cell for the pieces
    cuts_pieces = True

    def __init__( self, directions, offsets, density = 1.0, target_mass = 1.0, volume = None, **kwargs ):
        """`directions`: `[ nb_cuts, d ]`. `offsets`: `[ nb_cuts ]`: the half-spaces `direction . x <= offset`.
        `density`: the value on the polytope. `volume`: when known ( a copy of a polytope )."""
        dirs = numpy.asarray( directions, dtype = float )
        offs = numpy.asarray( offsets, dtype = float ).reshape( -1 )
        if dirs.ndim != 2 or dirs.shape[ 0 ] != offs.shape[ 0 ]:
            raise ValueError( f"Polytope : `directions` has to be [ nb_cuts, d ] and `offsets` [ nb_cuts ] ( got { dirs.shape } and { offs.shape } )" )
        norm = numpy.sqrt( ( dirs ** 2 ).sum( axis = 1 ) )
        if not numpy.all( norm > 0 ):
            raise ValueError( "Polytope : a direction is null" )
        self._dirs, self._offs = dirs / norm[ :, None ], offs / norm
        self._volume = _volume_of( self._dirs, self._offs ) if volume is None else float( volume )

        if hasattr( density, "reshape" ) and not isinstance( density, numpy.ndarray ):
            density = density.reshape( 1 )                  # a backend array, possibly traced
        else:
            density = numpy.asarray( density, dtype = float ).reshape( 1 )
        self.__base_init__( directions = self._dirs, offsets = self._offs, density = density, target_mass = target_mass, **kwargs )

    @property
    def volume( self ):
        """the volume of the polytope ( a host float )"""
        return self._volume

    def bounding_half_spaces( self ):
        """the polytope itself"""
        return self._dirs, self._offs

    def inscribed_box( self ):
        """the largest cube in the polytope ( the density is constant on it ), see `Distribution.inscribed_box`"""
        from ._inscribed import largest_cube
        cube = largest_cube( self._dirs, self._offs )
        return None if cube is None else cube[ :2 ]

    def normalized_version( self, nb_dims = None ):
        mass = self.mass
        if not self.target_mass.is_defined:
            return self
        return Polytope(
            self._dirs, self._offs,
            density = self.target_mass / mass * self.density,
            target_mass = self.target_mass,
            volume = self._volume,
            current_mass = self.target_mass,
            batch_axes = self.batch_axes,
        )

    def _update_current_mass( self ):
        self.current_mass = self.density.sum( axis = self.num_density ) * self._volume


def Polygon( points, density = 1.0, **kwargs ):
    """The `Polytope` of the CONVEX polygon with these `points` ( `[ n, 2 ]`, in any order ). Raises if a point is
    not a vertex of the convex hull -- a non convex polygon is not a polytope."""
    return _hull( points, 2, density, **kwargs )


def Polyhedron( points, density = 1.0, **kwargs ):
    """The `Polytope` of the CONVEX polyhedron with these `points` ( `[ n, 3 ]` ): the 3D sibling of `Polygon`"""
    return _hull( points, 3, density, **kwargs )


def _hull( points, d, density, **kwargs ):
    from scipy.spatial import ConvexHull
    pts = numpy.asarray( points, dtype = float )
    if pts.ndim != 2 or pts.shape[ 1 ] != d:
        raise ValueError( f"the points have to be [ n, { d } ] ( got { pts.shape } )" )
    hull = ConvexHull( pts )
    if len( hull.vertices ) != len( pts ):
        raise ValueError( "the points are not the vertices of a convex polytope" )
    # `ConvexHull.equations` : `n . x + c <= 0` inside, `n` a unit vector pointing outward
    return Polytope( hull.equations[ :, :d ], - hull.equations[ :, d ], density = density, volume = hull.volume, **kwargs )
