import numpy

from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar
from loom.util import ComputedAttribute

from ._bvh import build_bvh
from .Distribution import Distribution


class Mesh( Distribution ):
    """A PIECEWISE LINEAR density on a mesh of simplices, DISCONTINUOUS between them ( DG1 ): one value per corner of each simplex.

        rho( x ) = sum_i lambda_i( x ) values( e, i )      for x in the simplex e ( barycentric coordinates ),
                 = 0                                       outside the mesh.

    `nodes` is `[ nb_nodes, d ]`, `simplices` `[ nb_elems, d + 1 ]` ( indices of nodes ) -- triangles in 2D, tetrahedra in
    3D: the simplices of the dimension of the space. `values` is `[ nb_elems, d + 1 ]` ( a value per corner: a hole is a set of
    simplices whose values are zero ), or `[ nb_nodes ]` for a CONTINUOUS density ( P1: the value of the node at each corner );
    one is all by default. The total mass is exact, `sum_e |e| mean( values( e ) )`, and normalizing is a division, so autodiff
    goes through it.

    = The width continuation

    `spread( k )` is the same mesh with the values spread by the corner averages ( `sdot.corner_averages`: `k = 0` the density
    itself, then wider and wider, continuous in `k`, the mass and the positivity kept ), `spread_start()` the `k` where the
    density is flat enough to start from. The solver's continuation of a mesh goes down this path ( `SdotPlanNd._solve_by_spreading` ).

    = How a cell meets it

    The cell is cut by each simplex it meets: a piece is `cell INTERSECT simplex`, on which the density is AFFINE and
    integrated exactly ( `include/sdot/Mesh.h`). The simplices a cell meets are found in a bounding volume hierarchy
    built here, on the host, once ( `_bvh.py` ): the simplices are stored in the order of its leaves, `bvh_*` is the tree, and the
    cell's own box is all the query there is.

    = The support

    The box of the mesh, widened by `MARGIN` ( `bounding_half_spaces` ): the mesh itself is not convex in general, and what lies
    in the box without being in the mesh has a zero density -- a cell there has no mass. The margin keeps the border of the mesh
    away from the border of the domain, so that moving a seed moves the facets across the border of the mesh smoothly.
    """

    MARGIN           = 0.1

    nb_nodes         : ShapeVar
    nb_elems         : ShapeVar
    nb_corners       : ShapeVar
    nb_bnodes        : ShapeVar
    nb_links         : ShapeVar
    nb_dims          : CtShapeVar

    num_node         : Axis[ "nb_nodes" ]
    num_elem         : Axis[ "nb_elems" ]
    num_corner       : Axis[ "nb_corners" ]
    num_bnode        : Axis[ "nb_bnodes" ]
    num_link         : Axis[ "nb_links" ]
    dim              : Axis[ "nb_dims" ]

    nodes            : RealTensor[ "num_node", "dim" ]
    values           : RealTensor[ "num_elem", "num_corner" ]                 # DG1: a value per corner of each simplex
    simplices        : IntTensor[ "num_elem", "num_corner", dict( size = 32 ) ]
    grads            : RealTensor[ "num_elem", "num_corner", "dim" ]          # the gradients of the barycentric coordinates
    corner_weights   : RealTensor[ "num_elem", "num_corner" ]                 # `|e| / ( d + 1 )`: `mass = sum( values * corner_weights )`
    bvh_lo           : RealTensor[ "num_bnode", "dim" ]
    bvh_hi           : RealTensor[ "num_bnode", "dim" ]
    bvh_links        : IntTensor[ "num_bnode", "num_link", dict( size = 32 ) ]  # left, right, begin, end

    current_mass     : ComputedAttribute[ RealTensor, ( "values", ) ]

    cuts_pieces = True

    def __init__( self, nodes, simplices, values = None, target_mass = 1.0, **kwargs ):
        """see the class docstring. The other keyword arguments are for the copies that `normalized_version` makes ( the geometry
        already prepared )"""
        if "grads" in kwargs:
            self._bbox = kwargs.pop( "bbox" )
            self.__base_init__( nodes = nodes, simplices = simplices, values = values, target_mass = target_mass, **kwargs )
            return

        nd = numpy.asarray( nodes, dtype = float )
        sx = numpy.asarray( simplices )
        if nd.ndim != 2 or sx.ndim != 2 or sx.shape[ 1 ] != nd.shape[ 1 ] + 1:
            raise ValueError( f"Mesh : `nodes` has to be [ nb_nodes, d ] and `simplices` [ nb_elems, d + 1 ] ( got { nd.shape } and { sx.shape } )" )
        d = nd.shape[ 1 ]
        if d not in ( 2, 3 ):
            raise ValueError( f"Mesh : the dimension of the space has to be 2 or 3 ( got { d } )" )
        if sx.size and ( sx.min() < 0 or sx.max() >= len( nd ) ):
            raise ValueError( "Mesh : `simplices` refers to a node that does not exist" )
        sx = sx.astype( numpy.int64 )
        n_in = len( sx )

        # the geometry: edges from the first corner, the gradients of the barycentric coordinates, the volumes
        P = nd[ sx ]                                                           # [ E, d + 1, d ]
        A = P[ :, 1:, : ] - P[ :, :1, : ]
        det = numpy.linalg.det( A )
        keep = numpy.abs( det ) > 0                                            # ( a flat simplex has no volume to give )
        sx, P, A, det = sx[ keep ], P[ keep ], A[ keep ], det[ keep ]
        keep_perm = numpy.nonzero( keep )[ 0 ]
        if not len( sx ):
            raise ValueError( "Mesh : no simplex of positive volume" )
        factorial = float( numpy.prod( numpy.arange( 1, d + 1 ) ) )
        vol = numpy.abs( det ) / factorial
        G = numpy.empty( ( len( sx ), d + 1, d ) )
        G[ :, 1:, : ] = numpy.swapaxes( numpy.linalg.inv( A ), 1, 2 )
        G[ :, 0, : ] = - G[ :, 1:, : ].sum( axis = 1 )

        # the hierarchy, and the simplices in the order of its leaves
        lo, hi = P.min( axis = 1 ), P.max( axis = 1 )
        perm, bvh_lo, bvh_hi, links = build_bvh( lo, hi )
        sx, G, vol = sx[ perm ], G[ perm ], vol[ perm ]
        keep_perm = keep_perm[ perm ]                                         # ( the input simplex of each stored one )
        self._bbox = ( lo.min( axis = 0 ), hi.max( axis = 0 ) )

        weights = numpy.repeat( ( vol / ( d + 1 ) )[ :, None ], d + 1, axis = 1 )
        self.__base_init__( nodes = nd, simplices = sx.astype( numpy.int32 ), values = self._corner_values( values, sx, keep_perm, n_in ),
                            target_mass = target_mass, grads = G, corner_weights = weights, bvh_lo = bvh_lo, bvh_hi = bvh_hi, bvh_links = links.astype( numpy.int32 ), **kwargs )

    @staticmethod
    def from_contours( contours, box = None, mesh_size = 0.05, inside = 1.0, outside = 0.0, margin = 0.2 ):
        """a 2D mesh generated by `pygmsh` ( needs it installed ) from closed CONTOURS: `contours` is a list of `[ n, 2 ]` arrays of
        points ( in order, not repeated at the end ), each one the border of a region. The regions are paved with triangles, together
        with a box around them, whose edges follow the contours ( no triangle straddles one ). The density is `inside` ( 1 ) on the
        triangles within a contour and `outside` ( 0 ) on those of the box around.

        `box` is `( lo, hi )`, or `None` for the box of the contours widened by `margin` times its largest side. `mesh_size` is the
        target length of the edges ( a number, handed to `pygmsh` for the points of the contours and the box )."""
        import pygmsh                                            # ( heavy, and optional )
        contours = [ numpy.asarray( c, dtype = float ) for c in contours ]
        if not contours or any( c.ndim != 2 or c.shape[ 1 ] != 2 or len( c ) < 3 for c in contours ):
            raise ValueError( "Mesh.from_contours : `contours` has to be a list of [ n >= 3, 2 ] arrays" )
        if box is None:
            pts = numpy.concatenate( contours )
            lo, hi = pts.min( axis = 0 ), pts.max( axis = 0 )
            pad = margin * float( numpy.max( hi - lo ) )
            box = ( lo - pad, hi + pad )
        ( x0, y0 ), ( x1, y1 ) = numpy.asarray( box, dtype = float )

        with pygmsh.geo.Geometry() as geom:
            inner = [ geom.add_polygon( c, mesh_size = mesh_size ) for c in contours ]
            around = geom.add_polygon( [ [ x0, y0 ], [ x1, y0 ], [ x1, y1 ], [ x0, y1 ] ], mesh_size = mesh_size,
                                       holes = [ p.curve_loop for p in inner ] )
            geom.add_physical( [ p.surface for p in inner ], "inside" )
            geom.add_physical( around.surface, "outside" )
            mesh = geom.generate_mesh()

        block = [ c.type for c in mesh.cells ].index( "triangle" )
        simplices = mesh.cells[ block ].data
        is_inside = numpy.zeros( len( simplices ), bool )
        is_inside[ mesh.cell_sets[ "inside" ][ block ] ] = True
        values = numpy.where( is_inside, inside, outside )
        return Mesh( nodes = mesh.points[ :, :2 ], simplices = simplices, values = numpy.repeat( values[ :, None ], 3, axis = 1 ) )

    @staticmethod
    def _corner_values( values, sx, src, n_in ):
        """the DG1 values `[ E, d + 1 ]` of the stored simplices ( `sx`, the input simplex of each in `src` ) from what was given: `None`
        ( ones ), a value per node ( P1 ), or a value per corner of the input simplices ( DG1 ). An array of the backend stays one: the
        gathers are its own ( autodiff goes through them )"""
        k = sx.shape[ 1 ]
        if values is None:
            return numpy.ones( sx.shape )
        if isinstance( values, ( list, tuple ) ):
            values = numpy.asarray( values, dtype = float )
        shape = tuple( numpy.shape( values ) )
        if len( shape ) == 1:
            return values[ sx ]
        if shape == ( n_in, k ):
            return values[ src ]
        raise ValueError( f"Mesh : `values` has to be [ nb_nodes ] ( P1 ) or [ nb_elems, { k } ] ( DG1 ) ( got { shape } )" )

    # ---- the width continuation ( `SdotPlanNd._solve_by_spreading` ) -------------------------------------------------------------

    #: where the continuation starts: the first power of 2 `k` whose spread density is under this factor times its mean
    SPREAD_FLATNESS = 3.0
    #: the relative precision of the Chebyshev series of the spreading ( `sdot.corner_averages` )
    SPREAD_EPS = 1e-8

    def _host_geometry( self ):
        """`( nodes, simplices, values )` on the host, or `None` ( a tracer )"""
        try:
            return ( numpy.asarray( self.nodes, dtype = float ), numpy.asarray( self.simplices ),
                     numpy.asarray( self.values, dtype = float ).reshape( -1, int( self.nb_corners.value ) ) )
        except ( TypeError, ValueError, RuntimeError ):
            return None

    def spread( self, k ):
        """the same mesh, the values spread by `k` corner averages ( see the class docstring ); `self` for `k = 0`"""
        if k <= 0:
            return self
        from ..corner_averages import corner_averages
        nodes, sx, cv = self._host_geometry()
        return self.with_values( corner_averages( nodes, sx, cv, [ k ], method = "cheb", eps = self.SPREAD_EPS )[ 0 ] )

    def spread_start( self ):
        """the first power of 2 `k` whose spread density is under `SPREAD_FLATNESS` times its mean -- `None` when the mesh is not
        readable on the host ( a tracer: no continuation )"""
        geo = self._host_geometry()
        if geo is None:
            return None
        from ..corner_averages import corner_averages
        nodes, sx, cv = geo
        w = numpy.asarray( self.corner_weights, dtype = float )
        mean = float( ( w * cv ).sum() / w.sum() )
        if mean <= 0:
            return None
        k = 1
        while k < 1 << 22:
            if corner_averages( nodes, sx, cv, [ k ], method = "cheb", eps = self.SPREAD_EPS )[ 0 ].max() <= self.SPREAD_FLATNESS * mean:
                return k
            k *= 2
        return k

    def with_values( self, values, current_mass = None ):
        """the same mesh ( geometry, hierarchy ) with other DG1 values `[ nb_elems, d + 1 ]`, in the order of `self.simplices`"""
        return Mesh(
            self.nodes, self.simplices, values = values, target_mass = self.target_mass,
            grads = self.grads, corner_weights = self.corner_weights, bvh_lo = self.bvh_lo, bvh_hi = self.bvh_hi, bvh_links = self.bvh_links,
            nb_nodes = self.nb_nodes.value, nb_elems = self.nb_elems.value, nb_corners = self.nb_corners.value,
            nb_bnodes = self.nb_bnodes.value, nb_links = self.nb_links.value, nb_dims = self.nb_dims.value,
            batch_axes = self.batch_axes, bbox = self._bbox, **( {} if current_mass is None else dict( current_mass = current_mass ) ),
        )

    def bounding_half_spaces( self ):
        """the box of the mesh, widened by a margin ( `MARGIN` times its largest side ): the cells must be free to move
        past the border of the mesh, otherwise the border would look fixed to the derivatives while it is not"""
        lo, hi = self._bbox
        pad = self.MARGIN * float( numpy.max( hi - lo ) )
        lo, hi = lo - pad, hi + pad
        d = len( lo )
        return numpy.concatenate( [ numpy.eye( d ), -numpy.eye( d ) ] ), numpy.concatenate( [ hi, -lo ] )

    def inscribed_box( self ):
        """the largest cube that fits in ONE simplex where the density is positive ( see `Distribution.inscribed_box` ): a LP per
        simplex, the best kept. A cube that straddles several simplices could be larger, and this one is always inside"""
        from ._inscribed import largest_cube
        try:
            nodes, sx = numpy.asarray( self.nodes, dtype = float ), numpy.asarray( self.simplices )
            vals = numpy.asarray( self.values, dtype = float ).reshape( sx.shape )
        except ( TypeError, ValueError, RuntimeError ):      # not readable on the host ( a tracer )
            return None
        best = None
        # the 16 simplices of largest volume among those with a density ( a LP each: not all of a big mesh )
        P_all = nodes[ sx ]
        vol = numpy.abs( numpy.linalg.det( P_all[ :, 1:, : ] - P_all[ :, :1, : ] ) ) * ( vals.max( axis = 1 ) > 0 )
        for e in numpy.argsort( -vol )[ :16 ]:
            if vol[ e ] <= 0:
                break
            P = nodes[ sx[ e ] ]
            # the facet opposite to the corner i: `n . x <= n . p_j` for the other corners p_j, `n` outward
            dirs, offs = [], []
            for i in range( len( P ) ):
                F = numpy.delete( P, i, axis = 0 )
                u, _, vt = numpy.linalg.svd( F[ 1: ] - F[ 0 ] )
                n = vt[ -1 ]
                if n @ ( P[ i ] - F[ 0 ] ) > 0:
                    n = -n
                dirs.append( n ); offs.append( n @ F[ 0 ] )
            cube = largest_cube( numpy.array( dirs ), numpy.array( offs ) )
            if cube is not None and ( best is None or cube[ 2 ] > best[ 2 ] ):
                best = cube
        return None if best is None else best[ :2 ]

    def domain_box( self ):
        """`( lo, hi )` of the domain: the box of the mesh, widened by `MARGIN` ( see `bounding_half_spaces` )"""
        lo, hi = self._bbox
        pad = self.MARGIN * float( numpy.max( hi - lo ) )
        return lo - pad, hi + pad

    # ---- the display of the cells where the mesh has mass ( `PowerDiagram.support_pieces` ) ----------------------------------

    def display_blocks( self, threshold = 0.0 ):
        """`( dirs, offs, ids )`: one block per simplex with mass -- a corner `> threshold * max( values )` -- ( see
        `Distribution.display_blocks` ). Its `d + 1` facets are `SEAM`s when they are shared with another simplex and the density
        exceeds the threshold somewhere on them ( the neighbour then holds the same part of the facet ), `SUPPORT`s otherwise; a
        positive threshold adds the plane `rho = threshold` where the simplex has nodes under it ( a `SUPPORT`: the density is
        affine on a simplex, so `rho > threshold` is a half-space there )."""
        return self._display( threshold )[ "blocks" ]

    def blocks_of_cells( self, cells, threshold = 0.0 ):
        """`( owners, blocks )` for the cells of `cells` ( see `Distribution.blocks_of_cells` ). The border of the support is
        made of the `SUPPORT` facets ( and of the `rho = threshold` planes ): a cell whose box meets the box of none of them
        does not cross it, and is kept whole ( block `-1` ) if its centroid is in a block, dropped otherwise. The others are cut
        to each block their box meets ( `overlapping_pairs` )."""
        from ._display import cell_vertices, boxes_of, overlapping_pairs
        g = self._display( threshold )
        dirs, offs, _ = g[ "blocks" ]

        vp, live, nv = cell_vertices( cells )
        clo, chi = boxes_of( vp, live )
        n = len( nv )
        i, j = overlapping_pairs( clo, chi, g[ "box_lo" ], g[ "box_hi" ] )
        alive = nv[ i ] > 0
        i, j = i[ alive ], j[ alive ]

        on_rim = numpy.zeros( n, bool )
        on_rim[ overlapping_pairs( clo, chi, g[ "rim_lo" ], g[ "rim_hi" ] )[ 0 ] ] = True

        # the cells off the rim: in the support or out of it, as their centroid ( in a block: under all its walls )
        centroid = numpy.where( live[ ..., None ], vp, 0 ).sum( axis = 1 ) / numpy.maximum( nv, 1 )[ :, None ]
        off = ~on_rim[ i ]
        ci, cj = i[ off ], j[ off ]
        s = numpy.einsum( "pfc,pc->pf", dirs[ cj ], centroid[ ci ] ) - offs[ cj ]
        inside = numpy.zeros( n, bool )
        inside[ ci[ ( s <= 1e-12 * ( 1 + numpy.abs( offs[ cj ] ) ) ).all( axis = 1 ) ] ] = True
        whole = numpy.nonzero( ( nv > 0 ) & ~on_rim & inside )[ 0 ]

        cut = on_rim[ i ]
        return numpy.concatenate( [ whole, i[ cut ] ] ), numpy.concatenate( [ numpy.full( len( whole ), -1 ), j[ cut ] ] )

    #: beyond this number of simplices with mass, the display does not merge them into larger blocks ( a loop on the host )
    DISPLAY_MERGE_LIMIT = 200_000

    def _display( self, threshold ):
        """the blocks where the mesh has mass, their boxes, and the boxes of the border of the support -- once per `threshold`"""
        cache = self.__dict__.setdefault( "_display_cache", {} )
        key = float( threshold )
        if key in cache:
            return cache[ key ]
        from ..Cell import SEAM, SUPPORT
        from ._display import pack_blocks

        nodes = numpy.asarray( self.nodes, dtype = float )
        sx = numpy.asarray( self.simplices ).astype( numpy.int64 )
        G = numpy.asarray( self.grads, dtype = float )
        E, d = len( sx ), nodes.shape[ 1 ]
        v = numpy.asarray( self.values, dtype = float ).reshape( E, d + 1 )  # ( DG1 )
        thr = key * float( v.max( initial = 0 ) )
        kept = v.max( axis = 1 ) > thr

        # the facet opposite to corner `i`: shared ( two simplices hold the same nodes ) and with mass on it -> a seam
        opp = numpy.stack( [ numpy.sort( numpy.delete( sx, i, axis = 1 ), axis = 1 ) for i in range( d + 1 ) ], axis = 1 )
        _, inv, cnt = numpy.unique( opp.reshape( -1, d ), axis = 0, return_inverse = True, return_counts = True )
        inv = inv.reshape( E, d + 1 )
        shared = cnt[ inv ] >= 2
        fmax = numpy.stack( [ numpy.delete( v, i, axis = 1 ).max( axis = 1 ) for i in range( d + 1 ) ], axis = 1 )
        # ( the density is DG1: the neighbour has its own values on the facet, a jump to zero there is a border of the support )
        both = numpy.full( len( cnt ), numpy.inf )
        numpy.minimum.at( both, inv.reshape( -1 ), fmax.reshape( -1 ) )
        ids = numpy.where( shared & ( both[ inv ] > thr ), SEAM, SUPPORT ).astype( numpy.int32 )

        # the facets as half-spaces: `lambda_i >= 0`, `lambda_i( x ) = G_i . ( x - x_0 ) + delta_i0` ( see `Mesh.h` )
        P = nodes[ sx ]
        gn = numpy.linalg.norm( G, axis = 2 )
        dirs = - G / gn[ ..., None ]
        offs = ( numpy.eye( d + 1 )[ 0 ][ None, : ] - numpy.einsum( "ekc,ec->ek", G, P[ :, 0 ] ) ) / gn

        # `rho >= thr`: `rho( x ) = g . ( x - x_0 ) + v_0`, `g = sum_i v_i G_i` -- where a node is under the threshold
        gr = numpy.einsum( "ek,ekc->ec", v, G )
        ng = numpy.linalg.norm( gr, axis = 1 )
        plane = ( v.min( axis = 1 ) <= thr ) & ( ng > 0 ) & ( thr > 0 )
        safe = numpy.where( ng > 0, ng, 1 )
        rdir = - gr / safe[ :, None ]
        roff = ( v[ :, 0 ] - thr - numpy.einsum( "ec,ec->e", gr, P[ :, 0 ] ) ) / safe

        # the blocks: each simplex with mass, then merged with its neighbours while the union stays convex ( `_merge_blocks` )
        keep = numpy.nonzero( kept )[ 0 ]
        blocks = [ dict( nodes = set( sx[ e ].tolist() ),
                         walls = { int( inv[ e, i ] ): ( dirs[ e, i ], offs[ e, i ], int( ids[ e, i ] ) ) for i in range( d + 1 ) },
                         rho = ( rdir[ e ], roff[ e ] ) if plane[ e ] else None ) for e in keep ]
        if len( blocks ) <= self.DISPLAY_MERGE_LIMIT:
            blocks = _merge_blocks( blocks, nodes, SEAM )

        wd, wo, wi = [], [], []
        for blk in blocks:
            walls = list( blk[ "walls" ].values() )
            if blk[ "rho" ] is not None:
                walls.append( ( blk[ "rho" ][ 0 ], blk[ "rho" ][ 1 ], SUPPORT ) )
            wd.append( numpy.array( [ w[ 0 ] for w in walls ] ) )
            wo.append( numpy.array( [ w[ 1 ] for w in walls ] ) )
            wi.append( numpy.array( [ w[ 2 ] for w in walls ] ) )
        pts = [ nodes[ sorted( blk[ "nodes" ] ) ] for blk in blocks ]

        # the border of the support: the `SUPPORT` facets of the simplices with mass, and those that carry a `rho` plane
        e, i = numpy.nonzero( ( ids == SUPPORT ) & kept[ :, None ] )
        F = nodes[ opp[ e, i ] ]                                            # [ nb, d, d ]
        R = P[ plane & kept ]
        rim_lo = numpy.concatenate( [ F.min( axis = 1 ), R.min( axis = 1 ) ] ).reshape( -1, d )
        rim_hi = numpy.concatenate( [ F.max( axis = 1 ), R.max( axis = 1 ) ] ).reshape( -1, d )

        res = cache[ key ] = dict( rim_lo = rim_lo, rim_hi = rim_hi, blocks = pack_blocks( wd, wo, wi ),
                                   box_lo = numpy.array( [ p.min( axis = 0 ) for p in pts ] ).reshape( -1, d ),
                                   box_hi = numpy.array( [ p.max( axis = 0 ) for p in pts ] ).reshape( -1, d ) )
        return res

    def normalized_version( self, nb_dims = None ):
        mass = self.mass
        if not self.target_mass.is_defined:
            return self
        return self.with_values( self.target_mass / mass * self.values, current_mass = self.target_mass )

    def _update_current_mass( self ):
        self.current_mass = ( self.values * self.corner_weights ).sum( axis = self.num_corner ).sum( axis = self.num_elem )


def _merge_blocks( blocks, nodes, seam ):
    """Simplices with mass merged into larger CONVEX blocks, for the display ( `Mesh.display_blocks` ). A block is
    `dict( nodes, walls, rho )`: its nodes, its walls by facet number `{ facet: ( dir, off, id ) }` and the plane `rho = threshold`
    when it has one. Two blocks are merged, greedily, round after round, when they share a `SEAM` facet and

      * all their shared facets lie on ONE plane -- then the union is the intersection of their other walls as soon as
      * each one's nodes are under the other's other walls ( the union is convex ),
      * and two other walls on the same plane carry the same identifier ( a wall stays a seam or a border over all its extent ).

    A block with a `rho` plane is not merged: its region is not the hull of its nodes."""
    def coplanar( a, b, tol = 1e-9 ):
        return abs( abs( float( a[ 0 ] @ b[ 0 ] ) ) - 1 ) < tol and \
               abs( float( a[ 1 ] ) - float( b[ 1 ] ) * float( numpy.sign( a[ 0 ] @ b[ 0 ] ) ) ) < tol * ( 1 + abs( float( a[ 1 ] ) ) )

    def under( pts, walls ):
        return all( ( pts @ w[ 0 ] - w[ 1 ] <= 1e-9 * ( 1 + abs( w[ 1 ] ) ) ).all() for w in walls )

    def merged( A, B ):
        if A[ "rho" ] is not None or B[ "rho" ] is not None:
            return None
        common = A[ "walls" ].keys() & B[ "walls" ].keys()
        cw = [ A[ "walls" ][ f ] for f in common ]
        if not cw or any( w[ 2 ] != seam for w in cw ) or not all( coplanar( cw[ 0 ], w ) for w in cw[ 1: ] ):
            return None
        ea = [ w for f, w in A[ "walls" ].items() if f not in common ]
        eb = [ w for f, w in B[ "walls" ].items() if f not in common ]
        if not under( nodes[ list( B[ "nodes" ] ) ], ea ) or not under( nodes[ list( A[ "nodes" ] ) ], eb ):
            return None
        if any( wa[ 2 ] != wb[ 2 ] and coplanar( wa, wb ) for wa in ea for wb in eb ):
            return None
        walls = { f: w for f, w in A[ "walls" ].items() if f not in common }
        walls.update( { f: w for f, w in B[ "walls" ].items() if f not in common } )
        return dict( nodes = A[ "nodes" ] | B[ "nodes" ], walls = walls, rho = None )

    alive = list( blocks )
    while True:
        owner = {}                                          # facet -> the blocks that hold it
        for b, blk in enumerate( alive ):
            for f, w in blk[ "walls" ].items():
                if w[ 2 ] == seam:
                    owner.setdefault( f, [] ).append( b )
        used, out, changed = set(), {}, False
        for f, bs in owner.items():
            if len( bs ) != 2 or bs[ 0 ] in used or bs[ 1 ] in used:
                continue
            m = merged( alive[ bs[ 0 ] ], alive[ bs[ 1 ] ] )
            if m is None:
                continue
            used.update( bs )
            out[ bs[ 0 ] ] = m
            changed = True
        if not changed:
            return alive
        alive = [ out.get( b, blk ) for b, blk in enumerate( alive ) if b not in used or b in out ]
