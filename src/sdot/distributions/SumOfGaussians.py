import numpy

from loom.tensor import Axis, CtShapeVar, RealTensor, ShapeVar
from loom.util import ComputedAttribute

from .Distribution import Distribution


class SumOfGaussians( Distribution ):
    """A sum of ISOTROPIC gaussians, as a continuous density.

        rho( x ) = sum_i  weights( i ) * exp( - |x - positions( i )|^2 / ( 2 sigmas( i )^2 ) )
                            / ( 2 pi sigmas( i )^2 ) ^ ( d / 2 )

    `weights( i )` is therefore the MASS of gaussian `i`, not its height: the total mass is the
    sum of the weights, with no integral to compute, and normalizing is just a division.

    = Isotropic today, and how we will say otherwise

    The INTENT is read from the RANK of `sigmas`, not from a flag:

        sigmas : RealTensor[ "num_gaussian" ]                   -- a scalar  -> isotropic (here)
        sigmas : RealTensor[ "num_gaussian", "dim" ]            -- a vector  -> axis-aligned
        sigmas : RealTensor[ "num_gaussian", "dim", "dim" ]     -- a matrix  -> arbitrary

    It is the right place to say it because it is where the FFI reads it: the rank crosses over in the
    C++ type of the member, so `SumOfGaussians::value_at` can branch on it with an
    `if constexpr ( sigmas.ct_rank == 1 )` -- not a runtime test, not a second aggregate, and
    the isotropic case pays nothing for the existence of the others. Only rank 1 is written for now.

    = What it is used to test

    It is the first density that is NOT piecewise constant, hence the first to make the integrator
    go through its quadrature (see `PowerDiagram::integrate_into`). It cuts nothing --
    its piece is the whole cell -- and only provides three pointwise things: `value_at`,
    `gradient_at`, and where to accumulate `d rho / d parameters`. It knows nothing about cells, and
    `PowerDiagram` knows nothing about gaussians: it is exactly the split we want to put to the test.

    = The support

    A gaussian has none, but beyond a few standard deviations nothing is left: the support
    that the distribution DECLARES ( `bounding_half_spaces` ) is the box `[ c_i - k s_i, c_i + k s_i ]`
    united over the gaussians, `k = support_sigmas` ( 6 by default: the tail beyond weighs 2e-9 ).
    It is what bounds the domain of a transport ( `SdotPlanNd`: the domain comes from the density, and
    from it alone ) and what `PowerDiagram` adds to its half-spaces. `support_sigmas = None`: no
    declared support -- the boundary cells stay infinite and `measures` answers `TF::max` there,
    unless a `boundaries` is given. The mass outside the domain is lost, and the sum of the measures equals the
    target mass MINUS the tails: it is the truth of what was asked for, not a numerical error.
    """

    nb_gaussians     : ShapeVar
    nb_dims          : CtShapeVar

    num_gaussian     : Axis[ "nb_gaussians" ]
    dim              : Axis[ "nb_dims" ]

    positions        : RealTensor[ "num_gaussian", "dim" ]
    sigmas           : RealTensor[ "num_gaussian" ]
    weights          : RealTensor[ "num_gaussian" ]

    current_mass     : ComputedAttribute[ RealTensor, ( "weights", ) ]

    def __init__( self, positions, sigmas, weights = None, target_mass = 1.0, support_sigmas = 6.0, support_box = None, **kwargs ):
        """`positions`: `[ n, d ]`. `sigmas`: `[ n ]` (isotropic). `weights`: `[ n ]`, the MASS of
        each gaussian -- all equal by default. `support_sigmas`: the declared support, in
        standard deviations ( see the class docstring ); `None` to declare none. `support_box`: `( lo, hi )`, a box declared
        as the support instead ( the domain of a transport: the unit square of the old campaign's densities, § 9 ) -- the mass
        outside it is lost, as beyond `support_sigmas`."""
        self.__base_init__( positions = positions, sigmas = sigmas, target_mass = target_mass, **kwargs )
        self.support_sigmas = None if support_sigmas is None else float( support_sigmas )
        self.support_box = None if support_box is None else tuple( numpy.asarray( b, dtype = float ).reshape( -1 ) for b in support_box )
        if weights is not None:
            self.weights = weights
        elif self.weights.is_undefined:
            # all of the same mass: the exact value does not matter, `normalized_version` rescales
            # it -- what matters is that they are DEFINED, the kernel reading them
            # without a branch (unlike the `weights` of a `PowerDiagram`, where absence has a meaning).
            self.weights = RealTensor[ *self.batch_axes, self.num_gaussian ].full( 1.0 )

    def normalized_version( self, nb_dims = None ):
        mass = self.mass
        if not self.target_mass.is_defined:
            return self

        # the total mass is the sum of the weights (each gaussian is normalized to `weights( i )`),
        # so normalizing is a DIVISION and not an integral -- and autodiff goes through it.
        return SumOfGaussians(
            nb_gaussians = self.nb_gaussians.value,
            nb_dims = self.nb_dims.value,

            positions = self.positions,
            sigmas = self.sigmas,
            weights = self.target_mass / mass * self.weights,

            current_mass = self.target_mass,
            batch_axes = self.batch_axes,
            support_sigmas = self.support_sigmas,
            support_box = self.support_box,
        )

    def add_to_viz( self, viz, color = "#d62728", **kwargs ):
        """The centres of the gaussians, as points."""
        viz.add_points( numpy.asarray( self.positions ), color = color )
        return viz

    def bounding_half_spaces( self ):
        """the box `[ c_i - k s_i, c_i + k s_i ]` united over the gaussians ( see the class
        docstring ) -- `None` without `support_sigmas`, or when the parameters are not readable on the
        host side ( under `jit` ): bounding is an optimization, it must not break the call"""
        if self.support_box is not None:
            lo, hi = self.support_box
            d = len( lo )
            return numpy.concatenate( [ numpy.eye( d ), -numpy.eye( d ) ] ), numpy.concatenate( [ hi, -lo ] )
        if self.support_sigmas is None:
            return None
        try:
            d = int( self.nb_dims.value )
            c = numpy.asarray( self.positions, dtype = float ).reshape( -1, d )
            s = numpy.asarray( self.sigmas, dtype = float ).reshape( -1, 1 )
        except ( TypeError, ValueError, RuntimeError ):      # RuntimeError: a torch tensor that requires grad
            return None
        lo = ( c - self.support_sigmas * s ).min( axis = 0 )
        hi = ( c + self.support_sigmas * s ).max( axis = 0 )
        return numpy.concatenate( [ numpy.eye( d ), -numpy.eye( d ) ] ), numpy.concatenate( [ hi, -lo ] )

    def _update_current_mass( self ):
        # reduction over the GAUSSIAN axis only, so that a possible batch axis survives
        # (same reason as `SumOfDiracs._update_current_mass`).
        self.current_mass = self.weights.sum( axis = self.num_gaussian )
