"""`quantization` : OPTIMAL QUANTIZATION of a density by Lloyd's algorithm, in its transport form.

    pos = sdot.quantization( target, pos0 )
    pos = sdot.quantization( target, 1000 )         # 1000 diracs, starting uniformly at random in the support

The diracs keep their masses ( equal by default ) and only MOVE : at each iteration, the
transport from the diracs to `target` is solved, and each dirac goes to the barycenter of its
cell. It is the fixed point of `W_2( diracs, target )` with respect to the positions -- the step
`x <- b( x )` is exactly a gradient step of size `1 / ( 2 m_i )` on `W_2^2`, which therefore
decreases at each iteration. With equal masses, the result is a blue noise distributed like
`target` ( de Goes et al., "Blue noise through optimal transport", 2012 ).

= The stopping criterion

Lloyd converges LINEARLY, and slowly : the moves never become exactly zero, they get smaller
than what matters. What matters is the size of a cell, and `sqrt( W_2^2 )` is precisely the
typical distance between a point of the target and its dirac -- so the criterion is

    rms_i | b_i - x_i |  <=  tol * sqrt( W_2^2 )

scale free, independent of `n` and of the dimension. `tol = 2e-2` ( the default ) : the moves are
a fiftieth of a cell, the eye no longer sees them. Lower values cost many more iterations
( the rate depends on the conditioning of the target ), for a point set that does not look different.

= The cost of an iteration

The problem is KEPT from one iteration to the next ( `OtProblem` ) : the weights of the previous
solution are the starting point of the next one, and since the diracs moved a little, a few Newton steps
suffice -- the cost of an iteration is the cost of these few steps, not that of a fresh solve.

The transports need not be precise either : a mass error of `1e-4` times the mass of a dirac moves its
barycenter by far less than the moves Lloyd still makes. Hence the default tolerance of the solver,
`1e-4 / n` ( relative to the mass of a dirac, where `Iterative`'s default is absolute ), and a solve
that STAGNATES above it does not stop the iterations : its solution is used, with a warning.
"""

import numbers
import warnings

import loom
import numpy
from loom.boundary import _to_framework, export, universe_of

from .OtProblem import Direct, Iterative, OtNotConverged, OtProblem
from .distributions.SumOfDiracs import SumOfDiracs


@loom.public
def quantization( target, positions, masses = None, *, tol = 2e-2, max_iter = 200, callback = None, verbose = False, ot_tol = None,
                  seed = None, **settings ):
    """The positions of diracs that QUANTIZE `target` optimally, by Lloyd iterations starting from
    `positions` ( `[ n, d ]` ). Returns the final positions, `[ n, d ]`, in the caller's framework.

    `positions` can also be a NUMBER of diracs: they then start uniformly at random in the support of
    `target` ( its `bounding_half_spaces` ), drawn with `numpy.random.default_rng( seed )`.

    `masses` : `[ n ]`, the mass of each dirac ( kept during the iterations ) -- equal by default.
    `tol` : the relative stopping criterion ( see the module ) -- the root mean square move must be
    lower than `tol` times the square root of the transport cost. `max_iter` : the number of
    iterations at most ( a warning is issued if it is reached first ). `callback( iteration, positions, plan )`
    is called after each solve, with the positions FOR WHICH `plan` was solved ( `iteration = 0`:
    the starting positions ) -- the way to record the trajectory. `settings` : those of the solver
    of the regime ( `Iterative` for `d >= 2`, see `ot_solve` ) -- `tol` excepted, which is named
    `ot_tol` here ( `1e-4 / n` by default, see the module ).
    """
    if isinstance( positions, numbers.Integral ):
        positions = _to_framework( uniform_in_support( target, int( positions ), seed ), universe_of( ( target, masses ) ) )
    universe = universe_of( ( target, positions, masses ) ) if callback is not None else None

    pb = OtProblem( SumOfDiracs( positions, masses ), target )
    nb_diracs = pb.source.nb_diracs.value
    if pb.regime == "direct":
        solver_settings = Direct( with_barycenters = True, **settings )
    else:
        solver_settings = Iterative( tol = 1e-4 / nb_diracs if ot_tol is None else ot_tol, **settings )

    # RAW arrays ( of the caller's framework, on its device ): the barycenters' axes belong to the plan,
    # the positions' to the cloud -- two loom tensors with different axes would broadcast to an outer product
    x = pb.source.positions.value
    stagnations = []
    for iteration in range( max_iter + 1 ):
        try:
            plan = pb.solve( solver_settings, verbose )
        except OtNotConverged as e:
            if e.solution.stats[ "status" ] != "stagnation":
                raise
            plan = e.solution
            stagnations.append( plan.stats[ "residual" ] )
        if callback is not None:
            callback( iteration, export( pb.source.positions, universe ), plan )

        b = plan.barycenters.value
        ratio = ( float( ( ( b - x ) ** 2 ).sum() ) / nb_diracs / float( plan.cost ) ) ** 0.5
        if ratio <= tol or iteration == max_iter:
            break

        pb.source = SumOfDiracs( b, masses )
        x = b

    if stagnations:
        warnings.warn( f"sdot.quantization: { len( stagnations ) } of { iteration + 1 } transports stagnated above their tolerance "
                       f"( worst residual { max( stagnations ) :.3g}, for a dirac mass of { 1 / nb_diracs :.3g} ) -- their "
                       "solutions were used all the same", stacklevel = 3 )
    if ratio > tol:
        warnings.warn( f"sdot.quantization: { max_iter } iterations without reaching tol = { tol } ( the last "
                       f"move is { ratio :.3g} times the cell size )", stacklevel = 3 )
    return plan.barycenters


def uniform_in_support( target, n, seed = None ):
    """`n` points drawn uniformly at random in the support of `target` ( a host `[ n, d ]` array ): in the
    polytope of its `bounding_half_spaces`, by rejection in the bounding box of its vertices."""
    from .viz.polytope import vertices_of

    support = target.bounding_half_spaces()
    if support is None:
        raise ValueError( "sdot.quantization: the target declares no support ( `bounding_half_spaces` is None ), "
                          "the starting positions have to be given" )
    dirs, offs = ( numpy.asarray( a, dtype = float ) for a in support )
    offs = offs.reshape( -1 )
    verts, _ = vertices_of( dirs, offs )
    lo, hi = verts.min( axis = 0 ), verts.max( axis = 0 )

    rng = numpy.random.default_rng( seed )
    res = numpy.zeros( ( 0, len( lo ) ) )
    while len( res ) < n:
        x = rng.uniform( lo, hi, ( 2 * ( n - len( res ) ) + 16, len( lo ) ) )
        res = numpy.concatenate( [ res, x[ numpy.all( x @ dirs.T <= offs, axis = 1 ) ] ] )
    return res[ :n ]
