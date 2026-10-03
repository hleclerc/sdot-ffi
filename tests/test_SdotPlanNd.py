import numpy

from errand import Param, experiment, test
from loom.testing import need

from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                   box_half_spaces, ot_solve, write_convergence_html )


# the domain COMES FROM THE DENSITY, and from it alone: the tile of an image, `centers +- 6 sigma` for
# Gaussians ( `SumOfGaussians.bounding_half_spaces` ). The mass outside the domain is LOST ( 2e-9
# at 6 sigma ), and the solver rescales the target masses to what the domain contains
# ( `sdotplan/Solve.h` ). `atol` in the tests: the target mass of a dirac is `1 / n`, and Newton converges
# down to the kernel noise.


def _overlapping_target( d, nb_gaussians, seed, spread = 0.12 ):
    """A few Gaussians CLOSE to one another rather than separate bumps: their
    combined density vanishes nowhere on the useful area -- the SOFT case."""
    rng = numpy.random.default_rng( seed )
    center = numpy.full( d, 0.5 )
    pos = center + rng.uniform( -spread, spread, size = ( nb_gaussians, d ) )
    sigmas = rng.uniform( 0.18, 0.24, nb_gaussians )
    weights = rng.uniform( 0.6, 1.4, nb_gaussians )
    return SumOfGaussians( pos, sigmas, weights = weights )


def _scattered_target( d, nb_gaussians, seed, sigma = 0.13 ):
    """NARROW, SEPARATE bumps over all of `[ 0.3, 0.7 ]^d` -- the HARD case: a dirac drawn
    uniformly has a good chance of landing in a density DESERT, between two bumps. This is
    where KMT's damping ( the mass floor ) does its work."""
    rng = numpy.random.default_rng( seed )
    pos = rng.uniform( 0.3, 0.7, size = ( nb_gaussians, d ) )
    sigmas = numpy.full( nb_gaussians, sigma )
    weights = rng.uniform( 0.7, 1.3, nb_gaussians )
    return SumOfGaussians( pos, sigmas, weights = weights )


def _target_masses( plan ):
    return numpy.asarray( plan.target_masses ).reshape( -1 )


if test( "newton_matches_the_target_masses" ):
    need( "cpu" )
    # the basic test: the masses of the CELLS, once the fit is done, must fall back on
    # the masses of the DIRACS -- this is the only thing `SdotPlanNd` promises. ONE Gaussian, wide and
    # well centered on the cloud of diracs: the simplest case, with no density desert at all.
    rng = numpy.random.default_rng( 3 )
    pos = rng.uniform( 0.15, 0.85, size = ( 20, 2 ) )
    src = SumOfDiracs( pos )
    dst = SumOfGaussians( numpy.array( [ [ 0.5, 0.5 ] ] ), numpy.array( [ 0.15 ] ),
                          weights = numpy.array( [ 1.0 ] ) )

    plan = OtProblem( src, dst ).solve( Iterative( max_iter = 50, tol = 1e-12 ) )

    assert plan.converged, plan.stats
    got = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert numpy.allclose( got, _target_masses( plan ), atol = 1e-10 ), numpy.abs( got - _target_masses( plan ) ).max()
    # the target mass is that of the diracs, up to the tail beyond 6 sigma
    assert abs( plan.stats[ "domain_mass" ] - 1 ) < 1e-7, plan.stats[ "domain_mass" ]


if test( "ot_solve_and_the_warm_start_is_a_PLAN_not_weights" ):
    need( "cpu" )
    # `ot_solve` is ONLY the shortcut: same plan, up to the last digits. And the warm start
    # is a PLAN -- either given through `ot_plan`, or proposed by the `OtProblem` which keeps its last
    # solution. A plan and not weights: as soon as there are coincident seeds, `w` alone does not describe
    # the solution ( README § 23.11 ), and the plan also carries what is needed not to re-detect the clusters.
    rng = numpy.random.default_rng( 7 )
    pos = rng.uniform( 0.15, 0.85, size = ( 30, 2 ) )
    dst = SumOfGaussians( numpy.array( [ [ 0.5, 0.5 ] ] ), numpy.array( [ 0.2 ] ), weights = numpy.array( [ 1.0 ] ) )

    direct = OtProblem( SumOfDiracs( pos ), dst ).solve( Iterative( max_iter = 50, tol = 1e-12 ) )
    shortcut  = ot_solve( SumOfDiracs( pos ), dst, max_iter = 50, tol = 1e-12 )
    assert shortcut.converged, shortcut.stats
    assert numpy.allclose( numpy.asarray( shortcut.weights ), numpy.asarray( direct.weights ), atol = 1e-12 )
    assert shortcut.stats[ "nb_diag" ] == direct.stats[ "nb_diag" ], ( shortcut.stats[ "nb_diag" ], direct.stats[ "nb_diag" ] )

    # the RE-WARMING belongs to the problem, not to the shortcut: a barely moved cloud restarts from the
    # previous weights if we keep the problem, and from zero if we do not.
    moved = pos + 1e-4 * rng.normal( size = pos.shape )
    pb = OtProblem( SumOfDiracs( pos ), dst )
    pb.solve( Iterative( max_iter = 50, tol = 1e-12 ) )
    pb.source = SumOfDiracs( moved )
    rewarmed = pb.solve( Iterative( max_iter = 50, tol = 1e-12 ) )
    cold = ot_solve( SumOfDiracs( moved ), dst, max_iter = 50, tol = 1e-12 )
    assert rewarmed.stats[ "start" ] == "weights0", rewarmed.stats[ "start" ]
    assert cold.stats[ "start" ] == "voronoi", cold.stats[ "start" ]
    assert rewarmed.stats[ "nb_diag" ] < cold.stats[ "nb_diag" ], ( rewarmed.stats[ "nb_diag" ], cold.stats[ "nb_diag" ] )
    assert numpy.allclose( numpy.asarray( rewarmed.weights ), numpy.asarray( cold.weights ), atol = 1e-8 )

    # `ot_plan`: the start IS a plan, and it gives the same result as the kept problem
    resumed = ot_solve( SumOfDiracs( moved ), dst, ot_plan = direct, max_iter = 50, tol = 1e-12 )
    assert resumed.stats[ "warm_start" ] == "ot_plan", resumed.stats[ "warm_start" ]
    assert resumed.stats[ "start" ] == "weights0", resumed.stats[ "start" ]
    assert resumed.stats[ "nb_diag" ] == rewarmed.stats[ "nb_diag" ], ( resumed.stats[ "nb_diag" ], rewarmed.stats[ "nb_diag" ] )
    assert cold.stats[ "warm_start" ] == "none", cold.stats[ "warm_start" ]

    # a plan that does not carry the right number of weights cannot be used: IMPOSED, it raises ( the caller
    # believes it is restarting warm ); proposed by the problem, it is dropped and REPORTED
    try:
        ot_solve( SumOfDiracs( pos[ :20 ] ), dst, ot_plan = direct )
    except ValueError as e:
        assert "ot_plan" in str( e ), str( e )
    else:
        raise AssertionError( "an ot_plan of the wrong size should have raised" )
    pb2 = OtProblem( SumOfDiracs( pos ), dst )
    pb2.solve( Iterative( max_iter = 50, tol = 1e-12 ) )
    pb2.source = SumOfDiracs( pos[ :20 ] )                        # a multiscale stage
    smaller = pb2.solve( Iterative( max_iter = 50, tol = 1e-12 ) )
    assert smaller.stats[ "warm_start" ] == "none", smaller.stats[ "warm_start" ]
    assert smaller.stats[ "start" ] == "voronoi", smaller.stats[ "start" ]

    # both starts at once make no sense
    try:
        Iterative( ot_plan = direct, weights0 = numpy.zeros( len( pos ) ) )
    except ValueError as e:
        assert "ot_plan" in str( e ), str( e )
    else:
        raise AssertionError( "ot_plan AND weights0 should have raised" )

    # a setting that does not belong to this regime is REPORTED, not ignored
    try:
        ot_solve( SumOfDiracs( pos ), dst, with_barycenters = True )
    except TypeError as e:
        assert "Iterative" in str( e ), str( e )
    else:
        raise AssertionError( "`with_barycenters` belongs to the direct regime: ot_solve should have raised" )


if test( "starting_from_nonzero_weights_still_converges" ):
    need( "cpu" )
    # the starting point should only be a matter of speed, not of result -- here we
    # start already NEAR the solution ( `weights0` drawn at random but small ) rather than from zero, and
    # the solver keeps those weights ( `start = "weights0"` ): they empty no cell.
    rng = numpy.random.default_rng( 2 )
    pos = rng.uniform( 0.1, 0.9, size = ( 18, 2 ) )
    src = SumOfDiracs( pos )
    dst = _overlapping_target( 2, 2, seed = 3 )
    w0 = rng.uniform( -0.003, 0.003, 18 )

    plan = OtProblem( src, dst ).solve( Iterative( weights0 = w0, max_iter = 50, tol = 1e-12 ) )

    assert plan.converged and plan.stats[ "start" ] == "weights0", plan.stats
    got = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert numpy.allclose( got, _target_masses( plan ), atol = 1e-10 ), numpy.abs( got - _target_masses( plan ) ).max()


if test( "no_cell_dies_even_with_scattered_targets" ):
    need( "cpu" )
    # the HARD case ( `_scattered_target` ): without a floor, this scenario empties several cells and gets
    # stuck there. Here we check the TWO things the damping promises: no cell dies
    # ALONG THE WAY ( `min_measure` stays `> 0` at EVERY step of `plan.history` ), and
    # the fit still falls back on the target masses.
    rng = numpy.random.default_rng( 5 )
    pos = rng.uniform( 0.1, 0.9, size = ( 40, 2 ) )
    src = SumOfDiracs( pos )
    dst = _scattered_target( 2, 4, seed = 6 )

    plan = OtProblem( src, dst ).solve( Iterative( max_iter = 200, tol = 1e-12 ) )

    assert all( h[ "min_measure" ] > 0 for h in plan.history ), min( h[ "min_measure" ] for h in plan.history )
    assert plan.converged, plan.stats
    got = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert numpy.allclose( got, _target_masses( plan ), atol = 1e-10 ), numpy.abs( got - _target_masses( plan ) ).max()


if test( "the_hessian_rows_are_the_jacobian_of_the_measures" ):
    # `PowerDiagram.hessian_rows` against the finite difference of the measures with respect to the weights,
    # on an image ( flat facets with constant density ): symmetric, rows summing to zero
    rng = numpy.random.default_rng( 31 )
    n = 25
    pos = rng.uniform( 0.1, 0.9, size = ( n, 2 ) )
    img = Image( values = rng.uniform( 0.2, 1, size = ( 10, 10 ) ), origin = [ 0.0, 0.0 ],
                 frame = [ [ 0.1, 0 ], [ 0, 0.1 ] ] )
    w0 = rng.uniform( -0.01, 0.01, n )
    pd = PowerDiagram( pos, w0, distribution = img, kernel_dtype = "FP64" )
    counts, ids, vals = pd.hessian_rows()
    H = numpy.zeros( ( n, n ) )
    for i in range( n ):
        for q in range( counts[ i ] ):
            if ids[ i, q ] >= 0:
                H[ i, ids[ i, q ] ] -= vals[ i, q ]
                H[ i, i ] += vals[ i, q ]
    assert numpy.abs( H - H.T ).max() < 1e-12
    assert numpy.abs( H.sum( axis = 1 ) ).max() < 1e-12
    h = 1e-6
    for j in ( 0, 8, 24 ):
        e = numpy.zeros( n ); e[ j ] = h
        pd.weights = w0 + e; mp = numpy.asarray( pd.measures ).reshape( -1 )
        pd.weights = w0 - e; mm = numpy.asarray( pd.measures ).reshape( -1 )
        assert numpy.abs( ( mp - mm ) / ( 2 * h ) - H[ :, j ] ).max() < 1e-7


if test( "the_hessian_rows_hold_in_3d_too" ):
    # in 3D the facet is a FACE, whose area comes from the accumulation of `LocalN::measure_3d`
    # ( `for_each_facet` ): same finite-difference check, without a distribution ( Lebesgue )
    rng = numpy.random.default_rng( 32 )
    n = 14
    pos = rng.uniform( 0.1, 0.9, size = ( n, 3 ) )
    w0 = rng.uniform( -0.01, 0.01, n )
    pd = PowerDiagram( pos, w0, boundaries = box_half_spaces( [ 0 ] * 3, [ 1 ] * 3 ), kernel_dtype = "FP64" )
    counts, ids, vals = pd.hessian_rows()
    H = numpy.zeros( ( n, n ) )
    for i in range( n ):
        for q in range( counts[ i ] ):
            if ids[ i, q ] >= 0:
                H[ i, ids[ i, q ] ] -= vals[ i, q ]
                H[ i, i ] += vals[ i, q ]
    assert numpy.abs( H - H.T ).max() < 1e-12
    h = 1e-6
    for j in ( 0, 5, 13 ):
        e = numpy.zeros( n ); e[ j ] = h
        pd.weights = w0 + e; mp = numpy.asarray( pd.measures ).reshape( -1 )
        pd.weights = w0 - e; mm = numpy.asarray( pd.measures ).reshape( -1 )
        assert numpy.abs( ( mp - mm ) / ( 2 * h ) - H[ :, j ] ).max() < 1e-7


if test( "newton_converges_quadratically_on_an_image" ):
    need( "cpu" )
    # on an image, a few steps suffice, the residual drops quadratically at the end, and a warm
    # start ( the weights of a neighboring cloud ) needs only two or three -- which is what a
    # reconstruction lives on ( `otrec.models.ProjectedDiracModel` )
    rng = numpy.random.default_rng( 41 )
    n = 300
    pos = rng.uniform( 0.05, 0.95, size = ( n, 2 ) )
    img = Image( values = 1 + 0.5 * rng.random( ( 24, 24 ) ), origin = [ 0.0, 0.0 ],
                 frame = [ [ 1 / 24, 0 ], [ 0, 1 / 24 ] ] )
    plan = OtProblem( SumOfDiracs( pos ), img ).solve( Iterative( max_iter = 60, tol = 1e-10 / n ) )
    res = [ h[ "max_abs_residual" ] * n for h in plan.history ]
    assert plan.converged and res[ -1 ] < 1e-9 and len( res ) < 30, ( plan.stats, res[ -1 ], len( res ) )
    # the last two steps: at least one order of magnitude each ( the quadratic phase )
    assert res[ -1 ] < 0.1 * res[ -2 ] < 0.01 * res[ -3 ]
    got  = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert numpy.allclose( got, _target_masses( plan ), atol = 1e-11 )
    # one diagram per step: no backtracking on this soft case
    assert plan.stats[ "nb_backtracks" ] == 0 and plan.stats[ "nb_diag" ] == len( res ), plan.stats

    warm = OtProblem( SumOfDiracs( pos + 1e-4 * rng.normal( size = pos.shape ) ), img ).solve( Iterative( max_iter = 60, tol = 1e-10 / n, weights0 = plan.weights ) )
    assert warm.stats[ "start" ] == "weights0" and len( warm.history ) <= 6, ( warm.stats, len( warm.history ) )

    # a warm start that EMPTIES a cell ( weights that have nothing to do with the cloud any more )
    # is dropped in favor of the Voronoi, and we still converge
    bad = OtProblem( SumOfDiracs( pos ), img ).solve( Iterative( max_iter = 60, tol = 1e-10 / n, weights0 = rng.uniform( -1, 1, n ) ) )
    assert bad.converged and bad.stats[ "start" ] == "voronoi", bad.stats


if test( "newton_starts_from_a_similarity_when_the_voronoi_has_empty_cells" ):
    need( "cpu" )
    # diracs OUTSIDE the domain ( their Voronoi cell restricted to the domain is empty ): the
    # start is the Voronoi of the cloud brought back into the domain by a similarity, written as a
    # power diagram of the original cloud -- all cells fed, and Newton converges
    rng = numpy.random.default_rng( 51 )
    n = 40
    pos = rng.uniform( -2, 3, size = ( n, 2 ) )
    img = Image( values = 1 + 0.3 * rng.random( ( 16, 16 ) ), origin = [ 0.0, 0.0 ],
                 frame = [ [ 1 / 16, 0 ], [ 0, 1 / 16 ] ] )
    voronoi = PowerDiagram( pos, numpy.zeros( n ), distribution = img, kernel_dtype = "FP64" )
    assert ( numpy.asarray( voronoi.measures ) == 0 ).any()      # the problem does exist
    plan = OtProblem( SumOfDiracs( pos ), img ).solve( Iterative( max_iter = 80, tol = 1e-10 / n, keep_weights = True ) )
    assert plan.stats[ "start" ] == "similarity", plan.stats
    assert plan.history[ 0 ][ "min_measure" ] > 0                # ... and the start solved it
    assert plan.converged, plan.stats
    # the similarity itself: its weights are those of the Voronoi of the contracted cloud ( up to the gauge )
    w = numpy.asarray( plan.history[ 0 ][ "weights" ] ).reshape( -1 )
    q = 0.5 + 0.8 * ( pos - ( pos.min( axis = 0 ) + pos.max( axis = 0 ) ) / 2 ) / numpy.ptp( pos, axis = 0 ).max()
    a = 0.8 / numpy.ptp( pos, axis = 0 ).max()
    ws = ( pos ** 2 ).sum( 1 ) - ( q ** 2 ).sum( 1 ) / a
    assert numpy.allclose( w - w[ 0 ], ws - ws[ 0 ] )


if test( "newton_works_in_3d" ):
    need( "cpu" )
    # the same promise in 3D, without a distribution ( Lebesgue on the cube ): the facet is a face
    rng = numpy.random.default_rng( 61 )
    n = 200
    pos = rng.uniform( 0.05, 0.95, size = ( n, 3 ) )
    # ( `SdotPlanNd` requires a distribution: a constant one-tile image is the Lebesgue measure )
    img = Image( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) )
    plan = OtProblem( SumOfDiracs( pos ), img ).solve( Iterative( max_iter = 60, tol = 1e-10 / n ) )
    assert plan.converged and len( plan.history ) < 30, ( plan.stats, len( plan.history ) )
    got = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert numpy.allclose( got, _target_masses( plan ), atol = 1e-11 )


if test( "the_log_residual_and_the_lin_residual_reach_the_same_plan" ):
    need( "cpu" )
    # `residual = "log"` ( the default: the log residual, then the lin one as soon as `max|a-nu|/nu <= 2` ), `"lin"`
    # ( KMT ) and `"power"` change the path, never the solution ( `solvers_des_familles` README § 24.5 )
    rng = numpy.random.default_rng( 81 )
    pos = rng.uniform( 0.1, 0.9, size = ( 60, 2 ) )
    src = SumOfDiracs( pos )
    dst = _scattered_target( 2, 4, seed = 6 )
    plans = { r: OtProblem( src, dst ).solve( Iterative( max_iter = 200, tol = 1e-12, continuation = "never",
                                                         tuning = Tuning( step = "trials", residual = r ) ) )
              for r in ( "lin", "log", "power" ) }
    for r, plan in plans.items():
        assert plan.converged, ( r, plan.stats )
        assert numpy.allclose( numpy.asarray( plan.weights ), numpy.asarray( plans[ "lin" ].weights ), atol = 1e-9 ), r
    assert plans[ "lin" ].stats[ "it_switch" ] == -1, plans[ "lin" ].stats
    assert plans[ "log" ].stats[ "it_switch" ] >= 0, plans[ "log" ].stats        # the switch happened


if test( "the_limits_step_reaches_the_same_plan_with_fewer_diagrams" ):
    need( "cpu" )
    # `step = "limits"` ( the default in 2D ): the same weights as KMT's trials ( `"trials"` ), and
    # fewer diagrams -- on the HARD case, where the trials back off ( `solvers_des_familles` README § 7 )
    rng = numpy.random.default_rng( 81 )
    pos = rng.uniform( 0.1, 0.9, size = ( 60, 2 ) )
    src = SumOfDiracs( pos )
    dst = _scattered_target( 2, 4, seed = 6 )
    # ( without continuation: it is the direct Newton, and its backtracking, that we compare here; with the
    # KMT residual -- `residual = "lin"` -- the case where the trials back off: the `log` one, the default,
    # already saves most of the backtracks of this small case, and the limits have nothing left to correct )
    a = OtProblem( src, dst ).solve( Iterative( max_iter = 200, tol = 1e-12, continuation = "never", tuning = Tuning( step = "trials", residual = "lin" ) ) )
    b = OtProblem( src, dst ).solve( Iterative( max_iter = 200, tol = 1e-12, continuation = "never", tuning = Tuning( step = "limits", residual = "lin" ) ) )
    assert a.converged and b.converged, ( a.stats, b.stats )
    assert numpy.allclose( numpy.asarray( a.weights ), numpy.asarray( b.weights ), atol = 1e-9 )
    assert b.stats[ "nb_diag" ] <= a.stats[ "nb_diag" ], ( a.stats[ "nb_diag" ], b.stats[ "nb_diag" ] )
    assert b.stats[ "nb_cell_lim" ] > 0 and b.stats[ "nb_backtracks" ] == 0, b.stats


if test( "the_continuation_solves_what_direct_newton_cannot" ):
    need( "cpu" )
    # NARROW bumps ( the bench's hard case, `solvers_des_familles` README § 9 ): cells with no
    # mass at the start, direct Newton STAGNATES; the width continuation ( `sdotplan/Continuation.h` )
    # converges, and `"auto"` triggers it by itself based on the smallest mass at the start
    rng = numpy.random.default_rng( 91 )
    pos = rng.uniform( 0, 1, size = ( 400, 2 ) )
    centres = numpy.array( [ [ 0.3, 0.3 ], [ 0.7, 0.35 ], [ 0.4, 0.75 ], [ 0.75, 0.7 ] ] )
    dst = SumOfGaussians( centres, 0.04 * numpy.array( [ 1, 0.7, 1.3, 1 ] ), weights = numpy.array( [ 0.35, 0.25, 0.25, 0.15 ] ) )
    direct = OtProblem( SumOfDiracs( pos ), dst ).solve( Iterative( max_iter = 100, continuation = "never", tuning = Tuning( mass_rtol = 1e-6 ) ) )
    assert not direct.converged, direct.stats
    plan = OtProblem( SumOfDiracs( pos ), dst ).solve( Iterative( max_iter = 100, tuning = Tuning( mass_rtol = 1e-6 ) ) )
    assert plan.converged and plan.stats[ "nb_continuation_steps" ] > 1, plan.stats
    got = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert numpy.allclose( got, _target_masses( plan ), rtol = 1e-5 ), numpy.abs( got / _target_masses( plan ) - 1 ).max()
    # the history carries the width of each step, decreasing down to 0
    ss = [ h[ "s" ] for h in plan.history ]
    assert ss[ 0 ] > 0 and ss[ -1 ] == 0 and all( b <= a for a, b in zip( ss, ss[ 1: ] ) )

    # an IMAGE too ( blurred on its grid ): an almost empty image, except for two spots
    values = numpy.full( ( 32, 32 ), 1e-6 )
    values[ 6:10, 6:10 ] = 1.0
    values[ 20:26, 18:24 ] = 0.7
    img = Image( values = values, origin = [ 0.0, 0.0 ], frame = [ [ 1 / 32, 0 ], [ 0, 1 / 32 ] ] )
    plan = OtProblem( SumOfDiracs( pos ), img ).solve( Iterative( max_iter = 100, tuning = Tuning( mass_rtol = 1e-6 ) ) )
    assert plan.converged and plan.stats[ "nb_continuation_steps" ] > 1, plan.stats
    got = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert numpy.allclose( got, _target_masses( plan ), rtol = 1e-5 ), numpy.abs( got / _target_masses( plan ) - 1 ).max()


if test( "the_domain_comes_from_the_density_alone" ):
    need( "cpu" )
    # Gaussians: the domain is the support they DECLARE ( `centers +- 6 sigma` ), not
    # the envelope of the diracs -- diracs drawn in a corner of the domain have cells that go
    # far from them, and the transport sends them there ( the barycenters leave the box of the diracs )
    rng = numpy.random.default_rng( 3 )
    pos = rng.uniform( 0.4, 0.6, size = ( 60, 2 ) )
    dst = SumOfGaussians( numpy.array( [ [ 0.5, 0.5 ] ] ), numpy.array( [ 0.15 ] ), weights = numpy.array( [ 1.0 ] ) )
    plan = OtProblem( SumOfDiracs( pos ), dst ).solve( Iterative( max_iter = 100, tol = 1e-12 ) )
    assert plan.converged, plan.stats
    pd = plan._pd
    assert pd.box_min.is_defined and not pd.bnd_offsets.is_defined
    assert numpy.allclose( numpy.asarray( pd.box_min ), [ 0.5 - 0.9 ] * 2 ) and numpy.allclose( numpy.asarray( pd.box_max ), [ 0.5 + 0.9 ] * 2 )
    assert abs( plan.stats[ "domain_mass" ] - 1 ) < 1e-7
    _, bary, _ = plan.transport()
    bary = numpy.asarray( bary )
    assert bary.min() < 0.3 and bary.max() > 0.7, ( bary.min(), bary.max() )
    # without a declared support, no domain: we say so
    try:
        OtProblem( SumOfDiracs( pos ), SumOfGaussians( numpy.array( [ [ 0.5, 0.5 ] ] ), numpy.array( [ 0.15 ] ), support_sigmas = None ) ).solve()
        assert False, "an unbounded domain should have been refused"
    except ValueError:
        pass


if test( "the_plain_storage_gives_the_same_plan" ):
    need( "cpu" )
    # `accelerator = "plain"`: the same weights as the BSP tree, up to rounding ( the acceleration
    # only changes what the diagram costs )
    rng = numpy.random.default_rng( 71 )
    n = 60
    pos = rng.uniform( 0.05, 0.95, size = ( n, 2 ) )
    img = Image( values = 1 + 0.5 * rng.random( ( 8, 8 ) ), origin = [ 0.0, 0.0 ], frame = [ [ 1 / 8, 0 ], [ 0, 1 / 8 ] ] )
    a = OtProblem( SumOfDiracs( pos ), img ).solve( Iterative( max_iter = 60, tol = 1e-12 ) )
    b = OtProblem( SumOfDiracs( pos ), img ).solve( Iterative( max_iter = 60, tol = 1e-12, tuning = Tuning( accelerator = "plain" ) ) )
    assert a.converged and b.converged
    assert numpy.allclose( numpy.asarray( a.weights ), numpy.asarray( b.weights ), atol = 1e-10 )


# -- the moments, and what a transport cost draws from them ---------------------------------

if test( "moments_are_the_closed_forms" ):
    # ONE dirac in the unit square: its cell is the square, whose moments are known --
    # mass 1, barycenter ( 1/2, 1/2 ), `int |x|^2 = 2/3`. And on an image with ONE pixel lit, the
    # barycenter is the center of that pixel: this also checks the orientation of the grid
    # ( `values[ i, j ]` <-> `origin + i frame[ 0 ] + j frame[ 1 ]` ).
    pd = PowerDiagram( numpy.array( [ [ 0.3, 0.6 ] ] ), boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ),
                       kernel_dtype = "FP64" )
    mass, first, second = pd.moments
    assert abs( float( numpy.asarray( mass ).reshape( -1 )[ 0 ] ) - 1 ) < 1e-12
    assert numpy.allclose( numpy.asarray( first ).reshape( -1 ), [ 0.5, 0.5 ], atol = 1e-12 )
    assert abs( float( numpy.asarray( second ).reshape( -1 )[ 0 ] ) - 2 / 3 ) < 1e-12

    values = numpy.zeros( ( 4, 3 ) )
    values[ 3, 1 ] = 1.0
    img = Image( values = values, origin = [ 0.0, 0.0 ], frame = [ [ 0.5, 0.0 ], [ 0.0, 0.25 ] ] )
    pd = PowerDiagram( numpy.array( [ [ 0.3, 0.6 ] ] ), distribution = img, kernel_dtype = "FP64" )
    mass, first, second = pd.moments
    m = float( numpy.asarray( mass ).reshape( -1 )[ 0 ] )
    bary = numpy.asarray( first ).reshape( -1 ) / m
    assert abs( m - 1 ) < 1e-12, m                                    # normalized
    assert numpy.allclose( bary, [ 3.5 * 0.5, 1.5 * 0.25 ], atol = 1e-12 ), bary

    # several diracs: the order-0 moments are the measures, and the barycenters stay inside
    # the square, their weighted mean being the center of mass of the domain
    rng = numpy.random.default_rng( 11 )
    pos = rng.uniform( 0.1, 0.9, size = ( 15, 2 ) )
    pd = PowerDiagram( pos, boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ), kernel_dtype = "FP64" )
    mass, first, second = pd.moments
    m = numpy.asarray( mass ).reshape( -1 )
    mx = numpy.asarray( first ).reshape( -1, 2 )
    assert numpy.allclose( m, numpy.asarray( pd.measures ).reshape( -1 ), atol = 1e-12 )
    assert numpy.allclose( mx.sum( axis = 0 ), [ 0.5, 0.5 ], atol = 1e-12 )
    assert abs( float( numpy.asarray( second ).reshape( -1 ).sum() ) - 2 / 3 ) < 1e-12


if test( "the_transport_cost_derives_by_the_envelope_theorem" ):
    need( "cpu" )
    # `cost_and_position_grad`: the derivative of the cost with respect to the dirac positions, at the
    # fitted weights, against the finite difference of the cost itself ( each evaluation refitting
    # its weights, restarting from the previous ones ). On an IMAGE: its moments are exact ( those of a
    # Gaussian go through the adaptive quadrature, up to `rtol` ).
    rng = numpy.random.default_rng( 21 )
    pos = rng.uniform( 0.2, 0.8, size = ( 8, 2 ) )
    img = Image( values = 1 + 0.5 * rng.random( ( 12, 12 ) ), origin = [ 0.0, 0.0 ], frame = [ [ 1 / 12, 0 ], [ 0, 1 / 12 ] ] )

    def plan_at( p, w0 = None ):
        return OtProblem( SumOfDiracs( p ), img ).solve( Iterative( weights0 = w0, max_iter = 100, tol = 1e-14 ) )

    plan = plan_at( pos )
    cost, grad = plan.cost_and_position_grad()
    cost, grad = float( cost ), numpy.asarray( grad )
    assert cost > 0 and numpy.isfinite( grad ).all()

    # the cost is indeed `W_2^2`: the same thing as `sum_i m_i |p_i - b_i|^2 + sum_i var_i` -- we
    # at least check the bound `cost >= sum_i m_i |p_i - b_i|^2`
    _, bary, m = plan.transport()
    bary, m = numpy.asarray( bary ), numpy.asarray( m ).reshape( -1 )
    assert cost >= float( ( m * ( ( pos - bary ) ** 2 ).sum( axis = 1 ) ).sum() ) - 1e-12

    h = 1e-4
    for i, c in ( ( 0, 0 ), ( 3, 1 ), ( 7, 0 ) ):
        dp = numpy.zeros_like( pos ); dp[ i, c ] = h
        fd = ( float( plan_at( pos + dp, plan.weights ).cost ) - float( plan_at( pos - dp, plan.weights ).cost ) ) / ( 2 * h )
        assert abs( fd - grad[ i, c ] ) < 1e-4 * max( 1.0, abs( fd ) ), ( i, c, fd, grad[ i, c ] )


if test( "the_multigrid_reaches_the_cholesky_plan_in_2d_and_3d" ):
    need( "cpu" )
    # `linear_solver = "mg"` ( `sdotplan/Multigrid.h` ): the in-house multigrid. `mg_stop` small, so that the hierarchy has SEVERAL
    # levels on a test-sized problem ( the default stops coarsening under 1000 unknowns ): aggregation by the tree order,
    # smoothed prolongation, Galerkin product, Chebyshev, the recycled start and the reused hierarchy are all exercised.
    # The solver changes the path, never the solution: the same plan as Cholesky to the Newton tolerance.
    img3 = Image( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) )
    img2 = Image( values = numpy.ones( ( 1, 1 ) ), origin = [ 0.0, 0.0 ], frame = numpy.eye( 2 ) )
    for d, n, img in ( ( 2, 600, img2 ), ( 3, 500, img3 ) ):
        rng = numpy.random.default_rng( 91 + d )
        src = SumOfDiracs( rng.uniform( 0.05, 0.95, size = ( n, d ) ) )
        ref = OtProblem( src, img ).solve( Iterative( max_iter = 60, tol = 1e-12, continuation = "never",
                                                      tuning = Tuning( linear_solver = "cholesky", step = "trials" ) ) )
        for options in ( dict(), dict( mg_pack = 4, mg_recycle = 0, mg_rebuild = 1 ) ):
            got = OtProblem( src, img ).solve( Iterative( max_iter = 60, tol = 1e-12, continuation = "never",
                                                          tuning = Tuning( linear_solver = "mg", step = "trials", linear_tol = 1e-9, mg_stop = 40, **options ) ) )
            assert ref.converged and got.converged, ( d, options, ref.stats, got.stats )
            assert got.stats[ "lin_nb_hierarchies" ] >= 1 and got.stats[ "lin_nb_iter" ] > 0, got.stats
            assert numpy.allclose( numpy.asarray( got.weights ), numpy.asarray( ref.weights ), atol = 1e-8 ), ( d, options )
            assert numpy.allclose( numpy.asarray( got.cell_masses ), numpy.asarray( ref.cell_masses ), atol = 1e-10 ), ( d, options )


if test( "the_multigrid_without_a_tree_order_and_the_unknown_solver" ):
    need( "cpu" )
    # the `plain` storage has no tree: the aggregation then follows the order of the identifiers ( worse, but exact )
    rng = numpy.random.default_rng( 17 )
    pos = rng.uniform( 0.1, 0.9, size = ( 300, 2 ) )
    img2 = Image( values = numpy.ones( ( 1, 1 ) ), origin = [ 0.0, 0.0 ], frame = numpy.eye( 2 ) )
    a = OtProblem( SumOfDiracs( pos ), img2 ).solve( Iterative( max_iter = 60, tol = 1e-12, tuning = Tuning( linear_solver = "cholesky" ) ) )
    b = OtProblem( SumOfDiracs( pos ), img2 ).solve( Iterative( max_iter = 60, tol = 1e-12, tuning = Tuning( accelerator = "plain", linear_solver = "mg", linear_tol = 1e-9, mg_stop = 30 ) ) )
    assert a.converged and b.converged, ( a.stats, b.stats )
    assert numpy.allclose( numpy.asarray( a.weights ), numpy.asarray( b.weights ), atol = 1e-8 )
    try:
        OtProblem( SumOfDiracs( pos ), img2 ).solve( Iterative( tuning = Tuning( linear_solver = "nope" ) ) )
    except ValueError:
        pass
    else:
        raise AssertionError( "an unknown linear_solver must be refused" )


# -- what we LOOK AT -------------------------------------------------------------------------
#
#   ./run experiment test_SdotPlanNd

def _report( p, plan, pos, stem ):
    """Shared by the two experiments below: the same pair ( curve, animation ), the same
    reading of the history."""
    last = plan.history[ -1 ]
    print( f"  { last[ 'step' ] } steps, { plan.stats[ 'nb_diag' ] } diagrams, { plan.stats[ 'status' ] }"
           f", max residual { last[ 'max_abs_residual' ]:.3e}"
           f", final min measure { last[ 'min_measure' ]:.3e}" )

    write_convergence_html(
        { "l2 residual":                      [ h[ "residual_l2" ] for h in plan.history ],
          "max residual":                     [ h[ "max_abs_residual" ] for h in plan.history ],
          "minimum measure (never 0)":        [ h[ "min_measure" ] for h in plan.history ] },
        p.out_dir / f"{ stem }_convergence.html",
        title = f"SdotPlanNd 2D -- { len( pos ) } diracs" )

    idx = numpy.unique( numpy.linspace(
        0, len( plan.history ) - 1, min( 40, len( plan.history ) ) ).astype( int ) )

    viz = Visualizer( title = f"SdotPlanNd 2D, { len( pos ) } diracs -- convergence", frame_axis = "step" )
    for j, i in enumerate( idx ):
        if j:
            viz.new_frame( int( plan.history[ i ][ "step" ] ) )
        w = plan.history[ i ][ "weights" ]
        plan.power_diagram( w ).add_to_viz( viz )
        viz.add_points( pos, color = "#ffffff" )
    viz.write_html( p.out_dir / f"{ stem }_anim.html" )


if p := experiment( "ot 2D newton",
                    nb_points    = Param( 30, help = "number of diracs" ),
                    nb_gaussians = Param( 2, help = "number of Gaussians in the target" ),
                    max_iter     = Param( 100, help = "number of steps" ),
                    seed         = Param( 5, help = "random draw seed" ) ):
    # what `SdotPlanNd` does: START from zero weights ( the Voronoi -- each cell takes its purely
    # geometric share ) and make them SLIDE until each cell weighs, against the target
    # density, exactly what its dirac weighs. The convergence curve says WHETHER it converges
    # and how fast; the animation shows HOW: the PLANES slide from one step to the next,
    # not the seeds. SOFT target ( `_overlapping_target` ).
    pos = numpy.random.default_rng( p.seed ).uniform( 0.1, 0.9, size = ( p.nb_points, 2 ) )
    src = SumOfDiracs( pos )
    dst = _overlapping_target( 2, p.nb_gaussians, seed = p.seed + 1 )

    plan = OtProblem( src, dst ).solve( Iterative( max_iter = p.max_iter, keep_weights = True ), verbose = True )
    _report( p, plan, pos, "ot_2d_newton" )


if p := experiment( "ot 2D newton scattered",
                    nb_points    = Param( 40, help = "number of diracs" ),
                    nb_gaussians = Param( 4, help = "number of bumps, separate and narrow" ),
                    max_iter     = Param( 200, help = "number of steps" ),
                    seed         = Param( 5, help = "random draw seed" ) ):
    # the HARD case: narrow, separate bumps ( `_scattered_target` ). The `minimum
    # measure` curve is the one that matters here: it starts almost null ( a dirac in a density
    # desert, at the Voronoi ) and must CLIMB BACK without ever touching 0 again.
    pos = numpy.random.default_rng( p.seed ).uniform( 0.1, 0.9, size = ( p.nb_points, 2 ) )
    src = SumOfDiracs( pos )
    dst = _scattered_target( 2, p.nb_gaussians, seed = p.seed + 1 )

    plan = OtProblem( src, dst ).solve( Iterative( max_iter = p.max_iter, keep_weights = True ), verbose = True )
    _report( p, plan, pos, "ot_2d_newton_scattered" )
