
from errand import Param, experiment, test



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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    need( "cpu" )
    # NARROW bumps ( the bench's hard case, `solvers_des_familles` README § 9 ): cells with no
    # mass at the start, direct Newton STAGNATES; the width continuation ( `sdotplan/Continuation.h` )
    # converges, and `"auto"` triggers it by itself based on the smallest mass at the start
    rng = numpy.random.default_rng( 91 )
    pos = rng.uniform( 0, 1, size = ( 400, 2 ) )
    centres = numpy.array( [ [ 0.3, 0.3 ], [ 0.7, 0.35 ], [ 0.4, 0.75 ], [ 0.75, 0.7 ] ] )
    dst = SumOfGaussians( centres, 0.04 * numpy.array( [ 1, 0.7, 1.3, 1 ] ), weights = numpy.array( [ 0.35, 0.25, 0.25, 0.15 ] ) )
    direct = OtProblem( SumOfDiracs( pos ), dst ).solve( Iterative( max_iter = 100, continuation = "never", on_failure = "ignore", tuning = Tuning( mass_rtol = 1e-6 ) ) )
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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
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


# -- ON THE CARD ( `gpu/Newton2D.cuh` ) ------------------------------------------------------------
#
# On a CUDA device, a 2D transport in a box against a constant density is solved by the card's Newton ( the
# majorants, the cells, the laplacian, the linear solver, the step: all on the card, one ffi call ). These tests
# skip themselves without a CUDA device. The plan is checked against what does not depend on the solver: the
# measures of its cells through the GENERIC path of the diagram ( double, `use_card_cells = False` ), and the plans
# of the other variants ( step, linear solver, kernel float ) -- the CPU's own plans are compared by the bench
# ( `bench_newton --save-weights` on both environments ).

def _card():
    import loom
    return bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )


def _box_target():
    return Image( values = numpy.ones( ( 1, 1 ) ), origin = [ 0.0, 0.0 ], frame = numpy.eye( 2 ) )


def _generic_measures( pos, w ):
    """the measures of the cells of `( pos, w )` through the generic double path ( not the card's kernel )"""
    pd = PowerDiagram( pos, numpy.asarray( w, dtype = float ).reshape( -1 ), boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ),
                       kernel_dtype = "FP64" )
    pd.use_card_cells = False
    return numpy.asarray( pd.measures.value ).reshape( -1 )


if test( "the_card_solves_the_transport" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    rng = numpy.random.default_rng( 101 )
    n = 4000
    pos = rng.uniform( 0.001, 0.999, size = ( n, 2 ) )
    nu = rng.uniform( 0.5, 1.5, n )
    nu /= nu.sum()
    plans = {}
    for name, precision, tuning in ( ( "limits mg double", "fp64", Tuning( step = "limits" ) ),
                                     ( "trials mg double", "fp64", Tuning( step = "trials" ) ),
                                     ( "limits cg double", "fp64", Tuning( step = "limits", linear_solver = "cg", linear_tol = 1e-8 ) ),
                                     ( "limits host cholesky", "fp64", Tuning( step = "limits", linear_solver = "cholesky" ) ),
                                     ( "limits host mg", "fp64", Tuning( step = "limits", linear_solver = "mg", linear_host = True ) ),
                                     ( "limits mg float", "fp32", Tuning( step = "limits" ) ),
                                     ( "limits mixed ( auto )", "auto", Tuning( step = "limits" ) ),
                                     ( "limits mg double levels", "fp64", Tuning( step = "limits", mg_precision = "double" ) ),
                                     ( "limits mg plain aggregation", "fp64", Tuning( step = "limits", mg_smoothed = 0 ) ) ):
        plan = OtProblem( SumOfDiracs( pos, nu ), _box_target() ).solve( Iterative( tol = 1e-10 / n, max_iter = 60, precision = precision, tuning = tuning ) )
        assert plan.converged, ( name, plan.stats )
        w = numpy.asarray( plan.weights ).reshape( -1 )
        assert w[ 0 ] == 0, name                                       # the gauge
        m = numpy.asarray( plan.cell_masses ).reshape( -1 )
        assert numpy.abs( m - nu ).max() < 1e-10 / n, ( name, numpy.abs( m - nu ).max() * n )
        # the cells of these weights, measured by the generic path, are the targets
        g = _generic_measures( pos, w )
        assert numpy.abs( g - nu ).max() < 1e-8 / n, ( name, numpy.abs( g - nu ).max() * n )
        # the moments: barycentres inside the box, the cost positive, the history readable
        bary = numpy.asarray( plan.barycenters )
        assert bary.min() > 0 and bary.max() < 1 and float( plan.cost ) > 0, name
        assert plan.history[ -1 ][ "max_abs_residual" ] == plan.stats[ "residual" ], name
        plans[ name ] = w
        assert plan.stats[ "it_double" ] == ( -1 if precision == "auto" else 0 ), ( name, plan.stats[ "it_double" ] )   # ( no switch here )
        print( f"  { name }: { plan.stats[ 'nb_iter' ] } it, { plan.stats[ 'nb_diag' ] } diagrams, { plan.stats[ 'lin_nb_iter' ] } linear it, "
               f"residual { plan.stats[ 'residual' ] * n :.1e} ( relative ), limits rounds { plan.stats[ 'nb_limit_rounds' ] }" )
    ref = plans[ "limits mg double" ]
    for name, w in plans.items():
        assert numpy.abs( w - ref ).max() < 1e-9 / n, ( name, numpy.abs( w - ref ).max() * n )


def _card_csr( pos ):
    """the laplacian of the Voronoi diagram of `pos` in the unit box, on the card, in ranks: `( row, col, val, dia )`"""
    from sdot import AaBsp, PowerDiagram as Pd
    n = len( pos )
    pd = Pd( pos, boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ), kernel_dtype = "FP64", accelerator = AaBsp( pos ) )
    out = pd._card_cells( facets = True, moments = False )
    row = numpy.asarray( out[ "row" ].raw ).reshape( -1 ).astype( numpy.int64 )[ :n + 1 ]
    col = numpy.asarray( out[ "col" ].raw ).reshape( -1 ).astype( numpy.int64 )[ :row[ n ] ]
    val = numpy.asarray( out[ "val" ].raw ).reshape( -1 )[ :row[ n ] ]
    dia = numpy.asarray( out[ "dia" ].raw ).reshape( -1 )[ :n ]
    return row, col, val, dia


def _csr_product( row, col, val, dia, x ):
    """`L x` ( `y_i = dia_i x_i - sum_e val_e x_( col_e )` )"""
    n = len( dia )
    off = numpy.zeros( n )
    nz = numpy.repeat( numpy.arange( n ), numpy.diff( row ) )
    numpy.add.at( off, nz, val * x[ col ] )
    return dia * x - off


if test( "the_card_linear_solver_solves" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    from sdot.SdotPlanNd import _card_linear_solves
    rng = numpy.random.default_rng( 106 )
    # a small system against a dense solve ( the multigrid's variants and CG ), then a larger one ( the hash table of the
    # Galerkin product: coarse levels over 4096 rows ) against its own residual
    for n, dense in ( ( 1500, True ), ( 40000, False ) ):
        pos = rng.uniform( 0.001, 0.999, size = ( n, 2 ) )
        row, col, val, dia = _card_csr( pos )
        b = rng.normal( size = ( 3, n ) )
        b -= b.mean( axis = 1, keepdims = True )                     # the range of the laplacian
        ref = None
        if dense:
            L = numpy.diag( dia )
            for i in range( n ):
                L[ i, col[ row[ i ]:row[ i + 1 ] ] ] -= val[ row[ i ]:row[ i + 1 ] ]
            ref = numpy.linalg.lstsq( L, b.T, rcond = None )[ 0 ].T
            ref -= ref.mean( axis = 1, keepdims = True )
        for name, kw in ( ( "mg", {} ), ( "mg double levels", dict( precision = "double" ) ), ( "mg plain", dict( smoothed = 0 ) ),
                          ( "mg smoothed twice", dict( smoothed = 2 ) ), ( "cg", dict( method = "cg" ) ) ):
            for tol in ( 1e-6, 1e-10 ):
                d, its = _card_linear_solves( row, col, val, dia, b, tol = tol, **kw )
                assert ( its >= 0 ).all(), ( n, name, its )
                for j in range( len( b ) ):
                    assert abs( d[ j ].mean() ) < 1e-12 * numpy.abs( d[ j ] ).max(), ( n, name, "the gauge" )
                    res = numpy.linalg.norm( _csr_product( row, col, val, dia, d[ j ] ) - b[ j ] ) / numpy.linalg.norm( b[ j ] )
                    assert res < 2 * tol, ( n, name, tol, j, res )
                    if ref is not None and tol == 1e-10:
                        assert numpy.abs( d[ j ] - ref[ j ] ).max() < 1e-6 * numpy.abs( ref[ j ] ).max(), ( n, name, j )
                print( f"  n { n } { name } tol { tol :.0e}: iterations { its.tolist() }" )
        # the same calls give the same numbers, to the bit
        d1, _ = _card_linear_solves( row, col, val, dia, b )
        d2, _ = _card_linear_solves( row, col, val, dia, b )
        assert numpy.array_equal( d1, d2 ), n
        # RECYCLING: a right-hand side already solved ( kept in the subspace ) starts at its solution; without recycling,
        # the same system again is the same solve, to the bit
        bb = numpy.stack( [ b[ 0 ], b[ 1 ], b[ 0 ] ] )
        d, its = _card_linear_solves( row, col, val, dia, bb, recycle = 2 )
        assert its[ 2 ] <= 2 and its[ 0 ] > 3, its
        d, its = _card_linear_solves( row, col, val, dia, bb, recycle = 0 )
        assert its[ 2 ] == its[ 0 ] and numpy.array_equal( d[ 2 ], d[ 0 ] ), its


if test( "the_card_majorants_bound_the_weights" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    # the tree the solve leaves behind carries the majorants of the FITTED weights, redone on the card ( `Majorant2D.cuh` ):
    # each node bounds the weights of its seeds, `w( y ) <= a . y + b`, and the affine ones are tight
    rng = numpy.random.default_rng( 102 )
    n = 6000
    pos = rng.uniform( 0.001, 0.999, size = ( n, 2 ) )
    nu = 1 + 0.8 * numpy.sin( 6 * pos[ :, 0 ] ) * numpy.cos( 5 * pos[ :, 1 ] )        # a smooth target: weights with a slope
    plan = OtProblem( SumOfDiracs( pos, nu ), _box_target() ).solve( Iterative( tol = 1e-10 / n, max_iter = 60 ) )
    assert plan.converged, plan.stats
    pd = plan._pd
    tree = pd.tree
    w = numpy.asarray( plan.weights ).reshape( -1 )[ numpy.asarray( tree.seed_indices ).reshape( -1 ) ]
    p = pos[ numpy.asarray( tree.seed_indices ).reshape( -1 ) ]
    beg, end = numpy.asarray( tree.node_begin ).reshape( -1 ), numpy.asarray( tree.node_end ).reshape( -1 )
    wa, wb = numpy.asarray( tree.node_wa ).reshape( -1, 2 ), numpy.asarray( tree.node_wb ).reshape( -1 )
    nb_affine = 0
    for i in range( len( beg ) ):
        if end[ i ] <= beg[ i ]:
            continue
        sl = slice( beg[ i ], end[ i ] )
        bound = p[ sl ] @ wa[ i ] + wb[ i ]
        assert ( w[ sl ] <= bound ).all(), ( i, ( w[ sl ] - bound ).max() )
        nb_affine += bool( wa[ i ].any() )
        spread = w[ sl ].max() - w[ sl ].min()
        assert ( bound - w[ sl ] ).min() <= spread + 1e-5 * ( abs( wb[ i ] ) + spread + 1e-300 ), i
    assert nb_affine > 0.3 * len( beg ), nb_affine                   # the smooth potential: most nodes keep their slope


if test( "the_card_solve_runs_under_jit" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    import jax
    rng = numpy.random.default_rng( 103 )
    n = 3000
    pos = rng.uniform( 0.001, 0.999, size = ( n, 2 ) )
    nu = rng.uniform( 0.5, 1.5, n )

    def solve( masses ):
        plan = OtProblem( SumOfDiracs( pos, masses ), _box_target() ).solve( Iterative( tol = 1e-10 / n, max_iter = 60 ) )
        return plan.weights.raw, plan.cell_masses.raw, plan.stats[ "nb_diag" ]

    we, me, de = solve( nu )
    wj, mj, dj = jax.jit( solve )( nu )
    assert numpy.abs( numpy.asarray( wj ) - numpy.asarray( we ) ).max() == 0, numpy.abs( numpy.asarray( wj ) - numpy.asarray( we ) ).max()
    assert numpy.abs( numpy.asarray( mj ) - numpy.asarray( me ) ).max() == 0
    assert float( dj ) == float( de ), ( dj, de )
    # a second call of the compiled function: the same numbers ( a solve that is the same at every run )
    wj2, _, _ = jax.jit( solve )( nu )
    assert numpy.array_equal( numpy.asarray( wj2 ), numpy.asarray( wj ) )


if test( "the_card_solve_builds_its_tree_in_the_jitted_call" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    # the POSITIONS traced: the tree is built on the card inside the jitted program ( `gpu/Bsp2D.cuh` ), and the solve is the
    # eager one to the bit; a cold start and a warm one ( given weights: their majorants on the card too )
    import jax
    rng = numpy.random.default_rng( 107 )
    n = 5000
    pos = rng.uniform( 0.001, 0.999, size = ( n, 2 ) )
    nu = rng.uniform( 0.5, 1.5, n )
    w0 = 1e-3 * ( pos[ :, 0 ] - 0.5 ) / n

    # ( the masses are an argument too: a closure constant would be normalized by XLA's fused reduction under the jit and
    # op by op in eager, 1e-19 apart -- nothing to do with the tree )
    def solve( p, m, start ):
        plan = OtProblem( SumOfDiracs( p, m ), _box_target() ).solve( Iterative( tol = 1e-10 / n, max_iter = 60, weights0 = start ) )
        return plan.weights.raw, plan.stats[ "nb_diag" ], plan.stats[ "status" ]

    for start in ( None, w0 ):
        we, de, se = solve( pos, nu, start )
        assert se == "converged", se
        f = jax.jit( lambda p, m: solve( p, m, start ) )
        wj, dj, _ = f( pos, nu )
        assert numpy.array_equal( numpy.asarray( wj ), numpy.asarray( we ) ), numpy.abs( numpy.asarray( wj ) - numpy.asarray( we ) ).max()
        assert float( dj ) == float( de ), ( dj, de )
        # other positions through the same compiled function: another tree, the eager result again
        pos2 = numpy.clip( pos + 1e-3 * rng.normal( size = pos.shape ), 0.001, 0.999 )
        assert numpy.array_equal( numpy.asarray( f( pos2, nu )[ 0 ] ), numpy.asarray( solve( pos2, nu, start )[ 0 ] ) )


if test( "the_card_solve_on_rings_in_batches_and_past_the_limit" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    # seeds in the middle of rings of 600 ( 600-gons: past the shared-memory pass, into the fourth one, in global memory ),
    # six of them through FOUR slots: the fourth pass of every diagram works in batches, within its fixed budget ( no
    # capacity, no second run ), and the solve converges, eager and under jit
    import sys
    import jax
    from loom.drivers.CallArg_Errors import KernelFailure
    bsp = sys.modules[ "sdot.PowerDiagram_Bsp" ]
    rng = numpy.random.default_rng( 104 )
    k, r = 600, 0.1
    centres = numpy.array( [ [ 0.2, 0.25 ], [ 0.5, 0.25 ], [ 0.8, 0.25 ], [ 0.2, 0.75 ], [ 0.5, 0.75 ], [ 0.8, 0.75 ] ] )
    parts = [ centres ]
    for c in centres:
        a = 2 * numpy.pi * ( numpy.arange( k ) + rng.uniform( -0.1, 0.1, k ) ) / k
        parts.append( c + r * numpy.stack( [ numpy.cos( a ), numpy.sin( a ) ], axis = 1 ) )
    back = rng.uniform( 0.001, 0.999, size = ( 3000, 2 ) )
    parts.append( back[ numpy.linalg.norm( back[ :, None, : ] - centres[ None ], axis = 2 ).min( axis = 1 ) > 1.3 * r ] )
    pos = numpy.concatenate( parts )
    n = len( pos )
    original = bsp.PowerDiagram_Bsp.card_overflow_warps
    try:
        bsp.PowerDiagram_Bsp.card_overflow_warps = 4
        for precision in ( "fp64", "fp32" ):
            def solve( masses ):
                plan = OtProblem( SumOfDiracs( pos, masses ), _box_target() ).solve( Iterative( tol = 1e-9 / n, max_iter = 100, precision = precision ) )
                return plan
            plan = solve( numpy.ones( n ) )
            assert plan.converged, ( precision, plan.stats )
            w = numpy.asarray( plan.weights ).reshape( -1 )
            g = _generic_measures( pos, w )
            # ( the generic path is a reference on rings too since its cut picks the run of the farthest vertex: `Local2::cut_impl` )
            assert numpy.abs( g - 1 / n ).max() < 1e-7 / n, ( precision, numpy.abs( g - 1 / n ).max() * n )
            wj = jax.jit( lambda m: solve( m ).weights.raw )( numpy.ones( n ) )
            assert numpy.array_equal( numpy.asarray( wj ).reshape( -1 ), w ), precision
    finally:
        bsp.PowerDiagram_Bsp.card_overflow_warps = original
    # past the vertex limit: a `KernelFailure` naming a centre, eager and under jit ( never a NaN, never a second run )
    original = bsp.PowerDiagram_Bsp.card_max_vertices
    try:
        bsp.PowerDiagram_Bsp.card_max_vertices = 256
        for run in ( lambda m: OtProblem( SumOfDiracs( pos, m ), _box_target() ).solve( Iterative( tol = 1e-9 / n, max_iter = 100 ) ).weights.raw, ):
            for fn in ( run, jax.jit( run ) ):
                try:
                    numpy.asarray( fn( numpy.ones( n ) ) )
                    raise AssertionError( "a cell past the vertex limit went through" )
                except AssertionError:
                    raise
                except Exception as e:
                    assert "more than 256 vertices" in str( e ) and any( f"seed { c }" in str( e ) for c in range( len( centres ) ) ), str( e )
                    assert fn is not run or isinstance( e, KernelFailure ), type( e )
    finally:
        bsp.PowerDiagram_Bsp.card_max_vertices = original


if test( "the_card_refuses_what_it_does_not_solve" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    # an image on a ROTATED grid: its support is not a box ( and its rows are not the card's rows )
    rng = numpy.random.default_rng( 105 )
    pos = rng.uniform( 0.2, 0.8, size = ( 50, 2 ) )
    c, s = numpy.cos( 0.3 ), numpy.sin( 0.3 )
    for dst in ( Image( values = 1 + rng.random( ( 4, 4 ) ), origin = [ 0.0, 0.0 ], frame = [ [ 0.25 * c, 0.25 * s ], [ -0.25 * s, 0.25 * c ] ] ), ):
        try:
            OtProblem( SumOfDiracs( pos ), dst ).solve()
        except NotImplementedError as e:
            assert "box" in str( e ) or "distribution" in str( e ), str( e )
        else:
            raise AssertionError( "a domain that is not a box must be refused on the card" )


# -- ON THE CARD, A DENSITY ( `gpu/Density2D.cuh`, `gpu/DensityHost2D.cuh` ) -------------------------------------------------
#
# The card's plan against THE CPU's: the same problem solved by `sdotplan/Solve.h` in a subprocess on the CPU device
# ( `LOOM_DEVICE=cpu`: one process has one device ). With `step = "trials"` the two Newtons take the same decisions, so the
# same iterations and diagrams; the plans agree to the solve's tolerance, and the cells of the card's weights, measured by
# the GENERIC path ( `use_card_cells = False`, cutting the cells ), are the targets.

_CPU_SOLVE = """
import pickle, sys, numpy
from sdot import Image, Iterative, OtProblem, SumOfDiracs, SumOfGaussians, Tuning
d = pickle.load( open( sys.argv[ 1 ], "rb" ) )
kind, spec = d[ "target" ]
dst = Image( **spec ) if kind == "image" else SumOfGaussians( **spec )
plan = OtProblem( SumOfDiracs( d[ "pos" ], d[ "nu" ] ), dst ).solve( Iterative( **d[ "it" ], tuning = Tuning( **d[ "tun" ] ) ) )
out = dict( weights = numpy.asarray( plan.weights ).reshape( -1 ), stats = plan.stats, cost = float( plan.cost ),
            bary = numpy.asarray( plan.barycenters ), masses = numpy.asarray( plan.cell_masses ).reshape( -1 ),
            s = [ h[ "s" ] for h in plan.history ] )
pickle.dump( out, open( sys.argv[ 2 ], "wb" ) )
"""


def _cpu_solve( pos, nu, target, it, tun ):
    """the same solve on the CPU device, in a subprocess: a dict ( `weights`, `stats`, `cost`, `bary`, `masses`, `s` )"""
    import os, pickle, subprocess, sys, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        fi, fo = os.path.join( tmp, "in.pkl" ), os.path.join( tmp, "out.pkl" )
        pickle.dump( dict( pos = pos, nu = nu, target = target, it = it, tun = tun ), open( fi, "wb" ) )
        env = dict( os.environ, LOOM_DEVICE = "cpu" )
        r = subprocess.run( [ sys.executable, "-c", _CPU_SOLVE, fi, fo ], env = env, capture_output = True, text = True )
        assert r.returncode == 0, r.stdout[ -3000: ] + r.stderr[ -3000: ]
        return pickle.load( open( fo, "rb" ) )


def _make_target( target ):
    kind, spec = target
    return Image( **spec ) if kind == "image" else SumOfGaussians( **spec )


def _generic_density_measures( pos, w, target ):
    """the measures of the cells of `( pos, w )` against the density, through the GENERIC path ( the cells cut by the
    pixels, the gaussians' exact corners ), in the domain of the target"""
    from sdot import OtProblem
    dst = _make_target( target )
    prob = OtProblem( SumOfDiracs( pos ), dst )
    dirs, offs = prob.domain
    pd = PowerDiagram( pos, numpy.asarray( w, dtype = float ).reshape( -1 ), boundaries = ( dirs, offs ), distribution = prob.target,
                       kernel_dtype = "FP64" )
    pd.use_card_cells = False
    return numpy.asarray( pd.measures.value ).reshape( -1 )


def _card_vs_cpu( name, pos, nu, target, it, tun, counts = True, wtol = 1e-7, mtol = 1e-8 ):
    """the card's solve and the CPU's of the same problem: converged both, the same counts ( `counts` ), the same plan. The
    moments: exact on both sides for an image; for gaussians the card's are closed forms and the CPU's an adaptive
    quadrature ( `PointwiseDensity`, ~1e-4 relative: checked against a brute-force quadrature, the card's are right )"""
    n = len( pos )
    plan = OtProblem( SumOfDiracs( pos, nu ), _make_target( target ) ).solve( Iterative( **it, tuning = Tuning( **tun ) ) )
    cpu = _cpu_solve( pos, nu, target, it, tun )
    st, cs = plan.stats, cpu[ "stats" ]
    w, wc = numpy.asarray( plan.weights ).reshape( -1 ), cpu[ "weights" ]
    gap = numpy.abs( w - wc ).max() / max( numpy.abs( wc ).max(), 1e-300 )
    print( f"  { name }: card { st[ 'status' ] } { st[ 'nb_iter' ] } it / { st[ 'nb_diag' ] } diag / { st[ 'nb_continuation_steps' ] } stages, "
           f"cpu { cs[ 'status' ] } { cs[ 'nb_iter' ] } it / { cs[ 'nb_diag' ] } diag / { cs[ 'nb_continuation_steps' ] } stages; "
           f"weights gap { gap :.1e} ( relative to max |w| ), cost { float( plan.cost ) :.10e} vs { cpu[ 'cost' ] :.10e}, "
           f"domain mass { st[ 'domain_mass' ] :.12f} vs { cs[ 'domain_mass' ] :.12f}" )
    assert st[ "status" ] == "converged" and cs[ "status" ] == "converged", ( name, st, cs )
    assert st[ "nb_continuation_steps" ] == cs[ "nb_continuation_steps" ], name
    assert st[ "start" ] == cs[ "start" ], ( name, st[ "start" ], cs[ "start" ] )
    assert abs( st[ "domain_mass" ] - cs[ "domain_mass" ] ) < 1e-10, name
    if counts:
        assert ( st[ "nb_iter" ], st[ "nb_diag" ] ) == ( cs[ "nb_iter" ], cs[ "nb_diag" ] ), ( name, st[ "nb_iter" ], st[ "nb_diag" ], cs[ "nb_iter" ], cs[ "nb_diag" ] )
        hs = [ h[ "s" ] for h in plan.history ]
        assert numpy.allclose( hs, cpu[ "s" ], rtol = 1e-14, atol = 0 ), name
    assert gap < wtol, ( name, gap )
    exact = target[ 0 ] == "image"
    bgap = numpy.abs( numpy.asarray( plan.barycenters ) - cpu[ "bary" ] ).max() * n ** 0.5      # relative to the size of a cell
    print( f"    cost gap { abs( float( plan.cost ) / cpu[ 'cost' ] - 1 ) :.1e} ( relative ), barycentres gap { bgap :.1e} ( relative to a cell )" )
    assert abs( float( plan.cost ) / cpu[ "cost" ] - 1 ) < ( 1e-9 if exact else 1e-3 ), name
    # ( the CPU's quadrature, capped at 8 bisections, on the huge cells of the tails and on the needles that reach a narrow
    # bump: the card's closed forms were checked against a brute-force quadrature of the same polygons to 1e-7 )
    assert bgap < ( 1e-8 if exact else 0.25 ), name
    # the cells of the card's weights, measured by the generic path: the targets
    g = _generic_density_measures( pos, w, target )
    tm = numpy.asarray( plan.target_masses ).reshape( -1 )
    assert numpy.abs( g - tm ).max() < mtol / n, ( name, numpy.abs( g - tm ).max() * n )
    return plan, cpu


def _image_target( nx, ny, values ):
    return ( "image", dict( values = values, origin = [ 0.0, 0.0 ], frame = [ [ 1 / nx, 0 ], [ 0, 1 / ny ] ] ) )


if test( "the_card_solves_an_image_as_the_cpu" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    rng = numpy.random.default_rng( 201 )
    n = 1500
    pos = rng.uniform( 0.001, 0.999, size = ( n, 2 ) )
    nu = rng.uniform( 0.5, 1.5, n )
    nu /= nu.sum()
    # a soft image ( 3:1 ), on a grid that is not square, directly
    soft = _image_target( 24, 17, 1 + 2 * rng.random( ( 24, 17 ) ) )
    it = dict( tol = 1e-10 / n, max_iter = 80, continuation = "never" )
    plan, _ =_card_vs_cpu( "soft image, trials", pos, nu, soft, dict( it, precision = "fp64" ), dict( step = "trials" ) )
    # the float kernel ( its vertices re-solved in double ) and the limits step: the same plan, other counts
    for prec, step in ( ( "fp32", "trials" ), ( "fp64", "limits" ), ( "auto", "limits" ) ):
        p2 = OtProblem( SumOfDiracs( pos, nu ), _make_target( soft ) ).solve( Iterative( **dict( it, precision = prec ), tuning = Tuning( step = step ) ) )
        assert p2.converged, ( prec, step, p2.stats )
        gap = numpy.abs( numpy.asarray( p2.weights ) - numpy.asarray( plan.weights ) ).max() / numpy.abs( numpy.asarray( plan.weights ) ).max()
        print( f"  soft image, { step }, { prec }: { p2.stats[ 'nb_iter' ] } it / { p2.stats[ 'nb_diag' ] } diag, gap { gap :.1e}" )
        assert gap < 1e-7, ( prec, step, gap )
    # a contrasted image with a HOLE ( zeros ): the width continuation, forced and automatic
    vals = 0.1 + 0.9 * rng.random( ( 32, 32 ) )
    vals[ 4:12, 18:28 ] = 0
    vals[ 20:23, : ] += 4
    hard = _image_target( 32, 32, vals )
    _card_vs_cpu( "image with a hole, continuation always, trials", pos, nu, hard, dict( it, continuation = "always", precision = "fp64" ),
                  dict( step = "trials" ) )
    _card_vs_cpu( "image with a hole, continuation auto, limits", pos, nu, hard, dict( it, continuation = "auto", precision = "fp64" ),
                  dict( step = "limits" ), counts = False )


if test( "the_card_solves_gaussians_as_the_cpu" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    rng = numpy.random.default_rng( 202 )
    n = 1500
    nu = rng.uniform( 0.5, 1.5, n )
    nu /= nu.sum()
    it = dict( tol = 1e-10 / n, max_iter = 100, precision = "fp64" )
    # overlapping gaussians: no continuation needed
    soft = ( "gauss", dict( positions = 0.5 + rng.uniform( -0.12, 0.12, size = ( 3, 2 ) ), sigmas = rng.uniform( 0.18, 0.24, 3 ),
                            weights = rng.uniform( 0.6, 1.4, 3 ) ) )
    pos = rng.uniform( 0.3, 0.7, size = ( n, 2 ) )
    _card_vs_cpu( "overlapping gaussians, trials", pos, nu, soft, dict( it, continuation = "never" ), dict( step = "trials" ) )
    # narrow separate bumps ( § 9 of the old campaign, sigma = 0.04 ): the width continuation, automatic
    hard = ( "gauss", dict( positions = numpy.array( [ [ 0.26, 0.30 ], [ 0.72, 0.26 ], [ 0.34, 0.74 ], [ 0.76, 0.70 ] ] ),
                            sigmas = 0.04 * numpy.array( [ 1, 0.7, 1.3, 1 ] ), weights = numpy.array( [ 0.35, 0.25, 0.25, 0.15 ] ) ) )
    pos = rng.uniform( 0.05, 0.95, size = ( n, 2 ) )
    _card_vs_cpu( "narrow gaussians, continuation auto, trials", pos, nu, hard, dict( it, continuation = "auto", tol = 1e-9 / n ),
                  dict( step = "trials", mass_rtol = 0 ), wtol = 1e-6 )
    plan, cpu = _card_vs_cpu( "narrow gaussians, continuation always, limits", pos, nu, hard, dict( it, continuation = "always", tol = 1e-9 / n ),
                              dict( step = "limits" ), counts = False, wtol = 1e-6 )
    ss = [ h[ "s" ] for h in plan.history ]
    assert ss[ 0 ] > 0 and ss[ -1 ] == 0 and all( b <= a for a, b in zip( ss, ss[ 1: ] ) )


if test( "the_card_density_solves_run_under_jit" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    import jax
    rng = numpy.random.default_rng( 203 )
    n = 2000
    pos = rng.uniform( 0.001, 0.999, size = ( n, 2 ) )
    nu = rng.uniform( 0.5, 1.5, n )
    vals = 0.1 + rng.random( ( 20, 30 ) )
    vals[ 3:8, 10:20 ] = 0
    targets = { "image": _image_target( 20, 30, vals ),
                "gaussians": ( "gauss", dict( positions = numpy.array( [ [ 0.3, 0.3 ], [ 0.7, 0.6 ] ] ), sigmas = numpy.array( [ 0.05, 0.08 ] ),
                                              weights = numpy.array( [ 0.6, 0.4 ] ) ) ) }
    for name, target in targets.items():
        def solve( masses ):
            plan = OtProblem( SumOfDiracs( pos, masses ), _make_target( target ) ).solve( Iterative( tol = 1e-9 / n, max_iter = 100, continuation = "auto" ) )
            return plan.weights.raw, plan.cost.raw, plan.stats[ "nb_diag" ], plan.stats[ "status" ], plan.stats[ "nb_continuation_steps" ]
        we, ce, de, se, ke = solve( nu )
        assert se == "converged" and ke > 1, ( name, se, ke )
        wj, cj, dj, _, _ = jax.jit( solve )( nu )
        assert numpy.array_equal( numpy.asarray( wj ), numpy.asarray( we ) ), ( name, numpy.abs( numpy.asarray( wj ) - numpy.asarray( we ) ).max() )
        assert float( cj ) == float( ce ) and float( dj ) == float( de ), name
        print( f"  { name }: { de } diagrams, { ke } stages, jit == eager" )


# THE CARD'S MEMORY ( `CardMemory.py` ): what the solve takes from XLA's pool, its model, and the refusal. XLA's pool waits
# ~10 s before it refuses an allocation; a solve that does not fit must raise a clear error AT ONCE ( the model, checked
# before the call ), and at worst after ONE wait ( loom's `Scratch`: a refusal is final for the call ). The pool is made
# small in a subprocess ( `XLA_PYTHON_CLIENT_MEM_FRACTION`: one process, one pool ).

def _card_scratch_model( n, it ):
    from sdot.CardMemory import card_solve_bytes
    from sdot.SdotPlanNd import SdotPlanNd
    parts = card_solve_bytes( n, **SdotPlanNd._card_memory_kw( n, it ) )
    return sum( v for k, v in parts.items() if k not in ( "tree", "outputs" ) )


if test( "the_card_memory_model_follows_the_solve" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    rng = numpy.random.default_rng( 301 )
    n = 200_000
    pos = rng.uniform( 0.001, 0.999, size = ( n, 2 ) )
    for name, it in ( ( "mixed", Iterative( tol = 1e-10 / n, max_iter = 30 ) ),
                      ( "fp32", Iterative( tol = 1e-10 / n, max_iter = 30, precision = "fp32" ) ),
                      ( "fp64, double levels", Iterative( tol = 1e-10 / n, max_iter = 30, precision = "fp64", tuning = Tuning( mg_precision = "double" ) ) ),
                      ( "fp64, cg", Iterative( tol = 1e-10 / n, max_iter = 30, precision = "fp64", tuning = Tuning( linear_solver = "cg", linear_tol = 1e-8 ) ) ) ):
        plan = OtProblem( SumOfDiracs( pos ), _box_target() ).solve( it )
        assert plan.converged, ( name, plan.stats )
        took, model = plan.stats[ "scratch_bytes" ], _card_scratch_model( n, it )
        print( f"  { name }: took { took / n :.1f} bytes per seed, model { model / n :.1f}" )
        assert abs( model - took ) < 0.08 * took, ( name, took, model )


if test( "the_card_memory_check_counts_the_jitted_program" ):
    # pure Python ( the pool is faked ): while tracing, the pool cannot show the jitted program's own buffers ( positions,
    # masses, the weights returned: 40 bytes per seed ); the check counts them, and a pool that holds the eager solve just
    # makes the jitted one give up its recycled solutions -- at 1e7 seeds on a 7.85 GB pool, before: RESOURCE_EXHAUSTED
    # after XLA's 10 s, after: the solve runs with no recycled solution ( `calibration_lmo_today.md`, step 10 )
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    import sdot.CardMemory as cm
    from sdot.SdotPlanNd import SdotPlanNd
    n, it = 10_000_000, Iterative( tol = 1e-6 / 10_000_000, max_iter = 30 )
    kw = SdotPlanNd._card_memory_kw( n, it )
    eager = sum( cm.card_solve_bytes( n, **kw ).values() )
    assert sum( cm.card_solve_bytes( n, **kw, jitted = True ).values() ) - eager == 40 * n
    original = cm.card_pool
    try:
        cm.card_pool = lambda: ( eager + 10 * n, 0 )
        assert SdotPlanNd._check_card_memory( n, it ) is None                      # the default: two recycled solutions
        assert SdotPlanNd._check_card_memory( n, it, jitted = True ) == 0          # 40 B / seed more: none of them ( 16 B each )
        cm.card_pool = lambda: ( eager - 33 * n, 0 )
        try:
            SdotPlanNd._check_card_memory( n, it, jitted = True )
            raise AssertionError( "a jitted solve that does not fit went through" )
        except MemoryError:
            pass
    finally:
        cm.card_pool = original


if test( "the_card_finish_store_by_chunks_gives_the_same_plan" ):
    # `SDOT_CARD_DEFER_CAP`: the float kernel's finish store smaller than the cloud, so the first pass runs by chunks and the
    # second leaves its cells by position ( `Card::run` ): the same cells, so the same plan to the bit
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    import os, pickle, subprocess, sys, tempfile
    rng = numpy.random.default_rng( 302 )
    n = 6000
    pos = rng.uniform( 0.001, 0.999, size = ( n, 2 ) )
    nu = rng.uniform( 0.5, 1.5, n )
    script = """
import pickle, sys, numpy
from sdot import Image, Iterative, OtProblem, SumOfDiracs
pos, nu = pickle.load( open( sys.argv[ 1 ], "rb" ) )
box = Image( values = numpy.ones( ( 1, 1 ) ), origin = [ 0.0, 0.0 ], frame = numpy.eye( 2 ) )
out = {}
for precision in ( "fp32", "auto" ):
    plan = OtProblem( SumOfDiracs( pos, nu ), box ).solve( Iterative( tol = 1e-10 / len( nu ), max_iter = 60, precision = precision ) )
    out[ precision ] = ( numpy.asarray( plan.weights ).reshape( -1 ), plan.stats[ "nb_diag" ], plan.stats[ "scratch_bytes" ] )
pickle.dump( out, open( sys.argv[ 2 ], "wb" ) )
"""
    runs = {}
    with tempfile.TemporaryDirectory() as tmp:
        fi = os.path.join( tmp, "in.pkl" )
        pickle.dump( ( pos, nu ), open( fi, "wb" ) )
        for cap in ( "", "1000" ):
            fo = os.path.join( tmp, f"out{ cap }.pkl" )
            r = subprocess.run( [ sys.executable, "-c", script, fi, fo ], env = dict( os.environ, SDOT_CARD_DEFER_CAP = cap ),
                                capture_output = True, text = True )
            assert r.returncode == 0, r.stdout[ -3000: ] + r.stderr[ -3000: ]
            runs[ cap ] = pickle.load( open( fo, "rb" ) )
    for precision in ( "fp32", "auto" ):
        ( w0, d0, b0 ), ( w1, d1, b1 ) = runs[ "" ][ precision ], runs[ "1000" ][ precision ]
        assert numpy.array_equal( w0, w1 ) and d0 == d1, ( precision, numpy.abs( w0 - w1 ).max(), d0, d1 )
        assert b1 < b0, ( precision, b0, b1 )       # ( 148 bytes per cell of the store: 5000 cells less )
        print( f"  { precision }: the same plan, { ( b0 - b1 ) / 1e3 :.0f} kB less" )


_OOM_SOLVE = """
import os, sys, time, numpy
from sdot import Image, Iterative, OtProblem, SumOfDiracs
dim = int( os.environ.get( "SDOT_TEST_DIM", "2" ) )
box = Image( values = numpy.ones( ( 1, ) * dim ), origin = [ 0.0 ] * dim, frame = numpy.eye( dim ) )
def solve( n, jit = False ):
    pos = numpy.random.default_rng( 0 ).uniform( 0.001, 0.999, size = ( n, dim ) )
    it = Iterative( tol = 1e-6 / n, max_iter = 30 )
    if jit:
        import jax
        return jax.jit( lambda m: OtProblem( SumOfDiracs( pos, m ), box ).solve( it ).weights.raw )( numpy.ones( n ) ).block_until_ready()
    return OtProblem( SumOfDiracs( pos ), box ).solve( it ).weights
solve( 2000 )                                    # ( the kernels compiled, the pool opened )
for jit in sys.argv[ 2: ]:
    t = time.perf_counter()
    try:
        solve( int( sys.argv[ 1 ] ), jit == "jit" )
        print( "RESULT none", time.perf_counter() - t )
    except Exception as e:
        print( "RESULT", type( e ).__name__, time.perf_counter() - t, str( e ).replace( chr( 10 ), " " ) )
"""


def _small_pool_fraction( pool_bytes = 0.5e9 ):
    """The memory fraction that gives XLA's pool about `pool_bytes`, whatever the card ( 0.05 if it cannot be read )"""
    import subprocess
    try:
        out = subprocess.run( [ "nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0" ],
                              capture_output = True, text = True, timeout = 30 ).stdout
        return f"{ pool_bytes / ( float( out.split()[ 0 ] ) * 2**20 ) :.4f}"
    except ( OSError, ValueError, IndexError, subprocess.TimeoutExpired ):
        return "0.05"


def _oom_runs( n, modes, **env ):
    import os, subprocess, sys
    r = subprocess.run( [ sys.executable, "-c", _OOM_SOLVE, str( n ), *modes ], capture_output = True, text = True,
                        env = dict( os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION = _small_pool_fraction(), **env ), timeout = 900 )
    lines = [ ( l.split( " ", 3 ) + [ "" ] )[ 1:4 ] for l in r.stdout.splitlines() if l.startswith( "RESULT " ) ]
    assert len( lines ) == len( modes ), r.stdout[ -3000: ] + r.stderr[ -3000: ]
    return [ ( kind, float( t ), msg ) for kind, t, msg in lines ]


if test( "the_card_refuses_a_solve_that_does_not_fit_at_once" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    n = 1_000_000                                # ~1 GB for the solve, the pool ~0.5 GB
    # the model, before anything is launched: eager and while tracing
    for mode, ( kind, t, msg ) in zip( ( "eager", "jit" ), _oom_runs( n, [ "eager", "jit" ] ) ):
        print( f"  { mode }: { kind } after { t :.2f} s: { msg[ :200 ] }..." )
        assert kind == "MemoryError" and t < 5, ( mode, kind, t, msg )
        assert f"solve of { n } seeds needs about" in msg and "XLA_PYTHON_CLIENT_MEM_FRACTION" in msg and "LOOM_DEVICE=cpu" in msg, msg
    # without the check, the call itself: ONE wait of XLA's pool ( ~10 s ), then the refusal, with what was refused and why
    ( kind, t, msg ), = _oom_runs( n, [ "eager" ], SDOT_CARD_MEMORY_CHECK = "0" )
    print( f"  without the check: { kind } after { t :.2f} s: { msg[ :300 ] }..." )
    assert kind != "none" and t < 40, ( kind, t, msg )
    assert "RESOURCE_EXHAUSTED" in msg and "the scratch pool refused" in msg and f"Newton solve of { n } seeds" in msg, msg


if test( "a_seed_with_more_than_512_neighbours_is_solved" ):
    import numpy
    from sdot import OtProblem, SumOfDiracs, SumOfGaussians
    # two seeds off a line of 1000 aligned ones: each has a cell with ONE FACET PER SEED OF THE LINE ( ~900 ). The facets of a
    # cell were accumulated in a fixed array of 512 and a cell past it was taken for a scratch overflow: the scratch doubled
    # for ever ( 19 GB, no output ). The solve must converge, and the cells must tile the domain
    line = numpy.linspace( 0, 1, 1000 )
    pos = numpy.concatenate( [ [ [ 0, 1 ], [ 1, 0 ] ], numpy.stack( [ line, line ], axis = 1 ) ] )
    plan = OtProblem( SumOfDiracs( pos ), SumOfGaussians( [ [ 0.5, 0.5 ] ], [ 0.15 ] ) ).solve()
    assert plan.converged, plan.stats
    assert abs( float( plan.cost ) - 0.0916 ) < 1e-3, float( plan.cost )


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
        plan.power_diagram( w ).add_to_viz( viz, seed_color = "#ffffff" )
    viz.write_html( p.out_dir / f"{ stem }_anim.html" )

    # the quick ways, on the plan itself: the fitted diagram, its seeds, the transport to the barycenters
    plan.write_html( p.out_dir / f"{ stem }_plan.html", transport = True )
    print( "  pvd  :", plan.write_pvd( p.out_dir / f"{ stem }_plan.pvd", transport = True ) )


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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
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
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    pos = numpy.random.default_rng( p.seed ).uniform( 0.1, 0.9, size = ( p.nb_points, 2 ) )
    src = SumOfDiracs( pos )
    dst = _scattered_target( 2, p.nb_gaussians, seed = p.seed + 1 )

    plan = OtProblem( src, dst ).solve( Iterative( max_iter = p.max_iter, keep_weights = True ), verbose = True )
    _report( p, plan, pos, "ot_2d_newton_scattered" )


# ---- THE AGGREGATION of near-coincident seeds ( `sdotplan/Aggregation.h` ) ----------------------------------------------
#
# Two seeds `delta` apart are split by a plane whose offset is `( w_i - w_j ) / 2 delta`: below some `delta`, no weight
# vector reaches the tolerance there, and the solve stagnated. Merged, the tests read the mass of the cluster, the plane
# inside it placed as well as the doubles let it ( `stats[ "residual_full" ]` ). Exact duplicates are merged too, their
# cells kept empty, their representative carrying their mass.

def _degenerate_clouds( d, seed ):
    """`{ name: ( positions, groups ) }`: a uniform cloud with seeds moved onto others; `groups` the expected clusters
    ( `( members, exact )`, `exact`: equal positions )"""
    rng = numpy.random.default_rng( seed )
    n = 400 if d == 2 else 600
    base = rng.uniform( 0.05, 0.95, size = ( n, d ) )
    unit = numpy.ones( d ) / d ** 0.5
    side = numpy.eye( d )[ 0 ]
    def moved( *moves ):
        pos = base.copy()
        groups = []
        for members, offsets in moves:
            for m, off in zip( members[ 1: ], offsets ):
                pos[ m ] = pos[ members[ 0 ] ] + off
            groups.append( ( members, all( not numpy.any( off ) for off in offsets ) ) )
        return pos, groups
    # ( a triple is not ALIGNED here: the middle seed of three aligned 1e-10 apart starts with a Voronoi cell of 5e-9 of its
    # target, whose log step is refused: see `a_refused_log_step_falls_back_on_the_lin_residual` )
    return {
        "exact pair":         moved( ( [ 5, 7 ], [ 0.0 * unit ] ) ),
        "exact triple":       moved( ( [ 3, 11, 29 ], [ 0.0 * unit, 0.0 * unit ] ) ),
        "pair at 1e-9":       moved( ( [ 5, 7 ], [ 1e-9 * unit ] ) ),
        "pair at 1e-12":      moved( ( [ 8, 2 ], [ 1e-12 * unit ] ) ),
        "triple at 1e-10":    moved( ( [ 4, 9, 13 ], [ 1e-10 * unit, 2e-10 * side ] ) ),
        "pair and duplicate": moved( ( [ 1, 6 ], [ 3e-10 * unit ] ), ( [ 12, 15, 30 ], [ 0.0 * unit, 2e-10 * side ] ) ),
    }


def _check_aggregated( name, plan, pos, groups, tol ):
    """the plan of a cloud whose `groups` the doubles cannot separate: converged ( aggregated, unless only exact duplicates ),
    the clusters found, each cluster's mass its target, every other cell its own, an exact duplicate's cell empty"""
    n = len( pos )
    st = plan.stats
    nu = numpy.asarray( plan.target_masses ).reshape( -1 )
    m = numpy.asarray( plan.cell_masses ).reshape( -1 )
    print( f"  { name }: { st[ 'status' ] }, { st[ 'nb_iter' ] } it / { st[ 'nb_diag' ] } diag, aggregated { st[ 'residual' ] * n :.1e} "
           f"full { st[ 'residual_full' ] * n :.1e} ( relative ), { st[ 'aggregation' ] }" )
    assert plan.converged, ( name, st )
    only_exact = all( exact for _, exact in groups )
    assert st[ "status" ] == ( "converged" if only_exact else "converged (aggregated)" ), ( name, st[ "status" ] )
    assert st[ "residual" ] <= tol, ( name, st[ "residual" ] )
    cl = numpy.asarray( plan.clusters ).reshape( -1 )
    alone = numpy.ones( n, dtype = bool )
    for members, exact in groups:
        assert ( cl[ members ] == min( members ) ).all(), ( name, members, cl[ members ] )
        alone[ members ] = False
        # the cluster's mass is its target ( the test reads each member's SHARE of the cluster's gap: `k` members, `k tol` )
        assert abs( m[ members ].sum() - nu[ members ].sum() ) <= len( members ) * tol * ( 1 + 1e-9 ), ( name, members, m[ members ].sum() - nu[ members ].sum() )
        if exact:                                # the duplicates' cells are empty, their representative holds the mass
            assert ( m[ members[ 1: ] ] == 0 ).all() and ( numpy.asarray( plan.weights ).reshape( -1 )[ members[ 1: ] ] <
                                                            numpy.asarray( plan.weights ).reshape( -1 )[ members[ 0 ] ] ).all(), name
    assert ( cl[ alone ] == numpy.arange( n )[ alone ] ).all(), name
    assert numpy.abs( m[ alone ] - nu[ alone ] ).max() <= tol, ( name, numpy.abs( m[ alone ] - nu[ alone ] ).max() )
    assert st[ "nb_clusters" ] == len( groups ) and st[ "nb_aggregated" ] == sum( len( g ) for g, _ in groups ), ( name, st[ "aggregation" ] )


if test( "the_aggregation_merges_what_the_doubles_cannot_separate" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    need( "cpu" )
    for d, steps in ( ( 2, ( "trials", "limits" ) ), ( 3, ( "trials", ) ) ):
        for name, ( pos, groups ) in _degenerate_clouds( d, 17 ).items():
            n = len( pos )
            tol = 1e-11 / n
            for step in steps:
                it = Iterative( tol = tol, max_iter = 60, tuning = Tuning( step = step ) )
                plan = OtProblem( SumOfDiracs( pos ), _box_target() if d == 2 else Image( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0 ] * 3, frame = numpy.eye( 3 ) ) ).solve( it )
                _check_aggregated( f"{ d }D { name } { step }", plan, pos, groups, tol )
                # without the aggregation: the floor ( or worse: two equal points count their cell twice )
                it = Iterative( tol = tol, max_iter = 60, aggregate = False, on_failure = "ignore", tuning = Tuning( step = step ) )
                off = OtProblem( SumOfDiracs( pos ), _box_target() if d == 2 else Image( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0 ] * 3, frame = numpy.eye( 3 ) ) ).solve( it )
                assert not off.converged and off.stats[ "aggregation" ] == "off" and off.clusters is None, ( name, off.stats[ "status" ] )


if test( "a_refused_log_step_falls_back_on_the_lin_residual" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    need( "cpu" )
    # three ALIGNED seeds 1e-10 apart: the middle Voronoi cell holds 5e-9 of its target. The log residual asks it for
    # `a ( c - log x )`, 1e-10 next to the others' 1e-3, which the linear solver loses; and at the Voronoi start ( weights 0 )
    # the aggregation cannot see the cluster yet. Its merit refuses every step ( it stagnated in 1 it / 35 diagrams, the merit
    # flat under the 1e-8 floor of `g( a / nu )` ): the iteration is done again in lin, which converges, aggregated
    rng = numpy.random.default_rng( 17 )
    n = 400
    pos = rng.uniform( 0.05, 0.95, size = ( n, 2 ) )
    unit = numpy.ones( 2 ) / 2 ** 0.5
    pos[ 9 ] = pos[ 4 ] + 1e-10 * unit
    pos[ 13 ] = pos[ 4 ] - 2e-10 * unit
    tol = 1e-11 / n
    for step in ( "trials", "limits" ):
        runs = { r: OtProblem( SumOfDiracs( pos ), _box_target() ).solve( Iterative( tol = tol, continuation = "never", tuning = Tuning( step = step, residual = r ) ) )
                 for r in ( "log", "lin" ) }
        _check_aggregated( f"aligned triple { step } log", runs[ "log" ], pos, [ ( [ 4, 9, 13 ], False ) ], tol )
        _check_aggregated( f"aligned triple { step } lin", runs[ "lin" ], pos, [ ( [ 4, 9, 13 ], False ) ], tol )
        log, lin = runs[ "log" ].stats, runs[ "lin" ].stats
        assert log[ "it_switch" ] == 0, ( step, log[ "it_switch" ] )
        # the same iterations as lin; the refused log trials on top ( 3 diagrams with the limits, 8 with the trials )
        assert log[ "nb_iter" ] == lin[ "nb_iter" ] and log[ "nb_diag" ] <= lin[ "nb_diag" ] + 10, ( step, log[ "nb_diag" ], lin[ "nb_diag" ] )


if test( "the_aggregation_changes_nothing_without_such_seeds" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    need( "cpu" )
    # a cloud whose pairs the doubles separate: the same plan to the bit, the same counts, no cluster -- in 2D ( both steps )
    # and in 3D, and with a tolerance under what any mass can reach ( `MEASURE_PRECISION`: no merge either )
    for d, step, tol in ( ( 2, "limits", 1e-12 ), ( 2, "trials", 1e-17 ), ( 3, "trials", 1e-12 ) ):
        rng = numpy.random.default_rng( 23 + d )
        n = 3000
        pos = rng.uniform( 0.001, 0.999, size = ( n, d ) )
        target = _box_target() if d == 2 else Image( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0 ] * 3, frame = numpy.eye( 3 ) )
        plans = [ OtProblem( SumOfDiracs( pos ), target ).solve( Iterative( tol = tol / n, max_iter = 60, aggregate = agg, on_failure = "ignore", tuning = Tuning( step = step ) ) )
                  for agg in ( True, False ) ]
        on, off = plans
        assert tol < 1e-15 or on.stats[ "status" ] == "converged", on.stats
        assert on.clusters is None and on.stats[ "aggregation" ] == "none" and on.stats[ "duplicates" ] == "checked", on.stats
        assert numpy.array_equal( numpy.asarray( on.weights ), numpy.asarray( off.weights ) ), d
        for k in ( "nb_iter", "nb_diag", "residual", "residual_full" ):
            assert on.stats[ k ] == off.stats[ k ], ( d, k, on.stats[ k ], off.stats[ k ] )


if test( "the_card_aggregates_as_the_cpu" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    import jax
    for name, ( pos, groups ) in _degenerate_clouds( 2, 17 ).items():
        n = len( pos )
        # ( a pair 1e-12 apart: the card's cells, cut in each seed's frame, place the merged cell to ~5e-11 of its mass -- the
        # CPU's to 5e-14 )
        tol = ( 1e-9 if "1e-12" in name else 1e-11 ) / n
        for step, precision in ( ( "limits", "fp64" ), ( "trials", "fp64" ), ( "limits", "auto" ) ):
            it = Iterative( tol = tol, max_iter = 60, precision = precision, tuning = Tuning( step = step ) )
            plan = OtProblem( SumOfDiracs( pos ), _box_target() ).solve( it )
            _check_aggregated( f"card { name } { step } { precision }", plan, pos, groups, tol )
            # the cells of the card's weights, measured by the generic path: each cluster's mass, every other cell's
            g = _generic_measures( pos, numpy.asarray( plan.weights ).reshape( -1 ) )
            nu = numpy.asarray( plan.target_masses ).reshape( -1 )
            alone = numpy.ones( n, dtype = bool )
            for members, _ in groups:
                alone[ members ] = False
                assert abs( g[ members ].sum() - nu[ members ].sum() ) < 1e-8 / n, ( name, g[ members ].sum() - nu[ members ].sum() )
            assert numpy.abs( g[ alone ] - nu[ alone ] ).max() < 1e-8 / n, name
        # under `jax.jit` ( the masses traced, the positions constant: the duplicates are still found ): the eager plan
        def solve( masses ):
            plan = OtProblem( SumOfDiracs( pos, masses ), _box_target() ).solve( Iterative( tol = tol, max_iter = 60 ) )
            return plan.weights.raw, plan.stats[ "nb_clusters" ], plan.stats[ "status" ]
        we, ce, se = solve( numpy.ones( n ) )
        wj, cj, sj = jax.jit( solve )( numpy.ones( n ) )
        assert numpy.array_equal( numpy.asarray( wj ), numpy.asarray( we ) ) and float( cj ) == float( ce ) == len( groups ), name
    # a cloud without such seeds: the same plan as without the aggregation, to the bit ( at 1e-12 relative, the bound of
    # `TAU_REL_MIN`, the mixed kernel merges one ordinary pair of this cloud: the floor of its weights is then within 4 of the
    # tolerance -- the status stays `converged` )
    rng = numpy.random.default_rng( 29 )
    pos = rng.uniform( 0.001, 0.999, size = ( 4000, 2 ) )
    on, off = [ OtProblem( SumOfDiracs( pos ), _box_target() ).solve( Iterative( tol = 1e-10 / 4000, max_iter = 60, aggregate = agg ) ) for agg in ( True, False ) ]
    assert on.clusters is None and on.stats[ "status" ] == "converged", on.stats
    assert numpy.array_equal( numpy.asarray( on.weights ), numpy.asarray( off.weights ) )
    assert ( on.stats[ "nb_iter" ], on.stats[ "nb_diag" ] ) == ( off.stats[ "nb_iter" ], off.stats[ "nb_diag" ] )


# -- ON THE CARD IN 3D ( `gpu/Newton2D.cuh` with `Cell3D.cuh`'s cells ) ------------------------------------------------------
#
# The same solve in 3D: a box, a constant density, the `trials` step ( the `limits` one is 2D, on the CPU as on the card ).
# Checked against THE CPU's plan of the same problem ( `_cpu_solve`, a subprocess on the CPU device ): the same status, the
# weights within the tolerance, the cost and the barycentres; and against the measures of the card's weights through the
# PLAIN storage ( the generic 3D path with the BSP tree is wrong on a few cells of some clouds: `calibration_n22.md` ).



def _box_target_3d():
    return _make_target( _CUBE )


def _planes_cloud( n, rng, nb_planes = 4, sigma = 0.02 ):
    """`n` seeds around `nb_planes` planes through the cube ( `bench/cases.py::planes_cloud`, the old campaign's 3D case )"""
    pts = []
    for _ in range( nb_planes ):
        u = rng.normal( size = 3 )
        u /= numpy.linalg.norm( u )
        c = u @ numpy.full( 3, 0.5 ) + rng.uniform( -0.25, 0.25 )
        v1 = numpy.cross( u, [ 1.0, 0.0, 0.0 ] if abs( u[ 0 ] ) < 0.9 else [ 0.0, 1.0, 0.0 ] )
        v1 /= numpy.linalg.norm( v1 )
        v2 = numpy.cross( u, v1 )
        st = rng.uniform( -1.2, 1.2, size = ( 8 * n, 2 ) )
        P = c * u + st[ :, :1 ] * v1 + st[ :, 1: ] * v2
        P = P[ numpy.all( ( P > 0 ) & ( P < 1 ), axis = 1 ) ][ :n // nb_planes + 1 ]
        pts.append( P + rng.normal( 0.0, sigma, size = ( len( P ), 1 ) ) * u )
    P = numpy.concatenate( pts )
    rng.shuffle( P )
    return numpy.clip( P[ :n ], 1e-4, 1 - 1e-4 )


def _plain_measures_3d( pos, w ):
    """the measures of the cells of `( pos, w )` in the unit cube, through the PLAIN storage ( every seed cuts every cell:
    no tree, no card )"""
    pd = PowerDiagram( pos, numpy.asarray( w, dtype = float ).reshape( -1 ), boundaries = box_half_spaces( [ 0 ] * 3, [ 1 ] * 3 ),
                       kernel_dtype = "FP64", accelerator = "plain" )
    pd.use_card_cells = False
    return numpy.asarray( pd.measures.value ).reshape( -1 )


def _card_vs_cpu_3d( name, pos, nu, it, tun, wtol = 1e-7 ):
    """the card's 3D solve and the CPU's: converged both, the plans within `wtol` ( relative to max |w| ), the costs and the
    barycentres; the counts are printed ( the two multigrids give directions within their tolerance: the same decisions
    in general, not to the bit )"""
    n = len( pos )
    plan = OtProblem( SumOfDiracs( pos, nu ), _make_target( _CUBE ) ).solve( Iterative( **it, tuning = Tuning( **tun ) ) )
    cpu = _cpu_solve( pos, nu, _CUBE, it, tun )
    st, cs = plan.stats, cpu[ "stats" ]
    w, wc = numpy.asarray( plan.weights ).reshape( -1 ), cpu[ "weights" ]
    gap = numpy.abs( w - wc ).max() / max( numpy.abs( wc ).max(), 1e-300 )
    cgap = abs( float( plan.cost ) / cpu[ "cost" ] - 1 )
    bgap = numpy.abs( numpy.asarray( plan.barycenters ) - cpu[ "bary" ] ).max() * n ** ( 1 / 3 )     # relative to the size of a cell
    print( f"  { name }: card { st[ 'status' ] } { st[ 'nb_iter' ] } it / { st[ 'nb_diag' ] } diag / { st[ 'lin_nb_iter' ] } lin it, "
           f"cpu { cs[ 'status' ] } { cs[ 'nb_iter' ] } it / { cs[ 'nb_diag' ] } diag; weights gap { gap :.1e}, cost gap { cgap :.1e}, "
           f"barycentres gap { bgap :.1e}, residual { st[ 'residual' ] * n :.1e} ( relative )" )
    assert plan.converged and cs[ "status" ] in ( "converged", "converged (aggregated)" ), ( name, st, cs )
    assert st[ "start" ] == cs[ "start" ], ( name, st[ "start" ], cs[ "start" ] )
    assert abs( st[ "domain_mass" ] - cs[ "domain_mass" ] ) < 1e-10, name
    assert gap < wtol and cgap < 1e-9 and bgap < 1e-6, ( name, gap, cgap, bgap )
    m = numpy.asarray( plan.cell_masses ).reshape( -1 )
    tm = numpy.asarray( plan.target_masses ).reshape( -1 )
    assert numpy.abs( m - tm ).max() <= it[ "tol" ] * ( 1 + 1e-9 ), name
    return plan, cpu


if test( "the_card_solves_the_transport_in_3d_as_the_cpu" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    rng = numpy.random.default_rng( 401 )
    n = 3000
    pos = rng.uniform( 0.001, 0.999, size = ( n, 3 ) )
    nu = rng.uniform( 0.5, 1.5, n )
    nu /= nu.sum()
    it = dict( tol = 1e-10 / n, max_iter = 60 )
    plan, cpu = _card_vs_cpu_3d( "uniform, fp64", pos, nu, dict( it, precision = "fp64" ), dict( step = "trials" ) )
    # the cells of the card's weights, measured by the plain storage: the targets
    g = _plain_measures_3d( pos, plan.weights )
    tm = numpy.asarray( plan.target_masses ).reshape( -1 )
    assert numpy.abs( g - tm ).max() < 1e-8 / n, numpy.abs( g - tm ).max() * n
    assert numpy.asarray( plan.weights ).reshape( -1 )[ 0 ] == 0            # the gauge
    assert plan.history[ -1 ][ "max_abs_residual" ] == plan.stats[ "residual" ]
    # the other kernels and linear solvers: the same plan ( `auto` is the step `trials` in 3D )
    ref = numpy.asarray( plan.weights ).reshape( -1 )
    for name, precision, tuning in ( ( "fp32", "fp32", Tuning() ), ( "mixed ( auto )", "auto", Tuning( step = "auto" ) ),
                                     ( "cg", "fp64", Tuning( linear_solver = "cg", linear_tol = 1e-8 ) ),
                                     ( "host cholesky", "fp64", Tuning( linear_solver = "cholesky" ) ),
                                     ( "double levels", "fp64", Tuning( mg_precision = "double" ) ),
                                     ( "plain aggregation", "fp64", Tuning( mg_smoothed = 0 ) ) ):
        p2 = OtProblem( SumOfDiracs( pos, nu ), _box_target_3d() ).solve( Iterative( **it, precision = precision, tuning = tuning ) )
        gap = numpy.abs( numpy.asarray( p2.weights ).reshape( -1 ) - ref ).max() / numpy.abs( ref ).max()
        print( f"  { name }: { p2.stats[ 'status' ] } { p2.stats[ 'nb_iter' ] } it / { p2.stats[ 'nb_diag' ] } diag / { p2.stats[ 'lin_nb_iter' ] } lin it, "
               f"gap { gap :.1e}, double kernel from it { p2.stats[ 'it_double' ] }" )
        assert p2.converged and gap < 1e-7, ( name, p2.stats, gap )
    # given weights ( a warm start near the solution ), and diracs outside the domain ( the similarity start )
    warm = OtProblem( SumOfDiracs( pos, nu ), _box_target_3d() ).solve( Iterative( **it, weights0 = ref * ( 1 + 1e-3 ) ) )
    assert warm.converged and warm.stats[ "start" ] == "weights0" and warm.stats[ "nb_iter" ] <= 4, warm.stats
    out = 0.5 + 1.6 * ( pos[ :400 ] - 0.5 )
    _card_vs_cpu_3d( "seeds outside the cube ( similarity start )", out, numpy.ones( 400 ) / 400, dict( tol = 1e-10 / 400, max_iter = 80 ), {} )


if test( "the_card_solves_the_planes_in_3d_as_the_cpu" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    # the old campaign's 3D case ( seeds around four planes, sigma 0.02 ): equal volumes from the Voronoi ( `planes_voronoi` ),
    # then targets of varied masses
    rng = numpy.random.default_rng( 402 )
    n = 6000
    pos = _planes_cloud( n, rng )
    it = dict( tol = 1e-10 / n, max_iter = 80 )
    plan, _ = _card_vs_cpu_3d( "planes, equal volumes, mixed", pos, numpy.ones( n ) / n, dict( it, precision = "auto" ), {} )
    g = _plain_measures_3d( pos, plan.weights )
    assert numpy.abs( g - 1 / n ).max() < 1e-8 / n, numpy.abs( g - 1 / n ).max() * n
    nu = numpy.exp( rng.normal( 0, 0.5, n ) )
    _card_vs_cpu_3d( "planes, varied masses, fp64", pos, nu / nu.sum(), dict( it, precision = "fp64" ), {} )


if test( "the_card_3d_solve_runs_under_jit" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    # the eager solve and the jitted one, to the bit: the masses traced, then the positions too ( the tree built in the program )
    import jax
    rng = numpy.random.default_rng( 403 )
    n = 4000
    pos = rng.uniform( 0.001, 0.999, size = ( n, 3 ) )
    nu = rng.uniform( 0.5, 1.5, n )

    def solve( p, m ):
        plan = OtProblem( SumOfDiracs( p, m ), _box_target_3d() ).solve( Iterative( tol = 1e-10 / n, max_iter = 60 ) )
        return plan.weights.raw, plan.cost.raw, plan.stats[ "nb_diag" ], plan.stats[ "status" ]

    we, ce, de, se = solve( pos, nu )
    assert se == "converged", se
    wj, cj, dj, _ = jax.jit( lambda m: solve( pos, m ) )( nu )
    assert numpy.array_equal( numpy.asarray( wj ), numpy.asarray( we ) ), numpy.abs( numpy.asarray( wj ) - numpy.asarray( we ) ).max()
    assert float( cj ) == float( ce ) and float( dj ) == float( de ), ( cj, ce, dj, de )
    f = jax.jit( solve )
    assert numpy.array_equal( numpy.asarray( f( pos, nu )[ 0 ] ), numpy.asarray( we ) )
    pos2 = numpy.clip( pos + 1e-3 * rng.normal( size = pos.shape ), 0.001, 0.999 )
    assert numpy.array_equal( numpy.asarray( f( pos2, nu )[ 0 ] ), numpy.asarray( solve( pos2, nu )[ 0 ] ) )
    print( f"  { int( de ) } diagrams, jit == eager" )


if test( "the_card_3d_solve_refuses_what_it_does_not_solve" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    rng = numpy.random.default_rng( 404 )
    pos = rng.uniform( 0.1, 0.9, size = ( 200, 3 ) )
    # the step `limits`: 2D only ( the CPU's 3D has no area polynomials either )
    try:
        OtProblem( SumOfDiracs( pos ), _box_target_3d() ).solve( Iterative( tuning = Tuning( step = "limits" ) ) )
        raise AssertionError( "step = 'limits' must be refused in 3D" )
    except ValueError as e:
        assert "2D only" in str( e ) and "'trials'" in str( e ), str( e )
    # a density that is not a constant: an image of varied values, gaussians
    for dst in ( Image( values = 1 + rng.random( ( 4, 4, 4 ) ), origin = [ 0.0 ] * 3, frame = 0.25 * numpy.eye( 3 ) ),
                 SumOfGaussians( numpy.full( ( 1, 3 ), 0.5 ), numpy.array( [ 0.2 ] ) ) ):
        try:
            OtProblem( SumOfDiracs( pos ), dst ).solve()
            raise AssertionError( f"a 3D density { type( dst ).__name__ } must be refused on the card" )
        except NotImplementedError as e:
            assert "3D" in str( e ) and "constant density" in str( e ) and "LOOM_DEVICE=cpu" in str( e ), str( e )


if test( "the_card_3d_memory_model_follows_the_solve" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    from sdot.CardMemory import card_solve_bytes
    from sdot.SdotPlanNd import SdotPlanNd
    rng = numpy.random.default_rng( 405 )
    n = 200_000
    pos = rng.uniform( 0.001, 0.999, size = ( n, 3 ) )
    for name, it in ( ( "mixed", Iterative( tol = 1e-8 / n, max_iter = 30 ) ),
                      ( "fp64, double levels", Iterative( tol = 1e-8 / n, max_iter = 30, precision = "fp64", tuning = Tuning( mg_precision = "double" ) ) ),
                      ( "fp32", Iterative( tol = 1e-7 / n, max_iter = 30, precision = "fp32" ) ),   # ( the float kernel alone floors at 3e-8 here )
                      ( "fp64, cg", Iterative( tol = 1e-8 / n, max_iter = 30, precision = "fp64", tuning = Tuning( linear_solver = "cg", linear_tol = 1e-8 ) ) ) ):
        plan = OtProblem( SumOfDiracs( pos ), _box_target_3d() ).solve( it )
        assert plan.converged, ( name, plan.stats )
        parts = card_solve_bytes( n, **SdotPlanNd._card_memory_kw( n, it, 3 ) )
        took, model = plan.stats[ "scratch_bytes" ], sum( v for k, v in parts.items() if k not in ( "tree", "outputs" ) )
        print( f"  { name }: took { took / n :.1f} bytes per seed, model { model / n :.1f}" )
        assert abs( model - took ) < 0.08 * took, ( name, took, model )


if test( "the_card_refuses_a_3d_solve_that_does_not_fit_at_once" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    n = 1_000_000                                # ~1 GB for the 3D solve, the pool ~0.5 GB
    for mode, ( kind, t, msg ) in zip( ( "eager", "jit" ), _oom_runs( n, [ "eager", "jit" ], SDOT_TEST_DIM = "3" ) ):
        print( f"  { mode }: { kind } after { t :.2f} s: { msg[ :200 ] }..." )
        assert kind == "MemoryError" and t < 5, ( mode, kind, t, msg )
        assert f"solve of { n } seeds needs about" in msg and "LOOM_DEVICE=cpu" in msg, msg


if test( "the_card_aggregates_in_3d_as_the_cpu" ):
    import numpy
    from loom.testing import need
    from sdot import ( Image, Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians, Tuning, Visualizer,
                       box_half_spaces, ot_solve, write_convergence_html )
    _CUBE = ( "image", dict( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) ) )
    from errand import skip
    if not _card():
        skip( "the card's solver needs a CUDA device" )
    for name, ( pos, groups ) in _degenerate_clouds( 3, 17 ).items():
        n = len( pos )
        tol = ( 1e-9 if "1e-12" in name else 1e-11 ) / n
        for precision in ( "fp64", "auto" ):
            plan = OtProblem( SumOfDiracs( pos ), _box_target_3d() ).solve( Iterative( tol = tol, max_iter = 60, precision = precision ) )
            _check_aggregated( f"card 3D { name } { precision }", plan, pos, groups, tol )
            g = _plain_measures_3d( pos, plan.weights )
            nu = numpy.asarray( plan.target_masses ).reshape( -1 )
            alone = numpy.ones( n, dtype = bool )
            for members, _ in groups:
                alone[ members ] = False
                assert abs( g[ members ].sum() - nu[ members ].sum() ) < 1e-8 / n, ( name, g[ members ].sum() - nu[ members ].sum() )
            assert numpy.abs( g[ alone ] - nu[ alone ] ).max() < 1e-8 / n, name


if test( "triangle_mesh_starts_in_its_inscribed_box" ):
    import numpy
    from loom.testing import need
    from sdot import Mesh, OtProblem, SumOfDiracs
    need( "cpu" )
    # a density whose support is a TRIANGLE inside its bounding box: the Voronoi start leaves a third of the cells empty, and
    # the similarity into the box of the domain does not help -- the seeds are packed into the target's `inscribed_box` instead
    rng = numpy.random.default_rng( 3 )
    dst = Mesh( nodes = [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ] ], simplices = [ [ 0, 1, 2 ] ], values = [ 1, 0.1, 0.1 ] )
    lo, hi = dst.inscribed_box()
    assert numpy.all( lo >= -1e-9 ) and numpy.all( hi - lo > 0.4 ) and hi.sum() <= 1 + 1e-9, ( lo, hi )
    plan = OtProblem( SumOfDiracs( rng.uniform( 0, 1, size = ( 100, 2 ) ) ), dst ).solve()
    assert plan.converged and plan.stats[ "start" ] == "similarity", plan.stats


if test( "a_solve_that_misses_its_tolerance_raises" ):
    import numpy
    from loom.testing import need
    from sdot import Iterative, OtNotConverged, OtProblem, SumOfDiracs, SumOfGaussians
    need( "cpu" )
    # not silent: `OtNotConverged`, carrying the solution -- unless `on_failure = "ignore"`
    rng = numpy.random.default_rng( 5 )
    pos = rng.uniform( 0, 1, size = ( 200, 2 ) )
    dst = _scattered_target( 2, 4, 7 )
    try:
        OtProblem( SumOfDiracs( pos ), dst ).solve( Iterative( max_iter = 2, tol = 1e-14 ) )
        raise AssertionError( "no exception" )
    except OtNotConverged as e:
        assert not e.solution.converged
    assert not OtProblem( SumOfDiracs( pos ), dst ).solve( Iterative( max_iter = 2, tol = 1e-14, on_failure = "ignore" ) ).converged


if test( "a_mesh_the_plain_solve_misses_goes_through_its_spreading" ):
    import numpy
    from scipy.spatial import Delaunay
    from loom.testing import need
    from sdot import Iterative, Mesh, OtNotConverged, OtProblem, SumOfDiracs
    need( "cpu" )
    # TWO thin curved bands apart, in a mesh that paves their box ( DG1: one on the simplices of the bands, zero elsewhere ): the cells
    # that start between them are fed by neither, and the plain Newton stagnates. The width continuation spreads the mesh on itself
    # ( `Mesh.spread`, the corner averages ) and climbs back to it, stage by stage ( `SdotPlanNd._solve_by_spreading` )
    def ring( angle, shift ):
        r, t = numpy.linspace( 0.95, 1.0, 3 ), numpy.linspace( -angle, angle, 40 )
        R, T = numpy.meshgrid( r, t, indexing = "ij" )
        return numpy.stack( [ R * numpy.sin( T ), R * numpy.cos( T ) ], axis = -1 ).reshape( -1, 2 ) * numpy.array( [ 1, shift ] )
    def band( x, shift, dy ):
        y = ( x - [ 0, dy ] ) * [ 1, shift ]
        r, t = numpy.linalg.norm( y, axis = -1 ), numpy.arctan2( y[ ..., 0 ], y[ ..., 1 ] )
        return ( r > 0.95 - 1e-9 ) & ( r < 1 + 1e-9 ) & ( numpy.abs( t ) < 0.5 + 1e-9 )
    bands = numpy.concatenate( [ ring( 0.5, 1 ), ring( 0.5, -1 ) + [ 0, 1.2 ] ] )
    lo, hi = bands.min( axis = 0 ) - 0.1, bands.max( axis = 0 ) + 0.1
    g = numpy.stack( numpy.meshgrid( numpy.linspace( lo[ 0 ], hi[ 0 ], 16 ), numpy.linspace( lo[ 1 ], hi[ 1 ], 16 ), indexing = "ij" ), -1 ).reshape( -1, 2 )
    g = g[ ~band( g, 1, 0 ) & ~band( g, -1, 1.2 ) ]
    nodes = numpy.concatenate( [ bands, g ] )
    sx = Delaunay( nodes ).simplices
    c = nodes[ sx ].mean( axis = 1 )
    on = band( c, 1, 0 ) | band( c, -1, 1.2 )
    dst = Mesh( nodes, sx, values = numpy.repeat( on[ :, None ] * 1.0, 3, axis = 1 ) )
    src = SumOfDiracs( lo + ( hi - lo ) * numpy.random.default_rng( 0 ).random( ( 400, 2 ) ) )
    try:
        OtProblem( src, dst ).solve( Iterative( continuation = "never" ) )
        raise AssertionError( "the plain solve was expected to miss" )
    except OtNotConverged:
        pass
    plan = OtProblem( src, dst ).solve()                            # `auto`: the plain solve misses, the spreading takes over
    assert plan.converged and plan.stats[ "spread_stages" ][ -1 ] == 0 and plan.stats[ "start" ] == "weights0", plan.stats
    # the path: from the flat start down to the mesh, the mass and the positivity kept at every stage
    k0 = dst.spread_start()
    assert plan.stats[ "spread_start" ] == k0 and k0 >= 2
    w = numpy.asarray( dst.corner_weights )
    for k in ( k0, k0 / 7, 1.5, 0.25 ):
        v = numpy.asarray( dst.spread( k ).values )
        assert v.min() >= 0 and abs( ( w * v ).sum() / float( dst.mass ) - 1 ) < 1e-10, k


if test( "many_gaussians_go_through_the_tree_and_stay_exact" ):
    import numpy
    from loom.testing import need
    from sdot import Iterative, OtProblem, PowerDiagram, SumOfDiracs, SumOfGaussians
    need( "cpu" )
    # 400 gaussians along an arc ( a thin banana ): each cell of the solver sees only the gaussians near it, merged during the
    # continuation ( `GaussianTree.h` ) -- the fitted weights must still give the target masses on the EXACT density, every gaussian
    # integrated by every cell ( `PowerDiagram.measures`, no tree )
    rng = numpy.random.default_rng( 5 )
    G = 400
    t = rng.uniform( 0.2 * numpy.pi, 0.8 * numpy.pi, G )
    r = 0.35 + 0.02 * rng.standard_normal( G )
    dst = SumOfGaussians( numpy.c_[ 0.5 + r * numpy.cos( t ), 0.2 + r * numpy.sin( t ) ], numpy.full( G, 0.01 ), weights = numpy.ones( G ) )
    pos = rng.uniform( 0, 1, ( 100, 2 ) )
    plan = OtProblem( SumOfDiracs( pos ), dst ).solve( Iterative( max_iter = 200, tol = 1e-11, continuation = "always" ) )
    assert plan.converged and plan.stats[ "nb_continuation_steps" ] > 5, plan.stats
    pd = PowerDiagram( pos, numpy.asarray( plan.weights ).reshape( -1 ), distribution = dst.normalized_version(), kernel_dtype = "FP64" )
    got = numpy.asarray( pd.measures ).reshape( -1 )
    assert numpy.allclose( got, _target_masses( plan ), atol = 1e-9 ), numpy.abs( got - _target_masses( plan ) ).max()


if test( "the_display_keeps_the_cells_where_the_density_has_mass" ):
    import itertools, tempfile, pathlib
    import numpy
    from loom.testing import need
    from sdot import Image, Iterative, Mesh, OtProblem, PowerDiagram, SumOfDiracs
    from sdot.Cell import SEAM, SUPPORT
    need( "cpu" )
    # a density with ZERO regions ( a hole, and a corner ): the pieces of `support_pieces` tile exactly the region with mass,
    # cell by cell -- their measure, summed per seed, is the measure of the cell against the indicator of that region -- and
    # the walls between two blocks with mass are `SEAM`s, not drawn. An image is gathered into boxes ( far fewer pieces than
    # tiles ), a mesh is cut by its simplices.
    def image( d, s, rng ):
        x = ( numpy.indices( ( s, ) * d ) + 0.5 ) / s
        values = 1 + rng.random( ( s, ) * d )
        values[ ( ( x - 0.5 ) ** 2 ).sum( axis = 0 ) < 0.25 ** 2 ] = 0
        values[ ( x < 0.25 ).all( axis = 0 ) ] = 0
        frame = numpy.eye( d ) / s
        return Image( values = values, origin = numpy.zeros( d ), frame = frame ), \
               Image( values = ( values > 0 ) * 1.0, origin = numpy.zeros( d ), frame = frame ), ( values > 0 ).sum() / s ** d

    def mesh( d, s, rng, hole = True ):
        # a grid of cubes, each split into d! simplices along the paths of its diagonal ( conforming ), minus a corner; the
        # nodes in a ball have a zero value -- the simplices that only touch them have no mass
        nodes = numpy.indices( ( s + 1, ) * d ).reshape( d, -1 ).T / s
        idx = numpy.arange( len( nodes ) ).reshape( ( s + 1, ) * d )
        sx = []
        for c in itertools.product( range( s ), repeat = d ):
            for perm in itertools.permutations( range( d ) ):
                k = numpy.array( c )
                tri = [ idx[ tuple( k ) ] ]
                for a in perm:
                    k = k.copy(); k[ a ] += 1
                    tri.append( idx[ tuple( k ) ] )
                sx.append( tri )
        sx = numpy.array( sx )
        sx = sx[ ~( nodes[ sx ].mean( axis = 1 ) < 0.25 ).all( axis = 1 ) ]
        values = 1 + rng.random( len( nodes ) )
        if hole:
            values[ ( ( nodes - 0.5 ) ** 2 ).sum( axis = 1 ) < 0.25 ** 2 ] = 0
        kept = sx[ values[ sx ].max( axis = 1 ) > 0 ]
        P = nodes[ kept ]
        area = numpy.abs( numpy.linalg.det( P[ :, 1: ] - P[ :, :1 ] ) ).sum() / numpy.prod( numpy.arange( 1, d + 1 ) )
        return Mesh( nodes, sx, values = values ), Mesh( nodes, kept ), area

    for kind, d, s, n in ( ( image, 2, 24, 60 ), ( image, 3, 8, 40 ), ( mesh, 2, 12, 60 ), ( mesh, 3, 5, 40 ) ):
        rng = numpy.random.default_rng( 3 + d )
        dst, indicator, support = kind( d, s, rng )
        plan = OtProblem( SumOfDiracs( rng.uniform( 0.3, 0.7, ( n, d ) ) ), dst ).solve( Iterative( max_iter = 200, tol = 1e-9 / n ) )
        assert plan.converged, plan.stats
        w = numpy.asarray( plan.weights ).reshape( -1 )
        pos = numpy.asarray( plan._pd.positions ).reshape( -1, d )

        pd = PowerDiagram( pos, w, distribution = dst, kernel_dtype = "FP64" )
        pieces, owners = pd.support_pieces()
        vol = numpy.asarray( pieces.measure ).reshape( -1 )
        per_cell = numpy.bincount( owners, weights = vol, minlength = n )
        ind = numpy.asarray( PowerDiagram( pos, w, distribution = indicator, kernel_dtype = "FP64" ).measures ).reshape( -1 ) * support
        assert numpy.allclose( per_cell, ind, atol = 1e-12 ), ( kind.__name__, d, numpy.abs( per_cell - ind ).max() )
        assert abs( vol.sum() - support ) < 1e-12, ( kind.__name__, d, vol.sum(), support )

        ids = numpy.concatenate( [ it.cid for it in pieces._items() ] )
        assert ( ids == SEAM ).any() and ( ids == SUPPORT ).any()
        if kind is image:
            assert ( numpy.bincount( owners, minlength = n ) == 1 ).any()   # some cells are kept whole
            # the boxes: far fewer than the tiles with mass
            g = dst.normalized_version()._display( 0.0 )
            print( f"  { len( numpy.unique( owners ) ) } cells drawn, { len( g[ 'box_lo' ] ) } boxes for { int( ( g[ 'label' ] >= 0 ).sum() ) } tiles" )
            assert len( g[ "box_lo" ] ) < 0.5 * ( g[ "label" ] >= 0 ).sum()

        # the display: html and ParaView, from the plan
        out = pathlib.Path( tempfile.mkdtemp() )
        plan.write_html( out / f"support_{ kind.__name__ }_{ d }d.html", support_only = True )
        plan.write_pvd( out / f"support_{ kind.__name__ }_{ d }d.pvd", support_only = True )
        print( f"  { kind.__name__ } { d }D: { n } cells, { len( owners ) } pieces -> { out }" )

    # a threshold: the pieces cover where `rho > threshold * max` -- on a mesh, the plane `rho = threshold` of each simplex, checked
    # against the same area counted on random points ( the mesh of the loop above: a grid of squares cut along their diagonal )
    dst, _, _ = mesh( 2, 12, numpy.random.default_rng( 5 ) )
    pos = numpy.random.default_rng( 9 ).uniform( 0.1, 0.9, ( 40, 2 ) )
    pd = PowerDiagram( pos, numpy.zeros( 40 ), distribution = dst, kernel_dtype = "FP64" )
    pieces, owners = pd.support_pieces( threshold = 0.6 )
    vals = numpy.zeros( len( numpy.asarray( dst.nodes ) ) )                  # ( the node values back from the corners: continuous here )
    vals[ numpy.asarray( dst.simplices ).ravel() ] = numpy.asarray( dst.values ).ravel()
    x = numpy.random.default_rng( 10 ).random( ( 400_000, 2 ) )
    k, f = numpy.minimum( ( x * 12 ).astype( int ), 11 ), x * 12 - numpy.minimum( ( x * 12 ).astype( int ), 11 )
    node = lambda a, b: vals[ ( k[ :, 0 ] + a ) * 13 + k[ :, 1 ] + b ]
    up = f[ :, 0 ] >= f[ :, 1 ]                                        # the simplex of the path `e0` then `e1`, or the other one
    rho = numpy.where( up, ( 1 - f[ :, 0 ] ) * node( 0, 0 ) + ( f[ :, 0 ] - f[ :, 1 ] ) * node( 1, 0 ) + f[ :, 1 ] * node( 1, 1 ),
                           ( 1 - f[ :, 1 ] ) * node( 0, 0 ) + ( f[ :, 1 ] - f[ :, 0 ] ) * node( 0, 1 ) + f[ :, 0 ] * node( 1, 1 ) )
    rho[ ( k < 3 ).all( axis = 1 ) ] = 0                                 # the corner that is not meshed
    expected = ( rho > 0.6 * vals.max() ).mean()
    got = float( numpy.asarray( pieces.measure ).sum() )
    assert abs( got - expected ) < 4e-3, ( got, expected )

    # a mesh without a hole: the cells away from its border are kept whole ( their box meets no facet of the border )
    dst, indicator, support = mesh( 2, 12, numpy.random.default_rng( 7 ), hole = False )
    pos = numpy.random.default_rng( 8 ).uniform( 0.3, 0.7, ( 30, 2 ) )
    pd = PowerDiagram( pos, numpy.zeros( 30 ), distribution = dst, kernel_dtype = "FP64" )
    pieces, owners = pd.support_pieces()
    assert ( numpy.bincount( owners, minlength = 30 ) == 1 ).any(), numpy.bincount( owners, minlength = 30 )
    per_cell = numpy.bincount( owners, weights = numpy.asarray( pieces.measure ).reshape( -1 ), minlength = 30 )
    ind = numpy.asarray( PowerDiagram( pos, numpy.zeros( 30 ), distribution = indicator, kernel_dtype = "FP64" ).measures ).reshape( -1 ) * support
    assert numpy.allclose( per_cell, ind, atol = 1e-12 )


if test( "a_concave_crease_of_the_support_is_drawn" ):
    import numpy
    from loom.testing import need
    from sdot import Image, PowerDiagram, Visualizer
    need( "cpu" )
    # an L-shaped prism ( 2 x 2 x 1 voxels, one of them empty ) in ONE cell: the edges drawn are exactly the 18 edges of the
    # prism -- the CONCAVE one, where the border of the support folds on a seam, included, and none of the seams themselves
    values = numpy.ones( ( 2, 2, 1 ) )
    values[ 1, 1, 0 ] = 0
    pd = PowerDiagram( numpy.array( [ [ 0.7, 0.7, 0.5 ] ] ), numpy.zeros( 1 ), distribution = Image( values = values, origin = numpy.zeros( 3 ), frame = numpy.eye( 3 ) ),
                       kernel_dtype = "FP64" )
    viz = Visualizer()
    pd.add_to_viz( viz, support_only = True, seeds = False )
    pos, edges = numpy.asarray( viz.positions, float ), numpy.asarray( viz.edges )
    drawn = numpy.stack( [ pos[ edges[ :, 0 ] ], pos[ edges[ :, 1 ] ] ], axis = 1 )

    outline = numpy.array( [ [ 0, 0 ], [ 2, 0 ], [ 2, 1 ], [ 1, 1 ], [ 1, 2 ], [ 0, 2 ] ], float )
    true = [ ( numpy.r_[ outline[ i ], z ], numpy.r_[ outline[ ( i + 1 ) % 6 ], z ] ) for i in range( 6 ) for z in ( 0, 1 ) ]
    true += [ ( numpy.r_[ p, 0 ], numpy.r_[ p, 1 ] ) for p in outline ]

    def on( m, a, b ):
        t = numpy.clip( ( m - a ) @ ( b - a ) / ( ( b - a ) @ ( b - a ) ), 0, 1 )
        return numpy.linalg.norm( a + t * ( b - a ) - m ) < 1e-6
    for a, b in drawn:                                          # nothing drawn off the edges of the prism ( no seam )
        assert any( on( ( a + b ) / 2, p, q ) for p, q in true ), ( a, b )
    for p, q in true:                                           # and every edge of the prism drawn, the concave one at ( 1, 1 ) included
        assert any( on( ( p + q ) / 2, a, b ) for a, b in drawn ), ( p, q )


if test( "a_disconnected_laplacian_is_a_status_not_a_crash" ):
    import numpy
    from loom.testing import need
    from sdot import Iterative, Mesh, OtProblem, SumOfDiracs
    from sdot.OtProblem import Tuning
    need( "cpu" )
    # a mesh density ZERO outside two small far apart bumps, and a start KEPT as given ( `keep_start` ): most cells
    # have no mass, the facet graph falls apart into components, and every one that does not hold the gauge is a singular
    # block -- the coarse LU of AMGCL used to throw ( "Zero sum in skyline_lu factorization" ) through the FFI call. The
    # solve must come back with a status, the caller ( `on_failure = "ignore"` ) deciding what to do with it.
    g = numpy.linspace( 0, 1, 61 )
    X, Y = numpy.meshgrid( g, g, indexing = "ij" )
    nodes = numpy.stack( [ X.ravel(), Y.ravel() ], 1 )
    k = numpy.arange( 60 )[ :, None ] * 61 + numpy.arange( 60 )[ None, : ]
    k = k.ravel()
    sx = numpy.concatenate( [ numpy.stack( [ k, k + 61, k + 62 ], 1 ), numpy.stack( [ k, k + 62, k + 1 ], 1 ) ] )
    def bump( c, s ):
        return numpy.maximum( 0.0, 1 - numpy.linalg.norm( nodes - c, axis = 1 ) / s )
    vals = bump( [ 0.2, 0.25 ], 0.06 ) + 0.5 * bump( [ 0.8, 0.7 ], 0.04 )
    # ( 100 seeds and these small random weights: the smallest case found that reached the throw, the AMG forced -- AUTO only takes it
    # with OpenMP )
    pos = numpy.random.default_rng( 3 ).uniform( 0.02, 0.98, ( 100, 2 ) )
    w0 = numpy.random.default_rng( 3 ).uniform( 0, 1e-2, 100 )
    sol = OtProblem( SumOfDiracs( pos ), Mesh( nodes, sx, values = vals ) ).solve(
        Iterative( weights0 = w0, continuation = "never", on_failure = "ignore", max_iter = 20,
                   tuning = Tuning( keep_start = True, linear_solver = "amg" ) ) )
    assert not sol.converged and sol.stats[ "status" ] == "linear solver failure", sol.stats[ "status" ]


if test( "the_plan_is_differentiable_with_respect_to_all_its_inputs" ):
    import numpy
    from loom.testing import check_grad, need
    from sdot import Iterative, Mesh, OtProblem, SumOfDiracs, SumOfGaussians
    need( "cpu" )
    # `weights`, `cell_masses`, `cost` and `barycenters` against EVERY input -- the positions and the masses of the diracs, the parameters of
    # the target -- by the implicit function theorem ( `SdotPlanNd._attach_derivatives`: the solve itself has no adjoint, it runs on
    # detached inputs and the derivative is one solve with the laplacian ). Checked by finite differences on the whole solve.
    rng = numpy.random.default_rng( 0 )
    n = 24
    pos = rng.uniform( 0.2, 0.8, size = ( n, 2 ) )
    nu = rng.uniform( 0.5, 1.5, size = n )
    settings = Iterative( tol = 1e-13, max_iter = 200 )
    nodes, simplices = [ [ 0, 0 ], [ 1, 0 ], [ 1, 1 ], [ 0, 1 ] ], [ [ 0, 1, 2 ], [ 0, 2, 3 ] ]
    values = numpy.array( [ [ 1, 0.5, 0.2 ], [ 1, 0.2, 0.7 ] ] )

    def on_mesh( p, q, v ):
        return OtProblem( SumOfDiracs( p, weights = q ), Mesh( nodes = nodes, simplices = simplices, values = v ) ).solve( settings )

    # a mesh: closed forms, the finite differences are clean
    for output, tol in ( ( "weights", 1e-5 ), ( "cell_masses", 1e-5 ), ( "cost", 1e-3 ), ( "barycenters", 1e-3 ) ):
        check_grad( lambda p, q, v: getattr( on_mesh( p, q, v ), output ), pos, nu, values, seed = 3, eps = 1e-5, rtol = tol, atol = tol * 1e-2 )

    # gaussians: the support is DECLARED as a box ( a traced parameter has no support to read ), the quadrature of the moments flips its
    # subdivisions -- the finite differences of the cost and of the barycenters are only as smooth as its `rtol`
    def on_gaussians( p, q, c, s, m ):
        dst = SumOfGaussians( c, s, weights = m, support_box = ( [ 0, 0 ], [ 1, 1 ] ) )
        return OtProblem( SumOfDiracs( p, weights = q ), dst ).solve( settings )
    c, s, m = numpy.array( [ [ 0.4, 0.5 ], [ 0.7, 0.3 ] ] ), numpy.array( [ 0.25, 0.2 ] ), numpy.array( [ 1.0, 0.7 ] )
    for output, eps, tol in ( ( "weights", 1e-5, 1e-5 ), ( "cell_masses", 1e-5, 1e-5 ), ( "cost", 1e-4, 5e-3 ), ( "barycenters", 1e-4, 5e-3 ) ):
        check_grad( lambda p, q, c, s, m: getattr( on_gaussians( p, q, c, s, m ), output ), pos, nu, c, s, m, seed = 4, eps = eps, rtol = tol, atol = tol * 1e-2 )

    # and the values are those of the solve: nothing changes when nobody differentiates
    plain = OtProblem( SumOfDiracs( pos, weights = nu ), Mesh( nodes = nodes, simplices = simplices, values = values ) ).solve( settings )
    import jax
    traced = jax.jit( lambda p: on_mesh( p, nu, values ).cost.raw )( pos )
    assert abs( float( traced ) - float( plain.cost ) ) < 1e-5 * float( plain.cost ), ( float( traced ), float( plain.cost ) )
