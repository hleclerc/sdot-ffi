"""A sum of gaussians on the FACETS of a cell ( `SumOfGaussians::facet_mass` ): what the laplacian of a transport reads
( `sdotplan/Sweep.h`, `PowerDiagram.hessian_rows` ). In 3D a facet is a planar polygon, on which an isotropic gaussian splits
into a 1D factor across the plane and the exact 2D reduction in it."""

from errand import test


def _dense_hessian( pd, n ):
    import numpy
    counts, ids, vals = pd.hessian_rows()
    H = numpy.zeros( ( n, n ) )
    for i in range( n ):
        for q in range( counts[ i ] ):
            if ids[ i, q ] >= 0:
                H[ i, ids[ i, q ] ] -= vals[ i, q ]
                H[ i, i ] += vals[ i, q ]
    return H


if test( "the_3d_facet_mass_of_gaussians_is_exact_on_any_plane" ):
    # two seeds in the unit cube: their facet is the square `x = 1/2`, on which a gaussian is a product of `erf`s. Then the same
    # configuration ROTATED ( seeds, centers, the half-spaces of the cube ): an oblique facet, the same masses -- the fan of
    # triangles and the in-plane frame of `SumOfGaussians::facet_mass_3d`, against the closed form
    import numpy
    from math import erf, exp, pi, sqrt
    from sdot import PowerDiagram, SumOfGaussians, box_half_spaces
    c = numpy.array( [ [ 0.43, 0.58, 0.47 ], [ 0.52, 0.31, 0.66 ], [ 0.2, 0.7, 0.5 ] ] )
    s = numpy.array( [ 0.17, 0.05, 0.3 ] )                 # narrow ( the exact 2D reduction matters ), mid, wide
    w = numpy.array( [ 0.5, 1.0, 0.7 ] )
    pos = numpy.array( [ [ 0.25, 0.5, 0.5 ], [ 0.75, 0.5, 0.5 ] ] )
    seg = lambda a, s: 0.5 * ( erf( ( 1 - a ) / ( s * sqrt( 2 ) ) ) - erf( ( 0 - a ) / ( s * sqrt( 2 ) ) ) )
    facet = sum( w[ i ] / w.sum() * exp( - ( 0.5 - c[ i, 0 ] ) ** 2 / ( 2 * s[ i ] ** 2 ) ) / ( s[ i ] * sqrt( 2 * pi ) )
                 * seg( c[ i, 1 ], s[ i ] ) * seg( c[ i, 2 ], s[ i ] ) for i in range( 3 ) )
    ref = facet / ( 2 * 0.5 )                              # `d mass_0 / d w_1 = facet / ( 2 | p_1 - p_0 | )`
    dirs, offs = box_half_spaces( [ 0 ] * 3, [ 1 ] * 3 )
    rng = numpy.random.default_rng( 35 )
    for r in range( 3 ):
        R = numpy.eye( 3 ) if r == 0 else numpy.linalg.qr( rng.normal( size = ( 3, 3 ) ) )[ 0 ]
        pd = PowerDiagram( pos @ R.T, boundaries = ( dirs @ R.T, offs ), kernel_dtype = "FP64",
                           distribution = SumOfGaussians( positions = c @ R.T, sigmas = s, weights = w ) )
        H = _dense_hessian( pd, 2 )
        assert abs( - H[ 0, 1 ] - ref ) < 1e-12 * ref, ( r, - H[ 0, 1 ], ref )


if test( "the_3d_facet_mass_of_gaussians_is_symmetric" ):
    # many seeds, oblique facets cut by their neighbours: each facet is integrated from both sides ( two cells, two fans, two
    # frames ), and must agree. ( No finite difference here: the measures of a 3D gaussian come from an adaptive quadrature,
    # `PointwiseDensity`, whose noise is above the checks this one makes. )
    import numpy
    from sdot import PowerDiagram, SumOfGaussians, box_half_spaces
    rng = numpy.random.default_rng( 33 )
    n = 30
    pos = rng.uniform( 0.1, 0.9, size = ( n, 3 ) )
    w0 = rng.uniform( -0.01, 0.01, n )
    dens = SumOfGaussians( positions = rng.uniform( 0.3, 0.7, size = ( 3, 3 ) ), sigmas = [ 0.04, 0.2, 0.35 ], weights = [ 0.5, 1.0, 0.7 ] )
    pd = PowerDiagram( pos, w0, boundaries = box_half_spaces( [ 0 ] * 3, [ 1 ] * 3 ), distribution = dens, kernel_dtype = "FP64" )
    H = _dense_hessian( pd, n )
    assert ( H - numpy.diag( numpy.diag( H ) ) < 0 ).sum() > 2 * n, "too few facets"
    assert numpy.abs( H - H.T ).max() < 1e-12 * numpy.abs( H ).max(), numpy.abs( H - H.T ).max()


if test( "the_3d_transport_towards_gaussians_is_solved" ):
    # the solver's laplacian reads `facet_mass`: in 3D it did not compile before ( `sdotplan/Sweep.h`, `density_facet_mass` )
    import numpy
    from sdot import Iterative, OtProblem, SumOfDiracs, SumOfGaussians
    rng = numpy.random.default_rng( 34 )
    n = 40
    dens = SumOfGaussians( positions = rng.uniform( 0.35, 0.65, size = ( 4, 3 ) ), sigmas = [ 0.15, 0.2, 0.18, 0.25 ] )
    # ( a loose tolerance: the measures of a 3D gaussian come from the adaptive quadrature of `PointwiseDensity`, and Newton stagnates
    # around a residual of 1e-3 / n -- the facet masses are exact, see above )
    plan = OtProblem( SumOfDiracs( rng.uniform( 0.2, 0.8, size = ( n, 3 ) ) ), dens ).solve( Iterative( tol = 1e-2 / n ) )
    m = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert plan.converged and numpy.abs( m * n - 1 ).max() < 1e-2, ( plan.stats[ "status" ], numpy.abs( m * n - 1 ).max() )
