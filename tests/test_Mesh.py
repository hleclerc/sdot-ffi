"""A mesh of simplices as a density ( `include/sdot/Mesh.h`, `distributions/Mesh.py` ): continuous piecewise linear, one value per node.

The witness is that the density does not depend on HOW the domain is meshed: two triangulations with the values of the same
affine function are the same density, whatever the pieces a cell is cut into."""

from errand import test, skip


def _square_mesh( n, seed, lo = 0.0, hi = 1.0 ):
    """a Delaunay triangulation of the square with `n` random inner points, and its four corners"""
    import numpy
    from scipy.spatial import Delaunay
    rng = numpy.random.default_rng( seed )
    corners = numpy.array( [ [ lo, lo ], [ hi, lo ], [ lo, hi ], [ hi, hi ] ] )
    pts = numpy.concatenate( [ corners, rng.uniform( lo, hi, size = ( n, 2 ) ) ] )
    return pts, Delaunay( pts ).simplices


def _affine( x ):
    return 1.0 + 0.7 * x[ :, 0 ] + 1.9 * x[ :, 1 ]


def _seeds( n, seed ):
    import numpy
    return numpy.random.default_rng( seed ).uniform( 0.05, 0.95, size = ( n, 2 ) )


if test( "the_mass_of_a_mesh_is_exact" ):
    import numpy
    from sdot import Mesh
    nodes, sx = _square_mesh( 300, 1 )
    m = Mesh( nodes, sx, values = _affine( nodes ) )
    # int ( 1 + 0.7 x + 1.9 y ) over the unit square
    assert abs( float( m.mass ) - ( 1 + 0.35 + 0.95 ) ) < 1e-12, float( m.mass )
    # the box of the mesh, widened by the margin on each side
    pad = Mesh.MARGIN
    assert numpy.allclose( m.bounding_half_spaces()[ 1 ], [ 1 + pad, 1 + pad, pad, pad ] )


if test( "constant_values_are_lebesgue_whatever_the_triangulation" ):
    import numpy
    from sdot import Mesh, PowerDiagram, box_half_spaces
    nodes, sx = _square_mesh( 400, 2 )
    pos = _seeds( 40, 3 )
    w = numpy.random.default_rng( 4 ).uniform( -0.01, 0.01, 40 )
    ref = numpy.asarray( PowerDiagram( pos, w, boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ), kernel_dtype = "FP64" ).measures ).reshape( -1 )
    got = numpy.asarray( PowerDiagram( pos, w, distribution = Mesh( nodes, sx ), kernel_dtype = "FP64" ).measures ).reshape( -1 )
    assert abs( got.sum() - 1 ) < 1e-12 and numpy.abs( got - ref ).max() < 1e-12, numpy.abs( got - ref ).max()


if test( "an_affine_density_does_not_depend_on_the_triangulation" ):
    import numpy
    from sdot import Mesh, PowerDiagram
    two = numpy.array( [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ], [ 1, 1 ] ], dtype = float )
    coarse = Mesh( two, [ [ 0, 1, 2 ], [ 1, 3, 2 ] ], values = _affine( two ) )
    nodes, sx = _square_mesh( 500, 5 )
    fine = Mesh( nodes, sx, values = _affine( nodes ) )
    pos = _seeds( 30, 6 )
    w = numpy.random.default_rng( 7 ).uniform( -0.01, 0.01, 30 )
    res = []
    for dist in ( coarse, fine ):
        pd = PowerDiagram( pos, w, distribution = dist, kernel_dtype = "FP64" )
        mass, first, second = pd.moments
        res.append( ( numpy.asarray( pd.measures ).reshape( -1 ), numpy.asarray( mass ).reshape( -1 ),
                      numpy.asarray( first ).reshape( -1, 2 ), numpy.asarray( second ).reshape( -1 ) ) )
    for a, b, what in zip( res[ 0 ], res[ 1 ], ( "measures", "mass", "first moment", "second moment" ) ):
        assert numpy.abs( a - b ).max() < 1e-11, ( what, numpy.abs( a - b ).max() )
    assert numpy.abs( res[ 0 ][ 0 ] - res[ 0 ][ 1 ] ).max() < 1e-12            # the moments' mass is the measure
    assert abs( res[ 0 ][ 0 ].sum() - 1 ) < 1e-12


if test( "the_mesh_carries_its_own_derivative" ):
    import numpy
    from loom.testing import check_grad, need
    from sdot import Mesh, PowerDiagram, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    need( "grad" )
    nodes, sx = _square_mesh( 12, 8 )
    pos = _seeds( 6, 9 )
    w = numpy.random.default_rng( 10 ).uniform( -0.01, 0.01, 6 )
    vals = _affine( nodes )
    # with respect to the values of the nodes ( through the normalization ), the positions and the weights of the seeds
    check_grad( lambda v: PowerDiagram( pos, w, distribution = Mesh( nodes, sx, values = v ) ).measures, vals )
    check_grad( lambda p, q: PowerDiagram( p, weights = q, distribution = Mesh( nodes, sx, values = vals ) ).measures, pos, w )


if test( "the_transport_towards_a_mesh_is_solved" ):
    import numpy
    from sdot import Iterative, Mesh, OtProblem, SumOfDiracs
    nodes, sx = _square_mesh( 200, 11 )
    n = 60
    plan = OtProblem( SumOfDiracs( _seeds( n, 12 ) ), Mesh( nodes, sx, values = _affine( nodes ) ) ).solve( Iterative() )
    m = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert numpy.abs( m * n - 1 ).max() < 1e-6, numpy.abs( m * n - 1 ).max()


if test( "a_tetrahedral_mesh_is_a_density_too" ):
    import numpy
    from scipy.spatial import Delaunay
    from sdot import Mesh, PowerDiagram
    rng = numpy.random.default_rng( 13 )
    corners = numpy.array( [ [ x, y, z ] for x in ( 0, 1 ) for y in ( 0, 1 ) for z in ( 0, 1 ) ], dtype = float )
    nodes = numpy.concatenate( [ corners, rng.uniform( 0, 1, size = ( 60, 3 ) ) ] )
    mesh = Mesh( nodes, Delaunay( nodes ).simplices, values = 1.0 + nodes @ numpy.array( [ 0.5, 1.0, 1.5 ] ) )
    assert abs( float( mesh.mass ) - ( 1 + 0.25 + 0.5 + 0.75 ) ) < 1e-12
    pos = rng.uniform( 0.1, 0.9, size = ( 8, 3 ) )
    pd = PowerDiagram( pos, distribution = mesh, kernel_dtype = "FP64" )
    m = numpy.asarray( pd.measures ).reshape( -1 )
    assert abs( m.sum() - 1 ) < 1e-10, m.sum()
    # the same density on the two-tetrahedra-free coarse mesh: the corners alone ( a cube cut into 5 or 6 tetrahedra )
    cube = Delaunay( corners ).simplices
    coarse = Mesh( corners, cube, values = 1.0 + corners @ numpy.array( [ 0.5, 1.0, 1.5 ] ) )
    mc = numpy.asarray( PowerDiagram( pos, distribution = coarse, kernel_dtype = "FP64" ).measures ).reshape( -1 )
    assert numpy.abs( m - mc ).max() < 1e-10, numpy.abs( m - mc ).max()


def _lattice_mesh( k, flip = False ):
    """the unit square cut in k x k squares, each in two triangles ( the diagonal the other way if `flip` ), values of the affine density"""
    import numpy
    g = numpy.linspace( 0, 1, k + 1 )
    nodes = numpy.array( [ [ x, y ] for y in g for x in g ] )
    idx = lambda i, j: j * ( k + 1 ) + i
    tris = []
    for j in range( k ):
        for i in range( k ):
            if flip:
                tris += [ [ idx( i, j ), idx( i + 1, j ), idx( i, j + 1 ) ], [ idx( i + 1, j ), idx( i + 1, j + 1 ), idx( i, j + 1 ) ] ]
            else:
                tris += [ [ idx( i, j ), idx( i + 1, j ), idx( i + 1, j + 1 ) ], [ idx( i, j ), idx( i + 1, j + 1 ), idx( i, j + 1 ) ] ]
    return nodes, tris


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


if test( "the_hessian_of_a_mesh_is_its_finite_difference" ):
    # `hessian_rows` reads the density on each facet ( `facet_mass` ): the Jacobian of the measures with respect to the weights
    import numpy
    from sdot import Mesh, PowerDiagram
    nodes, sx = _square_mesh( 60, 19 )
    n = 20
    pos = _seeds( n, 20 )
    w0 = numpy.random.default_rng( 21 ).uniform( -0.01, 0.01, n )
    mesh = Mesh( nodes, sx, values = _affine( nodes ) )
    pd = PowerDiagram( pos, w0, distribution = mesh, kernel_dtype = "FP64" )
    H = _dense_hessian( pd, n )
    assert numpy.abs( H - H.T ).max() < 1e-10 and numpy.abs( H.sum( axis = 1 ) ).max() < 1e-12
    h = 1e-6
    for j in ( 0, 7, 19 ):
        e = numpy.zeros( n ); e[ j ] = h
        pd.weights = w0 + e; mp = numpy.asarray( pd.measures ).reshape( -1 )
        pd.weights = w0 - e; mm = numpy.asarray( pd.measures ).reshape( -1 )
        assert numpy.abs( ( mp - mm ) / ( 2 * h ) - H[ :, j ] ).max() < 1e-7, ( j, numpy.abs( ( mp - mm ) / ( 2 * h ) - H[ :, j ] ).max() )


if test( "a_facet_on_a_mesh_line_counts_once" ):
    # seeds on the centers of a 6 x 6 lattice: every bisector between neighbours lies EXACTLY on a line of the 6 x 6 mesh, where two
    # elements meet and hold the facet both. It counts once, by the strict convention on the nodes ( `Mesh.h::owns_facet` ): the
    # Hessian is that of a 7 x 7 mesh of the same affine density, whose lines meet no bisector, and of the other diagonal
    import numpy
    from sdot import Mesh, PowerDiagram
    k = 6
    c = ( numpy.arange( k ) + 0.5 ) / k
    seeds = numpy.array( [ [ x, y ] for y in c for x in c ] )
    H = []
    for kk, flip in ( ( 6, False ), ( 6, True ), ( 7, False ) ):
        nodes, tris = _lattice_mesh( kk, flip )
        H.append( _dense_hessian( PowerDiagram( seeds, numpy.zeros( k * k ), distribution = Mesh( nodes, tris, values = _affine( nodes ) ),
                                                kernel_dtype = "FP64" ), k * k ) )
    assert numpy.abs( H[ 1 ] - H[ 0 ] ).max() < 1e-10 * numpy.abs( H[ 0 ] ).max(), numpy.abs( H[ 1 ] - H[ 0 ] ).max()
    assert numpy.abs( H[ 2 ] - H[ 0 ] ).max() < 1e-10 * numpy.abs( H[ 0 ] ).max(), numpy.abs( H[ 2 ] - H[ 0 ] ).max()


if test( "a_solve_on_a_mesh_whose_lines_meet_the_bisectors" ):
    # seeds on a lattice moving along its lines only, so that the first bisectors lie on the lines of the mesh: same weights as on a mesh
    # whose lines do not meet them
    import numpy
    from sdot import Iterative, Mesh, OtProblem, SumOfDiracs
    k = 6
    c = ( numpy.arange( k ) + 0.5 ) / k
    seeds = numpy.array( [ [ x, y ] for y in c for x in c ] ) + numpy.array( [ 0.0, 1.0 ] ) * numpy.random.default_rng( 14 ).uniform( -0.03, 0.03, ( k * k, 1 ) )
    res = []
    for kk, flip in ( ( 6, False ), ( 7, False ) ):
        nodes, tris = _lattice_mesh( kk, flip )
        plan = OtProblem( SumOfDiracs( seeds ), Mesh( nodes, tris, values = _affine( nodes ) ) ).solve( Iterative() )
        m = numpy.asarray( plan.cell_masses ).reshape( -1 )
        assert numpy.abs( m * k * k - 1 ).max() < 1e-8, ( kk, numpy.abs( m * k * k - 1 ).max() )
        res.append( ( numpy.asarray( plan.weights ).reshape( -1 ), plan.stats[ "nb_iter" ] ) )
    assert numpy.abs( res[ 1 ][ 0 ] - res[ 0 ][ 0 ] ).max() < 1e-8
    assert abs( res[ 1 ][ 1 ] - res[ 0 ][ 1 ] ) <= 1 and max( r[ 1 ] for r in res ) < 12, [ r[ 1 ] for r in res ]


if test( "the_transport_towards_a_mesh_is_solved_on_the_card" ):
    import numpy
    import loom
    from sdot import Iterative, Mesh, OtProblem, PowerDiagram, SumOfDiracs
    if not getattr( loom.resolved_device(), "is_cuda_gpu", False ):
        skip( "the card's solve only exists on a CUDA device" )
    else:
        nodes, sx = _square_mesh( 3000, 15 )
        n = 20000
        mesh = Mesh( nodes, sx, values = _affine( nodes ) )
        pos = numpy.random.default_rng( 16 ).uniform( 0.05, 0.95, size = ( n, 2 ) )
        plan = OtProblem( SumOfDiracs( pos ), mesh ).solve( Iterative() )
        m = numpy.asarray( plan.cell_masses ).reshape( -1 )
        print( "mesh card solve: max relative mass error", numpy.abs( m * n - 1 ).max(), "iterations", plan.stats[ "nb_iter" ] )
        assert numpy.abs( m * n - 1 ).max() < 1e-4
        # the generic kernels ( the mesh is not a card density for the cells alone ) agree on the solved weights
        gen = numpy.asarray( PowerDiagram( pos, plan.weights, distribution = mesh, kernel_dtype = "FP64" ).measures ).reshape( -1 )
        assert numpy.abs( gen * n - 1 ).max() < 1e-3, numpy.abs( gen * n - 1 ).max()


if test( "the_transport_towards_a_tetrahedral_mesh_is_solved_on_the_card" ):
    # 3D on the card ( `gpu/Density3D.cuh`: the tetrahedra of the cells' fans against the mesh ): a DG1 density, affine with a
    # lighter ball ( a jump on its border ), against the generic kernels at the solved weights -- masses, barycentres, cost
    import numpy
    from scipy.spatial import Delaunay
    import loom
    from sdot import Iterative, Mesh, OtProblem, PowerDiagram, SumOfDiracs
    if not getattr( loom.resolved_device(), "is_cuda_gpu", False ):
        skip( "the card's solve only exists on a CUDA device" )
    else:
        rng = numpy.random.default_rng( 31 )
        corners = numpy.array( [ [ x, y, z ] for x in ( 0, 1 ) for y in ( 0, 1 ) for z in ( 0, 1 ) ], dtype = float )
        nodes = numpy.concatenate( [ corners, rng.uniform( 0, 1, size = ( 1500, 3 ) ) ] )
        sx = Delaunay( nodes ).simplices
        aff = 1.0 + nodes @ numpy.array( [ 0.5, 1.0, 1.5 ] )
        ball = numpy.linalg.norm( nodes[ sx ].mean( axis = 1 ) - 0.5, axis = 1 ) < 0.25
        mesh = Mesh( nodes, sx, values = aff[ sx ] * numpy.where( ball, 0.2, 1.0 )[ :, None ] )
        n = 2000
        pos = rng.uniform( 0.05, 0.95, size = ( n, 3 ) )
        plan = OtProblem( SumOfDiracs( pos ), mesh ).solve( Iterative( continuation = "never" ) )
        m = numpy.asarray( plan.cell_masses ).reshape( -1 )
        print( "tet mesh card solve: max relative mass error", numpy.abs( m * n - 1 ).max(), "iterations", plan.stats[ "nb_iter" ] )
        assert plan.converged and numpy.abs( m * n - 1 ).max() < 1e-4, ( plan.stats, numpy.abs( m * n - 1 ).max() )
        pd = PowerDiagram( pos, plan.weights, distribution = mesh, kernel_dtype = "FP64" )
        gen = numpy.asarray( pd.measures ).reshape( -1 )
        print( "generic kernels at the card's weights: max relative mass error", numpy.abs( gen * n - 1 ).max() )
        assert numpy.abs( gen * n - 1 ).max() < 1e-6, numpy.abs( gen * n - 1 ).max()
        mass, first, second = PowerDiagram( pos, plan.weights, distribution = mesh, kernel_dtype = "FP64" ).moments
        mass, first, second = ( numpy.asarray( a ).reshape( n, -1 ).squeeze() for a in ( mass, first, second ) )
        bary = numpy.asarray( plan.barycenters ).reshape( n, 3 )
        cost = float( numpy.sum( second - 2 * ( pos * first ).sum( axis = 1 ) + ( pos ** 2 ).sum( axis = 1 ) * mass ) )
        print( "barycentres", numpy.abs( bary - first / mass[ :, None ] ).max(), "cost", abs( float( plan.cost ) - cost ) / cost )
        assert numpy.abs( bary - first / mass[ :, None ] ).max() < 1e-6, numpy.abs( bary - first / mass[ :, None ] ).max()
        assert abs( float( plan.cost ) - cost ) < 1e-8 * cost, ( float( plan.cost ), cost )


if test( "the_barycenters_and_the_cost_of_a_solve_are_the_moments_of_the_mesh" ):
    # the solver's barycenters and `W_2^2` ( on the card: `DensMesh`'s closed forms for the moments, cell by cell; on the CPU:
    # `MeshElementDensity`'s ) against the moments of the generic path at the fitted weights: `int x rho`, and
    # `sum_i ( int |x|^2 rho - 2 p_i . int x rho + |p_i|^2 m_i )`
    import numpy
    from sdot import Iterative, Mesh, OtProblem, PowerDiagram, SumOfDiracs
    nodes, sx = _square_mesh( 400, 17 )
    n = 300
    mesh = Mesh( nodes, sx, values = _affine( nodes ) )
    pos = numpy.random.default_rng( 18 ).uniform( 0.05, 0.95, size = ( n, 2 ) )
    plan = OtProblem( SumOfDiracs( pos ), mesh ).solve( Iterative() )
    mass, first, second = PowerDiagram( pos, plan.weights, distribution = mesh, kernel_dtype = "FP64" ).moments
    mass, first, second = ( numpy.asarray( a ).reshape( n, -1 ).squeeze() for a in ( mass, first, second ) )
    bary = numpy.asarray( plan.barycenters ).reshape( n, 2 )
    cost = float( numpy.sum( second - 2 * ( pos * first ).sum( axis = 1 ) + ( pos ** 2 ).sum( axis = 1 ) * mass ) )
    assert numpy.abs( bary - first / mass[ :, None ] ).max() < 1e-6, numpy.abs( bary - first / mass[ :, None ] ).max()
    assert abs( float( plan.cost ) - cost ) < 1e-8 * cost, ( float( plan.cost ), cost )


if test( "the_width_continuation_of_a_tetrahedral_mesh_goes_through_its_spreading" ):
    # `continuation = "always"`: the path of the spread meshes ( `Mesh.spread`, the corner averages ), then the mesh
    import numpy
    from scipy.spatial import Delaunay
    from sdot import Iterative, Mesh, OtProblem, SumOfDiracs
    rng = numpy.random.default_rng( 23 )
    corners = numpy.array( [ [ x, y, z ] for x in ( 0, 1 ) for y in ( 0, 1 ) for z in ( 0, 1 ) ], dtype = float )
    nodes = numpy.concatenate( [ corners, rng.uniform( 0, 1, size = ( 300, 3 ) ) ] )
    mesh = Mesh( nodes, Delaunay( nodes ).simplices, values = 1.0 + nodes @ numpy.array( [ 0.5, 1.0, 1.5 ] ) )
    n = 60
    plan = OtProblem( SumOfDiracs( rng.uniform( 0.05, 0.95, size = ( n, 3 ) ) ), mesh ).solve( Iterative( continuation = "always" ) )
    m = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert plan.converged and numpy.abs( m * n - 1 ).max() < 1e-6, ( plan.stats, numpy.abs( m * n - 1 ).max() )
    assert plan.stats[ "spread_stages" ][ -1 ] == 0 and len( plan.stats[ "spread_stages" ] ) >= 2, plan.stats[ "spread_stages" ]


if test( "a_dg1_mesh_holds_a_hole" ):
    # a value per corner: zero on the simplices of a disk, a discontinuity on its border; the mass is exact, `spread( 0 )` is the
    # mesh, and spreading keeps the mass and the positivity and fills the hole
    import numpy
    from sdot import Mesh
    nodes, sx = _square_mesh( 400, 31 )
    cv = numpy.ones( sx.shape )
    hole = numpy.linalg.norm( nodes[ sx ].mean( 1 ) - 0.5, axis = 1 ) < 0.25
    cv[ hole ] = 0
    P = nodes[ sx ]
    vol = numpy.abs( numpy.linalg.det( P[ :, 1: ] - P[ :, :1 ] ) ) / 2
    m = Mesh( nodes, sx, values = cv )
    assert abs( float( m.mass ) - vol[ ~hole ].sum() ) < 1e-12
    assert m.spread( 0 ) is m
    w = numpy.asarray( m.corner_weights )
    for k in ( 0.5, 3, 40 ):
        v = numpy.asarray( m.spread( k ).values )
        assert v.min() >= 0 and abs( ( w * v ).sum() / float( m.mass ) - 1 ) < 1e-10, k
    inside = numpy.asarray( m.values ).max( axis = 1 ) == 0
    assert numpy.asarray( m.spread( 40 ).values )[ inside ].min() > 0


if test( "the_spreading_solves_what_the_plain_newton_does_not" ):
    # two small far apart bumps on a uniform mesh, zero elsewhere: the plain Newton does not converge, the continuation does
    import numpy
    from scipy.spatial import Delaunay
    from sdot import Iterative, Mesh, OtProblem, SumOfDiracs
    rng = numpy.random.default_rng( 1 )
    nodes = numpy.concatenate( [ numpy.array( [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ], [ 1, 1 ] ], float ), rng.uniform( 0, 1, ( 3000, 2 ) ) ] )
    sx = Delaunay( nodes ).simplices
    def bump( c, r ):
        return numpy.maximum( 0, 1 - numpy.linalg.norm( nodes[ sx ] - c, axis = -1 ) / r )
    mesh = Mesh( nodes, sx, values = bump( [ 0.2, 0.25 ], 0.06 ) + 0.5 * bump( [ 0.8, 0.7 ], 0.04 ) )
    n = 300
    pos = rng.uniform( 0.02, 0.98, ( n, 2 ) )
    plain = OtProblem( SumOfDiracs( pos ), mesh ).solve( Iterative( continuation = "never", on_failure = "ignore", max_iter = 100 ) )
    assert not plain.converged
    plan = OtProblem( SumOfDiracs( pos ), mesh ).solve( Iterative() )
    m = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert plan.converged and numpy.abs( m * n - 1 ).max() < 1e-5, ( plan.stats[ "status" ], numpy.abs( m * n - 1 ).max() )   # ( tol 1e-8 on the masses )
    assert plan.stats[ "spread_stages" ][ -1 ] == 0, plan.stats
