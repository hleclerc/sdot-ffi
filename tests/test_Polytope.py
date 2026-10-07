from errand import test, skip

if test( "measures_of_a_polygon_equal_the_lebesgue_ones_on_its_half_spaces" ):
    import numpy
    from sdot import Polygon, PowerDiagram, box_half_spaces
    rng = numpy.random.default_rng( 1 )
    n = 30
    pos = rng.uniform( 0.05, 0.6, size = ( n, 2 ) )
    w = rng.uniform( -0.01, 0.01, n )
    tri = Polygon( [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ] ] )
    assert abs( tri.volume - 0.5 ) < 1e-14 and abs( float( tri.mass ) - 0.5 ) < 1e-14
    ref = numpy.asarray( PowerDiagram( pos, w, boundaries = tri.bounding_half_spaces(), kernel_dtype = "FP64" ).measures ).reshape( -1 ) / 0.5
    # the polytope alone bounds the diagram, or cuts the cells of a bigger domain: same result
    for bnd in ( None, box_half_spaces( [ -5, -5 ], [ 5, 5 ] ) ):
        pd = PowerDiagram( pos, w, boundaries = bnd, distribution = tri, kernel_dtype = "FP64" )
        m = numpy.asarray( pd.measures ).reshape( -1 )
        assert abs( m.sum() - 1 ) < 1e-12 and numpy.abs( m - ref ).max() < 1e-12


if test( "a_polygon_must_be_convex" ):
    import pytest
    from sdot import Polygon
    with pytest.raises( ValueError ):
        Polygon( [ [ 0, 0 ], [ 2, 0 ], [ 2, 2 ], [ 0, 2 ], [ 1, 1 ] ] )


if test( "a_box_is_an_image_of_one_pixel" ):
    import numpy
    from sdot import Box
    b = Box( origin = [ 0.2, 0.1 ], frame = [ 0.5, 0.7 ] )
    assert abs( float( b.mass ) - 0.35 ) < 1e-14
    lo, hi = b.bounding_half_spaces()
    assert numpy.allclose( numpy.sort( hi / numpy.linalg.norm( lo, axis = 1 ) ), numpy.sort( [ 0.7, 0.8, -0.2, -0.1 ] ) )
    # a sheared box: the area is |det( frame )|
    assert abs( float( Box( frame = [ [ 1, 0 ], [ 1, 2 ] ] ).mass ) - 2 ) < 1e-14


if test( "the_transport_towards_a_polygon_is_solved" ):
    import numpy
    from sdot import Iterative, OtProblem, Polygon, SumOfDiracs
    rng = numpy.random.default_rng( 3 )
    n = 40
    hexa = Polygon( [ [ numpy.cos( t ), numpy.sin( t ) ] for t in numpy.arange( 6 ) * numpy.pi / 3 ] )
    pos = rng.uniform( -0.3, 0.3, size = ( n, 2 ) )
    plan = OtProblem( SumOfDiracs( pos ), hexa ).solve( Iterative() )
    m = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert numpy.abs( m - 1 / n ).max() < 1e-6, numpy.abs( m - 1 / n ).max()


# ---- the card: a polygon is the bounding box and the cuts the cells start from ( `gpu/Cell2D.cuh::Cuts` ) ----

def _regular_polygon( k, r = 0.45, c = 0.5 ):
    import numpy
    a = 2 * numpy.pi * numpy.arange( k ) / k + 0.3
    return numpy.stack( [ c + r * numpy.cos( a ), c + r * numpy.sin( a ) ], axis = 1 )


if test( "the_card_cells_of_a_polygon_are_the_generic_ones" ):
    import numpy
    import loom
    from sdot import AaBsp, Polygon, PowerDiagram
    if not getattr( loom.resolved_device(), "is_cuda_gpu", False ):
        skip( "the dedicated cell kernels only exist on a CUDA device" )
    else:
        rng = numpy.random.default_rng( 7 )
        n = 4000
        for k in ( 3, 5, 9 ):                     # 9 sides: the cells of the border overflow the first registers
            poly = Polygon( _regular_polygon( k ) )
            pos = rng.uniform( 0.0, 1.0, size = ( n, 2 ) )
            pos = pos[ numpy.all( poly.bounding_half_spaces()[ 0 ] @ pos.T <= poly.bounding_half_spaces()[ 1 ][ :, None ], axis = 0 ) ]
            w = rng.uniform( -0.2, 0.2, len( pos ) ) / len( pos )
            tree = AaBsp( pos, w )
            gen = PowerDiagram( pos, w, distribution = poly, kernel_dtype = "FP64", accelerator = tree )
            gen.use_card_cells = False
            ref = numpy.asarray( gen.measures.value ).reshape( -1 )
            for kernel in ( "FP64", "FP32" ):
                pd = PowerDiagram( pos, w, distribution = poly, kernel_dtype = kernel, accelerator = tree )
                assert pd._card_variant() is not None, ( k, "the card did not take the polygon" )
                assert pd.start_vertices.is_defined and pd.start_vertices.shape[ 0 ] == k, "the polygon is not the cell the cuts start from"
                m = numpy.asarray( pd.measures.value ).reshape( -1 )
                assert abs( m.sum() - 1 ) < 1e-6 and numpy.abs( m - ref ).max() < 1e-6 * ref.max(), ( k, kernel, numpy.abs( m - ref ).max() )


if test( "the_transport_towards_a_polygon_is_solved_on_the_card" ):
    import numpy
    import loom
    from sdot import Iterative, OtProblem, Polygon, SumOfDiracs
    if not getattr( loom.resolved_device(), "is_cuda_gpu", False ):
        skip( "the card's solve only exists on a CUDA device" )
    else:
        rng = numpy.random.default_rng( 8 )
        n = 20000
        poly = Polygon( _regular_polygon( 7 ) )
        pos = rng.uniform( 0.4, 0.6, size = ( n, 2 ) )
        plan = OtProblem( SumOfDiracs( pos ), poly ).solve( Iterative() )
        m = numpy.asarray( plan.cell_masses ).reshape( -1 )
        print( "2d card solve: max relative mass error", numpy.abs( m * n - 1 ).max() )
        assert numpy.abs( m * n - 1 ).max() < 1e-3, numpy.abs( m * n - 1 ).max()


def _sphere_points( k, seed = 0, r = 0.45, c = 0.5 ):
    import numpy
    p = numpy.random.default_rng( seed ).normal( size = ( k, 3 ) )
    return c + r * p / numpy.linalg.norm( p, axis = 1 )[ :, None ]


def _tangent_polytope( k, seed, r = 0.45, c = 0.5 ):
    """k random planes tangent to a sphere, and the six of the axes: a SIMPLE polyhedron ( three facets per vertex, generically )"""
    import numpy
    from sdot import Polytope
    n = numpy.random.default_rng( seed ).normal( size = ( k, 3 ) )
    n = numpy.concatenate( [ n / numpy.linalg.norm( n, axis = 1 )[ :, None ], numpy.eye( 3 ), -numpy.eye( 3 ) ] )
    return Polytope( n, n @ numpy.full( 3, c ) + r )


if test( "polyhedron_volume_and_cpu_measures" ):
    import numpy
    from sdot import Polyhedron, PowerDiagram
    rng = numpy.random.default_rng( 9 )
    cube = Polyhedron( [ [ x, y, z ] for x in ( 0, 2 ) for y in ( 0, 1 ) for z in ( 0, 1 ) ] )
    assert abs( cube.volume - 2 ) < 1e-12
    poly = Polyhedron( _sphere_points( 10 ) )
    pos = rng.uniform( 0.35, 0.65, size = ( 200, 3 ) )
    m = numpy.asarray( PowerDiagram( pos, distribution = poly, kernel_dtype = "FP64" ).measures ).reshape( -1 )
    assert abs( m.sum() - 1 ) < 1e-10, m.sum()


if test( "the_card_cells_of_a_polyhedron_are_the_generic_ones" ):
    # a simple polyhedron starts the cells ( `start_vertices` ); the convex hull of random points is not simple ( its vertices
    # have many facets ): it cuts the box of its vertices instead. Both are the same cells as the generic path's.
    import numpy
    import loom
    from sdot import AaBsp, Polyhedron, PowerDiagram
    if not getattr( loom.resolved_device(), "is_cuda_gpu", False ):
        skip( "the dedicated cell kernels only exist on a CUDA device" )
    else:
        rng = numpy.random.default_rng( 10 )
        for label, poly, starts in [ ( "simple 4", _tangent_polytope( 4, 1 ), True ), ( "simple 8", _tangent_polytope( 8, 2 ), True ),
                                     ( "simple 20", _tangent_polytope( 20, 3 ), True ), ( "hull 8", Polyhedron( _sphere_points( 8, seed = 8 ) ), False ),
                                     ( "hull 20", Polyhedron( _sphere_points( 20, seed = 20 ) ), False ) ]:
            pos = rng.uniform( 0.0, 1.0, size = ( 20000, 3 ) )
            dirs, offs = poly.bounding_half_spaces()
            pos = pos[ numpy.all( dirs @ pos.T <= offs[ :, None ], axis = 0 ) ]
            w = rng.uniform( -0.2, 0.2, len( pos ) ) / len( pos ) ** ( 2 / 3 )
            tree = AaBsp( pos, w )
            gen = PowerDiagram( pos, w, distribution = poly, kernel_dtype = "FP64", accelerator = tree )
            gen.use_card_cells = False
            ref = numpy.asarray( gen.measures.value ).reshape( -1 )
            for kernel in ( "FP64", "FP32" ):
                pd = PowerDiagram( pos, w, distribution = poly, kernel_dtype = kernel, accelerator = tree )
                assert pd._card_variant() is not None, ( label, "the card did not take the polyhedron" )
                assert pd.start_vertices.is_defined == starts, ( label, "the start cell is not what was expected" )
                m = numpy.asarray( pd.measures.value ).reshape( -1 )
                assert abs( m.sum() - 1 ) < 1e-5 and numpy.abs( m - ref ).max() < 1e-5 * ref.max(), ( label, kernel, numpy.abs( m - ref ).max() / ref.max() )


if test( "the_transport_towards_a_polyhedron_is_solved_on_the_card" ):
    import numpy
    import loom
    from sdot import Iterative, OtProblem, Polyhedron, SumOfDiracs
    if not getattr( loom.resolved_device(), "is_cuda_gpu", False ):
        skip( "the card's solve only exists on a CUDA device" )
    else:
        rng = numpy.random.default_rng( 11 )
        n = 20000
        for label, poly in [ ( "simple", _tangent_polytope( 10, 5 ) ), ( "hull", Polyhedron( _sphere_points( 12, seed = 5 ) ) ) ]:
            pos = rng.uniform( 0.4, 0.6, size = ( n, 3 ) )
            plan = OtProblem( SumOfDiracs( pos ), poly ).solve( Iterative() )
            m = numpy.asarray( plan.cell_masses ).reshape( -1 )
            print( f"3d card solve ( { label } ): max relative mass error", numpy.abs( m * n - 1 ).max() )
            assert numpy.abs( m * n - 1 ).max() < 1e-3, ( label, numpy.abs( m * n - 1 ).max() )


if test( "a_box_without_dimension_takes_it_where_it_is_used" ):
    import numpy
    import pytest
    from sdot import Box, Iterative, OtProblem, PowerDiagram, SumOfDiracs
    unit = Box()                                  # the unit cube, whatever the dimension
    with pytest.raises(ValueError):
        unit.bounding_half_spaces()
    for d in ( 2, 3 ):
        pos = numpy.random.default_rng( d ).uniform( 0.1, 0.9, size = ( 50, d ) )
        m = numpy.asarray( PowerDiagram( pos, distribution = unit, kernel_dtype = "FP64" ).measures ).reshape( -1 )
        assert abs( m.sum() - 1 ) < 1e-10, ( d, m.sum() )
    plan = OtProblem( SumOfDiracs( numpy.random.default_rng( 1 ).uniform( 0.2, 0.8, size = ( 30, 2 ) ) ), unit ).solve( Iterative() )
    assert numpy.abs( numpy.asarray( plan.cell_masses ).reshape( -1 ) * 30 - 1 ).max() < 1e-5


if test( "a_pyramid_has_no_start_cell_and_cuts_a_box_instead" ):
    # the apex has four facets: not simple, so the cells start from the box of the vertices and the facets cut it
    import numpy
    import loom
    from sdot import AaBsp, Polyhedron, PowerDiagram
    from sdot.StartCell import start_cell
    pyr = Polyhedron( [ [ 0, 0, 0 ], [ 1, 0, 0 ], [ 1, 1, 0 ], [ 0, 1, 0 ], [ 0.5, 0.5, 1 ] ] )
    cell = start_cell( *pyr.bounding_half_spaces() )
    assert cell is not None and "vertices" not in cell and numpy.allclose( cell[ "bounding_box" ][ 1 ], [ 1, 1, 1 ] )
    rng = numpy.random.default_rng( 12 )
    pos = rng.uniform( 0.0, 1.0, size = ( 3000, 3 ) )
    dirs, offs = pyr.bounding_half_spaces()
    pos = pos[ numpy.all( dirs @ pos.T <= offs[ :, None ], axis = 0 ) ]
    tree = AaBsp( pos )
    ref = numpy.asarray( PowerDiagram( pos, distribution = pyr, kernel_dtype = "FP64", accelerator = "plain" ).measures ).reshape( -1 )
    assert abs( ref.sum() - 1 ) < 1e-9
    if getattr( loom.resolved_device(), "is_cuda_gpu", False ):
        pd = PowerDiagram( pos, distribution = pyr, kernel_dtype = "FP64", accelerator = tree )
        assert pd._card_variant() is not None and not pd.start_vertices.is_defined
        m = numpy.asarray( pd.measures.value ).reshape( -1 )
        assert numpy.abs( m - ref ).max() < 1e-6 * ref.max()
