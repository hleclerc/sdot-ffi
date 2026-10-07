
from errand import Param, experiment, test


# Geometry is cut in FP32 by default ( see `Cell.py` ): the CORRECTNESS tests run with the
# kernel in double, and the `float` kernel has its own tests ( "the_float_kernel_*" ).


if test( "basic" ):
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 2, 0 ], [ 0, 1 ] ] )
    assert c.measure == 2

    # 2D: `vertex_positions` is already the polygon in cyclic order, so no lattice --
    # `Cell_2` does not even have these attributes
    from sdot import Cell_2
    assert isinstance( c, Cell_2 ) and not hasattr( c, "vertex_cuts" )
    assert numpy.allclose( numpy.asarray( c.vertex_positions ), [ [ 0, 0 ], [ 2, 0 ], [ 2, 1 ], [ 0, 1 ] ] )

if test( "basic_1D" ):
    # 1D: the cell is a SEGMENT ( `Cell_1` ) -- two ends, one cut per end
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    from sdot import Cell_1
    c = Cell.make_hypercube( 1, [ 0.5 ], [ [ 2 ] ] )
    assert isinstance( c, Cell_1 )
    assert c.measure == 2
    c.cut( [ 1 ], 1.5 )
    assert abs( float( c.measure ) - 1 ) < 1e-12
    assert numpy.allclose( c.vertices(), [ [ 0.5 ], [ 1.5 ] ] )
    c.cut( [ -1 ], -0.7 )
    assert abs( float( c.measure ) - 0.8 ) < 1e-12
    c.cut( [ 1 ], 0.6 )                                     # everything is outside
    assert int( c.nb_vertices.value ) == 0 and float( c.measure ) == 0

if test( "unbounded_1D" ):
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    c = Cell.make_unbounded( 1 )
    assert float( c.measure ) > 1e300 and not c.is_bounded
    c.cut( [ 1 ], 3.0 )
    assert float( c.measure ) > 1e300 and not c.is_bounded      # a half-line
    c.cut( [ -1 ], 1.0 )
    assert c.is_bounded and abs( float( c.measure ) - 4 ) < 1e-12

if test( "batch" ):
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 2, 0 ], [ 0, 1 ] ], batch_axes = [ new_batch_axis( 2 ) ] )
    assert tuple( c.measure.value ) == ( 2, 2 )
    assert c.nb_items == 2 and numpy.allclose( c.vertices( 1 ), c.vertices( 0 ) )

def _cell_with_coords( d, coords, cut_ids, vertex_cuts = None, vertex_nbrs = None ):
    """A cell set up DIRECTLY by its tensors -- which makes the measure differentiable with
    respect to the coordinates, `coords` being allowed to be a tracer."""
    c = Cell( d, init_as_unbounded = False )
    c.vertex_positions = coords
    c.cut_ids = cut_ids
    if vertex_cuts is not None:
        c.vertex_cuts = vertex_cuts
        c.vertex_nbrs = vertex_nbrs
    return c

if test( "grad_measure" ):
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    need( "grad" )
    # the adjoint of the shoelace formula, checked by finite differences on the coordinates themselves
    c = Cell.make_hypercube( 2, [ 0.3, -0.2 ], [ [ 2.0, 0.1 ], [ -0.3, 1.0 ] ] )
    c.cut( [ 1, 1 ], 1.5 )
    coords = loom.array( numpy.asarray( c.vertex_positions.value ) )
    ids = numpy.asarray( c.cut_ids.value )
    check_grad( lambda x: _cell_with_coords( 2, x, ids ).measure, coords )

if p := test( "cut" ):
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    c.cut( [ 1, 0 ], 0.1 )

    # a page written along the way: this test does not judge the IMAGE ( see the `experiment`s at the end of
    # the file for that ), it only checks that writing does not break.
    v = Visualizer()
    c.add_to_viz( v )
    v.write_html( p.out_dir / "cut.html" )

    assert c.measure == 0.1

if test( "cut_keeps_the_edge_to_cut_correspondence" ):
    # The 2D invariant `Local2` lives on: cut i carries the edge [ v_i, v_i+1 ], hence
    # `nb_cuts == nb_vertices`. We cut a corner off the unit square and check it edge by edge --
    # BOTH ends of edge i must be on plane i, read back from the geometry.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    c.cut( [ 1, 1 ], 1.5 )

    n = int( c.nb_vertices.value )
    assert n == 5 and int( c.nb_cuts.value ) == n
    assert abs( float( c.measure ) - 0.875 ) < 1e-12      # 1 - the triangle of side 1/2

    vp = c.vertices()
    cd, co = c.cut_planes
    for i in range( n ):
        j = ( i + 1 ) % n
        assert abs( cd[ i ] @ vp[ i ] - co[ i ] ) < 1e-12
        assert abs( cd[ i ] @ vp[ j ] - co[ i ] ) < 1e-12

    # the requested cut is found, normalized, among the planes read back
    assert any( numpy.allclose( cd[ i ], numpy.array( [ 1, 1 ] ) / numpy.sqrt( 2 ) ) and abs( co[ i ] - 1.5 / numpy.sqrt( 2 ) ) < 1e-12 for i in range( n ) )

if test( "cut_direction_is_not_normalized" ):
    # `offset` is the dot product as is, so ( 2n, 2o ) is the SAME half-space as
    # ( n, o ): a direction three times longer with an offset three times larger must
    # return exactly the same cell.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    a = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    b = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    a.cut( [ 1, 1 ], 1.5 )
    b.cut( [ 3, 3 ], 4.5 )

    assert int( a.nb_vertices.value ) == int( b.nb_vertices.value )
    assert numpy.allclose( a.vertices(), b.vertices() )
    assert abs( float( a.measure ) - float( b.measure ) ) < 1e-12

if test( "cut_degenerate" ):
    # two degeneracies. A cut that cuts nothing must add NOTHING; a cut that excludes
    # everything must empty the cell cleanly ( 0 vertices, 0 cuts, zero measure ).
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )

    c.cut( [ 1, 0 ], 5 )
    assert int( c.nb_vertices.value ) == 4 and int( c.nb_cuts.value ) == 4
    assert abs( float( c.measure ) - 1 ) < 1e-12

    c.cut( [ 1, 0 ], -1 )
    assert int( c.nb_vertices.value ) == 0 and int( c.nb_cuts.value ) == 0
    assert float( c.measure ) == 0

if test( "cut_capacity_is_exact" ):
    # The room a cut asks for is BOUNDED in advance ( `Cell_2._cut_capacities`: at most one vertex
    # more ), so a `cut` never has to make a second pass: the allocated capacity follows the
    # count one notch per cut, and a cut that adds nothing does not make it grow either.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    for k in range( 12 ):                                   # an inscribed dodecagon: 12 vertices
        t = 2 * numpy.pi * ( k + 0.5 ) / 12
        n = [ numpy.cos( t ), numpy.sin( t ) ]
        c.cut( n, float( numpy.dot( n, [ 0.5, 0.5 ] ) + 0.45 * numpy.cos( numpy.pi / 12 ) ) )
        assert c.nb_vertices.allocated_capacity() == 4 + k + 1
    assert int( c.nb_vertices.value ) == 12
    exp = 12 * ( 0.45 * numpy.cos( numpy.pi / 12 ) ) ** 2 * numpy.tan( numpy.pi / 12 )
    assert abs( float( c.measure ) - exp ) < 1e-12

if test( "cut_tangent" ):
    # The two tangencies, those where a vertex falls on the cutting plane. The clip only asks
    # one question, `s > 0`, and a vertex within epsilon of the plane answers like an inside vertex.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    c.cut( [ 1, 0 ], 1 )
    assert int( c.nb_vertices.value ) == 4 and int( c.nb_cuts.value ) == 4
    assert abs( float( c.measure ) - 1 ) < 1e-12

    # Tangent at a VERTEX: three vertices outside, the two edges toward corner ( 0, 0 ) each give
    # a crossing -- which falls on that corner. Zero area, correct position.
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
    c.cut( [ 1, 1 ], 0 )
    vp = c.vertices()
    assert int( c.nb_vertices.value ) == int( c.nb_cuts.value )     # the 2D invariant holds
    assert numpy.allclose( vp, 0 )                                  # all at the same point
    assert float( c.measure ) == 0

def _half_planes_of( *corners ):
    """The half-spaces `n . x <= o` of a convex polygon given by its vertices, in CCW order."""
    res = []
    for i in range( len( corners ) ):
        p, q = numpy.asarray( corners[ i ], float ), numpy.asarray( corners[ ( i + 1 ) % len( corners ) ], float )
        d = q - p
        n = numpy.array( [ d[ 1 ], - d[ 0 ] ] )
        res.append( ( list( n ), float( n @ p ) ) )
    return res


if test( "cut_unbounded_becomes_bounded" ):
    # The unbounded cell is a FAKE SIMPLEX ( `init_as_unbounded` ): its planes carry invented
    # offsets, and a cut would rank them according to the arbitrary scale of that simplex.
    # `cut` first pushes them back ( `Local2::grow_for` ) until the ranking is the one
    # it would be at infinity. The fake planes are eaten one by one and the unit square remains.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    c = Cell.make_unbounded( 2 )
    assert not c.is_bounded

    for n, o in ( ( [ -1, 0 ], 0 ), ( [ 1, 0 ], 1 ), ( [ 0, -1 ], 0 ) ):
        c.cut( n, o )
        assert not c.is_bounded                     # a strip, a half-plane: infinite
        assert float( c.measure ) > 1e300

    c.cut( [ 0, 1 ], 1 )    # the fourth one closes the cell
    assert c.is_bounded
    assert int( c.nb_vertices.value ) == 4
    assert float( c.measure ) == 1

if test( "cut_unbounded_triangle" ):
    # off the axes and with an arbitrary order of cuts: three half-planes cut out their
    # triangle in the whole plane. The push must find the RIGHT configuration.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    corners = ( ( 0, 0 ), ( 4, 1 ), ( 1, 3 ) )
    exact = 5.5

    for order in ( ( 0, 1, 2 ), ( 2, 0, 1 ), ( 1, 2, 0 ) ):
        c = Cell.make_unbounded( 2 )
        hp = _half_planes_of( *corners )
        for i in order:
            c.cut( *hp[ i ] )

        assert c.is_bounded
        assert int( c.nb_vertices.value ) == 3
        assert abs( float( c.measure ) - exact ) < 1e-12

if test( "cut_unbounded_far_from_the_origin" ):
    # The fake simplex is built at the ORIGIN and at scale 1: for a faraway cell, it has
    # to be pushed all the way there, and the result is correct to the precision of THOSE coordinates.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    shift = numpy.array( [ 1000.0, -500.0 ] )
    corners = [ tuple( numpy.array( p, float ) + shift ) for p in ( ( 0, 0 ), ( 4, 1 ), ( 1, 3 ) ) ]

    c = Cell.make_unbounded( 2 )
    for n, o in _half_planes_of( *corners ):
        c.cut( n, o )

    assert c.is_bounded
    assert int( c.nb_vertices.value ) == 3
    assert abs( float( c.measure ) - 5.5 ) < 1e-8

if test( "cut_batched" ):
    # a whole batch is cut in a single call, one item per work-item, each on its own stack
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    nb_items = 64
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ], batch_axes = [ new_batch_axis( nb_items ) ] )
    c.cut( [ 1, 1 ], 1.5 )

    m = numpy.asarray( c.measure.value )
    assert m.shape == ( nb_items, )
    assert numpy.allclose( m, 0.875 )

if test( "the_float_kernel_cuts_the_same_cells" ):
    # the default kernel is `float32`: same combinatorics, geometry to within 1e-6
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    corners = ( ( 0.1, 0.2 ), ( 0.9, 0.15 ), ( 0.8, 0.9 ), ( 0.2, 0.7 ) )
    for kd in ( "FP32", None ):
        a = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ], kernel_dtype = kd )
        b = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ], kernel_dtype = "FP64" )
        for n, o in _half_planes_of( *corners ):
            a.cut( n, o )
            b.cut( n, o )
        assert int( a.nb_vertices.value ) == int( b.nb_vertices.value ) == 4
        assert numpy.allclose( a.vertices(), b.vertices(), atol = 1e-6 )
        assert abs( float( a.measure ) - float( b.measure ) ) < 1e-6
        # ... and the measure comes out in the caller's float, not in the kernel's
        assert numpy.asarray( a.measure.value ).dtype == numpy.float64


# -- the oracle of the d > 2 tests ------------------------------------------------------------
# Beyond 2D, the cell is no longer described by its vertices alone: there is a LATTICE OF
# FACES to check, and comparing it by hand would not hold up. So we confront it with an
# independent reference, built by brute force from the H-representation alone -- solving
# all the d-tuples of planes and keeping the feasible solutions. It is O( nb_planes^d ), unrelated
# to the algorithm under test, which is precisely what one wants from an oracle.

def _reference_cell( planes, tol = 1e-9 ):
    """V-representation + "on which planes" for each vertex, from the H-representation."""
    from itertools import combinations

    # a plane cut twice is only added once: the second pass finds nothing
    # strictly outside and does nothing. The reference is thus deduplicated the same way.
    seen, uniq = set(), []
    for n, o in planes:
        k = tuple( numpy.round( numpy.append( numpy.asarray( n, float ), o ), 12 ) + 0.0 )
        if k not in seen:
            seen.add( k )
            uniq.append( ( numpy.asarray( n, float ), float( o ) ) )

    dirs = numpy.array( [ p[ 0 ] for p in uniq ] )
    offs = numpy.array( [ p[ 1 ] for p in uniq ] )
    d = dirs.shape[ 1 ]

    verts, on_planes = [], []
    for combo in combinations( range( len( offs ) ), d ):
        rows = list( combo )
        if abs( numpy.linalg.det( dirs[ rows ] ) ) < 1e-10:
            continue
        x = numpy.linalg.solve( dirs[ rows ], offs[ rows ] )
        if numpy.any( dirs @ x - offs > tol * max( 1.0, numpy.abs( x ).max() ) ):
            continue
        if any( numpy.allclose( x, v, atol = 1e-7 ) for v in verts ):
            continue
        verts.append( x )
        on_planes.append( frozenset( numpy.nonzero( numpy.abs( dirs @ x - offs ) <= 1e-7 )[ 0 ].tolist() ) )
    return dirs, offs, verts, on_planes


def alive_cuts( vi ):
    """the cuts a vertex still carries, sorted"""
    return sorted( set( int( r ) for row in vi for r in row ) )


def _check_against_reference( c, planes, name, exact = True ):
    """The cell IS the intersection of `planes`: vertices, cuts and edges, all three, read
    from the DERIVED tensors ( `vertex_positions`, `vertex_cut_indices`, `edges`, `cut_planes` ).

    `exact = False` when the cutting plane goes through existing vertices: the clip
    never tests `s == 0`, so it then produces vertices that COINCIDE to ~1e-16 and the zero-length
    edges that link them. The geometry is still checked but no longer the absence of duplicates.
    """
    nv, nc = int( c.nb_vertices.value ), int( c.nb_cuts.value )
    vp = c.vertices()
    vi = c.vertex_cut_indices
    ev = c.edges
    cd, co = c.cut_planes
    ci = numpy.asarray( c.cut_ids.value )
    d = vp.shape[ 1 ]

    dirs, offs, ref, ref_on = _reference_cell( planes )
    rnd = lambda v: tuple( numpy.round( v, 7 ) + 0.0 )

    # 1. the vertices, as a SET of points
    where = [ next( ( i for i, r in enumerate( ref ) if numpy.allclose( p, r, atol = 1e-7 ) ), None )
              for p in vp ]
    assert None not in where, f"{ name }: vertex outside the reference"
    assert set( where ) == set( range( len( ref ) ) ), f"{ name }: reference vertex missing"
    if exact:
        assert sorted( map( rnd, vp ) ) == sorted( map( rnd, ref ) ), f"{ name }: vertices"

    # 2. the lattice is consistent with the geometry: each vertex is indeed ON each of its
    #    d cuts, and inside for all the others ( the planes are read back from the geometry, so
    #    a dead cut -- without a vertex -- has a zero direction and constrains nothing ).
    #    A DEGENERATE face ( its vertices coincident or aligned, which is what a plane through
    #    existing vertices leaves ) has no plane to read back: zero direction, and nothing to check on it.
    flat = numpy.linalg.norm( cd, axis = 1 ) == 0
    assert exact <= ( not flat[ alive_cuts( vi ) ].any() ), f"{ name }: a live face without a plane"
    for k, p in enumerate( vp ):
        assert list( vi[ k ] ) == sorted( vi[ k ] ), f"{ name }: unsorted cut list at v{ k }"
        for r in vi[ k ]:
            assert flat[ r ] or abs( cd[ r ] @ p - co[ r ] ) < 1e-9, f"{ name }: v{ k } is not on cut { r }"
    assert numpy.all( vp @ cd.T - co < 1e-9 ), f"{ name }: a vertex is outside"

    # 3. the LIVE cuts are exactly the planes that a vertex touches ( a dead cut may
    #    linger in the list until the next compaction: it no longer has a vertex )
    used = sorted( set().union( *ref_on ) ) if ref_on else []
    alive = alive_cuts( vi )
    assert len( alive ) == len( used ), f"{ name }: { len( alive ) } live cuts, expected { len( used ) }"
    for r, u in zip( alive, used ):
        nrm = numpy.linalg.norm( dirs[ u ] )
        assert flat[ r ] or ( numpy.allclose( cd[ r ], dirs[ u ] / nrm ) and abs( co[ r ] - offs[ u ] / nrm ) < 1e-9 ), \
            f"{ name }: cut { r } is not plane { u }"

    # 4. the edges. Two vertices are neighbors when the SMALLEST FACE that contains them
    #    contains only them. A DEGENERATE edge ( its two ends on the same point ) is set aside.
    def _adjacent( a, b ):
        shared = ref_on[ a ] & ref_on[ b ]
        if len( shared ) < d - 1:
            return False
        return not any( shared <= ref_on[ c ] for c in range( len( ref ) ) if c != a and c != b )

    expected = { ( a, b ) for a in range( len( ref ) ) for b in range( a + 1, len( ref ) )
                 if _adjacent( a, b ) }
    got, nb_degenerate = set(), 0
    for row in ev:
        a, b = where[ row[ 0 ] ], where[ row[ 1 ] ]
        if a == b:
            nb_degenerate += 1
            continue
        got.add( ( min( a, b ), max( a, b ) ) )
    assert got == expected, (
        f"{ name }: edges -- extra { [ ( ref[ a ], ref[ b ] ) for a, b in sorted( got - expected ) ] }, "
        f"missing { [ ( ref[ a ], ref[ b ] ) for a, b in sorted( expected - got ) ] }" )
    if exact:
        assert nb_degenerate == 0 and len( got ) == len( ev ), f"{ name }: duplicate edge"


def _cube_planes( d ):
    """The 2d half-spaces of the unit cube, in the order in which `init_as_hypercube` writes them."""
    res = []
    for b in range( d ):
        res.append( ( [ -float( i == b ) for i in range( d ) ], 0.0 ) )
        res.append( ( [ +float( i == b ) for i in range( d ) ], 1.0 ) )
    return res


def _unit_cube( d ):
    return Cell.make_hypercube( d, [ 0 ] * d, numpy.eye( d ).tolist() )


if test( "basic_3D" ):
    # above 2D, the lattice EXISTS: the d cuts and the d neighbors of each vertex
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    c = _unit_cube( 3 )
    assert c.vertex_cuts.is_defined
    assert c.vertex_nbrs.is_defined
    assert int( c.nb_vertices.value ) == 8
    assert int( c.nb_cuts.value ) == 6
    assert len( c.edges ) == 12
    assert len( c.faces ) == 6 and all( len( f ) == 4 for f in c.faces )
    _check_against_reference( c, _cube_planes( 3 ), "cube 3D" )

if test( "cut_3D" ):
    # a sequence of generic cuts on the unit cube, each one fully rechecked
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    for extra in ( [ ( [ 1, 1, 1 ], 2.5 ) ],
                   [ ( [ 1, 1, 1 ], 2.5 ), ( [ -1, -1, -1 ], -0.4 ) ],
                   [ ( [ 1, 1, 1 ], 2.5 ), ( [ 1, -1, 0.5 ], 0.6 ), ( [ -0.3, 1, 0.7 ], 0.9 ) ],
                   [ ( [ 1, 1, 1 ], 10.0 ) ] ):          # this one misses the cell: no effect
        c, planes = _unit_cube( 3 ), _cube_planes( 3 )
        for n, o in extra:
            c.cut( n, o )
            planes = planes + [ ( n, o ) ]
            _check_against_reference( c, planes, f"cube 3D + { extra }" )

    # a cube corner sliced off: one vertex fewer, three more, one more cut
    c = _unit_cube( 3 )
    c.cut( [ 1, 1, 1 ], 2.5 )
    assert ( int( c.nb_vertices.value ), len( c.edges ), int( c.nb_cuts.value ) ) == ( 10, 15, 7 )

if test( "cut_3D_random" ):
    # arbitrary planes, in series: the combinatorics one would not write by hand
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    rng = numpy.random.default_rng( 12345 )
    for d in ( 3, 4 ):
        for trial in range( 12 ):
            c, planes = _unit_cube( d ), _cube_planes( d )
            for step in range( 5 ):
                n = rng.normal( size = d )
                o = float( n @ rng.uniform( 0.2, 0.8, size = d ) )   # passes through the interior
                c.cut( n.tolist(), o )
                if int( c.nb_vertices.value ) == 0:
                    break
                planes = planes + [ ( n.tolist(), o ) ]
                _check_against_reference( c, planes, f"random d={ d } t={ trial } s={ step }" )

if test( "cut_through_existing_vertices" ):
    # THE degenerate case: planes that go EXACTLY through vertices. The clip never tests
    # `s == 0`: what comes out has coincident vertices and zero-length edges, but it is
    # a COHERENT cell.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    for d in ( 3, 4 ):
        for n, o in ( ( [ 1, 1 ] + [ 0 ] * ( d - 2 ), 1.0 ),      # through an entire 2-face
                      ( [ 1 ] * d, float( d - 1 ) ),              # through an edge
                      ( [ 1 ] + [ 0 ] * ( d - 1 ), 1.0 ),         # COINCIDENT with a facet
                      ( [ -1 ] * ( d - 1 ) + [ 1 ], 0.0 ) ):      # through d vertices
            c = _unit_cube( d )
            c.cut( n, o )
            _check_against_reference( c, _cube_planes( d ) + [ ( n, o ) ],
                                      f"degenerate d={ d } n={ n }", exact = False )

if test( "cut_through_a_vertex_2D" ):
    # the same trap in 2D, on a cell whose coordinates do not come out round
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    origin, axes = [ 0.13, -0.27 ], [ [ 1.7, 0.29 ], [ -0.41, 1.13 ] ]
    o, a0, a1 = ( numpy.asarray( v, float ) for v in ( origin, axes[ 0 ], axes[ 1 ] ) )
    corners = ( o, o + a0, o + a0 + a1, o + a1 )
    for corner in corners:
        for n in ( [ 1.0, 0.37 ], [ -0.83, 1.0 ], [ 0.61, -1.19 ] ):
            offset = float( numpy.dot( n, corner ) )
            c = Cell.make_hypercube( 2, origin, axes )
            c.cut( n, offset )

            nv = int( c.nb_vertices.value )
            vp = c.vertices()
            cd, co = c.cut_planes
            assert nv == int( c.nb_cuts.value )                    # the 2D invariant still holds
            assert numpy.all( vp @ cd.T - co < 1e-9 )              # everything is inside all the cuts

            keep = [ p for p in corners if numpy.dot( n, p ) - offset <= 0 ]
            ref = []
            for a, b in zip( corners, corners[ 1 : ] + corners[ : 1 ] ):
                sa, sb = numpy.dot( n, a ) - offset, numpy.dot( n, b ) - offset
                if sa <= 0:
                    ref.append( a )
                if ( sa > 0 ) != ( sb > 0 ):
                    ref.append( ( sb * a - sa * b ) / ( sb - sa ) )
            area = abs( sum( ref[ i ][ 0 ] * ref[ ( i + 1 ) % len( ref ) ][ 1 ]
                           - ref[ ( i + 1 ) % len( ref ) ][ 0 ] * ref[ i ][ 1 ]
                             for i in range( len( ref ) ) ) ) / 2 if len( ref ) > 2 else 0.0
            assert abs( float( c.measure ) - area ) < 1e-9, \
                f"wrong area for n={ n } through { corner }: { float( c.measure ) } != { area }"

if test( "cut_5D" ):
    # nothing 3D-specific in the clip: the same loop holds in 5D
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    c, planes = _unit_cube( 5 ), _cube_planes( 5 )
    for n, o in ( ( [ 1, 1, 1, 1, 1 ], 4.5 ), ( [ 1, -0.4, 0.3, 0.2, 0.1 ], 0.75 ) ):
        c.cut( n, o )
        planes = planes + [ ( n, o ) ]
        _check_against_reference( c, planes, "cube 5D" )

if test( "cut_nd_empty" ):
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    c = _unit_cube( 3 )
    c.cut( [ 1, 1, 1 ], -1.0 )
    assert ( int( c.nb_vertices.value ), int( c.nb_cuts.value ) ) == ( 0, 0 )
    assert float( c.measure ) == 0

if test( "cut_nd_unbounded" ):
    # the fake simplex pushed back until the cut ranks it the way it would at
    # infinity -- in nD the push solves the same system as the vertex ( `LocalN::growth_rate` ).
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    for d in ( 3, 4 ):
        c, planes = Cell.make_unbounded( d ), []
        for n, o in _cube_planes( d ):
            c.cut( n, o )
            planes.append( ( n, o ) )
            assert c.is_bounded == ( len( planes ) == 2 * d )
        _check_against_reference( c, planes, f"unbounded { d }D -> cube" )
        assert int( c.nb_vertices.value ) == 2 ** d

if test( "cut_nd_capacity_is_exact" ):
    # in 3D the bound comes from Euler on a simple polytope ( `V = 2 F - 4` ): one more cut, and
    # as many vertices as such a polytope can have with those faces. Never a second pass.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    c = _unit_cube( 3 )
    cap_c = c.nb_cuts.allocated_capacity()
    for n, o in ( ( [ 1, 1, 1 ], 2.5 ), ( [ 1, -1, 0.5 ], 0.6 ), ( [ -0.3, 1, 0.7 ], 0.9 ) ):
        c.cut( n, o )
    _check_against_reference( c, _cube_planes( 3 ) + [ ( [ 1, 1, 1 ], 2.5 ), ( [ 1, -1, 0.5 ], 0.6 ), ( [ -0.3, 1, 0.7 ], 0.9 ) ], "three cuts" )
    assert c.nb_cuts.allocated_capacity() == cap_c + 3
    assert c.nb_vertices.allocated_capacity() == 2 * ( cap_c + 3 ) - 4

if test( "cut_nd_batched" ):
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    nb_items = 64
    c = Cell.make_hypercube( 3, [ 0, 0, 0 ], numpy.eye( 3 ).tolist(),
                             batch_axes = [ new_batch_axis( nb_items ) ] )
    c.cut( [ 1, 1, 1 ], 2.5 )

    nvs = numpy.asarray( c.nb_vertices.value )
    assert nvs.shape == ( nb_items, ) and ( nvs == 10 ).all()

    assert c.nb_items == nb_items
    for b in range( nb_items ):
        assert numpy.allclose( c.vertices( b ), c.vertices( 0 ) )


# -- the oracle of the MEASURE in d > 2 -----------------------------------------------------------
# The kernel splits the V-representation into simplices and sums determinants. The reference takes
# the other end: divergence theorem on the H-representation, `vol = ( 1/d ) sum_facets
# h_F A_F`, each facet area obtained by ELIMINATING a coordinate and then recursing. Not a single
# line of code in common, and it is exact.

def _reference_volume( dirs, offs ):
    from itertools import combinations

    dirs, offs = numpy.asarray( dirs, float ), numpy.asarray( offs, float )
    d = dirs.shape[ 1 ]

    # eliminating a coordinate makes planes that are already there REAPPEAR (on the facet x = 0 of a
    # prism, `x + y <= 1` becomes `y <= 1` again): without deduplication the corresponding facet
    # would be counted twice. Along the way we throw away the planes that became empty (`0 . x <= b`).
    seen, rows = set(), []
    for n, o in zip( dirs, offs ):
        norm = numpy.linalg.norm( n )
        if norm < 1e-12:
            continue
        key = tuple( numpy.round( numpy.append( n, o ) / norm, 9 ) + 0.0 )
        if key not in seen:
            seen.add( key )
            rows.append( ( n, o ) )
    dirs = numpy.array( [ r[ 0 ] for r in rows ] )
    offs = numpy.array( [ r[ 1 ] for r in rows ] )

    if d == 1:
        lo, hi = -numpy.inf, numpy.inf
        for n, o in zip( dirs[ :, 0 ], offs ):
            if n > 0: hi = min( hi, o / n )
            elif n < 0: lo = max( lo, o / n )
        return max( 0.0, hi - lo )

    verts = []
    for combo in combinations( range( len( offs ) ), d ):
        rows = list( combo )
        if abs( numpy.linalg.det( dirs[ rows ] ) ) < 1e-10:
            continue
        x = numpy.linalg.solve( dirs[ rows ], offs[ rows ] )
        if numpy.any( dirs @ x - offs > 1e-9 * max( 1.0, numpy.abs( x ).max() ) ):
            continue
        if not any( numpy.allclose( x, v, atol = 1e-7 ) for v in verts ):
            verts.append( x )       # a degenerate vertex comes out of several d-tuples of planes
    if len( verts ) < d + 1:
        return 0.0
    x0 = numpy.mean( verts, axis = 0 )

    total = 0.0
    for k in range( len( offs ) ):
        on = [ v for v in verts if abs( dirs[ k ] @ v - offs[ k ] ) < 1e-7 ]
        if len( on ) < d or numpy.linalg.matrix_rank( numpy.array( on ) - on[ 0 ], tol = 1e-7 ) < d - 1:
            continue                                       # redundant plane, not a facet
        # x_j = ( o_k - sum_{i != j} n_i x_i ) / n_j, carried over into all the other planes
        nk, ok = dirs[ k ], offs[ k ]
        j = int( numpy.argmax( numpy.abs( nk ) ) )
        keep = [ i for i in range( d ) if i != j ]
        others = [ i for i in range( len( offs ) ) if i != k ]
        A = dirs[ others ][ :, keep ] - numpy.outer( dirs[ others ][ :, j ] / nk[ j ], nk[ keep ] )
        b = offs[ others ] - dirs[ others ][ :, j ] * ok / nk[ j ]
        area = _reference_volume( A, b ) * numpy.linalg.norm( nk ) / abs( nk[ j ] )
        total += ( ok - nk @ x0 ) / numpy.linalg.norm( nk ) * area
    return total / d


if test( "measure_3D" ):
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    c = _unit_cube( 3 )
    assert float( c.measure ) == 1

    c.cut( [ 1, 1, 1 ], 2.5 )
    assert abs( float( c.measure ) - ( 1 - 0.5 ** 3 / 6 ) ) < 1e-15

if test( "measure_5D" ):
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    c = _unit_cube( 5 )
    assert float( c.measure ) == 1

    c.cut( [ 1, 1, 1, 1, 1 ], 4.5 )
    assert abs( float( c.measure ) - ( 1 - 0.5 ** 5 / 120 ) ) < 1e-15

if test( "measure_nd_vs_reference" ):
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    rng = numpy.random.default_rng( 7 )
    for d in ( 3, 4 ):
        for trial in range( 8 ):
            c, planes = _unit_cube( d ), _cube_planes( d )
            for step in range( 3 ):
                n = rng.normal( size = d )
                o = float( n @ rng.uniform( 0.2, 0.8, size = d ) )
                c.cut( n.tolist(), o )
                planes = planes + [ ( n.tolist(), o ) ]
                got = float( c.measure )
                exp = _reference_volume( [ p[ 0 ] for p in planes ], [ p[ 1 ] for p in planes ] )
                assert abs( got - exp ) < 1e-10 * max( 1.0, exp ), \
                    f"d={ d } t={ trial } s={ step }: { got } != { exp }"

if test( "measure_nd_is_additive" ):
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    rng = numpy.random.default_rng( 3 )
    for d in ( 3, 4 ):
        for _ in range( 6 ):
            n = rng.normal( size = d )
            o = float( n @ rng.uniform( 0.2, 0.8, size = d ) )
            a = _unit_cube( d ); a.cut( n.tolist(), o )
            b = _unit_cube( d ); b.cut( ( -n ).tolist(), -o )
            assert abs( float( a.measure ) + float( b.measure ) - 1 ) < 1e-12

if test( "measure_nd_degenerate" ):
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    for d in ( 3, 4 ):
        for n, o in ( ( [ 1, 1 ] + [ 0 ] * ( d - 2 ), 1.0 ),
                      ( [ 1 ] * d, float( d - 1 ) ),
                      ( [ -1 ] * ( d - 1 ) + [ 1 ], 0.0 ) ):
            c = _unit_cube( d )
            c.cut( n, o )
            exp = _reference_volume( [ p[ 0 ] for p in _cube_planes( d ) ] + [ n ],
                                     [ p[ 1 ] for p in _cube_planes( d ) ] + [ o ] )
            assert abs( float( c.measure ) - exp ) < 1e-11, f"d={ d } n={ n }"

if test( "measure_nd_unbounded_and_empty" ):
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    assert float( Cell.make_unbounded( 3 ).measure ) > 1e300
    c = _unit_cube( 3 )
    c.cut( [ 1, 1, 1 ], -1.0 )
    assert float( c.measure ) == 0

if test( "grad_measure_nd" ):
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    need( "grad" )
    # the adjoint of the splitting: the triangulation is COMBINATORIAL ( it does not move under a
    # small perturbation ), so the backward replays the same traversal and only differentiates the
    # determinants. Checked by finite differences on the coordinates.
    for d in ( 3, 4 ):
        c = Cell.make_hypercube( d, numpy.zeros( d ), numpy.eye( d ) + 0.07 * numpy.arange( d * d ).reshape( d, d ) / ( d * d ) )
        c.cut( numpy.linspace( 0.8, 1.3, d ).tolist(), float( d ) - 0.7 )
        coords = loom.array( numpy.asarray( c.vertex_positions.value ) )
        ids = numpy.asarray( c.cut_ids.value )
        vc, vn = numpy.asarray( c.vertex_cuts.value ), numpy.asarray( c.vertex_nbrs.value )
        check_grad( lambda x: _cell_with_coords( d, x, ids, vc, vn ).measure, coords )

if test( "measure_nd_batched" ):
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    nb_items = 48
    c = Cell.make_hypercube( 3, [ 0, 0, 0 ], numpy.eye( 3 ).tolist(),
                             batch_axes = [ new_batch_axis( nb_items ) ] )
    c.cut( [ 1, 1, 1 ], 2.5 )

    m = numpy.asarray( c.measure.value )
    assert m.shape == ( nb_items, )
    assert numpy.allclose( m, 1 - 0.5 ** 3 / 6 )


# -- what the display of an UNBOUNDED cell drops --------------------------------------------------
# An image cannot be asserted, but these rules can: they concern WHAT is sent to the
# visualizer, and the visualizer can tell what it received. See `Cell.add_to_viz`.

if test( "viz_drops_the_infinite_planes" ):
    # the walls of the fake simplex are not faces of the cell -- they have neither the right
    # position (their offset is an invention that the cuts push back) nor any existence. What
    # goes out is exactly the set of requested half-spaces, and nothing else.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    asked = ( ( [ 1.0, 0.0 ], 1.0 ), ( [ 0.0, 1.0 ], 1.0 ), ( [ 0.0, -1.0 ], 0.0 ) )
    c = Cell.make_unbounded( 2 )
    for n, o in asked:
        c.cut( n, o )
    assert not c.is_bounded                                       # a half-strip: unbounded

    v = Visualizer()
    c.add_to_viz( v )
    assert len( v.polytopes ) == 1
    dirs, offs, _, with_edges = v.polytopes[ 0 ]
    assert not with_edges          # in 2D/3D it is the cell that draws its edges, not the cut
    # the planes go out NORMALIZED ( read back from the geometry ): we compare at the same scale
    got = sorted( tuple( numpy.round( numpy.append( d, o ), 6 ) ) for d, o in zip( dirs, offs ) )
    exp = sorted( tuple( numpy.round( numpy.append( n, o ) / numpy.linalg.norm( n ), 6 ) ) for n, o in asked )
    assert got == exp, ( got, exp )

if test( "viz_dashes_what_runs_off_and_hides_what_is_made_up" ):
    # the same half-strip `{ x <= 1, 0 <= y <= 1 }`, which runs off toward negative x. Three kinds
    # of edges, and we tell them apart by the GEOMETRY of what reaches the visualizer:
    #
    # - `x = 1, 0 <= y <= 1`: both its ends are real vertices -> drawn solid;
    # - `y = 0` and `y = 1`: one real end, the other on a fake wall -> dashed, hence a
    #   bundle of small segments (7 dashes, cf. `Visualizer.add_edges( dashed = True )`);
    # - the closure of the fake simplex -> nothing at all.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    c = Cell.make_unbounded( 2 )
    for n, o in ( ( [ 1, 0 ], 1.0 ), ( [ 0, 1 ], 1.0 ), ( [ 0, -1 ], 0.0 ) ):
        c.cut( n, o )

    v = Visualizer()
    c.add_to_viz( v )
    segs = numpy.asarray( v.positions )[ numpy.asarray( v.edges ) ]      # [ m, 2, 2 ]

    def _is( seg, a, b ):
        return ( numpy.allclose( seg, [ a, b ], atol = 1e-6 )
              or numpy.allclose( seg, [ b, a ], atol = 1e-6 ) )

    full = [ s for s in segs if _is( s, [ 1, 0 ], [ 1, 1 ] ) ]
    rest = [ s for s in segs if not _is( s, [ 1, 0 ], [ 1, 1 ] ) ]
    assert len( full ) == 1, f"the real edge should be drawn solid, once ({ len( full ) })"

    # everything else is a dash, on one of the two half-lines -- nothing of the fake closure
    # (`x` very negative closed by an oblique plane) must be passed through.
    assert len( rest ) == 2 * 7, f"{ len( rest ) } dashes, expected 14"
    for s in rest:
        on_line = numpy.allclose( s[ :, 1 ], 0, atol = 1e-6 ) or numpy.allclose( s[ :, 1 ], 1, atol = 1e-6 )
        assert on_line and ( s[ :, 0 ] <= 1 + 1e-6 ).all(), s

if test( "viz_the_clipping_box_is_not_geometry" ):
    # An OPEN polytope has nothing to show as is: it is the scene's box that gives it
    # vertices (`polytope.clip_planes`, and the same thing on the page side at every cut). It is a
    # means of seeing it, not a part of it -- so it closes the face through which it leaves the field,
    # but it gives NO edge. Otherwise the box draws itself, and ends up
    # alone on screen as soon as the faces are unchecked.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    from sdot.viz.polytope import polytope_mesh

    lim = 2.0
    verts, edges, faces = polytope_mesh( [ [ 1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ] ], [ 1.0, 1.0, 1.0 ],
                                         bounds = [ [ -lim, lim ] ] * 3 )
    assert len( faces ) > 0                          # closed: there is indeed something to fill

    # the three real edges, and nothing else: those that leave the corner ( 1, 1, 1 ) along
    # the three axes ( pairwise intersections of the three requested planes ).
    assert len( edges ) == 3, f"{ len( edges ) } edges, expected 3"
    for a, b in edges:
        for k in range( 3 ):
            for side in ( -lim, lim ):
                assert not ( abs( verts[ a ][ k ] - side ) < 1e-9
                         and abs( verts[ b ][ k ] - side ) < 1e-9 ), \
                    f"edge lying on the box: { verts[ a ] } -> { verts[ b ] }"

if test( "viz_of_a_bounded_cell_is_untouched" ):
    # no fake cut on a bounded cell: nothing to remove, nothing to dash. The unit
    # square must come out as ONE face of 4 vertices and 4 solid edges, as before.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    v = Visualizer()
    Cell.make_hypercube( 2, [ 0, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] ).add_to_viz( v )
    assert len( v.polytopes ) == 0
    assert len( v.polygons ) == 1 and len( v.polygons[ 0 ] ) == 4
    assert len( v.edges ) == 4

if test( "viz_of_a_batch_uses_each_item_own_counts" ):
    # two cells of DIFFERENT SIZES in the same batch: the arrays are dense to the largest,
    # so drawing the small one on the large one's count would make it drag along slots
    # it does not use. This is the usual case of a Voronoi diagram.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    from sdot import Voronoi
    pos = numpy.array( [ [ 0.2, 0.5 ], [ 0.55, 0.5 ], [ 0.9, 0.2 ], [ 0.9, 0.8 ] ] )
    cs = Voronoi( pos, boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ) ).cells
    nvs = numpy.asarray( cs.nb_vertices.value )
    assert nvs.min() < nvs.max(), f"different sizes are needed for the test to say something ({ nvs })"

    v = Visualizer()
    cs.add_to_viz( v )
    assert [ len( f ) for f in v.polygons ] == list( nvs )

if test( "viz_colors_are_the_seed_and_nothing_else" ):
    # a color says WHICH seed. So it is taken from the item's rank, not from the number of things
    # already drawn -- and the proof is an EMPTY cell: the middle seed, dominated by a low weight,
    # has nothing left to show, and the other two must keep EXACTLY the color they
    # had without it. With a call counter, the third one inherited the second one's.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    from sdot import PowerDiagram
    from sdot.viz.Visualizer import scale_color

    pos = numpy.array( [ [ 0.2, 0.5 ], [ 0.5, 0.5 ], [ 0.8, 0.5 ] ] )
    box = ( [ 0, 0 ], [ 1, 1 ] )

    def face_colors( viz ):
        return [ tuple( viz.colors[ c ][ :3 ] ) for c in viz.polygon_colors ]

    full = Visualizer()
    PowerDiagram( pos, boundaries = box_half_spaces( *box )).add_to_viz( full )
    assert len( full.polygons ) == 3
    assert numpy.allclose( face_colors( full ), [ scale_color( i ) for i in range( 3 ) ] )

    holed = Visualizer()
    PowerDiagram( pos, weights = numpy.array( [ 0.0, -1.0, 0.0 ] ), boundaries = box_half_spaces( *box )).add_to_viz( holed )
    assert len( holed.polygons ) == 2                      # the middle one has disappeared
    assert numpy.allclose( face_colors( holed ), [ scale_color( 0 ), scale_color( 2 ) ] )

if test( "viz_colors_do_not_drift_from_one_frame_to_the_next" ):
    # the same diagram, redrawn frame after frame: each seed must get ITS color back.
    # This is what the per-frame reset buys -- without it, frame `k` started where the
    # previous one had stopped and the whole animation flickered.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    from sdot import Voronoi
    pos = numpy.array( [ [ 0.2, 0.2 ], [ 0.75, 0.3 ], [ 0.45, 0.8 ], [ 0.9, 0.85 ] ] )
    n, nb_frames = len( pos ), 3

    v = Visualizer( frame_axis = "step" )
    for k in range( nb_frames ):
        if k:
            v.new_frame( k )
        Voronoi( pos + 0.02 * k, boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ) ).add_to_viz( v )

    cols = [ tuple( v.colors[ c ][ :3 ] ) for c in v.polygon_colors ]
    assert len( cols ) == n * nb_frames
    for k in range( 1, nb_frames ):
        assert cols[ k * n : ( k + 1 ) * n ] == cols[ : n ], k
    assert len( set( cols[ : n ] ) ) == n                  # and the four are indeed distinct


# -- what we LOOK AT -----------------------------------------------------------------------------
# `experiment`s and not `test`s: an image cannot be asserted. What we check here is that the
# two OUTPUTS of the visualizer (the standalone HTML page and ParaView's VTK) come out properly for
# each of the regimes of `Cell.add_to_viz` -- V-representation in 2D, face lattice in 3D,
# H-representation beyond and on an unbounded cell -- plus the image series.
#
#   ./run experiment test_Cell                    # all of them
#   ./run experiment "test_Cell::viz 3D"          # just one
#   ./run experiment "test_Cell::viz cut*" --nb-cuts=4,8    # a sweep: one output per value
#
# Each entry writes into its `p.out_dir` -- tmp/experiment/test_Cell__viz_3D/<env>/, without a date:
# the path does not move from one day to the next, so the tab open on it reloads.

def _write_both( p, viz, stem ):
    """The two outputs, side by side -- this is the gesture that these experiments check."""
    print( "  html :", viz.write_html( p.out_dir / f"{ stem }.html" ) )
    print( "  vtk  :", viz.write_vtk( p.out_dir / f"{ stem }.vtu" ) )


if p := experiment( "viz 2D" ):
    # the 2D regime: `vertex_coords` IS the polygon, in cyclic order -- one face and its ring
    # of edges, nothing to rebuild. Two cells to see the automatic palette at work.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    v = Visualizer( title = "Cell 2D" )
    for shift, normal in ( ( 0.0, [ 1, 1 ] ), ( 1.3, [ -1, 2 ] ) ):
        c = Cell.make_hypercube( 2, [ shift, 0 ], [ [ 1, 0 ], [ 0, 1 ] ] )
        c.cut( normal, float( numpy.dot( normal, [ shift + 0.75, 0.75 ] ) ) )
        c.add_to_viz( v, points = True )
    _write_both( p, v, "cell_2d" )

if p := experiment( "viz 3D" ):
    # the 3D regime: the faces are read back from the lattice ( `Cell.faces` ). This is THE case where the page has something to hide -- solid faces,
    # back edges -- and the one you open in ParaView to turn around it.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    c = _unit_cube( 3 )
    c.cut( [ 1, 1, 1 ], 2.5 )
    v = Visualizer( title = "Cube 3D, one corner sliced off" )
    c.add_to_viz( v, opacity = 0.55, points = True )
    _write_both( p, v, "cell_3d" )

if p := experiment( "viz 5D" ):
    # beyond 3D there is no lattice to send: `add_to_viz` passes the H-representation,
    # the page shows a 3D SLICE of it ( cutting half-spaces gives half-spaces again ) and adds
    # the PROJECTED wireframe to it. On the VTK side, the coordinates beyond the 3rd go out as point data,
    # and it is up to ParaView to make its own cuts.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    c = _unit_cube( 5 )
    c.cut( [ 1, 1, 1, 1, 1 ], 4.5 )
    c.cut( [ 1, -0.4, 0.3, 0.2, 0.1 ], 0.75 )
    v = Visualizer( title = "Cell 5D ( slice )" )
    c.add_to_viz( v )
    _write_both( p, v, "cell_5d" )

if p := experiment( "viz unbounded" ):
    # an UNBOUNDED cell has no vertices to show -- the fake simplex of
    # `init_as_unbounded` has some, but they are those of a stand-in, not of the cell. So it is the
    # H-representation that goes out, and it is the visualizer that closes it on the scene's box.
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    v = Visualizer( title = "Unbounded cells" )
    c = Cell.make_unbounded( 3 )
    for n, o in ( ( [ -1, 0, 0 ], 0 ), ( [ 0, -1, 0 ], 0 ), ( [ 0, 0, -1 ], 0 ) ):
        c.cut( n, o )
    assert not c.is_bounded                                    # the positive octant: a cone
    c.add_to_viz( v, opacity = 0.55 )
    _write_both( p, v, "cell_unbounded" )

if p := experiment( "viz cut by cut",
                    dim     = Param( 3, help = "dimension of the cell" ),
                    nb_cuts = Param( 4, help = "number of random cuts" ),
                    seed    = Param( 0, help = "seed for drawing the planes" ) ):
    # ONE IMAGE per cut: the unit cube trimmed plane after plane. The page plays by itself
    # ( "cut" axis, play/pause ) and ParaView receives a `.pvd`, that is to say a time
    # series -- hence one `.vtu` per image, which the single-image outputs above do not cover.
    # We start from the cube, bounded, and not from the infinite cell: the scene's framing is
    # COMMON to all the images, so a first image as large as the world would crush all the
    # following ones into a line ( `viz unbounded` is what shows that regime ).
    import numpy
    from loom.testing import need
    from loom.testing import check_grad
    import loom
    from loom import new_batch_axis
    from sdot import Cell, Visualizer, box_half_spaces, set_kernel_dtype
    set_kernel_dtype( "FP64" )
    d   = p.dim
    rng = numpy.random.default_rng( p.seed )
    c   = _unit_cube( d )
    v   = Visualizer( title = f"cube { d }D, cut by cut", frame_axis = "cut" )
    c.add_to_viz( v, opacity = 0.55 )
    nb_drawn = 1
    for k in range( p.nb_cuts ):
        n = rng.normal( size = d )
        c.cut( n.tolist(), float( n @ rng.uniform( 0.2, 0.8, size = d ) ) )   # passes through the interior
        # randomly drawn planes end up excluding everything; the empty cell is a
        # legitimate state, but it does not make an image -- we stop there rather than add one.
        if int( numpy.max( numpy.asarray( c.nb_vertices.value ) ) ) == 0:
            print( f"  empty cell at cut { k }" )
            break
        v.new_frame( k + 1 )
        c.add_to_viz( v, opacity = 0.55 )
        nb_drawn += 1
    print( f"  { nb_drawn } image(s)" )
    _write_both( p, v, f"cut_by_cut_{ d }d" )
