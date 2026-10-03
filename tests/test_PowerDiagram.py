import numpy

from loom import driver
from errand import Param, bench, experiment, test
from loom.testing import need
from loom.testing import check_grad

from sdot import AaBsp, Image, PowerDiagram, SumOfGaussians, Visualizer, Voronoi, box_half_spaces, set_kernel_dtype

# Geometry is cut in FP32 by default ( see `Cell.py` ), which is enough for an optimal transport
# but not for comparing a sum of measures to 1e-10 nor a Jacobian to 1e-9: the CORRECTNESS
# tests run with the double kernel, and the `float` kernel has its own tests further down
# ( "the_float_kernel_*" ), with tolerances of its own.
set_kernel_dtype( "FP64" )


def _measures( v ):
    return numpy.asarray( v.measures.value ).reshape( -1 )


def _monte_carlo_measures( pos, mi, ma, weights = None, nb_samples = 200000, seed = 0 ):
    """Measures by SAMPLING: we draw points from the box and count, for each
    seed, the fraction that is closer to it than to any other IN THE POWER SENSE.

    No geometric code in common with the kernel -- this is the DEFINITION of the diagram
    ( "`|x - d|² - w` minimal" ), not another way of cutting half-spaces. What it
    checks is therefore not the cutting but what we claim to cut: the weight
    convention is written here a second time, and without a single plane.
    """
    rng = numpy.random.default_rng( seed )
    mi, ma = numpy.asarray( mi, float ), numpy.asarray( ma, float )
    pts = rng.uniform( mi, ma, size = ( nb_samples, mi.size ) )
    d2 = ( ( pts[ :, None, : ] - pos[ None, :, : ] ) ** 2 ).sum( axis = 2 )
    if weights is not None:
        d2 = d2 - numpy.asarray( weights, float )[ None, : ]
    nearest = d2.argmin( axis = 1 )
    volume = float( numpy.prod( ma - mi ) )
    return numpy.bincount( nearest, minlength = len( pos ) ) / nb_samples * volume


if test( "basic_2D" ):
    # two symmetric seeds in the unit square: each takes half.
    v = Voronoi( numpy.array( [ [ 0.25, 0.5 ], [ 0.75, 0.5 ] ] ), boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ) )
    assert numpy.allclose( _measures( v ), 0.5 )

if test( "one_seed_takes_everything" ):
    # no bisector at all: only the domain remains.
    for d in ( 2, 3, 4 ):
        v = Voronoi( numpy.full( ( 1, d ), 0.3 ), boundaries = box_half_spaces( numpy.zeros( d ), numpy.full( d, 2.0 ) ) )
        assert abs( float( _measures( v )[ 0 ] ) - 2.0 ** d ) < 1e-12

if test( "sum_is_the_domain" ):
    # the cells TILE the domain: their sum is its volume, whatever the dimension.
    # This is the only test that takes the diagram as a whole -- a cell that is wrong on one side and
    # wrong the other way round on the other is what it catches.
    for d, n in ( ( 2, 60 ), ( 3, 40 ), ( 4, 20 ) ):
        rng = numpy.random.default_rng( 12 + d )
        pos = rng.uniform( 0.05, 0.95, size = ( n, d ) )
        v = Voronoi( pos, boundaries = box_half_spaces( numpy.zeros( d ), numpy.ones( d ) ) )
        m = _measures( v )
        assert m.shape == ( n, )
        assert ( m > 0 ).all()
        assert abs( float( m.sum() ) - 1 ) < 1e-10

if test( "matches_the_cell_by_cell_build" ):
    # the same geometry through an entirely different orchestration: `Voronoi.cell` rebuilds the
    # cell on the Python side, one `driver.call` per cut, on the buffers of an ordinary `Cell`.
    for d, n in ( ( 2, 12 ), ( 3, 10 ), ( 4, 6 ) ):
        rng = numpy.random.default_rng( 5 + d )
        pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
        v = Voronoi( pos, boundaries = box_half_spaces( numpy.zeros( d ), numpy.ones( d ) ) )
        m = _measures( v )
        ref = numpy.array( [ float( v.cell( i ).measure ) for i in range( n ) ] )
        assert numpy.allclose( m, ref, atol = 1e-12 ), ( d, m - ref )

if test( "is_the_nearest_seed_partition" ):
    # what the cutting claims to cut. Sampling error in 1/sqrt(N): the tolerance
    # is loose on purpose, this test does not look for precision but for the right partition.
    for d, n in ( ( 2, 8 ), ( 3, 8 ) ):
        rng = numpy.random.default_rng( 100 + d )
        pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
        v = Voronoi( pos, boundaries = box_half_spaces( numpy.zeros( d ), numpy.ones( d ) ) )
        m = _measures( v )
        mc = _monte_carlo_measures( pos, numpy.zeros( d ), numpy.ones( d ) )
        assert numpy.abs( m - mc ).max() < 5e-3, ( d, m, mc )

if test( "an_off_centre_box" ):
    # the domain has nothing to do with the origin nor with the seeds: the half-spaces are cut
    # BEFORE the bisectors (that is what makes the cell bounded), and an arbitrary box shows it.
    d = 3
    mi, ma = numpy.array( [ -2.0, 1.0, 0.5 ] ), numpy.array( [ 1.0, 4.0, 2.5 ] )
    rng = numpy.random.default_rng( 7 )
    pos = rng.uniform( mi + 0.1, ma - 0.1, size = ( 15, d ) )
    v = Voronoi( pos, boundaries = box_half_spaces( mi, ma ) )
    assert abs( float( _measures( v ).sum() ) - float( numpy.prod( ma - mi ) ) ) < 1e-10

if test( "a_box_domain_survives_driver_jit" ):
    # `box_min` / `box_max` (the shortcut for a box domain, see `axis_aligned_box`) are
    # CONSTANTS closed over, never a function of what a `driver.jit` traces -- but
    # `_domain_cell` reads them back on the host side (`np.asarray`), which used to fail (`TracerArrayConversionError`)
    # as soon as the diagram CONSTRUCTION took place INSIDE a trace: `jax.jit` turns
    # every jax call executed in its lexical extent into a tracer of ITS trace, even a call on
    # a pure constant (see `JaxDriver.array`). The same result, jitted or not, shows
    # that the domain did not change meaning along the way.
    rng = numpy.random.default_rng( 0 )
    pos = rng.uniform( 0.05, 0.95, size = ( 10, 2 ) )
    bnd = box_half_spaces( [ 0, 0 ], [ 1, 1 ] )

    def loss( w ):
        return PowerDiagram( pos, w, boundaries = bnd ).measures.value.sum()

    w0 = numpy.zeros( 10 )
    eager  = float( loss( w0 ) )
    jitted = float( driver.jit( loss )( w0 ) )
    assert abs( eager - jitted ) < 1e-10, ( eager, jitted )

if test( "a_domain_that_is_not_a_box" ):
    # `box` is only a shortcut: the domain is a LIST OF HALF-SPACES, hence any
    # polyhedral convex set. Here the simplex `x, y, z >= 0, x + y + z <= 1` (volume 1/6).
    dirs = numpy.array( [ [ -1.0, 0, 0 ], [ 0, -1.0, 0 ], [ 0, 0, -1.0 ], [ 1.0, 1, 1 ] ] )
    offs = numpy.array( [ 0.0, 0, 0, 1 ] )
    rng = numpy.random.default_rng( 3 )
    pos = rng.dirichlet( numpy.ones( 4 ), size = 12 )[ :, :3 ] * 0.9
    v = Voronoi( pos, boundaries = ( dirs, offs ) )
    assert abs( float( _measures( v ).sum() ) - 1 / 6 ) < 1e-10

if test( "no_domain_leaves_the_cells_infinite" ):
    # without a domain, all the cells of such a small cloud go off to infinity -- and `Cell::measure`
    # says so as it does elsewhere (`TF::max`), without `Voronoi` having to know about it.
    v = Voronoi( numpy.array( [ [ 0.0, 0 ], [ 1.0, 0 ], [ 0.0, 1 ] ] ) )
    assert ( _measures( v ) > 1e300 ).all()

if test( "more_seeds_than_work_items" ):
    # the buffers are PER WORK-ITEM and passed on from one seed to the next: a cell that would leave
    # a remainder behind (a count, a scratch row) would show up here and not on 10 seeds.
    d, n = 3, 500
    rng = numpy.random.default_rng( 42 )
    pos = rng.uniform( 0.02, 0.98, size = ( n, d ) )
    v = Voronoi( pos, boundaries = box_half_spaces( numpy.zeros( d ), numpy.ones( d ) ) )
    m = _measures( v )
    assert ( m > 0 ).all()
    assert abs( float( m.sum() ) - 1 ) < 1e-9

if test( "a_capacity_too_small_is_grown_and_retried" ):
    # `scratch_capacity` is only a guess. Too small, the kernel records what would have
    # been needed and writes nothing wrong; the platform reserves twice as much and retries (`driver.call`).
    # The result must be the SAME as the one obtained straight away with enough room.
    d, n = 3, 30
    rng = numpy.random.default_rng( 8 )
    pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
    box = ( numpy.zeros( d ), numpy.ones( d ) )
    tight = _measures( Voronoi( pos, boundaries = box_half_spaces( *box ), scratch_capacity = 4 ) )
    roomy = _measures( Voronoi( pos, boundaries = box_half_spaces( *box ), scratch_capacity = 128 ) )
    assert numpy.allclose( tight, roomy )
    assert abs( float( tight.sum() ) - 1 ) < 1e-10

if test( "seeds_on_a_grid" ):
    # the DEGENERATE configuration par excellence: a regular grid, where four cells
    # meet at one point and where the bisectors pass exactly through vertices. The
    # clip never tests `s == 0` (cf. `Cell::cut`) and thus returns a possibly
    # degenerate cell, but never a wrong one: the squares are always 1/9.
    xs = numpy.array( [ 1, 3, 5 ] ) / 6
    pos = numpy.array( [ [ x, y ] for x in xs for y in xs ] )
    v = Voronoi( pos, boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ) )
    assert numpy.allclose( _measures( v ), 1 / 9 )

if test( "seeds_on_a_3D_grid" ):
    xs = numpy.array( [ 1, 3 ] ) / 4
    pos = numpy.array( [ [ x, y, z ] for x in xs for y in xs for z in xs ] )
    v = Voronoi( pos, boundaries = box_half_spaces( [ 0, 0, 0 ], [ 1, 1, 1 ] ) )
    assert numpy.allclose( _measures( v ), 1 / 8 )

if test( "a_seed_outside_the_domain_has_no_cell" ):
    # nothing abnormal: the cell empties out, and an empty cell measures zero. What matters is that
    # the others are none the worse for it -- they still tile the domain.
    pos = numpy.array( [ [ 0.5, 0.5 ], [ 0.2, 0.2 ], [ 5.0, 5.0 ] ] )
    v = Voronoi( pos, boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ) )
    m = _measures( v )
    assert abs( float( m[ 2 ] ) ) < 1e-14
    assert abs( float( m.sum() ) - 1 ) < 1e-12


# -- the weights ---------------------------------------------------------------------------------
# Everything above is the Euclidean case, obtained WITHOUT weights (`Voronoi`). What follows fixes the
# convention -- `|x - d_i|² - w_i <= |x - d_j|² - w_j` -- and nothing else: the geometry is
# already tested, the weights only move planes.

if test( "the_bisector_is_shifted_by_the_weight_gap" ):
    # THE convention test, and it can be computed by hand. Two seeds on the x axis, at 0 and 1:
    # the plane is at `( 1 + w_0 - w_1 ) / 2`, so the left cell measures exactly that in
    # the unit square. A sign flipped, a forgotten factor 2, a normalization lingering -- the
    # three possible mistakes show up here, in numbers.
    for gap in ( -0.6, -0.25, 0.0, 0.25, 0.6 ):
        pd = PowerDiagram( numpy.array( [ [ 0.0, 0.5 ], [ 1.0, 0.5 ] ] ),
                           weights = numpy.array( [ gap, 0.0 ] ),
                           boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ) )
        m = _measures( pd )
        assert abs( float( m[ 0 ] ) - ( 1 + gap ) / 2 ) < 1e-12, ( gap, m )
        assert abs( float( m.sum() ) - 1 ) < 1e-12

if test( "equal_weights_are_no_weights" ):
    # what `Voronoi.py` says: only the weight DIFFERENCES reach the planes. Weights
    # all equal -- to any value -- must give the Euclidean diagram bit for bit,
    # which at the same time checks that the added term is indeed a difference and not a sum.
    for d, n in ( ( 2, 30 ), ( 3, 20 ) ):
        rng = numpy.random.default_rng( 20 + d )
        pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
        box = ( numpy.zeros( d ), numpy.ones( d ) )
        ref = _measures( Voronoi( pos, boundaries = box_half_spaces( *box )) )
        for w in ( 0.0, 1.7, -3.0 ):
            got = _measures( PowerDiagram( pos, weights = numpy.full( n, w ), boundaries = box_half_spaces( *box )) )
            assert numpy.allclose( got, ref, atol = 1e-12 ), ( d, w, got - ref )

if test( "a_shift_of_every_weight_changes_nothing" ):
    # the same invariant, but on ARBITRARY weights: the diagram only depends on `w` modulo
    # constants. This cannot be deduced from the previous test, which only sees the flat case.
    d, n = 2, 25
    rng = numpy.random.default_rng( 33 )
    pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
    w = rng.uniform( -0.05, 0.05, size = n )
    box = ( numpy.zeros( d ), numpy.ones( d ) )
    a = _measures( PowerDiagram( pos, weights = w, boundaries = box_half_spaces( *box )) )
    b = _measures( PowerDiagram( pos, weights = w + 2.5, boundaries = box_half_spaces( *box )) )
    assert numpy.allclose( a, b, atol = 1e-12 ), a - b

if test( "weighted_cells_still_tile_the_domain" ):
    # power cells TILE the domain like Voronoi ones -- including when
    # some are empty, which weights make possible and the Euclidean case does not.
    for d, n in ( ( 2, 40 ), ( 3, 25 ), ( 4, 12 ) ):
        rng = numpy.random.default_rng( 50 + d )
        pos = rng.uniform( 0.05, 0.95, size = ( n, d ) )
        w = rng.uniform( -0.1, 0.1, size = n )
        m = _measures( PowerDiagram( pos, weights = w, boundaries = box_half_spaces( numpy.zeros( d ), numpy.ones( d ) ) ) )
        assert ( m >= -1e-14 ).all(), ( d, m.min() )
        assert abs( float( m.sum() ) - 1 ) < 1e-10, ( d, m.sum() )

if test( "is_the_least_power_distance_partition" ):
    # what the cutting claims to cut, taken from the Euclidean case with the power in place of
    # the distance. Sampling error in 1/sqrt(N): the tolerance is loose on purpose.
    for d, n in ( ( 2, 8 ), ( 3, 8 ) ):
        rng = numpy.random.default_rng( 70 + d )
        pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
        w = rng.uniform( -0.15, 0.15, size = n )
        m = _measures( PowerDiagram( pos, weights = w, boundaries = box_half_spaces( numpy.zeros( d ), numpy.ones( d ) ) ) )
        mc = _monte_carlo_measures( pos, numpy.zeros( d ), numpy.ones( d ), weights = w )
        assert numpy.abs( m - mc ).max() < 5e-3, ( d, m, mc )

if test( "matches_the_cell_by_cell_build_with_weights" ):
    # the oracle of the other tests, on the Python side: `PowerDiagram.cell` rebuilds the cell one
    # `driver.call` per cut. The weight convention is written there a second time, in numpy --
    # if the kernel and it agree, it is the same formula twice and not the same bug twice.
    for d, n in ( ( 2, 12 ), ( 3, 10 ), ( 4, 6 ) ):
        rng = numpy.random.default_rng( 90 + d )
        pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
        w = rng.uniform( -0.08, 0.08, size = n )
        pd = PowerDiagram( pos, weights = w, boundaries = box_half_spaces( numpy.zeros( d ), numpy.ones( d ) ) )
        ref = numpy.array( [ float( pd.cell( i ).measure ) for i in range( n ) ] )
        assert numpy.allclose( _measures( pd ), ref, atol = 1e-12 ), ( d, _measures( pd ) - ref )

if test( "a_dominated_seed_loses_its_cell" ):
    # what a Voronoi diagram cannot do: a seed INSIDE the domain, and yet without a cell.
    # A low enough weight makes it lose everywhere -- and the others still tile the domain, which
    # is the real claim here (an empty cell must leave nothing behind).
    pos = numpy.array( [ [ 0.25, 0.5 ], [ 0.5, 0.5 ], [ 0.75, 0.5 ] ] )
    m = _measures( PowerDiagram( pos, weights = numpy.array( [ 0.0, -1.0, 0.0 ] ),
                                 boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ) ) )
    assert abs( float( m[ 1 ] ) ) < 1e-14, m
    assert abs( float( m.sum() ) - 1 ) < 1e-12, m

if test( "a_big_enough_weight_swallows_the_domain" ):
    # the other extreme, and the bound of the previous one: a high enough weight takes EVERYTHING, the other
    # cells empty out. The tiling must hold there too.
    pos = numpy.array( [ [ 0.25, 0.4 ], [ 0.5, 0.6 ], [ 0.8, 0.3 ] ] )
    m = _measures( PowerDiagram( pos, weights = numpy.array( [ 0.0, 10.0, 0.0 ] ),
                                 boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ) ) )
    assert abs( float( m[ 1 ] ) - 1 ) < 1e-12, m
    assert float( m[ 0 ] ) + float( m[ 2 ] ) < 1e-14, m

if test( "the_memory_changes_nothing_but_the_cost" ):
    # yesterday's neighbours proposed first ( `memory` ): the same cells, exactly -- one more or one fewer cut
    # in the order only changes the rounding --, the memories fill up on the
    # first `measures`, survive fresh weights, and are erased along with the tree.
    for d in ( 2, 3 ):
        rng = numpy.random.default_rng( 5 + d )
        n = 400
        pos = rng.uniform( 0.05, 0.95, size = ( n, d ) )
        h = n ** ( -1.0 / d )
        w = rng.uniform( -0.3, 0.3, n ) * h * h
        box = box_half_spaces( numpy.zeros( d ), numpy.ones( d ) )
        ref = PowerDiagram( pos, weights = w, boundaries = box, memory = 0 )
        pd = PowerDiagram( pos, weights = w, boundaries = box, memory = 32 )
        assert not numpy.asarray( pd.memo_counts ).any()          # nothing yet
        assert numpy.abs( _measures( pd ) - _measures( ref ) ).max() < 1e-12
        counts = numpy.asarray( pd.memo_counts ).reshape( -1 )
        assert counts.mean() > ( 4 if d == 2 else 10 )             # filled at the first call
        assert numpy.abs( _measures( pd ) - _measures( ref ) ).max() < 1e-12
        ref.weights = 2 * w
        pd.weights = 2 * w                                         # the stale memories remain useful
        assert numpy.abs( _measures( pd ) - _measures( ref ) ).max() < 1e-12
        pd.positions = pos[ ::-1 ]                                  # the tree is rebuilt: the memory goes away
        assert not numpy.asarray( pd.memo_counts ).any()
        ref.positions = pos[ ::-1 ]
        assert numpy.abs( _measures( pd ) - _measures( ref ) ).max() < 1e-12

if test( "voronoi_is_the_power_diagram_without_weights" ):
    # `Voronoi` is not a class: it is `PowerDiagram` without the `weights` member (see
    # `Voronoi.py`). So it does build one, it refuses weights, and it returns the same
    # diagram as explicitly zero weights.
    pos = numpy.random.default_rng( 4 ).uniform( 0.1, 0.9, size = ( 15, 2 ) )
    box = ( [ 0, 0 ], [ 1, 1 ] )
    v = Voronoi( pos, boundaries = box_half_spaces( *box ))
    assert isinstance( v, PowerDiagram )
    assert not v.weights.is_defined              # `Unbound` -> `NoneTensor`, not a buffer of zeros
    zeros = PowerDiagram( pos, weights = numpy.zeros( 15 ), boundaries = box_half_spaces( *box ))
    assert numpy.allclose( _measures( v ), _measures( zeros ), atol = 1e-12 )

    try:
        Voronoi( pos, weights = numpy.zeros( 15 ), boundaries = box_half_spaces( *box ))
    except TypeError:
        pass
    else:
        raise AssertionError( "Voronoi should refuse weights" )


# -- the derivatives ----------------------------------------------------------------------------
# `measures` is differentiated with respect to the SEEDS, positions and weights (see
# `PowerDiagram.cxx::measures_bwd`). Two families of tests, and they do not overlap:
# `check_grad` compares the adjoint to a finite difference -- it catches a sign, a factor, a
# forgotten term -- whereas the two analytic tests below confront the FULL Jacobian
# with its classical formula, written in terms of facets (area and barycentre) where the kernel
# goes through a small linear system per vertex. Two derivations with nothing in common.

def _in_fp64( f ):
    """`f()` in double precision. The kernel runs in FP32 by default, which is enough for a
    volume but not for comparing a Jacobian to within 1e-9."""
    previous = driver.ftype
    driver.ftype = "FP64"
    try:
        return f()
    finally:
        driver.ftype = previous


def _facets_2d( pd, n ):
    """`( lengths, midpoints )` of each seed-seed facet of the 2D diagram, `[ n, n ]` and
    `[ n, n, 2 ]`, read from `pd.cells`. The 2D invariant does all the work: cut `c` carries
    the edge `[ v_c, v_c+1 ]`, so the facet between `i` and `cut_ids( c )` is that segment.
    """
    cs = pd.cells
    nvs = numpy.asarray( cs.nb_vertices.value ).reshape( -1 )
    cis = numpy.asarray( cs.cut_ids.raw )

    length = numpy.zeros( ( n, n ) )
    middle = numpy.zeros( ( n, n, 2 ) )
    for i in range( n ):
        nv = int( nvs[ i ] )
        vp = cs.vertices( i )
        for c in range( nv ):
            j = int( cis[ i, c ] )
            if j < 0:                       # the domain: a constant, with no share of the gradient
                continue
            a, b = vp[ c ], vp[ ( c + 1 ) % nv ]
            length[ i, j ] = float( numpy.linalg.norm( b - a ) )
            middle[ i, j ] = ( a + b ) / 2
    return length, middle


def _jacobian( f, x, nb_rows ):
    """The Jacobian of `f` at `x`, row by row: `nb_rows` passes of the adjoint, each with
    a one-hot cotangent. We materialize it because we have something to compare it with term by term."""
    _, pullback = driver.vjp( lambda a: f( a ).value, x )
    rows = []
    for i in range( nb_rows ):
        seed = numpy.zeros( nb_rows )
        seed[ i ] = 1.0
        rows.append( numpy.asarray( pullback( seed )[ 0 ] ) )
    return numpy.array( rows )


if test( "measures_derive_wrt_the_seeds" ):
    need( "grad" )
    # the adjoint against the finite difference, positions AND weights at the same time -- `check_grad` draws
    # a random tangent on each, so a missing cross term shows up too.
    for d, n in ( ( 2, 10 ), ( 3, 8 ) ):
        rng = numpy.random.default_rng( 200 + d )
        pos = rng.uniform( 0.15, 0.85, size = ( n, d ) )
        w = rng.uniform( -0.02, 0.02, size = n )
        box = ( numpy.zeros( d ), numpy.ones( d ) )
        check_grad( lambda p, q: PowerDiagram( p, weights = q, boundaries = box_half_spaces( *box )).measures, pos, w )

if test( "measures_derive_wrt_positions_alone" ):
    need( "grad" )
    # no weights at all: `weights` is a `NoneTensor`, so its gradient too, and the branch
    # that writes it must vanish at compile time without taking that of the positions with it.
    for d, n in ( ( 2, 10 ), ( 3, 8 ) ):
        rng = numpy.random.default_rng( 220 + d )
        pos = rng.uniform( 0.15, 0.85, size = ( n, d ) )
        box = ( numpy.zeros( d ), numpy.ones( d ) )
        check_grad( lambda p: Voronoi( p, boundaries = box_half_spaces( *box )).measures, pos )

if test( "measures_derive_wrt_weights_alone" ):
    need( "grad" )
    # the other half: positions frozen, only the weights move. This is the derivative a semi-discrete
    # optimal transport solver lives on, hence the one we want to isolate.
    for d, n in ( ( 2, 12 ), ( 3, 8 ) ):
        rng = numpy.random.default_rng( 240 + d )
        pos = rng.uniform( 0.15, 0.85, size = ( n, d ) )
        w = rng.uniform( -0.02, 0.02, size = n )
        box = ( numpy.zeros( d ), numpy.ones( d ) )
        check_grad( lambda q: PowerDiagram( pos, weights = q, boundaries = box_half_spaces( *box )).measures, w )

if test( "the_weight_jacobian_is_the_facet_over_the_gap" ):
    need( "grad" )
    # the classical formula: `dm_i/dw_j = -|F_ij| / ( 2 |d_i - d_j| )` for `j != i`, and the
    # diagonal is the opposite of the sum of its row -- moving all the weights together changes
    # nothing, so each row sums to zero. This is the Newton matrix of a semi-discrete
    # optimal transport, and it comes out here of a kernel that has never computed a facet area.
    def body():
        n = 8
        rng = numpy.random.default_rng( 260 )
        pos = rng.uniform( 0.15, 0.85, size = ( n, 2 ) )
        w = rng.uniform( -0.02, 0.02, size = n )
        box = ( numpy.zeros( 2 ), numpy.ones( 2 ) )
        pd = PowerDiagram( pos, weights = w, boundaries = box_half_spaces( *box ))

        length, _ = _facets_2d( pd, n )
        gap = numpy.linalg.norm( pos[ :, None, : ] - pos[ None, :, : ], axis = 2 )
        expected = numpy.zeros( ( n, n ) )
        off = length > 0
        expected[ off ] = - length[ off ] / ( 2 * gap[ off ] )
        expected[ numpy.diag_indices( n ) ] = - expected.sum( axis = 1 )

        got = _jacobian( lambda q: PowerDiagram( pos, weights = q, boundaries = box_half_spaces( *box )).measures, w, n )
        assert numpy.abs( got - expected ).max() < 1e-9, numpy.abs( got - expected ).max()
        assert numpy.abs( got.sum( axis = 1 ) ).max() < 1e-9      # translation invariance
        assert numpy.abs( got - got.T ).max() < 1e-9              # and it is symmetric
    _in_fp64( body )

if test( "the_position_jacobian_is_the_facet_moment" ):
    need( "grad" )
    # the other half of the same derivation: `dm_i/dd_j = |F_ij| ( d_j - b_ij ) / |d_j - d_i|`,
    # where `b_ij` is the BARYCENTRE of the facet (its midpoint, in 2D). The barycentre term is what
    # distinguishes a plane that translates from a plane that pivots -- it is what a naive
    # derivative forgets, and it appears nowhere in the kernel.
    def body():
        n = 8
        rng = numpy.random.default_rng( 280 )
        pos = rng.uniform( 0.15, 0.85, size = ( n, 2 ) )
        w = rng.uniform( -0.02, 0.02, size = n )
        box = ( numpy.zeros( 2 ), numpy.ones( 2 ) )
        pd = PowerDiagram( pos, weights = w, boundaries = box_half_spaces( *box ))

        length, middle = _facets_2d( pd, n )
        expected = numpy.zeros( ( n, n, 2 ) )
        for i in range( n ):
            for j in range( n ):
                if i == j or length[ i, j ] == 0:
                    continue
                nrm = float( numpy.linalg.norm( pos[ j ] - pos[ i ] ) )
                expected[ i, j ] = length[ i, j ] * ( pos[ j ] - middle[ i, j ] ) / nrm
                expected[ i, i ] += length[ i, j ] * ( middle[ i, j ] - pos[ i ] ) / nrm

        got = _jacobian( lambda p: PowerDiagram( p, weights = w, boundaries = box_half_spaces( *box )).measures, pos, n )
        assert numpy.abs( got - expected ).max() < 1e-9, numpy.abs( got - expected ).max()
    _in_fp64( body )

if test( "the_domain_carries_no_gradient" ):
    need( "grad" )
    # a LIMIT, not a property: a domain cut carries `cut_id == BOUNDARY`, which says
    # "not a seed" without saying WHICH one, so the backward has nowhere to put its share and
    # leaves it at zero. The test pins it down -- so that it remains a decision and not a surprise.
    # (The gradient with respect to the SEEDS, on the other hand, is complete: a domain facet depends
    # on no seed, its contribution there is truly zero.)
    n = 6
    rng = numpy.random.default_rng( 300 )
    pos = rng.uniform( 0.2, 0.8, size = ( n, 2 ) )
    dirs = numpy.array( [ [ 1.0, 0 ], [ 0, 1.0 ], [ -1.0, 0 ], [ 0, -1.0 ] ] )
    offs = numpy.array( [ 1.0, 1.0, 0.0, 0.0 ] )
    _, pullback = driver.vjp(
        lambda o: PowerDiagram( pos, boundaries = ( dirs, o ) ).measures.value, offs )
    assert numpy.abs( numpy.asarray( pullback( numpy.ones( n ) )[ 0 ] ) ).max() == 0


# -- all the cells at once -----------------------------------------------------------------------
# `measures` reduces each cell to a number and forgets it; `cells` KEEPS them, for which there is
# only one use -- drawing them. Both go through the same `make_cell`, so what remains to
# be checked here is the copy to the output (`Cell::copy_into`) and the item space (one work-item
# per seed, plus the strided loop of `measures`).

if test( "cells_are_the_cells" ):
    # confronted with the oracle of the other tests: the cell built ON THE PYTHON SIDE, one call per
    # cut. The same geometry, two orchestrations with nothing in common.
    for d in ( 2, 3 ):
        rng = numpy.random.default_rng( 5 )
        pos = rng.uniform( 0.1, 0.9, size = ( 9, d ) )
        v = Voronoi( pos, boundaries = box_half_spaces( [ 0 ] * d, [ 1 ] * d ) )

        cs = v.cells
        nvs = numpy.asarray( cs.nb_vertices.value )
        for i in range( len( pos ) ):
            ref = v.cell( i )
            nv = int( ref.nb_vertices.value )
            assert nvs[ i ] == nv, ( d, i, nvs[ i ], nv )
            # in ARBITRARY order: the two paths do not apply the cuts in the same
            # order, so the vertex numbering has no reason to coincide.
            a = numpy.sort( ref.vertices().round( 9 ), axis = 0 )
            b = numpy.sort( cs.vertices( i ).round( 9 ), axis = 0 )
            assert numpy.allclose( a, b ), ( d, i )

if test( "cells_measure_like_measures" ):
    # the copy carries EVERYTHING needed to measure -- including the face lattice in d > 2,
    # which nothing else reads back. A batched cell knows how to measure itself, and must return exactly
    # what the path that keeps nothing returns.
    for d in ( 2, 3 ):
        rng = numpy.random.default_rng( 11 )
        pos = rng.uniform( 0.1, 0.9, size = ( 20, d ) )
        v = Voronoi( pos, boundaries = box_half_spaces( [ 0 ] * d, [ 1 ] * d ) )
        got = numpy.asarray( v.cells.measure.value ).reshape( -1 )
        assert numpy.allclose( got, _measures( v ) ), d
        assert abs( float( got.sum() ) - 1 ) < 1e-10

if test( "cells_of_an_unbounded_diagram" ):
    # without a domain, the border cells remain infinite: they keep `INFINITE` planes,
    # and it is the display that knows what to do with them (cf. `Cell.add_to_viz`). What must hold here
    # is that the flag tells the truth -- an interior cell is bounded, a border one is not.
    pos = numpy.array( [ [ x, y ] for x in ( 0.25, 0.5, 0.75 ) for y in ( 0.25, 0.5, 0.75 ) ] )
    bounded = numpy.asarray( Voronoi( pos ).cells.is_bounded ).reshape( -1 )
    assert bounded[ 4 ] == 1                    # the one in the centre, surrounded
    assert bounded.sum() == 1                   # the other eight go off to infinity


if test( "weighted_cells_are_the_weighted_cells" ):
    # the display path goes through the same `make_cell`, so the weights are there by construction --
    # what remains to be checked is that the copy to the output does not lose them along the way, and an
    # EMPTY cell (which weights make possible) is the case that would tell.
    for d in ( 2, 3 ):
        rng = numpy.random.default_rng( 320 + d )
        pos = rng.uniform( 0.1, 0.9, size = ( 14, d ) )
        w = rng.uniform( -0.06, 0.06, size = 14 )
        w[ 3 ] = -2.0                                  # this one wins nowhere
        pd = PowerDiagram( pos, weights = w, boundaries = box_half_spaces( [ 0 ] * d, [ 1 ] * d ) )
        got = numpy.asarray( pd.cells.measure.value ).reshape( -1 )
        assert numpy.allclose( got, _measures( pd ), atol = 1e-12 ), d
        assert abs( float( got[ 3 ] ) ) < 1e-14, ( d, got[ 3 ] )
        assert abs( float( got.sum() ) - 1 ) < 1e-10, d

# -- SPATIAL ACCELERATION ------------------------------------------------------------------------
# An accelerator cannot change the diagram: it can only silence cuts that would
# have removed nothing. That is what everything that follows checks, and it is the ONLY thing to
# check -- the geometry is already tested above, and it is the same.


def _cut_sets( pd ):
    """For each cell, the set of seeds it actually touches.

    Stronger than a measure: two different cuttings can give the same volume (a tangent cut
    removes nothing), but not the same SET of neighbours. It is thus the direct witness of
    "no useful cut was killed" -- which neither the measures nor the vertices would say as
    precisely.
    """
    cs = pd.cells
    nbc = numpy.asarray( cs.nb_cuts.value ).reshape( -1 )
    ids = numpy.asarray( cs.cut_ids.raw )
    return [ frozenset( int( j ) for j in ids[ i, : int( nbc[ i ] ) ] if j >= 0 )
             for i in range( len( nbc ) ) ]


def _both_ways( pos, weights = None, boundaries = None, **kwargs ):
    """The same diagram, once with a full sweep and once accelerated."""
    plain = PowerDiagram( pos, weights = weights, boundaries = boundaries, accelerator = "plain" )
    acc = AaBsp.of( plain, **kwargs )
    return plain, PowerDiagram( pos, weights = weights, boundaries = boundaries, accelerator = acc )


if test( "the_bsp_holds_every_seed_exactly_once" ):
    # the structural property EVERYTHING else depends on: a seed silenced by the tree would be a
    # lost cut, and nothing in the walk could recover it.
    for d, n, leaf in ( ( 2, 200, 30 ), ( 3, 137, 8 ), ( 2, 1, 30 ), ( 4, 40, 1 ) ):
        rng = numpy.random.default_rng( 400 + d * 17 + n )
        pos = rng.uniform( 0, 1, size = ( n, d ) )
        bsp = AaBsp( pos, max_seeds_per_leaf = leaf )

        order = numpy.asarray( bsp.seed_indices ).reshape( -1 )
        assert sorted( order.tolist() ) == list( range( n ) ), ( d, n, leaf )

        left = numpy.asarray( bsp.node_left ).reshape( -1 )
        beg = numpy.asarray( bsp.node_begin ).reshape( -1 )
        end = numpy.asarray( bsp.node_end ).reshape( -1 )
        # the leaves PARTITION `seed_indices`, in order: that is what makes a leaf
        # readable in one piece.
        slices = sorted( ( int( beg[ k ] ), int( end[ k ] ) ) for k in range( len( left ) ) if left[ k ] < 0 )
        assert slices[ 0 ][ 0 ] == 0 and slices[ -1 ][ 1 ] == n, ( d, n, leaf )
        assert all( a[ 1 ] == b[ 0 ] for a, b in zip( slices, slices[ 1: ] ) ), ( d, n, leaf )
        assert all( b - a <= max( leaf, 1 ) for a, b in slices ), ( d, n, leaf )


if test( "the_tree_shape_does_not_depend_on_the_data" ):
    # the cut is at the MEDIAN, so depth and number of nodes are functions of `n` alone. That is what
    # says a kernel-side construction would have no capacity to guess -- neither for the descent
    # stack, nor for the node arrays. Tested even on the clouds one would think worse:
    # ten positions repeated `n` times, where the tree closes leaves early and thus only
    # shrinks.
    for n in ( 1, 7, 30, 31, 100, 1057, 5000 ):
        for leaf in ( 1, 8, 30 ):
            for d in ( 2, 3 ):
                for kind in ( "uniform", "degenerate" ):
                    rng = numpy.random.default_rng( 401 + n + leaf )
                    if kind == "uniform":
                        pos = rng.uniform( 0, 1, size = ( n, d ) )
                    else:
                        pos = rng.uniform( 0, 1, size = ( 10, d ) )[ rng.integers( 0, 10, n ) ]
                    bsp = AaBsp( pos, max_seeds_per_leaf = leaf )
                    want = AaBsp.max_depth_for( n, leaf )
                    # REACHED when the seeds are distinct, only bounded above when they are not:
                    # coincident positions close leaves early, so the tree
                    # shrinks -- which is the right direction for a capacity.
                    if kind == "uniform":
                        assert bsp.max_depth == want, ( n, leaf, d, kind, bsp.max_depth, want )
                    else:
                        assert bsp.max_depth <= want, ( n, leaf, d, kind, bsp.max_depth, want )
                    nb = int( numpy.asarray( bsp.node_left ).size )
                    assert nb <= AaBsp.max_nb_nodes_for( n, leaf ), ( n, leaf, d, kind, nb )


if test( "a_bsp_node_contains_its_subtree" ):
    # the box of a node must contain ALL the seeds of the subtree, not only those of its
    # direct leaves: it is the one the walk tests before refusing to descend.
    rng = numpy.random.default_rng( 411 )
    pos = rng.normal( size = ( 300, 3 ) )
    bsp = AaBsp( pos, max_seeds_per_leaf = 7 )

    left = numpy.asarray( bsp.node_left ).reshape( -1 )
    right = numpy.asarray( bsp.node_right ).reshape( -1 )
    beg = numpy.asarray( bsp.node_begin ).reshape( -1 )
    end = numpy.asarray( bsp.node_end ).reshape( -1 )
    box = numpy.asarray( bsp.node_box ).reshape( -1, 2, 3 )
    lo, hi = box[ :, 0 ], box[ :, 1 ]
    order = numpy.asarray( bsp.seed_indices ).reshape( -1 )

    def seeds_of( k ):
        if left[ k ] < 0:
            return list( order[ int( beg[ k ] ) : int( end[ k ] ) ] )
        return seeds_of( int( left[ k ] ) ) + seeds_of( int( right[ k ] ) )

    total = 0
    for k in range( len( left ) ):
        sub = pos[ seeds_of( k ) ]
        assert ( sub >= lo[ k ] - 1e-12 ).all() and ( sub <= hi[ k ] + 1e-12 ).all(), k
        total += 1
    assert total == len( left )
    assert len( seeds_of( 0 ) ) == len( pos )          # the root is everybody


if test( "the_weight_majorant_majorates" ):
    # `w_i <= a . y_i + b` for every seed of the subtree: without it the walk would prune a node
    # that nevertheless contained a cut. Tested on TRENDING weights (the regime where the affine one
    # is useful) as well as on pure noise (the one where it must keep quiet).
    rng = numpy.random.default_rng( 420 )
    pos = rng.uniform( 0, 1, size = ( 400, 2 ) )
    # and seeds CLAMPED at the border, `x` equal to within 1e-8, with varying weights: the normal
    # matrix is nearly singular there and the affine bound came out at 1e13 -- valid (raised) but a `b`
    # at 1e9 that no longer bounds anything useful, and a bound that no longer "touches".
    rng2 = numpy.random.default_rng( 421 )             # separate: do not shift the draws above
    clamped = pos.copy()
    clamped[ :, 0 ] = 1e-4 + 1e-8 * rng2.uniform( 0, 1, size = 400 )
    for name, pos, w in ( ( "affine", pos, 0.5 * pos[ :, 0 ] - 0.3 * pos[ :, 1 ] ),
                          ( "noise", pos, rng.normal( size = 400 ) ),
                          ( "noisy affine", pos, 0.5 * pos[ :, 0 ] + 0.02 * rng.normal( size = 400 ) ),
                          ( "clamped border", clamped, 0.1 * numpy.sin( 7 * clamped[ :, 1 ] ) + 0.02 * rng2.normal( size = 400 ) ) ):
        bsp = AaBsp( pos, w, max_seeds_per_leaf = 12 )
        left = numpy.asarray( bsp.node_left ).reshape( -1 )
        right = numpy.asarray( bsp.node_right ).reshape( -1 )
        beg = numpy.asarray( bsp.node_begin ).reshape( -1 )
        end = numpy.asarray( bsp.node_end ).reshape( -1 )
        wa = numpy.asarray( bsp.node_wa )
        wb = numpy.asarray( bsp.node_wb ).reshape( -1 )
        order = numpy.asarray( bsp.seed_indices ).reshape( -1 )

        def seeds_of( k ):
            if left[ k ] < 0:
                return list( order[ int( beg[ k ] ) : int( end[ k ] ) ] )
            return seeds_of( int( left[ k ] ) ) + seeds_of( int( right[ k ] ) )

        nb_affine = nb_nodes = 0
        for k in range( len( left ) ):
            sub = seeds_of( k )
            # an EMPTY slot: the right child of a node that passed everything to the left (see
            # `AaBsp.py`). It bounds nothing, there is nothing to bound.
            if not sub:
                continue
            nb_nodes += 1
            slack = wa[ k ] @ pos[ sub ].T + wb[ k ] - w[ sub ]
            assert slack.min() >= 0, ( name, k, slack.min() )
            # RAISED until it touches: a bound that never touches is a loose bound. What
            # keeps it from touching exactly is the rounding margin of `_weight_majorant`,
            # relative -- so that is what we compare against, not zero.
            room = 1e-4 * ( 1 + numpy.abs( w[ sub ] ).max() )
            assert slack.min() < room, ( name, k, slack.min(), room )
            nb_affine += bool( numpy.abs( wa[ k ] ).max() > 0 )

        # the affine/constant choice is made node by node: on a trend it must be taken,
        # on pure noise it must remain marginal. "Marginal" and not "never": the threshold
        # corrects the tightening that chance gives ON AVERAGE, not that of a particular
        # draw, and a node that goes through anyway remains perfectly valid -- the
        # block above says so.
        frac = nb_affine / nb_nodes
        if name == "noise":
            assert frac < 0.2, ( name, frac )
        elif name != "clamped border":
            assert frac > 0.5, ( name, frac )


if test( "an_accelerator_for_other_seeds_is_refused" ):
    # an accelerator INDEXES the seeds: that of another cloud would designate something else, and the
    # answer would be wrong with nothing to signal it. The common case is an `AaBsp` kept from a previous
    # step where the cloud changed size -- that is the one the count catches.
    rng = numpy.random.default_rng( 425 )
    pos = rng.uniform( 0, 1, size = ( 20, 2 ) )
    bsp = AaBsp( pos )
    try:
        PowerDiagram( pos[ :15 ], boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ), accelerator = bsp ).measures
        raise AssertionError( "expected a ValueError" )
    except ValueError as e:
        assert "20 seeds" in str( e ) and "15" in str( e ), str( e )


if test( "the_accelerator_changes_nothing_to_the_measures" ):
    # the central test, sweeping the regimes: dimension, weights, and a VERY
    # inhomogeneous distribution (clusters), which is the case where a median tree and a geometric tree diverge.
    cases = []
    for d in ( 2, 3 ):
        rng = numpy.random.default_rng( 430 + d )
        cases.append( ( f"uniform {d}D", rng.uniform( 0.02, 0.98, size = ( 120, d ) ), None ) )
        pos = rng.uniform( 0.02, 0.98, size = ( 120, d ) )
        cases.append( ( f"weights {d}D", pos, rng.uniform( -0.02, 0.02, 120 ) ) )
        # clusters: ten tight bunches, hence very unequal boxes
        centres = rng.uniform( 0.15, 0.85, size = ( 10, d ) )
        clust = ( centres[ rng.integers( 0, 10, 150 ) ] + 0.02 * rng.normal( size = ( 150, d ) ) ).clip( 0.01, 0.99 )
        cases.append( ( f"clusters {d}D", clust, None ) )
        cases.append( ( f"clusters+weights {d}D", clust, 0.01 * rng.normal( size = 150 ) ) )

    for name, pos, w in cases:
        d = pos.shape[ 1 ]
        box = ( [ 0 ] * d, [ 1 ] * d )
        plain, fast = _both_ways( pos, w, box_half_spaces( *box ) )
        a, b = _measures( plain ), _measures( fast )
        assert numpy.allclose( a, b, rtol = 0, atol = 1e-9 ), ( name, numpy.abs( a - b ).max() )


if test( "the_accelerator_keeps_every_facet" ):
    # the same thing, but at the level of NEIGHBOURS and not volumes: that is what says no useful
    # cut was killed, including those that remove almost nothing.
    for d in ( 2, 3 ):
        rng = numpy.random.default_rng( 440 + d )
        pos = rng.uniform( 0.02, 0.98, size = ( 60, d ) )
        w = rng.uniform( -0.02, 0.02, 60 )
        for weights in ( None, w ):
            plain, fast = _both_ways( pos, weights, box_half_spaces( [ 0 ] * d, [ 1 ] * d ) )
            assert _cut_sets( plain ) == _cut_sets( fast ), ( d, weights is not None )


if test( "the_accelerator_survives_the_degenerate_layouts" ):
    # the cases where the tree itself is odd: all the seeds at the same place (a leaf that cannot
    # be cut), a single seed, aligned seeds, a leaf of ONE seed only (hence
    # the deepest possible tree, and the most stressed stack).
    box2 = ( [ 0, 0 ], [ 1, 1 ] )

    same = numpy.full( ( 12, 2 ), 0.5 )
    plain, fast = _both_ways( same, None, box_half_spaces( *box2 ) )
    assert numpy.allclose( _measures( plain ), _measures( fast ), atol = 1e-9 )

    one = numpy.array( [ [ 0.3, 0.7 ] ] )
    plain, fast = _both_ways( one, None, box_half_spaces( *box2 ) )
    assert numpy.allclose( _measures( fast ), 1.0, atol = 1e-9 )
    assert numpy.allclose( _measures( plain ), _measures( fast ), atol = 1e-9 )

    line = numpy.stack( [ numpy.linspace( 0.05, 0.95, 40 ), numpy.full( 40, 0.5 ) ], axis = 1 )
    plain, fast = _both_ways( line, None, box_half_spaces( *box2 ) )
    assert numpy.allclose( _measures( plain ), _measures( fast ), atol = 1e-9 )

    rng = numpy.random.default_rng( 451 )
    pos = rng.uniform( 0.02, 0.98, size = ( 90, 2 ) )
    plain, fast = _both_ways( pos, None, box_half_spaces( *box2 ), max_seeds_per_leaf = 1 )
    assert numpy.allclose( _measures( plain ), _measures( fast ), atol = 1e-9 )
    # same seeds, opposite grain: a single leaf, so the tree is reduced to its root
    plain, fast = _both_ways( pos, None, box_half_spaces( *box2 ), max_seeds_per_leaf = 10 ** 6 )
    assert numpy.allclose( _measures( plain ), _measures( fast ), atol = 1e-9 )


if test( "an_unbounded_diagram_falls_back_to_the_full_sweep" ):
    # without a domain, a cell is not the hull of its vertices as long as it is not
    # bounded, so there is nothing to prune against: the walk must visit everything, and the
    # result remain the right one. It is the only regime where the accelerator does not accelerate, and that is
    # on purpose (see `cell_may_be_cut`).
    rng = numpy.random.default_rng( 460 )
    pos = rng.uniform( 0, 1, size = ( 30, 2 ) )
    plain = PowerDiagram( pos, accelerator = "plain" )
    fast = PowerDiagram( pos, accelerator = AaBsp.of( plain ) )
    a, b = _measures( plain ), _measures( fast )
    # `Cell::measure` says "infinite" through `TF::max`, as everywhere else
    inf_a, inf_b = a > 1e300, b > 1e300
    assert ( inf_a == inf_b ).all() and inf_a.sum() > 0
    assert numpy.allclose( a[ ~inf_a ], b[ ~inf_b ], atol = 1e-9 )
    assert _cut_sets( plain ) == _cut_sets( fast )


if test( "a_domain_that_is_not_a_box_is_accelerated_too" ):
    # pruning only knows the cell and the tree boxes: the SHAPE of the domain tells it
    # nothing, it is enough that it bounds. A simplex checks this.
    rng = numpy.random.default_rng( 470 )
    pos = rng.uniform( 0.05, 0.4, size = ( 50, 2 ) )
    bnd = ( numpy.array( [ [ -1.0, 0 ], [ 0, -1.0 ], [ 1.0, 1.0 ] ] ), numpy.array( [ 0.0, 0.0, 1.0 ] ) )
    plain = PowerDiagram( pos, boundaries = bnd, accelerator = "plain" )
    fast = PowerDiagram( pos, boundaries = bnd, accelerator = AaBsp.of( plain ) )
    assert numpy.allclose( _measures( plain ), _measures( fast ), atol = 1e-9 )
    assert abs( float( _measures( fast ).sum() ) - 0.5 ) < 1e-9


if test( "an_accelerator_built_on_other_weights_is_still_right" ):
    # a tree coming from outside may have been built on other weights, or without: the diagram redoes
    # its bound on ITS weights ( `PowerDiagram_Bsp._init_seeds` ), and that is what makes it right
    # -- a null bound, or one that is too low, would silence cuts. A tree built WITHOUT weights thus carries a
    # diagram WITH some, and so does a tree built on other weights.
    rng = numpy.random.default_rng( 480 )
    pos = rng.uniform( 0.02, 0.98, size = ( 70, 2 ) )
    w = rng.uniform( -0.03, 0.03, 70 )
    box = ( [ 0, 0 ], [ 1, 1 ] )
    ref = _measures( PowerDiagram( pos, weights = w, boundaries = box_half_spaces( *box ), accelerator = "plain" ) )

    bare = AaBsp( pos )
    assert bare.node_wa.is_undefined                       # no weights, no bound at all
    no_w = PowerDiagram( pos, weights = w, boundaries = box_half_spaces( *box ), accelerator = bare )
    assert bare.node_wa.is_defined                         # ... until a diagram needs one
    assert numpy.allclose( ref, _measures( no_w ), atol = 1e-9 )

    other = PowerDiagram( pos, weights = w, boundaries = box_half_spaces( *box ),
                          accelerator = AaBsp( pos, w - 0.05 ) )     # built too low: redone
    assert numpy.allclose( ref, _measures( other ), atol = 1e-9 )

    # and setting fresh weights on an existing diagram redoes the bound without rebuilding the tree
    tree = other.tree
    other.weights = w[ ::-1 ].copy()
    assert other.tree is tree
    ref2 = _measures( PowerDiagram( pos, weights = w[ ::-1 ], boundaries = box_half_spaces( *box ), accelerator = "plain" ) )
    assert numpy.allclose( ref2, _measures( other ), atol = 1e-9 )
    assert numpy.allclose( numpy.asarray( other.weights ).reshape( -1 ), w[ ::-1 ] )
    assert numpy.allclose( numpy.asarray( other.positions ).reshape( -1, 2 ), pos )


if test( "the_accelerator_changes_nothing_to_the_derivatives" ):
    need( "grad" )
    # the derivatives go through the SURVIVING cuts, so they can only differ if a
    # cut was lost. The full Jacobian, term by term, says so plainly -- and it is the
    # test that protects the tree against a subtly wrong backward pass rather than just a volume
    # that happens to come out right.
    def run():
        rng = numpy.random.default_rng( 490 )
        n = 12
        pos = rng.uniform( 0.1, 0.9, size = ( n, 2 ) )
        w = rng.uniform( -0.02, 0.02, n )
        box = ( [ 0, 0 ], [ 1, 1 ] )
        bsp = AaBsp( pos, w, max_seeds_per_leaf = 3 )

        for name, f_plain, f_fast, x in (
            ( "positions",
              lambda a: PowerDiagram( a, weights = w, boundaries = box_half_spaces( *box )).measures,
              lambda a: PowerDiagram( a, weights = w, boundaries = box_half_spaces( *box ), accelerator = bsp ).measures,
              pos ),
            ( "weights",
              lambda a: PowerDiagram( pos, weights = a, boundaries = box_half_spaces( *box )).measures,
              lambda a: PowerDiagram( pos, weights = a, boundaries = box_half_spaces( *box ), accelerator = bsp ).measures,
              w ),
        ):
            ja = _jacobian( f_plain, x, n )
            jb = _jacobian( f_fast, x, n )
            assert numpy.abs( ja - jb ).max() < 1e-9, ( name, numpy.abs( ja - jb ).max() )

        # and the adjoint against the finite difference, on the accelerated path this time: the
        # comparison above would say "same" if both were wrong in the same way.
        check_grad( lambda a, b: PowerDiagram( a, weights = b, boundaries = box_half_spaces( *box ), accelerator = bsp ).measures,
                    pos, w )
    _in_fp64( run )


if test( "an_accelerated_diagram_draws_the_same_cells" ):
    # the display path (`cells`) has its own call and its own per-work-item scratch:
    # so it must be checked too, and not only `measures`.
    rng = numpy.random.default_rng( 495 )
    pos = rng.uniform( 0.05, 0.95, size = ( 40, 2 ) )
    w = rng.uniform( -0.02, 0.02, 40 )
    plain, fast = _both_ways( pos, w, box_half_spaces( [ 0, 0 ], [ 1, 1 ] ) )
    a = numpy.asarray( plain.cells.measure.value ).reshape( -1 )
    b = numpy.asarray( fast.cells.measure.value ).reshape( -1 )
    assert numpy.allclose( a, b, atol = 1e-9 )
    assert numpy.allclose( b, _measures( fast ), atol = 1e-9 )


# -- what it COSTS --------------------------------------------------------------------------------
#   ./run bench test_PowerDiagram --nb-points=2000,32000,128000

if p := bench( "pd accelerated",
               nb_points = Param( 32000, help = "number of seeds" ),
               nb_dims   = Param( 2, help = "dimension" ),
               leaf_size = Param( 10, help = "seeds per BSP leaf" ),
               weights   = Param( 0, help = "1 for a power diagram" ),
               memory    = Param( -1, help = "memories per seed ( -1: the default for the dimension, 0: none )" ),
               plain     = Param( 1, help = "0 to NOT time the full sweep" ),
               reps      = Param( 3, help = "timed repetitions (we keep the minimum)" ),
               seed      = Param( 0, help = "seed of the draw" ),
               kernel    = Param( "FP32", help = "the kernel float type ( FP32 or FP64 )" ) ):
    # A real loop, and the MINIMUM: batch axis names are borrowed from a pool
    # (`loom.tensor.batch`), so two identical calls produce the same C++ source and the second
    # hits the compilation cache. The first call of each variant still compiles -- hence
    # the warm-up round outside the timer.
    import time

    d, n = p.nb_dims, p.nb_points
    rng = numpy.random.default_rng( p.seed )
    pos = rng.uniform( 0.01, 0.99, size = ( n, d ) )
    # the weights AT SCALE: a plane is shifted by `dw / ( 2 |d1 - d0| )`, so for the
    # shift to be a fraction of the spacing `h` we need `dw ~ h²`. Weights "at random between
    # -0.002 and 0.002" would empty almost all the cells for large `n` -- and a sweep that
    # returns right away measures nothing anymore.
    h = n ** ( -1.0 / d )
    w = rng.uniform( -0.3, 0.3, n ) * h * h if p.weights else None
    box = ( [ 0 ] * d, [ 1 ] * d )

    # a warm-up for the TREE too, and for the same reason as for the sweep further down:
    # `AaBsp` is now built by kernels (one call per level, see
    # `AaBsp._build_in_kernel`), so the first one goes through the kernel compiler. Without it `t_build` measured a
    # compilation -- 11.8 s where the tree costs 0.25.
    AaBsp( pos[ :1000 ], None if w is None else w[ :1000 ], max_seeds_per_leaf = p.leaf_size )

    t = time.perf_counter()
    bsp = AaBsp( pos, w, max_seeds_per_leaf = p.leaf_size )
    t_build = time.perf_counter() - t

    def run( acc ):
        # the diagram is built ONCE ( storing the seeds in tree order, redoing the
        # bound: `t_ctor` ); what we time is what a fitting step pays, setting
        # the weights and measuring -- without weights, measuring only.
        t = time.perf_counter()
        pd = PowerDiagram( pos, weights = w, boundaries = box_half_spaces( *box ), accelerator = acc,
                           kernel_dtype = p.kernel, memory = None if p.memory < 0 else p.memory )
        p.results[ f"t_ctor_{ acc if isinstance( acc, str ) else 'bsp' }" ] = time.perf_counter() - t
        def once():
            t = time.perf_counter()
            if w is not None:
                pd.weights = w
            m = numpy.asarray( pd.measures.value )
            return time.perf_counter() - t, m.reshape( -1 )
        once()                                          # warm-up: this is the one that compiles
        once()                                          # and this one that fills the memory ( `memory` )
        best, m = once()
        for _ in range( p.reps - 1 ):
            best = min( best, once()[ 0 ] )
        return best, m

    # The full sweep is `n^2`: at 1e6 seeds it does not finish, and timing it would have
    # no interest anyway. `--plain=0` skips it -- we then measure ONLY the path we
    # would really use, and there is no oracle left to compare the measures with (that is the role of the
    # tests, not of a bench).
    t_acc, m_acc = run( bsp )
    p.results[ "t_accelerated" ] = t_acc
    p.results[ "t_bsp_build" ] = t_build
    p.results[ "bsp_depth" ] = bsp.max_depth
    p.results[ "bsp_leaves" ] = bsp.nb_leaves
    p.results[ "ns_per_seed" ] = t_acc / n * 1e9

    if p.plain:
        t_plain, m_plain = run( "plain" )
        p.results[ "t_plain" ] = t_plain
        p.results[ "speedup" ] = t_plain / t_acc
        p.results[ "max_abs_diff" ] = float( numpy.abs( m_plain - m_acc ).max() )
        print( f"  {n} seeds in {d}D : {t_plain:.2f} s -> {t_acc:.2f} s"
               f" (x{t_plain / t_acc:.0f}), tree built in {t_build * 1e3:.0f} ms"
               f", max deviation {numpy.abs( m_plain - m_acc ).max():.1e}" )
    else:
        print( f"  {n} seeds in {d}D : {t_acc:.3f} s ({t_acc / n * 1e9:.0f} ns/seed)"
               f", tree built in {t_build * 1e3:.0f} ms"
               f", sum of the measures {m_acc.sum():.6f}" )


# -- what we LOOK AT -----------------------------------------------------------------------------
# A diagram is judged by eye far more than by a number. One `experiment` per regime, each
# writing the HTML page and the ParaView VTK -- see `Cell.add_to_viz` for what an unbounded
# cell shows of itself (the fake planes are not sent, truncated edges are
# dotted, those that are only the fake closure are not drawn).
#
#   ./run experiment test_PowerDiagram                      # all of them
#   ./run experiment "test_PowerDiagram::vor 2D open"       # just one
#   ./run experiment test_PowerDiagram --nb-points=20,200   # a sweep

def _write_both( p, viz, stem ):
    viz.write_html( p.out_dir / f"{ stem }.html" )
    v = viz.write_vtk( p.out_dir / f"{ stem }.vtu" )
    print( "  vtk  :", v )


def _seeds( d, n, seed ):
    return numpy.random.default_rng( seed ).uniform( 0.05, 0.95, size = ( n, d ) )


if p := experiment( "vor 2D",
                    nb_points = Param( 40, help = "number of seeds" ),
                    seed      = Param( 0, help = "seed of the draw" ) ):
    # the reference case: a diagram bounded by a square. All the cells are real
    # polygons, one colour each, and the tiling is visible.
    v = Voronoi( _seeds( 2, p.nb_points, p.seed ), boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ) )
    viz = Visualizer( title = f"Voronoi 2D, { p.nb_points } seeds" )
    v.add_to_viz( viz )
    viz.add_points( numpy.asarray( v.positions ), color = "#ffffff" )
    _write_both( p, viz, "vor_2d" )

if p := experiment( "vor 2D open",
                    nb_points = Param( 25, help = "number of seeds" ),
                    seed      = Param( 1, help = "seed of the draw" ) ):
    # WITHOUT a domain: the border cells go off to infinity. This is the experiment that shows the
    # three rules of a fake background -- no invented wall, a truncated edge dotted,
    # and nothing at all where the cell is closed only by the replacement simplex.
    v = Voronoi( _seeds( 2, p.nb_points, p.seed ) )
    viz = Visualizer( title = f"Voronoi 2D unbounded, { p.nb_points } seeds" )
    v.add_to_viz( viz )
    viz.add_points( numpy.asarray( v.positions ), color = "#ffffff" )
    _write_both( p, viz, "vor_2d_open" )

if p := experiment( "vor 3D",
                    nb_points = Param( 30, help = "number of seeds" ),
                    seed      = Param( 2, help = "seed of the draw" ) ):
    # in 3D each cell is a solid polyhedron: this is the case where opacity is useful, and the one one
    # opens in ParaView to slice the tiling rather than look at it from outside.
    v = Voronoi( _seeds( 3, p.nb_points, p.seed ), boundaries = box_half_spaces( [ 0 ] * 3, [ 1 ] * 3 ) )
    viz = Visualizer( title = f"Voronoi 3D, { p.nb_points } seeds" )
    v.add_to_viz( viz, opacity = 0.55 )
    _write_both( p, viz, "vor_3d" )

if p := experiment( "vor 3D open",
                    nb_per_side = Param( 3, help = "seeds per side of the grid" ),
                    jitter      = Param( 0.25, help = "disorder, as a fraction of the step" ),
                    seed        = Param( 4, help = "seed of the draw" ) ):
    # a JITTERED grid rather than a uniform draw, and for a reason that is not at all
    # cosmetic: a Voronoi vertex is a circumscribed sphere centre, and four nearly
    # coplanar seeds give one very far away. It is real, so it frames the scene -- and in 3D
    # without a domain, with few seeds, it is likely and it crushes everything else in the picture.
    # A jittered grid produces none -- but an EXACT grid is the worst case
    # of all: its quadruplets are strictly coplanar, their circumscribed sphere centre
    # is at infinity, and the clip returns an honest finite value for it, around 1e16 (try
    # `--jitter=0`). There is a disorder to aim for, neither zero nor too much.
    rng = numpy.random.default_rng( p.seed )
    k = p.nb_per_side
    xs = ( numpy.arange( k ) + 0.5 ) / k
    pos = numpy.array( [ [ x, y, z ] for x in xs for y in xs for z in xs ] )
    pos = pos + rng.uniform( -p.jitter, p.jitter, size = pos.shape ) / k
    v = Voronoi( pos )
    viz = Visualizer( title = f"Voronoi 3D unbounded, { len( pos ) } seeds" )
    v.add_to_viz( viz, opacity = 0.45 )
    _write_both( p, viz, "vor_3d_open" )

if p := experiment( "vor moving seeds",
                    nb_points = Param( 30, help = "number of seeds" ),
                    nb_frames = Param( 12, help = "number of frames" ),
                    seed      = Param( 7, help = "seed of the draw" ) ):
    # one FRAME per step: the seeds rotate, the diagram is entirely rebuilt each time --
    # which is literally what the class does, no diagram being kept. The page
    # plays by itself, ParaView receives a time series.
    rng = numpy.random.default_rng( p.seed )
    pos = rng.uniform( 0.15, 0.85, size = ( p.nb_points, 2 ) )
    dirs = rng.normal( size = pos.shape )
    dirs /= numpy.linalg.norm( dirs, axis = 1, keepdims = True )

    viz = Visualizer( title = "Voronoi 2D, moving seeds", frame_axis = "step" )
    for k in range( p.nb_frames ):
        if k:
            viz.new_frame( k )
        # a circular trajectory: the seeds come back, so the animation loops cleanly
        a = 2 * numpy.pi * k / p.nb_frames
        cur = pos + 0.08 * ( numpy.cos( a ) - 1 ) * dirs + 0.08 * numpy.sin( a ) * dirs[ :, ::-1 ]
        Voronoi( cur, boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ) ).add_to_viz( viz )
    _write_both( p, viz, "vor_moving" )


if p := experiment( "pd 2D weights",
                    nb_points = Param( 30, help = "number of seeds" ),
                    spread    = Param( 0.02, help = "amplitude of the weights" ),
                    seed      = Param( 3, help = "seed of the draw" ) ):
    # what the weights DO, with the seeds fixed. Two frames: the first without weights (the Voronoi), the
    # second with -- the seeds have not moved an inch, only the planes have slid. At high
    # amplitude (`--spread=0.1`) one sees what a Voronoi diagram never produces: seeds
    # outside their own cell, and seeds that no longer have one at all.
    pos = _seeds( 2, p.nb_points, p.seed )
    w = numpy.random.default_rng( p.seed + 1 ).uniform( -p.spread, p.spread, p.nb_points )
    box = ( [ 0, 0 ], [ 1, 1 ] )

    viz = Visualizer( title = f"Voronoi -> power, { p.nb_points } seeds", frame_axis = "weights" )
    for k, weights in enumerate( ( None, 0.2 * w, 0.4 * w, 0.6 * w, 0.8 * w, w ) ):
        if k:
            viz.new_frame( k )
        PowerDiagram( pos, weights = weights, boundaries = box_half_spaces( *box )).add_to_viz( viz )
        viz.add_points( pos, color = "#ffffff" )
    _write_both( p, viz, "pd_2d_weights" )


# ---- integrating against a DISTRIBUTION ------------------------------------------------------
# `measures` does not necessarily return the VOLUME of a cell: with a distribution it returns
# the integral of its density over it. The contract is in `distributions/Distribution.py` -- the
# distribution CUTS the cell into pieces where its density is simple, the diagram integrates over
# a piece -- and "no distribution" is an ordinary case of it (`UnitDensity`), not an
# absence of code. What follows checks both ends: that the ordinary case has not moved, and
# that the image case says what an independent definition says.


def _unit_box_image( shape, values ):
    """An image over `[ 0, shape_0 ] x ...`: origin 0, identity frame, knots 0, 1, 2, ... --
    the defaults, so the box `k` is literally the unit cube at `k`."""
    return Image( values = numpy.asarray( values, float ).reshape( shape ) )


def _mc_image_measures( pos, values, weights = None, nb_samples = 300000, seed = 0 ):
    """Measures by SAMPLING, density included.

    Like `_monte_carlo_measures`, but each point weighs the density of its box instead of 1 --
    and the density is normalized to mass 1, which `PowerDiagram` does on its side
    (`normalized_version`). No geometry in common with the kernel: no cut, no piece,
    just "which seed wins" and "which box did I land in"."""
    values = numpy.asarray( values, float )
    d = values.ndim
    mi = numpy.zeros( d )
    ma = numpy.asarray( values.shape, float )

    rng = numpy.random.default_rng( seed )
    pts = rng.uniform( mi, ma, size = ( nb_samples, d ) )
    d2 = ( ( pts[ :, None, : ] - pos[ None, :, : ] ) ** 2 ).sum( axis = 2 )
    if weights is not None:
        d2 = d2 - numpy.asarray( weights, float )[ None, : ]
    nearest = d2.argmin( axis = 1 )

    # the box of each point (the knots are 0, 1, 2, ... : the index is the integer part)
    idx = tuple( numpy.clip( pts[ :, k ].astype( int ), 0, values.shape[ k ] - 1 ) for k in range( d ) )
    rho = values[ idx ] / values.sum()          # mass of a box = value * 1, hence the sum

    res = numpy.zeros( len( pos ) )
    numpy.add.at( res, nearest, rho )
    return res * float( numpy.prod( ma - mi ) ) / nb_samples


if test( "a_constant_image_is_the_volume_up_to_its_mass" ):
    # the setting where the two definitions must coincide EXACTLY (no sampling): a
    # constant density integrated over a cell is its volume -- normalized, hence divided by
    # the total volume. If the cutting into pieces lost, duplicated or shifted anything
    # at all, this is where it would show, to the last digit.
    for d, shape in ( ( 2, ( 3, 4 ) ), ( 3, ( 2, 3, 2 ) ) ):
        rng = numpy.random.default_rng( 900 + d )
        n = 12
        pos = rng.uniform( 0.2, numpy.array( shape ) - 0.2, size = ( n, d ) )
        box = ( numpy.zeros( d ), numpy.array( shape, float ) )

        plain = _measures( PowerDiagram( pos, boundaries = box_half_spaces( *box )) )
        img = _unit_box_image( shape, numpy.ones( shape ) )
        with_img = _measures( PowerDiagram( pos, boundaries = box_half_spaces( *box ), distribution = img ) )

        total = float( numpy.prod( shape ) )
        assert numpy.allclose( with_img, plain / total, atol = 1e-9 ), ( d, with_img, plain / total )
        # ... and the mass is found again: the cells tile the domain, so the whole image.
        assert abs( with_img.sum() - 1 ) < 1e-9, ( d, with_img.sum() )


if test( "no_distribution_is_still_the_plain_volume" ):
    # the other end of the same contract: `UnitDensity` must change NOTHING. The line of code
    # `measures` is the same in both cases anyway (one piece, density 1) -- this test says
    # that going through `for_each_piece` does not cost a digit.
    rng = numpy.random.default_rng( 901 )
    for d in ( 2, 3 ):
        pos = rng.uniform( 0, 1, size = ( 20, d ) )
        box = ( numpy.zeros( d ), numpy.ones( d ) )
        m = _measures( PowerDiagram( pos, boundaries = box_half_spaces( *box )) )
        assert abs( m.sum() - 1 ) < 1e-9, ( d, m.sum() )


if test( "an_image_integrates_what_a_sampling_says" ):
    # the INDEPENDENT check: a non-trivial image (arbitrary values, zeros
    # included) against a uniform draw. The tolerance is that of a Monte Carlo, not that of the
    # kernel -- what we look for here is a convention error (a box shifted by one notch, a
    # value read in the wrong place), not a ulp.
    for d, shape, n in ( ( 2, ( 4, 3 ), 10 ), ( 3, ( 2, 3, 2 ), 8 ) ):
        rng = numpy.random.default_rng( 902 + d )
        values = rng.uniform( 0, 1, size = shape )
        values.reshape( -1 )[ 0 ] = 0.0                     # an empty box: it must contribute nothing
        pos = rng.uniform( 0.1, numpy.array( shape ) - 0.1, size = ( n, d ) )
        box = ( numpy.zeros( d ), numpy.array( shape, float ) )

        got = _measures( PowerDiagram( pos, boundaries = box_half_spaces( *box ), distribution = _unit_box_image( shape, values ) ) )
        ref = _mc_image_measures( pos, values, nb_samples = 400000, seed = 7 )

        assert abs( got.sum() - 1 ) < 1e-9, ( d, got.sum() )
        assert numpy.abs( got - ref ).max() < 0.02, ( d, got, ref )


if test( "an_image_integrates_what_a_sampling_says_with_weights" ):
    # the weights move the bisectors; the density does not move. The two mechanisms
    # are independent and compose -- still, this has to be checked once.
    d, shape, n = 2, ( 4, 4 ), 9
    rng = numpy.random.default_rng( 903 )
    values = rng.uniform( 0.2, 1, size = shape )
    pos = rng.uniform( 0.3, numpy.array( shape ) - 0.3, size = ( n, d ) )
    w = rng.uniform( -0.3, 0.3, size = n )
    box = ( numpy.zeros( d ), numpy.array( shape, float ) )

    got = _measures( PowerDiagram( pos, weights = w, boundaries = box_half_spaces( *box ),
                                   distribution = _unit_box_image( shape, values ) ) )
    ref = _mc_image_measures( pos, values, weights = w, nb_samples = 400000, seed = 11 )
    assert abs( got.sum() - 1 ) < 1e-9, got.sum()
    assert numpy.abs( got - ref ).max() < 0.02, ( got, ref )


def _mc_image_measures_general( pos, values, origin, frame, knots, nb_samples = 400000, seed = 0 ):
    """`_mc_image_measures`, but for an image that is not the unit cube: `x = origin + F^T t`
    and arbitrary knots. Drawing uniformly in `t` amounts to drawing uniformly in `x` (the
    Jacobian is constant), and `|det F|` disappears from the normalization -- it multiplies all the
    boxes equally."""
    values = numpy.asarray( values, float )
    d = values.ndim
    lo = numpy.array( [ knots[ a ][ 0 ] for a in range( d ) ] )
    hi = numpy.array( [ knots[ a ][ values.shape[ a ] ] for a in range( d ) ] )

    rng = numpy.random.default_rng( seed )
    t = rng.uniform( lo, hi, size = ( nb_samples, d ) )
    pts = numpy.asarray( origin ) + t @ numpy.asarray( frame )

    d2 = ( ( pts[ :, None, : ] - pos[ None, :, : ] ) ** 2 ).sum( axis = 2 )
    nearest = d2.argmin( axis = 1 )

    idx = tuple( numpy.clip( numpy.searchsorted( knots[ a ][ : values.shape[ a ] + 1 ], t[ :, a ],
                                                 side = "right" ) - 1, 0, values.shape[ a ] - 1 )
                 for a in range( d ) )

    # the mass of a box is `value * |det F| * product of the steps`; the `|det F|` cancels out
    spacing = numpy.ones( values.shape )
    for a in range( d ):
        w = numpy.diff( numpy.asarray( knots[ a ][ : values.shape[ a ] + 1 ], float ) )
        spacing = spacing * w.reshape( [ -1 if k == a else 1 for k in range( d ) ] )
    rho = values[ idx ] / float( ( values * spacing ).sum() )

    res = numpy.zeros( len( pos ) )
    numpy.add.at( res, nearest, rho )
    return res * float( numpy.prod( hi - lo ) ) / nb_samples


if test( "an_image_honors_its_origin_frame_and_knots" ):
    # the image geometry is not necessarily the unit cube: `origin` shifts it, `frame`
    # deforms it, `knots` de-regularizes it. The planes of a box are written from `F^-1` (see
    # `Image::_for_each_piece`), so a forgotten transpose would go unnoticed on an identity
    # and not here -- and the unequal knots say that the box index is indeed SEARCHED and not
    # computed by a division.
    d, n = 2, 8
    shape = ( 3, 3 )
    rng = numpy.random.default_rng( 904 )
    values = rng.uniform( 0.2, 1, size = shape )

    origin = numpy.array( [ -1.0, 2.0 ] )
    frame = numpy.array( [ [ 2.0, 0.5 ], [ 0.0, 1.5 ] ] )               # `x = origin + F^T t`
    knots = numpy.array( [ [ 0.0, 1.0, 2.5, 4.0 ], [ -1.0, 0.5, 1.0, 3.0 ] ] )

    # the seeds, drawn in grid coordinates then transported: so they fall inside the image.
    t = rng.uniform( knots[ :, 0 ] + 0.1, knots[ :, -1 ] - 0.1, size = ( n, d ) )
    pos = origin + t @ frame

    img = Image( values = values, origin = origin, frame = frame, knots = knots )

    # the domain, written in the physical frame: column `a` of `F^-1` is the normal of axis
    # `a`, and the useful band goes from `knots( a, 0 )` to `knots( a, shape_a )`.
    inv = numpy.linalg.inv( frame )
    sh = inv.T @ origin                                                 # `n_a . origin`, per axis
    dirs = numpy.concatenate( [ inv.T, -inv.T ] )
    offs = numpy.concatenate( [ knots[ :, -1 ] + sh, -( knots[ :, 0 ] + sh ) ] )

    got = _measures( PowerDiagram( pos, boundaries = ( dirs, offs ), distribution = img ) )
    ref = _mc_image_measures_general( pos, values, origin, frame, knots, nb_samples = 500000, seed = 17 )

    assert abs( got.sum() - 1 ) < 1e-6, got.sum()
    assert numpy.abs( got - ref ).max() < 0.02, ( got, ref )


if test( "an_unbounded_diagram_is_finite_against_an_image" ):
    # without a domain, the border cells are infinite and `measures` returns `TF::max`. With an
    # image, not at all anymore: the image has compact support, so the integral is finite -- and the
    # sum is always the mass. This is the case where `for_each_piece` has no bounding box to
    # read and sweeps the whole image (see `Image::_for_each_piece`).
    d, shape, n = 2, ( 3, 3 ), 6
    rng = numpy.random.default_rng( 905 )
    values = rng.uniform( 0.2, 1, size = shape )
    pos = rng.uniform( 0.5, numpy.array( shape ) - 0.5, size = ( n, d ) )

    plain = _measures( PowerDiagram( pos ) )
    assert ( plain > 1e300 ).any(), plain          # some cells are indeed infinite

    got = _measures( PowerDiagram( pos, distribution = _unit_box_image( shape, values ) ) )
    assert numpy.isfinite( got ).all(), got
    assert abs( got.sum() - 1 ) < 1e-9, got.sum()

    ref = _mc_image_measures( pos, values, nb_samples = 400000, seed = 13 )
    assert numpy.abs( got - ref ).max() < 0.02, ( got, ref )


if test( "an_image_leaves_the_derivatives_right" ):
    need( "grad" )
    # the adjoint against the finite difference. A piece is a polytope whose cuts carry
    # the index of the seed they face -- it is THIS fact that makes it possible to reuse
    # `scatter_cell_grad` as is on top of it -- and that is what this test puts to the test, positions and
    # weights at the same time.
    d, shape, n = 2, ( 3, 3 ), 6
    rng = numpy.random.default_rng( 906 )
    values = rng.uniform( 0.3, 1, size = shape )
    pos = rng.uniform( 0.4, numpy.array( shape ) - 0.4, size = ( n, d ) )
    w = rng.uniform( -0.1, 0.1, size = n )
    box = ( numpy.zeros( d ), numpy.array( shape, float ) )

    check_grad( lambda p, q: PowerDiagram( p, weights = q, boundaries = box_half_spaces( *box ),
                    distribution = _unit_box_image( shape, values ) ).measures, pos, w )


if test( "the_density_carries_its_own_derivative" ):
    need( "grad" )
    # `d mass / d value of a box` is the VOLUME of the piece -- the only thing the
    # distribution has to know how to accumulate, and the only one the diagram cannot guess.
    # The normalization (`values / total mass`) is on the Python side, so autodiff goes through it
    # by itself: what this test checks is the WHOLE chain, not just the kernel.
    d, shape, n = 2, ( 3, 3 ), 5
    rng = numpy.random.default_rng( 907 )
    values = rng.uniform( 0.3, 1, size = shape )
    pos = rng.uniform( 0.4, numpy.array( shape ) - 0.4, size = ( n, d ) )
    box = ( numpy.zeros( d ), numpy.array( shape, float ) )

    check_grad( lambda v: PowerDiagram( pos, boundaries = box_half_spaces( *box ),
                    distribution = Image( values = v ) ).measures, values )


if test( "the_support_of_a_distribution_bounds_the_cells" ):
    # an image has a compact support, so clipping against it is an IDENTITY, not an
    # approximation -- and it bounds the cells for free. Two witnesses: without an explicit domain
    # the cells become finite, and the result is the same as with the domain written by hand.
    d, shape, n = 2, ( 3, 3 ), 7
    rng = numpy.random.default_rng( 908 )
    values = rng.uniform( 0.2, 1, size = shape )
    pos = rng.uniform( 0.4, numpy.array( shape ) - 0.4, size = ( n, d ) )
    img = lambda: _unit_box_image( shape, values )

    free = PowerDiagram( pos, distribution = img() )
    boxed = PowerDiagram( pos, boundaries = box_half_spaces( numpy.zeros( d ), numpy.array( shape, float ) ), distribution = img() )

    assert numpy.allclose( _measures( free ), _measures( boxed ), atol = 1e-7 )

    # the cells themselves are bounded, which is what an accelerator requires in order to prune
    # (`cell_may_be_cut` has nothing to bite on with an infinite cell) -- and what the cutting
    # requires in order to have a bounding box to read.
    assert numpy.asarray( free.cells.is_bounded ).reshape( -1 ).all()
    assert not numpy.asarray( PowerDiagram( pos ).cells.is_bounded ).reshape( -1 ).all()


if test( "the_support_intersects_a_given_domain" ):
    # the caller's domain AND the support: the half-spaces ADD UP, they do not replace each other.
    # Here the requested box is smaller than the image, so it is the one that must win.
    d, shape, n = 2, ( 4, 4 ), 6
    rng = numpy.random.default_rng( 909 )
    values = numpy.ones( shape )
    pos = rng.uniform( 1.1, 2.9, size = ( n, d ) )
    small = ( numpy.ones( d ), 3 * numpy.ones( d ) )

    m = _measures( PowerDiagram( pos, boundaries = box_half_spaces( *small ), distribution = _unit_box_image( shape, values ) ) )
    # constant image of mass 1 over 16 boxes: the small box covers 4 of them, so a quarter.
    assert abs( m.sum() - 0.25 ) < 1e-6, m.sum()


# ---- a SMOOTH density: quadrature, and the delegation of the derivatives ----------------------
# `SumOfGaussians` cuts nothing and knows nothing about cells: it only answers at a POINT
# (`value_at`, `gradient_at`, `add_value_grad_at`). It is `PowerDiagram` that, seeing
# `is_constant == false`, cuts the piece into simplices and does its quadrature there. What follows
# puts that split to the test -- especially on the derivative side, where each must produce ONLY its half.


def _gaussian_density( x, centers, sigmas, weights ):
    """rho( x ) written in numpy, independently of the kernel. `x`: `[ m, d ]`."""
    d = x.shape[ 1 ]
    r2 = ( ( x[ :, None, : ] - centers[ None, :, : ] ) ** 2 ).sum( axis = 2 )
    norm = ( 2 * numpy.pi * sigmas ** 2 ) ** ( -d / 2 )
    return ( weights * norm * numpy.exp( - r2 / ( 2 * sigmas ** 2 ) ) ).sum( axis = 1 )


def _mc_gaussian_measures( pos, centers, sigmas, weights, mi, ma, nb_samples = 400000, seed = 0 ):
    """Measures by sampling, Gaussian density included -- normalized to mass 1 as
    `PowerDiagram` does, and TRUNCATED to the domain as it does too (the tails that leave the
    box are in no cell)."""
    mi, ma = numpy.asarray( mi, float ), numpy.asarray( ma, float )
    rng = numpy.random.default_rng( seed )
    pts = rng.uniform( mi, ma, size = ( nb_samples, mi.size ) )

    rho = _gaussian_density( pts, centers, sigmas, weights / weights.sum() )
    nearest = ( ( pts[ :, None, : ] - pos[ None, :, : ] ) ** 2 ).sum( axis = 2 ).argmin( axis = 1 )

    res = numpy.zeros( len( pos ) )
    numpy.add.at( res, nearest, rho )
    return res * float( numpy.prod( ma - mi ) ) / nb_samples


if test( "a_smooth_density_integrates_what_a_sampling_says" ):
    # the quadrature is exact to degree 2 on each simplex: with Gaussians that are WIDE compared to
    # the cells -- the regime of the applications -- it must land on the reference to the
    # tolerance of the Monte Carlo and not to its own.
    for d, n in ( ( 2, 12 ), ( 3, 10 ) ):
        rng = numpy.random.default_rng( 910 + d )
        mi, ma = numpy.zeros( d ), numpy.ones( d )
        centers = rng.uniform( 0.2, 0.8, size = ( 3, d ) )
        sigmas = rng.uniform( 0.35, 0.6, size = 3 )
        weights = rng.uniform( 0.5, 1.5, size = 3 )
        pos = rng.uniform( 0.05, 0.95, size = ( n, d ) )

        sog = SumOfGaussians( positions = centers, sigmas = sigmas, weights = weights )
        got = _measures( PowerDiagram( pos, boundaries = box_half_spaces( mi, ma ), distribution = sog ) )
        ref = _mc_gaussian_measures( pos, centers, sigmas, weights, mi, ma,
                                     nb_samples = 500000, seed = 23 )

        assert numpy.isfinite( got ).all(), ( d, got )
        assert numpy.abs( got - ref ).max() < 0.01, ( d, got, ref )
        # the mass OUTSIDE the domain is lost, so the sum is the target mass minus the tails:
        # less than 1, and equal to what the same sampling says about it.
        assert got.sum() < 1.0 and abs( got.sum() - ref.sum() ) < 0.01, ( d, got.sum(), ref.sum() )


if test( "a_smooth_density_is_exact_on_an_affine_one" ):
    # the quadrature rule is exact up to degree 2, so on an affine density it must make
    # NO error AT ALL. A very wide Gaussian is one, up to second order: we
    # rather take the clean test -- a single centred, very wide Gaussian, whose
    # sum of measures we compare to its exact integral over the box (erf), not to a draw.
    from math import erf, sqrt
    d, n = 2, 9
    rng = numpy.random.default_rng( 912 )
    mi, ma = numpy.zeros( d ), numpy.ones( d )
    centers = numpy.full( ( 1, d ), 0.5 )
    sigmas = numpy.array( [ 4.0 ] )                     # very flat compared to the box
    weights = numpy.array( [ 1.0 ] )
    pos = rng.uniform( 0.05, 0.95, size = ( n, d ) )

    sog = SumOfGaussians( positions = centers, sigmas = sigmas, weights = weights )
    got = _measures( PowerDiagram( pos, boundaries = box_half_spaces( mi, ma ), distribution = sog ) )

    # the exact integral of the normalized Gaussian over the square: a product of erf per axis
    part = erf( 0.5 / ( sigmas[ 0 ] * sqrt( 2.0 ) ) ) ** d
    assert abs( got.sum() - part ) < 2e-4, ( got.sum(), part )


if test( "a_smooth_density_leaves_the_derivatives_right" ):
    need( "grad" )
    # THE delegation test. The adjoint goes through three separate responsibilities:
    #   * the distribution only returns `gradient_at` (how the density varies at a point);
    #   * the integrator knows that a quadrature node is a FIXED barycentric combination of the
    #     simplex vertices, hence how the cotangent of a point falls back on them, and how
    #     the volume moves with them;
    #   * `scatter_cell_grad` goes back up from the vertices to the seeds, knowing nothing about the density.
    # If one of the three forgets its half, the finite difference says so. Positions AND weights together.
    d, n = 2, 7
    rng = numpy.random.default_rng( 913 )
    mi, ma = numpy.zeros( d ), numpy.ones( d )
    centers = rng.uniform( 0.2, 0.8, size = ( 2, d ) )
    sigmas = numpy.array( [ 0.4, 0.55 ] )
    weights = numpy.array( [ 1.0, 0.7 ] )
    pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
    w = rng.uniform( -0.01, 0.01, size = n )

    check_grad( lambda p, q: PowerDiagram( p, weights = q, boundaries = box_half_spaces( mi, ma ),
                    distribution = SumOfGaussians( positions = centers, sigmas = sigmas,
                                                   weights = weights ) ).measures, pos, w )


if test( "the_gaussians_carry_their_own_derivatives" ):
    need( "grad" )
    # the other half: `d measure / d ( centres, sigmas, weights )`. It does NOT go through the
    # geometry -- the cells do not move when the density changes -- so it tests
    # exactly `SumOfGaussians::add_value_grad_at`, and the normalization (`w / sum( w )`) that
    # autodiff goes through on the Python side.
    d, n = 2, 6
    rng = numpy.random.default_rng( 914 )
    mi, ma = numpy.zeros( d ), numpy.ones( d )
    pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
    centers = rng.uniform( 0.25, 0.75, size = ( 2, d ) )
    sigmas = numpy.array( [ 0.45, 0.6 ] )
    weights = numpy.array( [ 1.0, 0.8 ] )
    box = ( mi, ma )

    check_grad( lambda c, s, q: PowerDiagram( pos, boundaries = box_half_spaces( *box ),
                    distribution = SumOfGaussians( positions = c, sigmas = s, weights = q ) ).measures,
                centers, sigmas, weights )


def _fine_triangle_integral( A, B, C, centers, sigmas, weights, depth ):
    """The integral of the density over a triangle, by recursive subdivision + 3-node rule.

    A reference INDEPENDENT of the kernel: no special function, no closed formula -- just
    brute force, `4 ** depth` sub-triangles."""
    if depth == 0:
        # the 2D cross product written by hand: `numpy.cross` on dimension-2 vectors
        # is REMOVED since NumPy 2 (it only takes dimension 3 now). It worked locally
        # with a warning, failed in the container, whose numpy is more recent.
        u, v = B - A, C - A
        area = abs( u[ 0 ] * v[ 1 ] - u[ 1 ] * v[ 0 ] ) / 2
        pts = numpy.array( [ A, B, C ] )
        al, be = 2 / 3, 1 / 6
        s = 0.0
        for q in range( 3 ):
            bar = numpy.full( 3, be ); bar[ q ] = al
            s += _gaussian_density( ( bar @ pts )[ None, : ], centers, sigmas, weights )[ 0 ]
        return area * s / 3
    AB, BC, CA = ( A + B ) / 2, ( B + C ) / 2, ( C + A ) / 2
    return ( _fine_triangle_integral( A, AB, CA, centers, sigmas, weights, depth - 1 )
           + _fine_triangle_integral( AB, B, BC, centers, sigmas, weights, depth - 1 )
           + _fine_triangle_integral( CA, BC, C, centers, sigmas, weights, depth - 1 )
           + _fine_triangle_integral( AB, BC, CA, centers, sigmas, weights, depth - 1 ) )


def _brute_force_cell_integrals( pd, centers, sigmas, weights, depth = 5 ):
    """The integral of the density over EACH cell of `pd`, computed in numpy on the geometry that
    the kernel produced. What is tested this way is the INTEGRATION alone, the geometry being the
    same on both sides."""
    cs = pd.cells
    nvs = numpy.asarray( cs.nb_vertices.value ).reshape( -1 )
    res = numpy.zeros( len( nvs ) )
    for i in range( len( nvs ) ):
        v = cs.vertices( i )
        for k in range( 1, len( v ) - 1 ):      # fan from vertex 0, like the kernel
            res[ i ] += _fine_triangle_integral( v[ 0 ], v[ k ], v[ k + 1 ],
                                                 centers, sigmas, weights, depth )
    return res


if test( "the_exact_2d_gaussian_holds_where_a_quadrature_could_not" ):
    # THE test that tells the exact from the quadrature. A NARROW Gaussian compared to the cells:
    # a degree-2 rule on the cell makes an error in `( size / sigma ) ^ 4` there, i.e.
    # huge, whereas the 1D reduction does not depend at all on the shape of the cell.
    #
    # Two independent witnesses: the total mass, which has a closed form (a product of erf over the
    # box), and each cell taken separately against a brute-force subdivision.
    from math import erf, sqrt
    d, n = 2, 6
    rng = numpy.random.default_rng( 920 )
    mi, ma = numpy.zeros( d ), numpy.ones( d )
    centers = numpy.array( [ [ 0.5, 0.5 ] ] )
    sigmas = numpy.array( [ 0.09 ] )            # cells ~0.4: sigma 4x smaller
    weights = numpy.array( [ 1.0 ] )
    pos = rng.uniform( 0.08, 0.92, size = ( n, d ) )

    sog = lambda: SumOfGaussians( positions = centers, sigmas = sigmas, weights = weights )
    pd = PowerDiagram( pos, boundaries = box_half_spaces( mi, ma ), distribution = sog() )
    got = _measures( pd )

    # the mass in the box, in closed form -- the Gaussian is normalized to 1
    part = erf( 0.5 / ( sigmas[ 0 ] * sqrt( 2.0 ) ) ) ** d
    assert abs( got.sum() - part ) < 1e-5, ( got.sum(), part )

    ref = _brute_force_cell_integrals( PowerDiagram( pos, boundaries = box_half_spaces( mi, ma ) ),
                                       centers, sigmas, weights / weights.sum(), depth = 6 )
    assert numpy.abs( got - ref ).max() < 1e-5, ( got, ref )


if test( "the_exact_2d_gaussian_survives_the_awkward_placements" ):
    # the configurations that trip up the reduction if the signs or the tails are wrong:
    # a seed EXACTLY on the centre of a Gaussian (the origin falls on a cell vertex
    # and the distances to the lines become tiny), and cells that are huge compared to sigma.
    d = 2
    mi, ma = numpy.zeros( d ), numpy.ones( d )
    centers = numpy.array( [ [ 0.30, 0.42 ], [ 0.75, 0.60 ] ] )
    sigmas = numpy.array( [ 0.05, 0.35 ] )
    weights = numpy.array( [ 1.0, 0.6 ] )
    # the first seed IS the first centre; the other three are far away -> large cells
    pos = numpy.array( [ [ 0.30, 0.42 ], [ 0.95, 0.05 ], [ 0.05, 0.95 ], [ 0.95, 0.95 ] ] )

    sog = SumOfGaussians( positions = centers, sigmas = sigmas, weights = weights )
    got = _measures( PowerDiagram( pos, boundaries = box_half_spaces( mi, ma ), distribution = sog ) )
    ref = _brute_force_cell_integrals( PowerDiagram( pos, boundaries = box_half_spaces( mi, ma ) ),
                                       centers, sigmas, weights / weights.sum(), depth = 7 )

    assert numpy.isfinite( got ).all(), got
    assert numpy.abs( got - ref ).max() < 1e-5, ( got, ref )


if test( "the_exact_2d_gaussian_derives_right_too" ):
    need( "grad" )
    # the adjoint of the EXACT path is a completely different code from that of the quadrature: boundary
    # integrals (`erf`) instead of a cofactor plus a gradient at the nodes. It therefore deserves its own
    # finite difference -- and on a narrow Gaussian, where the two paths cannot be
    # confused.
    d, n = 2, 6
    rng = numpy.random.default_rng( 921 )
    mi, ma = numpy.zeros( d ), numpy.ones( d )
    centers = numpy.array( [ [ 0.42, 0.55 ], [ 0.68, 0.30 ] ] )
    sigmas = numpy.array( [ 0.12, 0.20 ] )
    weights = numpy.array( [ 1.0, 0.7 ] )
    pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
    w = rng.uniform( -0.01, 0.01, size = n )

    check_grad( lambda p, q: PowerDiagram( p, weights = q, boundaries = box_half_spaces( mi, ma ),
                    distribution = SumOfGaussians( positions = centers, sigmas = sigmas,
                                                   weights = weights ) ).measures, pos, w )

    check_grad( lambda c, s, q: PowerDiagram( pos, boundaries = box_half_spaces( mi, ma ),
                    distribution = SumOfGaussians( positions = c, sigmas = s, weights = q ) ).measures,
                centers, sigmas, weights )


if test( "the_quadrature_subdivides_until_it_sees_the_density" ):
    # In 3D we do not know how to integrate exactly, so it is `PointwiseDensity` that works -- but a rule
    # of degree 2 placed on a tetrahedron wider than sigma is simply BLIND to the density.
    # Hence the adaptive bisection: we cut the longest edge as long as the simplex and its two
    # halves do not agree.
    #
    # Witness: the total mass in the box, which has a closed form (a product of erf per axis) and owes
    # nothing to the kernel. The cells tiling the box, the sum of the measures is that mass.
    from math import erf, sqrt
    d, n = 3, 8
    rng = numpy.random.default_rng( 930 )
    mi, ma = numpy.zeros( d ), numpy.ones( d )
    centers = numpy.full( ( 1, d ), 0.5 )
    sigmas = numpy.array( [ 0.15 ] )            # cells ~0.5: the raw rule would see nothing
    weights = numpy.array( [ 1.0 ] )
    pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )

    sog = SumOfGaussians( positions = centers, sigmas = sigmas, weights = weights )
    got = _measures( PowerDiagram( pos, boundaries = box_half_spaces( mi, ma ), distribution = sog ) )

    part = erf( 0.5 / ( sigmas[ 0 ] * sqrt( 2.0 ) ) ) ** d
    assert numpy.isfinite( got ).all(), got
    assert abs( got.sum() - part ) < 2e-3, ( got.sum(), part )


if test( "the_subdivided_quadrature_derives_right" ):
    need( "grad" )
    # The adjoint of the quadrature path: it must redo EXACTLY the same subdivision as the forward
    # (the criterion is deterministic), differentiate each leaf, and bring the cotangents back to the
    # original vertices through their barycentric coordinates.
    #
    # = The tolerance has an ABSOLUTE part, and it is what carries everything
    #
    # An adaptive quadrature BY VALUE is only smooth up to `rtol`: when a sub-simplex
    # flips from "refine" to "accept", the value JUMPS by that much. The adjoint, for its part, differentiates the
    # locally smooth branch -- which is correct, and what an optimizer needs. The
    # finite difference, for its part, divides this jump by `2 eps`: its error is thus a JUMP OVER ONE STEP,
    # an ABSOLUTE quantity, which does not shrink when the measured derivative is small.
    #
    # Now `check_grad` projects the Jacobian onto a direction, and this projection may come out
    # small. MEASURED, narrow regime, 40 directions: the absolute gap stays within `[ 3e-5, 6e-3 ]`
    # whatever the direction, whereas the projection goes down to 4e-4 -- the RELATIVE
    # gap therefore climbs to 165 % with nothing being wrong. A purely relative tolerance was
    # the wrong form, and it is exactly how this test failed in the full suite while passing
    # alone: `check_grad` drew its direction from a process counter, hence from a
    # number of draws made by the tests BEFORE. Hence the `seed`, which is not an
    # adjustment but what makes the check reproducible.
    #
    # = The STEP is per CALL, and it is measured
    #
    # The jump error goes as `1/eps`, the truncation as `eps^2`: there is an optimum, and it is
    # not the same depending on what is perturbed. Over 10 directions, max gap:
    #
    #                       eps=2e-3   5e-3     1e-2     2e-2
    #     smooth,  positions  2.7e-5   5.1e-5   3.7e-5   2.2e-4
    #     smooth,  distrib.   4.2e-5   2.6e-5   3.3e-5   1.0e-4
    #     narrow,  positions  7.8e-3   4.1e-3   3.6e-3   7.2e-3
    #     narrow,  distrib.   2.2e-3   3.9e-3   8.2e-3   1.3e-2
    #
    # Perturbing the sigmas requires a shorter step than perturbing the positions: `2e-2` on a
    # sigma of 0.18 is 11 % of the parameter, and truncation wins. Each `atol` below
    # is taken at ~3x the worst gap of its row.
    #
    # Tuned this way, the check passes for ALL 10 directions tried (so the `seed` does not pick
    # a lucky draw, only a FIXED draw), and it detects an adjoint error of 0.6 % /
    # 0.4 % / 2.8 % / 1.6 % depending on the call. The smooth regime gains a lot from it: it used to share
    # the tolerances of the narrow regime until now, whereas its gap there is 100x smaller.
    d, n = 3, 6
    rng = numpy.random.default_rng( 931 )
    mi, ma = numpy.zeros( d ), numpy.ones( d )
    pos = rng.uniform( 0.15, 0.85, size = ( n, d ) )

    #                sigmas                          ( eps, rtol, atol, seed ) positions | distribution
    regimes = ( ( numpy.array( [ 0.7, 0.9 ] ),   ( 2e-3, 3e-3, 1e-4, 1 ), ( 5e-3, 3e-3, 1e-4, 6 ) ),
                ( numpy.array( [ 0.18, 0.30 ] ), ( 1e-2, 1e-2, 1.2e-2, 8 ), ( 2e-3, 1e-2, 7e-3, 8 ) ) )

    def _tol( c ):
        return dict( eps = c[ 0 ], rtol = c[ 1 ], atol = c[ 2 ], seed = c[ 3 ] )

    for sigmas, by_pos, by_dist in regimes:
        centers = rng.uniform( 0.3, 0.7, size = ( 2, d ) )
        weights = numpy.array( [ 1.0, 0.6 ] )

        check_grad( lambda p: PowerDiagram( p, boundaries = box_half_spaces( mi, ma ),
                        distribution = SumOfGaussians( positions = centers, sigmas = sigmas,
                                                       weights = weights ) ).measures,
                    pos, **_tol( by_pos ) )

        check_grad( lambda c, s, q: PowerDiagram( pos, boundaries = box_half_spaces( mi, ma ),
                        distribution = SumOfGaussians( positions = c, sigmas = s, weights = q ) ).measures,
                    centers, sigmas, weights, **_tol( by_dist ) )


# -- the `float` kernel, which is the default ---------------------------------------------------
# The geometry is cut in FP32 ( `kernel_dtype`, see `Cell.py` ) and what is derived from it is computed
# in the float type of the positions. What that is worth, and what it is not worth, is measured here: the
# tiling holds to 1e-6, the gradient to a few 1e-4 -- enough for an optimal transport, not for
# the 1e-10 witnesses of the other tests, which require the double kernel.

if test( "the_float_kernel_tiles_the_domain" ):
    for d, n in ( ( 2, 300 ), ( 3, 120 ) ):
        rng = numpy.random.default_rng( 900 + d )
        pos = rng.uniform( 0.02, 0.98, size = ( n, d ) )
        w = rng.uniform( -0.01, 0.01, n )
        for kd in ( "FP32", None ):
            pd = PowerDiagram( pos, weights = w, boundaries = box_half_spaces( [ 0 ] * d, [ 1 ] * d ), kernel_dtype = kd )
            m = _measures( pd )
            assert abs( float( m.sum() ) - 1 ) < 1e-6, ( d, kd, m.sum() )
            # the measure itself is in the float type of the positions, not in that of the kernel
            assert numpy.asarray( pd.measures.value ).dtype == numpy.float64
            ref = _measures( PowerDiagram( pos, weights = w, boundaries = box_half_spaces( [ 0 ] * d, [ 1 ] * d ), kernel_dtype = "FP64" ) )
            assert numpy.abs( m - ref ).max() < 2e-6, ( d, kd )

if test( "the_float_kernel_is_accelerated_the_same" ):
    # pruning is done in the float type of the kernel ( `cell/Pruning.h` ): it must not silence any
    # useful cut for all that
    d, n = 2, 400
    rng = numpy.random.default_rng( 910 )
    pos = rng.uniform( 0.02, 0.98, size = ( n, d ) )
    plain = PowerDiagram( pos, boundaries = box_half_spaces( [ 0 ] * d, [ 1 ] * d ), kernel_dtype = "FP32",
                          accelerator = "plain" )
    fast = PowerDiagram( pos, boundaries = box_half_spaces( [ 0 ] * d, [ 1 ] * d ), kernel_dtype = "FP32",
                         accelerator = AaBsp.of( plain, max_seeds_per_leaf = 8 ) )
    assert numpy.abs( _measures( plain ) - _measures( fast ) ).max() < 1e-6
    assert _cut_sets( plain ) == _cut_sets( fast )

if test( "the_float_kernel_derives_well_enough" ):
    need( "grad" )
    d, n = 2, 12
    rng = numpy.random.default_rng( 920 )
    pos = rng.uniform( 0.1, 0.9, size = ( n, d ) )
    w = rng.uniform( -0.01, 0.01, n )
    box = box_half_spaces( [ 0 ] * d, [ 1 ] * d )
    check_grad( lambda p, q: PowerDiagram( p, q, boundaries = box, kernel_dtype = "FP32" ).measures,
                driver.array( pos ), driver.array( w ), rtol = 1e-2, atol = 1e-3, seed = 3 )
