"""THE 2D CELLS OF THE CARD ( `include/sdot/gpu/Cell2D.cuh`, `Laplacian2D.cuh`, `PowerDiagram_Bsp._card_variant` ).

On a CUDA device, a 2D power diagram in a box goes through a dedicated kernel ( the cell in registers, overflow
passes, the tree as records in the kernel's float, the float accuracy fixes of the old GPU campaign ): its measures,
their ADJOINT ( positions and weights ), the facets that make Newton's Hessian ( the laplacian of the Laguerre graph,
in CSR ) and the moments of the cells. These tests compare it with the GENERIC path of the same diagram
( `use_card_cells = False`, double kernel ): the same cells, to the rounding of the kernel's float -- which, for the
float kernel, is the DOUBLE's rounding ( the vertices are re-solved in double ), except where the float decides a
different topology.

The four properties the dedicated path owes ( and that these tests pin ): it runs UNDER `jit` and under a
derivative ( an ffi call with its own adjoint ); nothing is read back and nothing runs again, the last overflow pass
working through its cells in batches within a fixed budget; the index types follow the tree ( no depth limit ); a cell
past the vertex limit is an ERROR ( `KernelFailure`, eager and traced ), never a NaN.

Every test skips itself without a CUDA device ( the path does not exist on a CPU ) -- but the one on the choice of the
variant, which is pure Python.
"""
import numpy

from loom import driver
from errand import test, skip

from sdot import AaBsp, PowerDiagram, box_half_spaces
from sdot.PowerDiagram_Bsp import card_variant_for

_GPU = bool( getattr( driver.device, "is_cuda_gpu", False ) )
_NO_GPU = "the dedicated 2D cell kernel only exists on a CUDA device"


def _pd( pos, w = None, kernel = "FP32", card = True, mi = ( 0, 0 ), ma = ( 1, 1 ), tree = None, **kw ):
    pd = PowerDiagram( pos, weights = w, boundaries = box_half_spaces( mi, ma ), kernel_dtype = kernel,
                       accelerator = tree, **kw )
    pd.use_card_cells = card
    return pd


def _m( pd ):
    return numpy.asarray( pd.measures.value ).reshape( -1 )


def _rel( m, ref ):
    """the gap relative to the cell, a cell 1e-3 of the mean cell or smaller counting as that ( an almost empty cell
    is decided at the rounding of the planes, generic path included )"""
    ok = ref > 0                                         # ( a NaN of the reference is not a cell to compare )
    floor = 1e-3 * numpy.nansum( ref ) / len( ref )
    return numpy.abs( m[ ok ] - ref[ ok ] ) / numpy.maximum( ref[ ok ], floor )


def _check( pos, w = None, mi = ( 0, 0 ), ma = ( 1, 1 ), tol64 = 1e-9, tol32_med = 1e-11, tol32_max = 1e-6, label = "", plain = False ):
    """the card ( float and double kernels ) against the generic double path, on the same tree -- or, `plain`, on the plain
    storage ( every seed cuts every cell: exact, and blind to the tree; the generic BSP path gets a few cells of the rings
    wrong on the card, NaN or a wrong area depending on how the tree orders its halves -- a defect of that path, not
    fixed here )"""
    tree = AaBsp( pos, w )
    ref = _m( _pd( pos, w, "FP64", False, mi, ma, "plain" if plain else tree ) )
    for kernel in ( "FP64", "FP32" ):
        pd = _pd( pos, w, kernel, True, mi, ma, tree )
        assert pd._card_variant() is not None, ( label, "the dedicated path did not take the call" )
        m = _m( pd )
        assert m.dtype == numpy.float64
        assert not numpy.isnan( m ).any(), label
        # the empty cells are empty on both sides
        assert ( ( m > 0 ) == ( ref > 0 ) ).mean() > 0.999, ( label, kernel )
        r = _rel( m, ref )
        vol = float( numpy.prod( numpy.asarray( ma, float ) - numpy.asarray( mi, float ) ) )
        if kernel == "FP64":
            assert r.max() < tol64, ( label, kernel, r.max() )
            assert abs( m.sum() - vol ) < 1e-10 * vol, ( label, kernel, m.sum() )
        else:
            assert numpy.median( r ) < tol32_med, ( label, kernel, numpy.median( r ) )
            assert r.max() < tol32_max, ( label, kernel, r.max() )
            assert abs( m.sum() - vol ) < 1e-6 * vol, ( label, kernel, m.sum() )
        print( f"  { label } { kernel }: rel. gap to the generic double median { numpy.median( r ) :.1e}, max { r.max() :.1e}" )
    return ref


def _ring( k, rng, n_back = 2000 ):
    """a seed in the middle of a ring of `k` seeds ( its cell is a k-gon ), plus a uniform background"""
    a = 2 * numpy.pi * ( numpy.arange( k ) + rng.uniform( -0.1, 0.1, k ) ) / k
    ring = 0.5 + 0.2 * numpy.stack( [ numpy.cos( a ), numpy.sin( a ) ], axis = 1 )
    back = rng.uniform( 0.001, 0.999, size = ( n_back, 2 ) )
    back = back[ numpy.linalg.norm( back - 0.5, axis = 1 ) > 0.3 ]
    return numpy.concatenate( [ [ [ 0.5, 0.5 ] ], ring, back ] )


def _lines( n, rng ):
    """the clustered cloud of the campaign's `lines` cases: five lines, a 0.005 normal spread ( thin, long cells )"""
    ends = rng.uniform( 0.05, 0.95, size = ( 5, 2, 2 ) )
    which = rng.integers( 0, 5, n )
    t = rng.uniform( 0, 1, n )[ :, None ]
    pos = ends[ which, 0 ] + t * ( ends[ which, 1 ] - ends[ which, 0 ] ) + rng.normal( 0, 0.005, size = ( n, 2 ) )
    return numpy.clip( pos, 0.001, 0.999 )


def _clouds( rng ):
    """`( label, positions, weights )`: uniform, clustered ( lines ), weighted ( lines with weights of the order of
    their cells, and a smooth potential: the `lines_equal` regime )"""
    n = 6000
    uni = rng.uniform( 0.001, 0.999, size = ( n, 2 ) )
    lin = _lines( n, rng )
    h2 = 1.0 / n
    return [ ( "uniform", uni, None ),
             ( "lines", lin, None ),
             ( "lines weighted", lin, rng.uniform( -0.5, 0.5, n ) * h2 + 0.05 * numpy.sin( 4 * lin[ :, 0 ] ) ) ]


# ---- the measures ------------------------------------------------------------------------------------------------

if test( "the_card_cells_are_the_generic_cells_voronoi" ):
    if not _GPU:
        skip( _NO_GPU )
    else:
        rng = numpy.random.default_rng( 0 )
        _check( rng.uniform( 0.001, 0.999, size = ( 20000, 2 ) ), label = "uniform" )
        # a clustered cloud ( thin, long cells ), and a box that is not the unit square
        t = rng.uniform( 0, 1, 5000 )
        pos = numpy.stack( [ t, 0.5 + 0.2 * t + rng.normal( 0, 0.005, t.size ) ], axis = 1 )
        _check( numpy.clip( pos, 0.001, 0.999 ), label = "line" )
        _check( rng.uniform( [ -3, 2 ], [ 5, 2.5 ], size = ( 3000, 2 ) ), mi = ( -3, 2 ), ma = ( 5, 2.5 ), label = "off-centre box" )

if test( "the_card_cells_are_the_generic_cells_with_weights" ):
    if not _GPU:
        skip( _NO_GPU )
    else:
        rng = numpy.random.default_rng( 1 )
        n = 20000
        pos = rng.uniform( 0.001, 0.999, size = ( n, 2 ) )
        # weights of the order of a cell ( a Laguerre diagram ), then LARGE in the absolute ( a smooth potential:
        # what the weight difference carried exactly is for ), some cells empty
        _check( pos, rng.uniform( -0.3, 0.3, n ) / n, label = "w ~ h^2" )
        _check( pos, 0.13 * numpy.sin( 3 * pos[ :, 0 ] ) + rng.uniform( -1, 1, n ) / n, label = "w ~ 0.13" )
        _check( pos, rng.uniform( -3, 3, n ) / n, label = "w ~ 3 h^2 ( empty cells )" )

if test( "the_card_cells_overflow_into_the_later_passes" ):
    if not _GPU:
        skip( _NO_GPU )
    else:
        # k > 8: second pass, k > 16: third pass ( shared memory ), k > 256 ( double ) / 384 ( float ): the fourth one
        # ( global memory, slots of `card_max_vertices` vertices )
        rng = numpy.random.default_rng( 2 )
        for k in ( 12, 40, 100, 300 ):
            pos = _ring( k, rng )
            ref = _check( pos, label = f"ring of { k }", plain = True )
            assert abs( ref[ 0 ] - k * 0.1 ** 2 * numpy.tan( numpy.pi / k ) ) < 0.2 * ref[ 0 ]   # ~ the k-gon of apothem 0.1
        # and weighted: the central seed heavier, its k-gon wider
        w = numpy.zeros( len( pos ) )
        w[ 0 ] = 0.03
        _check( pos, w, label = "heavy centre", plain = True )

if test( "the_card_cells_small_and_degenerate_clouds" ):
    if not _GPU:
        skip( _NO_GPU )
    else:
        rng = numpy.random.default_rng( 3 )
        # one seed ( the root is a leaf ), fewer seeds than a leaf, a grid ( cocircular vertices everywhere )
        for pos in ( numpy.array( [ [ 0.3, 0.6 ] ] ), rng.uniform( 0, 1, size = ( 7, 2 ) ),
                     numpy.stack( numpy.meshgrid( ( numpy.arange( 40 ) + 0.5 ) / 40, ( numpy.arange( 30 ) + 0.5 ) / 30 ), axis = -1 ).reshape( -1, 2 ) ):
            _check( pos, label = f"n = { len( pos ) }" )
        # a seed outside the domain, and duplicated seeds with different weights ( the lighter one is empty )
        pos = rng.uniform( 0.1, 0.9, size = ( 500, 2 ) )
        pos[ 0 ] = [ 1.5, 0.5 ]
        pos = numpy.concatenate( [ pos, pos[ 1:20 ] ] )
        w = numpy.zeros( len( pos ) )
        w[ 500: ] = -1e-4
        _check( pos, w, label = "outside + duplicates" )

if test( "the_card_cells_follow_the_weights" ):
    if not _GPU:
        skip( _NO_GPU )
    else:
        # `pd.weights = ...` refreshes the majorants of the tree; the card rebuilds its nodes at every call
        rng = numpy.random.default_rng( 4 )
        n = 5000
        pos = rng.uniform( 0, 1, size = ( n, 2 ) )
        pd = _pd( pos, numpy.zeros( n ), "FP32" )
        gen = _pd( pos, numpy.zeros( n ), "FP64", False )
        for s in range( 3 ):
            w = rng.uniform( -1, 1, n ) * ( s + 1 ) / n
            pd.weights = w
            gen.weights = w
            r = _rel( _m( pd ), _m( gen ) )
            assert numpy.median( r ) < 1e-9 and r.max() < 1e-4, ( s, numpy.median( r ), r.max() )

if test( "the_card_takes_a_constant_density" ):
    if not _GPU:
        skip( _NO_GPU )
    else:
        # an `Image` whose values are all equal on exactly the box is a constant density: the card takes it ( `rho`,
        # read on the card ), and integrates the same thing as the generic path
        from sdot import Image
        rng = numpy.random.default_rng( 5 )
        pos = rng.uniform( 0.001, 0.999, size = ( 3000, 2 ) )
        img = Image( numpy.full( ( 2, 3 ), 2.5 ) )
        pos_img = pos * numpy.array( [ 2.0, 3.0 ] )   # the image's frame: pixel units, the box [ 0, 2 ] x [ 0, 3 ]
        pd = PowerDiagram( pos_img, distribution = img, kernel_dtype = "FP64" )
        gen = PowerDiagram( pos_img, distribution = img, kernel_dtype = "FP64" )
        gen.use_card_cells = False
        assert pd._card_variant() is not None and abs( pd._card_variant()[ 1 ] - 1 / 6 ) < 1e-15, pd._card_variant()
        r = _rel( _m( pd ), _m( gen ) )
        assert r.max() < 1e-9 and abs( _m( pd ).sum() - 1 ) < 1e-12, ( r.max(), _m( pd ).sum() )


# ---- the facets: Newton's Hessian ----------------------------------------------------------------------------------

def _generic_laplacian( gen, tree, n ):
    """the generic path's facets ( `hessian_rows`, user ids ) as `{ ( rank, rank ): c }`"""
    cnt, ids, vals = gen.hessian_rows()
    rank = tree.rank_of_seeds()
    res = {}
    for i in range( n ):
        for r in range( cnt[ i ] ):
            j = ids[ i, r ]
            if j >= 0:
                key = ( int( rank[ i ] ), int( rank[ j ] ) )
                res[ key ] = res.get( key, 0.0 ) + float( vals[ i, r ] )
    return res


def _card_laplacian( out, n ):
    row = numpy.asarray( out[ "row" ].raw ).reshape( -1 ).astype( numpy.int64 )
    col = numpy.asarray( out[ "col" ].raw ).reshape( -1 ).astype( numpy.int64 )
    val = numpy.asarray( out[ "val" ].raw ).reshape( -1 )
    dia = numpy.asarray( out[ "dia" ].raw ).reshape( -1 )
    assert row[ 0 ] == 0 and row[ n ] == out[ "nnz" ], ( row[ n ], out[ "nnz" ] )
    return row, col[ :row[ n ] ], val[ :row[ n ] ], dia


if test( "the_card_laplacian_is_the_generic_one" ):
    if not _GPU:
        skip( _NO_GPU )
    else:
        rng = numpy.random.default_rng( 10 )
        for label, pos, w in _clouds( rng ) + [ ( "ring of 300", _ring( 300, rng ), None ) ]:
            n = len( pos )
            tree = AaBsp( pos, w )
            gen = _pd( pos, w, "FP64", False, tree = tree )
            ref = _generic_laplacian( gen, tree, n )
            ref_m = _m( gen )
            # the generic GPU path has NaN cells on the rings ( a defect of its own ): their rows are not a reference
            broken = set( int( r ) for r in tree.rank_of_seeds()[ numpy.isnan( ref_m ) ] )
            ref = { k: v for k, v in ref.items() if numpy.isfinite( v ) and not ( set( k ) & broken ) }
            for kernel in ( "FP64", "FP32" ):
                pd = _pd( pos, w, kernel, True, tree = tree )
                out = pd._card_cells( facets = True, moments = False )
                assert out is not None, label
                assert _rel( numpy.asarray( out[ "measures" ].raw ).reshape( -1 ), ref_m ).max() < ( 1e-9 if kernel == "FP64" else 1e-6 )
                row, col, val, dia = _card_laplacian( out, n )
                # sorted columns, no duplicate, no self loop
                for i in range( n ):
                    c = col[ row[ i ]:row[ i + 1 ] ]
                    assert ( numpy.diff( c ) > 0 ).all() and not ( c == i ).any(), ( label, kernel, "row", i, c )
                got = { ( i, int( col[ p ] ) ): float( val[ p ] ) for i in range( n ) for p in range( row[ i ], row[ i + 1 ] ) }
                cmp = { k: v for k, v in got.items() if not ( set( k ) & broken ) }
                # SYMMETRIC TO THE BIT, and each row sums to its diagonal ( `L 1 = 0` ) in its own order
                assert all( got[ ( j, i ) ] == v for ( i, j ), v in got.items() ), ( label, kernel, "not symmetric" )
                sums = numpy.zeros( n )
                for i in range( n ):                         # in the row's order, one addition after the other
                    for p in range( row[ i ], row[ i + 1 ] ):
                        sums[ i ] += val[ p ]
                bad = numpy.where( ~ ( ( sums == dia ) | ( ( sums == 0 ) & ( dia == 1 ) ) ) )[ 0 ]
                assert not len( bad ), ( label, kernel, "L 1 != 0", bad[ :5 ], sums[ bad[ :5 ] ], dia[ bad[ :5 ] ] )
                # the same graph as the generic path's ( a sliver seen from one side only may differ in float ), the same
                # values to the kernel's precision
                missing, extra = set( ref ) - set( cmp ), set( cmp ) - set( ref )
                scale = numpy.median( list( ref.values() ) )
                small = [ max( ref.get( k, 0 ), got.get( k, 0 ) ) / scale for k in missing | extra ]
                if kernel == "FP64":
                    assert not missing and not extra, ( label, kernel, "graph", len( missing ), len( extra ), small[ :5 ], sorted( missing | extra )[ :5 ] )
                else:
                    assert len( missing | extra ) <= 2e-4 * len( ref ) and all( s < 1e-3 for s in small ), ( label, kernel, len( missing ), len( extra ), small[ :5 ] )
                common = set( ref ) & set( cmp )
                err = numpy.array( [ abs( got[ k ] - ref[ k ] ) / max( ref[ k ], 1e-3 * scale ) for k in common ] )
                tol_max = 1e-8 if kernel == "FP64" else 1e-4
                assert err.max() < tol_max and numpy.median( err ) < ( 1e-12 if kernel == "FP64" else 1e-9 ), ( label, kernel, "values", err.max(), numpy.median( err ) )
                print( f"  { label } { kernel }: { len( got ) } entries ( generic { len( ref ) } compared, { len( broken ) } NaN rows, { len( missing ) } missing, "
                       f"{ len( extra ) } extra ), rel. gap median { numpy.median( err ) :.1e} max { err.max() :.1e}" )

if test( "the_card_moments_are_the_generic_ones" ):
    if not _GPU:
        skip( _NO_GPU )
    else:
        rng = numpy.random.default_rng( 11 )
        for label, pos, w in _clouds( rng ):
            tree = AaBsp( pos, w )
            gen = _pd( pos, w, "FP64", False, tree = tree )
            mg, fg, sg = ( numpy.asarray( t ) for t in gen.moments )
            mg = mg.reshape( -1 )
            full = mg > 0
            bary_ref = numpy.where( full[ :, None ], fg / numpy.where( full, mg, 1 )[ :, None ], pos )
            cost_ref = sg.reshape( -1 ) - 2 * ( pos * fg ).sum( 1 ) + ( pos * pos ).sum( 1 ) * mg
            for kernel in ( "FP64", "FP32" ):
                out = _pd( pos, w, kernel, True, tree = tree )._card_cells( facets = False, moments = True )
                bary = numpy.asarray( out[ "bary" ].raw ).reshape( -1, 2 )
                cost = numpy.asarray( out[ "cost" ].raw ).reshape( -1 )
                h = 1 / numpy.sqrt( len( pos ) )
                eb = numpy.abs( bary - bary_ref ).max() / h
                ec = numpy.abs( cost - cost_ref ).max() / numpy.abs( cost_ref ).max()
                assert eb < ( 1e-8 if kernel == "FP64" else 1e-5 ) and ec < ( 1e-8 if kernel == "FP64" else 1e-5 ), ( label, kernel, eb, ec )
                print( f"  { label } { kernel }: barycentres { eb :.1e} h, costs { ec :.1e}" )


# ---- the adjoint, the trace ----------------------------------------------------------------------------------------

def _measures_fn( pos, w, tree, kernel, card, wrt ):
    def f( x ):
        p, q = ( x, w ) if wrt == "positions" else ( pos, x )
        pd = PowerDiagram( p, weights = q, boundaries = box_half_spaces( ( 0, 0 ), ( 1, 1 ) ), kernel_dtype = kernel, accelerator = tree,
                           scratch_capacity = None if card else 1024 )   # the generic path cannot grow its scratch under a trace
        pd.use_card_cells = card
        if card:
            assert pd._card_variant() is not None
        return pd.measures.value
    return f


if test( "the_card_adjoint_is_the_generic_one" ):
    if not _GPU:
        skip( _NO_GPU )
    else:
        # the adjoint of the card ( the facets, gathered per cell ) against the generic one ( a small system per vertex,
        # scattered ), positions and weights, on the three clouds and both kernels
        rng = numpy.random.default_rng( 12 )
        for label, pos, w in _clouds( rng ) + [ ( "ring of 300", _ring( 300, rng ), numpy.zeros( 0 ) ) ]:
            n = len( pos )
            w = numpy.zeros( n ) if w is None or len( w ) == 0 else w
            tree = AaBsp( pos, w )
            g = rng.normal( size = n )
            for wrt, x in ( ( "weights", w ), ( "positions", pos ) ):
                _, pb = driver.vjp( _measures_fn( pos, w, tree, "FP64", False, wrt ), x )
                ref = numpy.asarray( pb( g )[ 0 ] )
                for kernel in ( "FP64", "FP32" ):
                    _, pb = driver.vjp( _measures_fn( pos, w, tree, kernel, True, wrt ), x )
                    got = numpy.asarray( pb( g )[ 0 ] )
                    err = numpy.abs( got - ref ).max() / numpy.abs( ref ).max()
                    assert err < ( 1e-9 if kernel == "FP64" else 1e-5 ), ( label, wrt, kernel, err )
                    print( f"  { label } d/d{ wrt } { kernel }: max gap { err :.1e} of the largest entry" )

if test( "the_card_adjoint_is_the_finite_difference" ):
    if not _GPU:
        skip( _NO_GPU )
    else:
        from loom.testing import check_grad
        rng = numpy.random.default_rng( 13 )
        n = 40
        pos = rng.uniform( 0.1, 0.9, size = ( n, 2 ) )
        w = rng.uniform( -0.02, 0.02, n )
        tree = AaBsp( pos, w, max_seeds_per_leaf = 3 )

        def f( p, q ):
            pd = PowerDiagram( p, weights = q, boundaries = box_half_spaces( ( 0, 0 ), ( 1, 1 ) ), kernel_dtype = "FP64", accelerator = tree )
            assert pd._card_variant() is not None
            return pd.measures
        check_grad( f, pos, w, seed = 13 )

if test( "the_card_runs_under_jit" ):
    if not _GPU:
        skip( _NO_GPU )
    else:
        # the forward and its adjoint, traced: the same numbers as eager ( an ffi call like the others, no bypass )
        import jax
        rng = numpy.random.default_rng( 14 )
        n = 5000
        pos = rng.uniform( 0.001, 0.999, size = ( n, 2 ) )
        w = rng.uniform( -0.3, 0.3, n ) / n
        tree = AaBsp( pos, w )
        g = rng.normal( size = n )
        for kernel in ( "FP64", "FP32" ):
            f = _measures_fn( pos, w, tree, kernel, True, "weights" )
            eager = numpy.asarray( f( w ) )
            jitted = numpy.asarray( jax.jit( f )( w ) )
            assert numpy.abs( jitted - eager ).max() < 1e-15, ( kernel, numpy.abs( jitted - eager ).max() )
            loss = lambda x: ( f( x ) * g ).sum()
            ge = numpy.asarray( jax.grad( loss )( w ) )
            gj = numpy.asarray( jax.jit( jax.grad( loss ) )( w ) )
            assert numpy.abs( gj - ge ).max() <= 1e-12 * numpy.abs( ge ).max(), kernel
            # the positions too, traced
            fp = _measures_fn( pos, w, tree, kernel, True, "positions" )
            gp = numpy.asarray( jax.jit( jax.grad( lambda x: ( fp( x ) * g ).sum() ) )( pos ) )
            assert numpy.isfinite( gp ).all() and numpy.abs( gp ).max() > 0, kernel


# ---- capacities, limits, variants ----------------------------------------------------------------------------------

if test( "the_card_fourth_pass_works_in_batches" ):
    if not _GPU:
        skip( _NO_GPU )
    else:
        # MANY cells past the shared-memory pass ( 24 centres of rings of 400: 400-gons, over the 384 / 256 vertices of the
        # third pass ), through FOUR slots of the fourth pass: six batches in one launch, no capacity, no second run. The
        # same cells as the generic path, and the same bits as with 64 slots ( one batch ), for the measures and the adjoint.
        import sys
        bsp = sys.modules[ "sdot.PowerDiagram_Bsp" ]           # the module ( `sdot.PowerDiagram_Bsp` is also the class )
        rng = numpy.random.default_rng( 14 )
        k, r = 400, 0.04
        centres = numpy.stack( numpy.meshgrid( ( numpy.arange( 6 ) + 0.5 ) / 6, ( numpy.arange( 4 ) + 0.5 ) / 4 ), axis = -1 ).reshape( -1, 2 )
        rings = []
        for c in centres:
            a = 2 * numpy.pi * ( numpy.arange( k ) + rng.uniform( -0.1, 0.1, k ) ) / k
            rings.append( c + r * numpy.stack( [ numpy.cos( a ), numpy.sin( a ) ], axis = 1 ) )
        pos = numpy.concatenate( [ centres ] + rings )
        n = len( pos )
        w = rng.uniform( -0.1, 0.1, n ) * r * r / k
        w[ :len( centres ) ] = 0
        tree = AaBsp( pos, w )
        g = rng.normal( size = n )
        assert bsp.card_overflow_warps_for( "sdot::gpu2d::Variant<double, int, int, 32>", n, 1 << 15, 4 ) == 4
        original = bsp.PowerDiagram_Bsp.card_overflow_warps
        try:
            got = {}
            for warps in ( 4, 64 ):
                bsp.PowerDiagram_Bsp.card_overflow_warps = warps
                if warps == 4:
                    ref = _check( pos, w, label = "24 rings of 400, 4 slots", plain = True )
                    apothem = 0.5 * r                    # ( the centres' weights are zero, the ring's tiny )
                    assert numpy.abs( ref[ :len( centres ) ] / ( k * apothem ** 2 * numpy.tan( numpy.pi / k ) ) - 1 ).max() < 0.05
                for kernel in ( "FP64", "FP32" ):
                    m = _m( _pd( pos, w, kernel, True, tree = tree ) )
                    _, pb = driver.vjp( _measures_fn( pos, w, tree, kernel, True, "weights" ), w )
                    got[ warps, kernel ] = ( m, numpy.asarray( pb( g )[ 0 ] ) )
        finally:
            bsp.PowerDiagram_Bsp.card_overflow_warps = original
        for kernel in ( "FP64", "FP32" ):
            for a, b in zip( got[ 4, kernel ], got[ 64, kernel ] ):
                assert numpy.isfinite( a ).all() and numpy.array_equal( a, b ), kernel
            assert numpy.abs( got[ 4, kernel ][ 1 ] ).max() > 0, kernel

if test( "the_card_raises_past_the_vertex_limit" ):
    if not _GPU:
        skip( _NO_GPU )
    else:
        from loom.drivers.CallArg_Errors import KernelFailure
        import jax
        rng = numpy.random.default_rng( 15 )
        pos = _ring( 3000, rng )
        tree = AaBsp( pos, None )
        ref = _m( _pd( pos, None, "FP64", False, tree = tree ) )
        for kernel in ( "FP64", "FP32" ):
            # a 3000-gon under the default limit: through the fourth pass, at once
            pd = _pd( pos, None, kernel, True, tree = tree )
            m = _m( pd )
            r = _rel( m, ref )
            assert not numpy.isnan( m ).any() and r.max() < ( 1e-9 if kernel == "FP64" else 1e-6 ), ( kernel, r.max() )
            assert abs( m[ 0 ] - 3000 * 0.01 * numpy.tan( numpy.pi / 3000 ) ) < 1e-3 * m[ 0 ]
            # past the hard limit: an error that says so and names the seed ( and never a NaN )
            pd = _pd( pos, None, kernel, True, tree = tree )
            pd.card_max_vertices = 1024
            try:
                pd.measures.value
                raise AssertionError( "a cell past the vertex limit went through" )
            except KernelFailure as e:
                assert "more than 1024 vertices" in str( e ) and "seed 0" in str( e ), str( e )
            # ... under a trace too, and under a derivative ( a tree of its own: setting traced weights leaves tracers in
            # the tree's majorants )
            pd = _pd( pos, numpy.zeros( len( pos ) ), kernel, True, tree = AaBsp( pos, numpy.zeros( len( pos ) ) ) )
            pd.card_max_vertices = 1024

            def f( q ):
                pd.weights = q
                return pd.measures.value
            for fn in ( jax.jit( f ), jax.jit( jax.grad( lambda q: ( f( q ) * q ).sum() ) ) ):
                try:
                    numpy.asarray( fn( numpy.zeros( len( pos ) ) ) )
                    raise AssertionError( "a cell past the vertex limit went through under jit" )
                except AssertionError:
                    raise
                except Exception as e:
                    assert "more than 1024 vertices" in str( e ) and "seed 0" in str( e ), str( e )

if test( "the_card_variant_follows_the_inputs" ):
    # pure Python: the variant chosen from the seeds and the tree, synthetic sizes past 32-bit indices and depth 32
    assert card_variant_for( 32, 10 ** 6, 2 ** 18 - 1 ) == "sdot::gpu2d::Variant<float, int, int, 32>"
    assert card_variant_for( 64, 10 ** 6, 2 ** 18 - 1 ) == "sdot::gpu2d::Variant<double, int, int, 32>"
    assert card_variant_for( 32, 2 ** 31 - 9, 2 ** 31 - 1 ) == "sdot::gpu2d::Variant<float, int, int, 32>"     # depth 31
    assert card_variant_for( 32, 2 ** 31, 2 ** 31 - 1 ) == "sdot::gpu2d::Variant<float, long long, int, 32>"   # ranks past int
    assert card_variant_for( 32, 10 ** 10, 2 ** 32 - 1 ) == "sdot::gpu2d::Variant<float, long long, long long, 32>"   # depth 32
    assert card_variant_for( 64, 10 ** 11, 2 ** 40 - 1 ) == "sdot::gpu2d::Variant<double, long long, long long, 64>"  # depth 40
    # the depth `AaBsp` gives is what sizes the stack: 28 levels ( past the old packed limit of 27 ) at 6.7e8 seeds
    assert AaBsp.max_depth_for( 7 * 10 ** 8 ) == 28 and card_variant_for( 32, 7 * 10 ** 8, AaBsp.max_nb_nodes_for( 7 * 10 ** 8 ) ).endswith( "int, int, 32>" )
    try:
        card_variant_for( 32, 10, 2 ** 65 - 1 )
        raise AssertionError( "a tree deeper than 64 levels" )
    except ValueError:
        pass

if test( "the_wide_card_variant_computes_the_same_cells" ):
    if not _GPU:
        skip( _NO_GPU )
    else:
        # the 64-bit ranks and node indices and the 64-slot stack, forced on a small cloud ( a tree deep enough to need
        # them does not fit on the card ): the same cells, the same laplacian, the same adjoint as the narrow variant
        import sys
        bsp = sys.modules[ "sdot.PowerDiagram_Bsp" ]           # the module ( `sdot.PowerDiagram_Bsp` is also the class )
        rng = numpy.random.default_rng( 16 )
        pos = _ring( 100, rng, n_back = 6000 )
        w = rng.uniform( -0.3, 0.3, len( pos ) ) / len( pos )
        tree = AaBsp( pos, w, max_seeds_per_leaf = 2 )
        narrow = _pd( pos, w, "FP32", True, tree = tree )._card_cells( facets = True )
        g = rng.normal( size = len( pos ) )
        _, pb = driver.vjp( _measures_fn( pos, w, tree, "FP32", True, "weights" ), w )
        gn = numpy.asarray( pb( g )[ 0 ] )
        original = bsp.card_variant_for
        try:
            bsp.card_variant_for = lambda fp, n, nodes: f"sdot::gpu2d::Variant<{ 'float' if fp == 32 else 'double' }, long long, long long, 64>"
            wide = _pd( pos, w, "FP32", True, tree = tree )._card_cells( facets = True )
            _, pb = driver.vjp( _measures_fn( pos, w, tree, "FP32", True, "weights" ), w )
            gw = numpy.asarray( pb( g )[ 0 ] )
        finally:
            bsp.card_variant_for = original
        for key in ( "measures", "row", "dia" ):
            assert numpy.array_equal( numpy.asarray( narrow[ key ].raw ), numpy.asarray( wide[ key ].raw ) ), key
        nnz = narrow[ "nnz" ]
        assert nnz == wide[ "nnz" ]
        for key in ( "col", "val" ):
            assert numpy.array_equal( numpy.asarray( narrow[ key ].raw )[ :nnz ], numpy.asarray( wide[ key ].raw )[ :nnz ] ), key
        assert numpy.array_equal( gn, gw )
