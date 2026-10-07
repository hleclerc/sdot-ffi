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

THE 3D CELLS ( `include/sdot/gpu/Cell3D.cuh`, the last section ): a warp per cell, the same outputs, the same four
properties, checked the same way against the generic double path ( or the plain storage where the cell is large ).
"""

from errand import test, skip


_NO_GPU = "the dedicated cell kernels only exist on a CUDA device"


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
    storage ( every seed cuts every cell: exact, and blind to the tree; in 3D the generic BSP path still gets a few cells
    of the spheres wrong -- in 2D its rings are right since its cut takes the run of the farthest vertex )"""
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
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
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
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        rng = numpy.random.default_rng( 1 )
        n = 20000
        pos = rng.uniform( 0.001, 0.999, size = ( n, 2 ) )
        # weights of the order of a cell ( a Laguerre diagram ), then LARGE in the absolute ( a smooth potential:
        # what the weight difference carried exactly is for ), some cells empty
        # ( `tol64`: the generic path's own gap to the exact plain storage is 2e-9 on the almost empty cells of this cloud,
        # and it moves with the majorants -- they change the order of the cuts, hence their rounding: 0.98e-9 with the
        # host's per-node majorants, 1.1e-9 with the build's )
        _check( pos, rng.uniform( -0.3, 0.3, n ) / n, label = "w ~ h^2", tol64 = 3e-9 )
        _check( pos, 0.13 * numpy.sin( 3 * pos[ :, 0 ] ) + rng.uniform( -1, 1, n ) / n, label = "w ~ 0.13" )
        _check( pos, rng.uniform( -3, 3, n ) / n, label = "w ~ 3 h^2 ( empty cells )" )

if test( "the_card_cells_overflow_into_the_later_passes" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        # k > 8: second pass, k > 16: third pass ( shared memory ), k > 256 ( double ) / 384 ( float ): the fourth one
        # ( global memory, slots of `card_max_vertices` vertices )
        rng = numpy.random.default_rng( 2 )
        for k in ( 12, 40, 100, 300 ):
            pos = _ring( k, rng )
            ref = _check( pos, label = f"ring of { k }" )
            assert abs( ref[ 0 ] - k * 0.1 ** 2 * numpy.tan( numpy.pi / k ) ) < 0.2 * ref[ 0 ]   # ~ the k-gon of apothem 0.1
        # and weighted: the central seed heavier, its k-gon wider
        w = numpy.zeros( len( pos ) )
        w[ 0 ] = 0.03
        _check( pos, w, label = "heavy centre" )

if test( "the_card_cells_small_and_degenerate_clouds" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
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
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
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
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
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
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
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
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
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
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
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
                _, pb = loom.vjp( _measures_fn( pos, w, tree, "FP64", False, wrt ), x )
                ref = numpy.asarray( pb( g )[ 0 ] )
                for kernel in ( "FP64", "FP32" ):
                    _, pb = loom.vjp( _measures_fn( pos, w, tree, kernel, True, wrt ), x )
                    got = numpy.asarray( pb( g )[ 0 ] )
                    err = numpy.abs( got - ref ).max() / numpy.abs( ref ).max()
                    assert err < ( 1e-9 if kernel == "FP64" else 1e-5 ), ( label, wrt, kernel, err )
                    print( f"  { label } d/d{ wrt } { kernel }: max gap { err :.1e} of the largest entry" )

if test( "the_card_adjoint_is_the_finite_difference" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
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
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
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


# ---- the tree, built on the card ( `gpu/Bsp2D.cuh` ) -------------------------------------------------------------
#
# The card's tree is the host's ( the per-level kernel `bsp_build_level.h`, with numpy for the top levels on a card ) as
# soon as the coordinates along every cut are distinct: the same slices, boxes, cuts, the same seeds in each leaf ( their
# order inside a leaf differs ). On ties at a median the two pick different halves of the tied seeds, so a tied cloud is
# checked on the RULE instead: tight boxes, the median cut on the first longest axis, the halves on either side of it.

def _tree( t ):
    """the tensors of a tree, on the host"""
    a = lambda x: numpy.asarray( x ).reshape( -1 )
    d = int( t.nb_dims.value )
    return dict( seed = a( t.seed_indices ).astype( numpy.int64 ), beg = a( t.node_begin ), end = a( t.node_end ),
                 left = a( t.node_left ), right = a( t.node_right ), box = numpy.asarray( t.node_box ).reshape( -1, 2, d ),
                 rank = a( t.rank_of_seeds() ).astype( numpy.int64 ) )


def _host_tree( pos, leaf = 10 ):
    """the host's tree ( `AaBsp._build_in_kernel`, what a card built before )"""
    from sdot.AaBsp import _build_in_kernel
    r = _build_in_kernel( pos, None, leaf )
    return dict( seed = r[ "seed_indices" ], beg = r[ "node_begin" ], end = r[ "node_end" ], left = r[ "node_left" ], right = r[ "node_right" ],
                 box = r[ "node_box" ] )


def _check_tree_rule( pos, t, leaf = 10 ):
    """the tree of `pos` by its rule ( see above ), whatever the side of the tied seeds"""
    from sdot import AaBsp as Bsp
    n = len( pos )
    depth = Bsp.max_depth_for( n, leaf )
    assert len( t[ "beg" ] ) == 2 ** depth - 1
    assert numpy.array_equal( numpy.sort( t[ "seed" ] ), numpy.arange( n ) ), "a permutation"
    assert numpy.array_equal( t[ "rank" ][ t[ "seed" ] ], numpy.arange( n ) ), "its inverse"
    p = pos[ t[ "seed" ] ]
    stack = [ ( 0, depth, 0, n ) ]                       # ( preorder index, height, the slice the parent gave )
    while stack:
        i, h, b, e = stack.pop()
        assert ( t[ "beg" ][ i ], t[ "end" ][ i ] ) == ( b, e ), ( i, h )
        if e > b:
            lo, hi = p[ b:e ].min( axis = 0 ), p[ b:e ].max( axis = 0 )
            assert numpy.array_equal( t[ "box" ][ i, 0 ], lo ) and numpy.array_equal( t[ "box" ][ i, 1 ], hi ), ( i, "the box" )
        if h == 1:
            assert t[ "left" ][ i ] == -1 and t[ "right" ][ i ] == -1
            continue
        assert t[ "left" ][ i ] == i + 1 and t[ "right" ][ i ] == i + 2 ** ( h - 1 ), i
        m = e
        if e > b:
            ax = int( numpy.argmax( hi - lo ) )
            if e - b > leaf and hi[ ax ] > lo[ ax ]:
                m = b + ( e - b ) // 2
                assert p[ b:m, ax ].max() <= p[ m:e, ax ].min(), ( i, "the median cut" )
        stack += [ ( i + 1, h - 1, b, m ), ( i + 2 ** ( h - 1 ), h - 1, m, e ) ]


def _same_tree( a, b ):
    """the same nodes, and the same seeds in each leaf"""
    for key in ( "beg", "end", "left", "right" ):
        assert numpy.array_equal( a[ key ], b[ key ] ), key
    assert numpy.array_equal( a[ "box" ], b[ "box" ] ), "the boxes"
    for i in numpy.nonzero( ( a[ "left" ] < 0 ) & ( a[ "end" ] > a[ "beg" ] ) )[ 0 ]:
        sl = slice( a[ "beg" ][ i ], a[ "end" ][ i ] )
        assert numpy.array_equal( numpy.sort( a[ "seed" ][ sl ] ), numpy.sort( b[ "seed" ][ sl ] ) ), ( i, "a leaf's seeds" )


def _tied_clouds( rng ):
    """clouds with ties at the medians: a grid ( equal coordinates ), the grid's points repeated ( equal seeds ), a vertical
    line, a heap of one point plus a few others ( a node that cannot be cut, propagated ), lines on a coarse lattice"""
    g = numpy.stack( numpy.meshgrid( numpy.arange( 60 ), numpy.arange( 40 ) ), axis = -1 ).reshape( -1, 2 ) / 61.0 + 0.01
    line = numpy.stack( [ numpy.full( 3000, 0.5 ), rng.uniform( 0, 1, 3000 ) ], axis = 1 )
    heap = numpy.concatenate( [ numpy.full( ( 700, 2 ), 0.25 ), rng.uniform( 0, 1, size = ( 300, 2 ) ) ] )
    lat = numpy.round( _lines( 5000, rng ) * 64 ) / 64
    return [ ( "grid", g ), ( "grid repeated", numpy.concatenate( [ g, g, g ] ) ), ( "vertical line", line ), ( "heap", heap ),
             ( "lines on a lattice", lat ), ( "negative and zero", numpy.concatenate( [ g - 0.5, -g, numpy.zeros( ( 30, 2 ) ) ] ) ) ]


if test( "the_card_tree_is_the_host_tree" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        rng = numpy.random.default_rng( 20 )
        clouds = [ ( label, pos ) for label, pos, _ in _clouds( rng ) ] + [
            ( "uniform 30000", rng.uniform( 0, 1, size = ( 30000, 2 ) ) ),
            ( "all in the per-level kernel", rng.uniform( -2, 3, size = ( 500, 2 ) ) ),
            ( "clustered", numpy.concatenate( [ c + 1e-4 * rng.normal( size = ( 2000, 2 ) ) for c in rng.uniform( 0, 1, size = ( 6, 2 ) ) ] ) ) ]
        clouds += [ ( f"n = { n }", rng.uniform( 0, 1, size = ( n, 2 ) ) ) for n in ( 1, 2, 9, 10, 11, 21, 64, 1000 ) ]
        for label, pos in clouds:
            for leaf in ( 10, 3 ):
                card = _tree( AaBsp( pos, max_seeds_per_leaf = leaf ) )
                _check_tree_rule( pos, card, leaf )
                _same_tree( card, _host_tree( pos, leaf ) )
            print( f"  { label }: the host's tree" )

if test( "the_card_tree_follows_the_rule_on_ties" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        rng = numpy.random.default_rng( 21 )
        for label, pos in _tied_clouds( rng ):
            t = _tree( AaBsp( pos ) )
            _check_tree_rule( pos, t )
            # the same slice SIZES as the host's where no tie decides which nodes hold only equal seeds ( the heap: the same
            # propagated nodes; elsewhere the sizes follow from `n` alone )
            if label in ( "grid", "grid repeated", "vertical line", "heap" ):
                h = _host_tree( pos )
                assert numpy.array_equal( t[ "end" ] - t[ "beg" ], h[ "end" ] - h[ "beg" ] ), label
            # the same bits at every call
            again = _tree( AaBsp( pos ) )
            assert all( numpy.array_equal( t[ k ], again[ k ] ) for k in t ), label
            print( f"  { label }: the rule holds" )
        # the cells of a tied cloud ( the grid: equal coordinates, distinct seeds ) through the card's tree are the generic ones
        g = _tied_clouds( rng )[ 0 ][ 1 ]
        _check( g + 1e-3 * rng.uniform( 0, 1, size = g.shape ) * ( rng.uniform( size = ( len( g ), 1 ) ) < 0.5 ), label = "grid, half jittered" )
        _check( g, label = "grid" )

if test( "the_card_tree_majorants_bound_the_weights" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        # a tree built with weights: its majorants are `Majorant2D.cuh`'s, valid on every node, and those of the same
        # weights refreshed on a tree built without
        rng = numpy.random.default_rng( 22 )
        n = 8000
        pos = rng.uniform( 0.001, 0.999, size = ( n, 2 ) )
        for label, w in ( ( "smooth", 0.1 * numpy.sin( 5 * pos[ :, 0 ] ) + 0.05 * pos[ :, 1 ] ), ( "random", rng.uniform( -1, 1, n ) / n ),
                          ( "constant", numpy.full( n, 0.25 ) ) ):
            tree = AaBsp( pos, w )
            t = _tree( tree )
            wa, wb = numpy.asarray( tree.node_wa ).reshape( -1, 2 ), numpy.asarray( tree.node_wb ).reshape( -1 )
            p, q = pos[ t[ "seed" ] ], w[ t[ "seed" ] ]
            for i in range( len( t[ "beg" ] ) ):
                sl = slice( t[ "beg" ][ i ], t[ "end" ][ i ] )
                if sl.stop > sl.start:
                    assert ( q[ sl ] <= p[ sl ] @ wa[ i ] + wb[ i ] ).all(), ( label, i )
            bare = AaBsp( pos )
            bare.refresh_weight_majorants( p, q )
            assert numpy.array_equal( numpy.asarray( bare.node_wb ), numpy.asarray( tree.node_wb ) ), label
            print( f"  { label }: { int( ( numpy.abs( wa ).sum( axis = 1 ) > 0 ).sum() ) } affine nodes of { len( wb ) }" )

if test( "the_card_tree_is_built_under_jit_from_traced_positions" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        # the tree of TRACED positions, in the jitted program: the same bits as eager; then a diagram on it ( the gathers
        # through the traced order ), its measures and their adjoint with respect to the positions, eager == jit
        import jax
        rng = numpy.random.default_rng( 23 )
        n = 7000
        pos = _lines( n, rng )
        w = rng.uniform( -0.3, 0.3, n ) / n
        keys = ( "seed_indices", "node_begin", "node_end", "node_left", "node_right", "node_box" )

        def build( p, q ):
            t = AaBsp( p, q )
            return [ getattr( t, k ).raw for k in keys ] + [ t.rank_of_seeds(), t.node_wa.raw, t.node_wb.raw ]

        eager = [ numpy.asarray( x ) for x in build( pos, w ) ]
        jitted = [ numpy.asarray( x ) for x in jax.jit( build )( pos, w ) ]
        for k, a, b in zip( keys + ( "rank", "wa", "wb" ), eager, jitted ):
            assert numpy.array_equal( a, b ), k

        g = rng.normal( size = n )
        for kernel in ( "FP64", "FP32" ):
            def measures( p ):
                pd = PowerDiagram( p, weights = w, boundaries = box_half_spaces( ( 0, 0 ), ( 1, 1 ) ), kernel_dtype = kernel, accelerator = AaBsp( p, w ) )
                assert pd._card_variant() is not None
                return pd.measures.value
            me, mj = numpy.asarray( measures( pos ) ), numpy.asarray( jax.jit( measures )( pos ) )
            assert numpy.array_equal( me, mj ), kernel
            ref = _m( _pd( pos, w, "FP64", False, tree = AaBsp( pos, w ) ) )
            assert numpy.median( _rel( me, ref ) ) < 1e-9, kernel
            loss = lambda p: ( measures( p ) * g ).sum()
            ge, gj = numpy.asarray( jax.grad( loss )( pos ) ), numpy.asarray( jax.jit( jax.grad( loss ) )( pos ) )
            assert numpy.array_equal( ge, gj ) and numpy.abs( ge ).max() > 0, kernel


# ---- capacities, limits, variants ----------------------------------------------------------------------------------

if test( "the_card_fourth_pass_works_in_batches" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
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
                    ref = _check( pos, w, label = "24 rings of 400, 4 slots" )
                    apothem = 0.5 * r                    # ( the centres' weights are zero, the ring's tiny )
                    assert numpy.abs( ref[ :len( centres ) ] / ( k * apothem ** 2 * numpy.tan( numpy.pi / k ) ) - 1 ).max() < 0.05
                for kernel in ( "FP64", "FP32" ):
                    m = _m( _pd( pos, w, kernel, True, tree = tree ) )
                    _, pb = loom.vjp( _measures_fn( pos, w, tree, kernel, True, "weights" ), w )
                    got[ warps, kernel ] = ( m, numpy.asarray( pb( g )[ 0 ] ) )
        finally:
            bsp.PowerDiagram_Bsp.card_overflow_warps = original
        for kernel in ( "FP64", "FP32" ):
            for a, b in zip( got[ 4, kernel ], got[ 64, kernel ] ):
                assert numpy.isfinite( a ).all() and numpy.array_equal( a, b ), kernel
            assert numpy.abs( got[ 4, kernel ][ 1 ] ).max() > 0, kernel

if test( "the_card_raises_past_the_vertex_limit" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
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
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    assert card_variant_for( 32, 10 ** 6, 2 ** 18 - 1 ) == "sdot::gpu2d::Variant<float, int, int, 32>"
    assert card_variant_for( 32, 10 ** 6, 2 ** 18 - 1, 3 ) == "sdot::gpu3d::Variant<float, int, int, 32>"
    assert card_variant_for( 64, 10 ** 11, 2 ** 40 - 1, 3 ) == "sdot::gpu3d::Variant<double, long long, long long, 64>"
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
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
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
        _, pb = loom.vjp( _measures_fn( pos, w, tree, "FP32", True, "weights" ), w )
        gn = numpy.asarray( pb( g )[ 0 ] )
        original = bsp.card_variant_for
        try:
            bsp.card_variant_for = lambda fp, n, nodes, dim = 2: f"sdot::gpu2d::Variant<{ 'float' if fp == 32 else 'double' }, long long, long long, 64>"
            wide = _pd( pos, w, "FP32", True, tree = tree )._card_cells( facets = True )
            _, pb = loom.vjp( _measures_fn( pos, w, tree, "FP32", True, "weights" ), w )
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


# ---- THE 3D CELLS ( `include/sdot/gpu/Cell3D.cuh` ) ----------------------------------------------------------------------
#
# A warp per cell ( 64 vertices on the lanes' registers, then 128, then a slot of global memory ), the tree built on the card
# ( `Bsp2D.cuh::build_tree< 3 >`, `Majorant3D.cuh` ). The reference is the generic double path on the same tree, or the plain
# storage ( exact, every seed cuts every cell ) where a cell is large.

_BOX3 = ( ( 0, 0, 0 ), ( 1, 1, 1 ) )


def _check3( pos, w = None, mi = ( 0, 0, 0 ), ma = ( 1, 1, 1 ), tol64 = 1e-9, tol32_med = 1e-9, tol32_max = 1e-5, label = "", plain = False ):
    """`_check` in 3D: the card ( float and double kernels ) against the generic double path"""
    return _check( pos, w, mi, ma, tol64, tol32_med, tol32_max, label, plain )


def _planes3( n, rng, sigma = 0.02 ):
    """the campaign's `planes` cloud ( `bench/cases.py::planes_cloud` ): four planes through the cube, a normal spread"""
    out = []
    while len( out ) < 4:
        u = rng.normal( size = 3 )
        u /= numpy.linalg.norm( u )
        out.append( ( u, u @ numpy.full( 3, 0.5 ) + rng.uniform( -0.25, 0.25 ) ) )
    pts = []
    for u, c in out:
        a = numpy.array( [ 0.0, 1.0, 0.0 ] ) if abs( u[ 0 ] ) > 0.9 else numpy.array( [ 1.0, 0.0, 0.0 ] )
        v1 = numpy.cross( u, a )
        v1 /= numpy.linalg.norm( v1 )
        v2 = numpy.cross( u, v1 )
        st = rng.uniform( -1.2, 1.2, size = ( 8 * n, 2 ) )
        P = c * u + st[ :, :1 ] * v1 + st[ :, 1: ] * v2
        P = P[ numpy.all( ( P > 0 ) & ( P < 1 ), axis = 1 ) ][ :n // 4 + 1 ]
        pts.append( P + rng.normal( 0, sigma, size = ( len( P ), 1 ) ) * u )
    P = numpy.concatenate( pts )[ :n ]
    rng.shuffle( P )
    return numpy.clip( P, 0.001, 0.999 )


def _clouds3( rng, n = 8000 ):
    """`( label, positions, weights )`: uniform, clustered ( planes ), weighted ( planes with weights of the order of their
    cells and a smooth potential: the `planes_equal` regime )"""
    uni = rng.uniform( 0.001, 0.999, size = ( n, 3 ) )
    pla = _planes3( n, rng )
    h2 = n ** ( -2 / 3 )
    return [ ( "uniform 3d", uni, None ),
             ( "planes", pla, None ),
             ( "planes weighted", pla, rng.uniform( -0.3, 0.3, n ) * h2 + 0.05 * numpy.sin( 4 * pla[ :, 0 ] ) ) ]


def _sphere3( k, rng, n_back = 1500, r = 0.2 ):
    """a seed in the middle of a sphere of `k` seeds ( its cell has `k` faces, `2 k - 4` vertices ), plus a uniform background"""
    i = numpy.arange( k ) + 0.5
    phi = numpy.arccos( 1 - 2 * i / k )
    th = numpy.pi * ( 1 + 5 ** 0.5 ) * i + rng.uniform( -0.05, 0.05, k )
    sph = 0.5 + r * numpy.stack( [ numpy.cos( th ) * numpy.sin( phi ), numpy.sin( th ) * numpy.sin( phi ), numpy.cos( phi ) ], axis = 1 )
    back = rng.uniform( 0.001, 0.999, size = ( n_back, 3 ) )
    back = back[ numpy.linalg.norm( back - 0.5, axis = 1 ) > r + 0.1 ]
    return numpy.concatenate( [ [ [ 0.5, 0.5, 0.5 ] ], sph, back ] )


if test( "the_card_cells_3d_are_the_generic_cells_voronoi" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        rng = numpy.random.default_rng( 30 )
        _check3( rng.uniform( 0.001, 0.999, size = ( 20000, 3 ) ), label = "uniform 3d" )
        _check3( _planes3( 20000, rng ), label = "planes" )
        _check3( rng.uniform( [ -3, 2, 0 ], [ 5, 2.5, 1 ], size = ( 3000, 3 ) ), mi = ( -3, 2, 0 ), ma = ( 5, 2.5, 1 ), label = "off-centre box" )

if test( "the_card_cells_3d_are_the_generic_cells_with_weights" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        rng = numpy.random.default_rng( 31 )
        n = 20000
        h2 = n ** ( -2 / 3 )
        pos = rng.uniform( 0.001, 0.999, size = ( n, 3 ) )
        _check3( pos, rng.uniform( -0.3, 0.3, n ) * h2, label = "w ~ h^2" )
        _check3( pos, 0.13 * numpy.sin( 3 * pos[ :, 0 ] ) + rng.uniform( -1, 1, n ) * h2, label = "w ~ 0.13" )
        _check3( pos, rng.uniform( -3, 3, n ) * h2, label = "w ~ 3 h^2 ( empty cells )" )
        for label, p, w in _clouds3( rng )[ 2: ]:
            _check3( p, w, label = label )

if test( "the_card_cells_3d_overflow_into_the_later_passes" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        # k faces: 2 k - 4 vertices. k = 20 fits the first pass, 50 the second ( 128 vertices ), 120 and 400 the global one
        rng = numpy.random.default_rng( 32 )
        for k in ( 20, 50, 120, 400 ):
            pos = _sphere3( k, rng )
            ref = _check3( pos, label = f"sphere of { k }", plain = True )
            assert 4.1e-3 < ref[ 0 ] < 6.5e-3, ref[ 0 ]   # the polytope around the ball of radius 0.1 ( 4.19e-3 )
        w = numpy.zeros( len( pos ) )
        w[ 0 ] = 0.01
        _check3( pos, w, label = "heavy centre", plain = True )

if test( "the_card_cells_3d_small_and_degenerate_clouds" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        rng = numpy.random.default_rng( 33 )
        g = numpy.stack( numpy.meshgrid( ( numpy.arange( 12 ) + 0.5 ) / 12, ( numpy.arange( 10 ) + 0.5 ) / 10, ( numpy.arange( 8 ) + 0.5 ) / 8 ),
                         axis = -1 ).reshape( -1, 3 )
        for label, pos in ( ( "n = 1", numpy.array( [ [ 0.3, 0.6, 0.2 ] ] ) ), ( "n = 7", rng.uniform( 0, 1, size = ( 7, 3 ) ) ) ):
            _check3( pos, label = label )
        # GRIDS: eight cells at every vertex, cuts through vertices that are on them ( the ties of `Cell3D.cuh::widen` ). The
        # reference is the exact box ( the generic path and the plain storage get some of these cells wrong ), and on the
        # jittered grid the generic cells where the plain storage agrees with them
        g2 = numpy.stack( numpy.meshgrid( *[ ( numpy.arange( 20 ) + 0.5 ) / 20 ] * 3 ), axis = -1 ).reshape( -1, 3 )
        gj = g + 1e-3 * rng.uniform( 0, 1, size = g.shape ) * ( rng.uniform( size = ( len( g ), 1 ) ) < 0.5 )
        gen = _m( _pd( gj, None, "FP64", False, *_BOX3, tree = AaBsp( gj ) ) )
        plain = _m( _pd( gj, None, "FP64", False, *_BOX3, tree = "plain" ) )
        trusted = numpy.abs( gen - plain ) <= 1e-9 * plain
        assert trusted.mean() > 0.99
        for label, pos, ref, ok in ( ( "grid", g, numpy.full( len( g ), 1 / len( g ) ), None ), ( "grid 20^3", g2, numpy.full( len( g2 ), 1 / len( g2 ) ), None ),
                                     ( "grid, half jittered", gj, gen, trusted ) ):
            for leaf in ( 10, 2000 ):                    # ( with and without the pruning )
                tree = AaBsp( pos, max_seeds_per_leaf = leaf )
                for kernel in ( "FP64", "FP32" ):
                    m = _m( _pd( pos, None, kernel, True, *_BOX3, tree = tree ) )
                    r = numpy.abs( m - ref ) / ref
                    r = r if ok is None else r[ ok ]
                    # ( a jittered vertex near a tie is decided in float in the float kernel: a sliver, at its rounding )
                    tol = 1e-6 if ( kernel == "FP32" and ok is not None ) else 1e-10
                    assert r.max() < tol and abs( m.sum() - 1 ) < 1e-3 * tol, ( label, leaf, kernel, r.max(), m.sum() )
                    print( f"  { label } ( leaf { leaf } ) { kernel }: max rel. gap { r.max() :.1e}" )
        pos = rng.uniform( 0.1, 0.9, size = ( 500, 3 ) )
        pos[ 0 ] = [ 1.5, 0.5, 0.5 ]
        pos = numpy.concatenate( [ pos, pos[ 1:20 ] ] )
        w = numpy.zeros( len( pos ) )
        w[ 500: ] = -1e-4
        _check3( pos, w, label = "outside + duplicates" )

if test( "the_card_cells_3d_follow_the_weights" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        rng = numpy.random.default_rng( 34 )
        n = 5000
        h2 = n ** ( -2 / 3 )
        pos = rng.uniform( 0, 1, size = ( n, 3 ) )
        pd = _pd( pos, numpy.zeros( n ), "FP32", True, *_BOX3 )
        gen = _pd( pos, numpy.zeros( n ), "FP64", False, *_BOX3 )
        for s in range( 3 ):
            w = rng.uniform( -1, 1, n ) * ( s + 1 ) * h2
            pd.weights = w
            gen.weights = w
            r = _rel( _m( pd ), _m( gen ) )
            assert numpy.median( r ) < 1e-9 and r.max() < 1e-4, ( s, numpy.median( r ), r.max() )

if test( "the_card_laplacian_3d_is_the_generic_one" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        rng = numpy.random.default_rng( 35 )
        for label, pos, w in _clouds3( rng, 4000 ) + [ ( "sphere of 120", _sphere3( 120, rng ), None ) ]:
            n = len( pos )
            tree = AaBsp( pos, w )
            gen = _pd( pos, w, "FP64", False, *_BOX3, tree = tree )
            ref = _generic_laplacian( gen, tree, n )
            ref_m = _m( gen )
            # the generic path gets a few cells of the spheres wrong ( as of the rings in 2D ): their rows are not a reference
            plain = _m( _pd( pos, w, "FP64", False, *_BOX3, tree = "plain" ) ) if label.startswith( "sphere" ) else ref_m
            wrong = numpy.isnan( ref_m ) | ( numpy.abs( ref_m - plain ) > 1e-9 * numpy.abs( plain ) )
            broken = set( int( r ) for r in numpy.asarray( tree.rank_of_seeds() )[ wrong ] )
            ref = { k: v for k, v in ref.items() if numpy.isfinite( v ) and not ( set( k ) & broken ) }
            for kernel in ( "FP64", "FP32" ):
                out = _pd( pos, w, kernel, True, *_BOX3, tree = tree )._card_cells( facets = True, moments = False )
                assert out is not None, label
                assert _rel( numpy.asarray( out[ "measures" ].raw ).reshape( -1 ), plain ).max() < ( 1e-9 if kernel == "FP64" else 1e-6 )
                row, col, val, dia = _card_laplacian( out, n )
                for i in range( n ):
                    c = col[ row[ i ]:row[ i + 1 ] ]
                    assert ( numpy.diff( c ) > 0 ).all() and not ( c == i ).any(), ( label, kernel, "row", i, c )
                got = { ( i, int( col[ p ] ) ): float( val[ p ] ) for i in range( n ) for p in range( row[ i ], row[ i + 1 ] ) }
                cmp = { k: v for k, v in got.items() if not ( set( k ) & broken ) }
                assert all( got[ ( j, i ) ] == v for ( i, j ), v in got.items() ), ( label, kernel, "not symmetric" )
                sums = numpy.zeros( n )
                for i in range( n ):
                    for p in range( row[ i ], row[ i + 1 ] ):
                        sums[ i ] += val[ p ]
                bad = numpy.where( ~ ( ( sums == dia ) | ( ( sums == 0 ) & ( dia == 1 ) ) ) )[ 0 ]
                assert not len( bad ), ( label, kernel, "L 1 != 0", bad[ :5 ] )
                # the same graph ( but slivers: a facet of a vanishing area may be seen or not at the rounding ), the same values
                scale = numpy.median( list( ref.values() ) )
                missing, extra = set( ref ) - set( cmp ), set( cmp ) - set( ref )
                small = [ max( ref.get( k, 0 ), got.get( k, 0 ) ) / scale for k in missing | extra ]
                tiny = 1e-9 if kernel == "FP64" else 1e-3
                assert all( s < tiny for s in small ), ( label, kernel, len( missing ), len( extra ), sorted( small )[ -5: ] )
                assert len( missing | extra ) <= 1e-3 * len( ref ), ( label, kernel, len( missing ), len( extra ) )
                common = set( ref ) & set( cmp )
                err = numpy.array( [ abs( got[ k ] - ref[ k ] ) / max( ref[ k ], 1e-3 * scale ) for k in common ] )
                assert err.max() < ( 1e-8 if kernel == "FP64" else 1e-4 ) and numpy.median( err ) < ( 1e-11 if kernel == "FP64" else 1e-8 ), \
                    ( label, kernel, "values", err.max(), numpy.median( err ) )
                print( f"  { label } { kernel }: { len( got ) } entries ( generic { len( ref ) } compared, { len( missing ) } missing, "
                       f"{ len( extra ) } extra ), rel. gap median { numpy.median( err ) :.1e} max { err.max() :.1e}" )

if test( "the_card_moments_3d_are_the_generic_ones" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        rng = numpy.random.default_rng( 36 )
        for label, pos, w in _clouds3( rng, 4000 ):
            tree = AaBsp( pos, w )
            gen = _pd( pos, w, "FP64", False, *_BOX3, tree = tree )
            mg, fg, sg = ( numpy.asarray( t ) for t in gen.moments )
            mg = mg.reshape( -1 )
            full = mg > 0
            bary_ref = numpy.where( full[ :, None ], fg / numpy.where( full, mg, 1 )[ :, None ], pos )
            cost_ref = sg.reshape( -1 ) - 2 * ( pos * fg ).sum( 1 ) + ( pos * pos ).sum( 1 ) * mg
            for kernel in ( "FP64", "FP32" ):
                out = _pd( pos, w, kernel, True, *_BOX3, tree = tree )._card_cells( facets = False, moments = True )
                bary = numpy.asarray( out[ "bary" ].raw ).reshape( -1, 3 )
                cost = numpy.asarray( out[ "cost" ].raw ).reshape( -1 )
                h = len( pos ) ** ( -1 / 3 )
                # ( the barycentre of a vanishing cell is that of a sliver decided at the rounding: not compared )
                seen = mg > 1e-9 * mg.mean()
                eb = numpy.abs( bary - bary_ref )[ seen ].max() / h
                ec = numpy.abs( cost - cost_ref ).max() / numpy.abs( cost_ref ).max()
                assert eb < ( 1e-8 if kernel == "FP64" else 1e-5 ) and ec < ( 1e-8 if kernel == "FP64" else 1e-5 ), ( label, kernel, eb, ec )
                print( f"  { label } { kernel }: barycentres { eb :.1e} h, costs { ec :.1e}" )


def _measures_fn3( pos, w, tree, kernel, card, wrt ):
    def f( x ):
        p, q = ( x, w ) if wrt == "positions" else ( pos, x )
        pd = PowerDiagram( p, weights = q, boundaries = box_half_spaces( *_BOX3 ), kernel_dtype = kernel, accelerator = tree,
                           scratch_capacity = None if card else 1024, memory = 0 )
        pd.use_card_cells = card
        if card:
            assert pd._card_variant() is not None
        return pd.measures.value
    return f


if test( "the_card_adjoint_3d_is_the_generic_one" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        rng = numpy.random.default_rng( 37 )
        for label, pos, w in _clouds3( rng, 4000 ):
            n = len( pos )
            w = numpy.zeros( n ) if w is None else w
            tree = AaBsp( pos, w )
            g = rng.normal( size = n )
            for wrt, x in ( ( "weights", w ), ( "positions", pos ) ):
                _, pb = loom.vjp( _measures_fn3( pos, w, tree, "FP64", False, wrt ), x )
                ref = numpy.asarray( pb( g )[ 0 ] )
                for kernel in ( "FP64", "FP32" ):
                    _, pb = loom.vjp( _measures_fn3( pos, w, tree, kernel, True, wrt ), x )
                    got = numpy.asarray( pb( g )[ 0 ] )
                    err = numpy.abs( got - ref ).max() / numpy.abs( ref ).max()
                    assert err < ( 1e-9 if kernel == "FP64" else 1e-5 ), ( label, wrt, kernel, err )
                    print( f"  { label } d/d{ wrt } { kernel }: max gap { err :.1e} of the largest entry" )

if test( "the_card_adjoint_3d_is_the_finite_difference" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        from loom.testing import check_grad
        rng = numpy.random.default_rng( 38 )
        n = 40
        pos = rng.uniform( 0.1, 0.9, size = ( n, 3 ) )
        w = rng.uniform( -0.02, 0.02, n )
        tree = AaBsp( pos, w, max_seeds_per_leaf = 3 )

        def f( p, q ):
            pd = PowerDiagram( p, weights = q, boundaries = box_half_spaces( *_BOX3 ), kernel_dtype = "FP64", accelerator = tree )
            assert pd._card_variant() is not None
            return pd.measures
        check_grad( f, pos, w, seed = 38 )
        # the cells of the second and third passes ( a sphere of 80: 156 vertices )
        pos = _sphere3( 80, rng, n_back = 60 )
        w = rng.uniform( -1e-3, 1e-3, len( pos ) )
        tree = AaBsp( pos, w, max_seeds_per_leaf = 3 )
        check_grad( f, pos, w, seed = 39 )

if test( "the_card_3d_runs_under_jit" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        import jax
        rng = numpy.random.default_rng( 39 )
        n = 5000
        pos = rng.uniform( 0.001, 0.999, size = ( n, 3 ) )
        w = rng.uniform( -0.3, 0.3, n ) * n ** ( -2 / 3 )
        tree = AaBsp( pos, w )
        g = rng.normal( size = n )
        for kernel in ( "FP64", "FP32" ):
            f = _measures_fn3( pos, w, tree, kernel, True, "weights" )
            eager = numpy.asarray( f( w ) )
            jitted = numpy.asarray( jax.jit( f )( w ) )
            assert numpy.array_equal( jitted, eager ), ( kernel, numpy.abs( jitted - eager ).max() )
            loss = lambda x: ( f( x ) * g ).sum()
            ge = numpy.asarray( jax.grad( loss )( w ) )
            gj = numpy.asarray( jax.jit( jax.grad( loss ) )( w ) )
            assert numpy.array_equal( gj, ge ), kernel
            fp = _measures_fn3( pos, w, tree, kernel, True, "positions" )
            gp = numpy.asarray( jax.jit( jax.grad( lambda x: ( fp( x ) * g ).sum() ) )( pos ) )
            assert numpy.isfinite( gp ).all() and numpy.abs( gp ).max() > 0, kernel
        # the tree of TRACED positions, built in the jitted program, and the cells on it
        def measures( p ):
            pd = PowerDiagram( p, weights = w, boundaries = box_half_spaces( *_BOX3 ), kernel_dtype = "FP32", accelerator = AaBsp( p, w ) )
            assert pd._card_variant() is not None
            return pd.measures.value
        me, mj = numpy.asarray( measures( pos ) ), numpy.asarray( jax.jit( measures )( pos ) )
        assert numpy.array_equal( me, mj )

if test( "the_card_tree_3d_is_the_host_tree" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        rng = numpy.random.default_rng( 40 )
        clouds = [ ( label, pos ) for label, pos, _ in _clouds3( rng ) ] + [
            ( f"n = { n }", rng.uniform( 0, 1, size = ( n, 3 ) ) ) for n in ( 1, 2, 9, 10, 11, 64, 1000 ) ]
        for label, pos in clouds:
            for leaf in ( 10, 3 ):
                card = _tree( AaBsp( pos, max_seeds_per_leaf = leaf ) )
                _check_tree_rule( pos, card, leaf )
                _same_tree( card, _host_tree( pos, leaf ) )
            print( f"  { label }: the host's tree" )
        # the majorants of the card ( `Majorant3D.cuh` ) bound the weights on every node
        pos = rng.uniform( 0.001, 0.999, size = ( 8000, 3 ) )
        for label, w in ( ( "smooth", 0.1 * numpy.sin( 5 * pos[ :, 0 ] ) + 0.05 * pos[ :, 1 ] - 0.02 * pos[ :, 2 ] ),
                          ( "random", rng.uniform( -1, 1, len( pos ) ) / len( pos ) ), ( "constant", numpy.full( len( pos ), 0.25 ) ) ):
            tree = AaBsp( pos, w )
            t = _tree( tree )
            wa, wb = numpy.asarray( tree.node_wa ).reshape( -1, 3 ), numpy.asarray( tree.node_wb ).reshape( -1 )
            p, q = pos[ t[ "seed" ] ], w[ t[ "seed" ] ]
            for i in range( len( t[ "beg" ] ) ):
                sl = slice( t[ "beg" ][ i ], t[ "end" ][ i ] )
                if sl.stop > sl.start:
                    assert ( q[ sl ] <= p[ sl ] @ wa[ i ] + wb[ i ] ).all(), ( label, i )
            print( f"  { label }: { int( ( numpy.abs( wa ).sum( axis = 1 ) > 0 ).sum() ) } affine nodes of { len( wb ) }" )

if test( "the_card_3d_third_pass_works_in_batches" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        # 8 cells past the register passes ( spheres of 150 seeds: 296 vertices ) through FOUR slots: two batches in one
        # launch. The same bits as with 64 slots, for the measures and the adjoint.
        import sys
        bsp = sys.modules[ "sdot.PowerDiagram_Bsp" ]
        rng = numpy.random.default_rng( 41 )
        k, r = 150, 0.06
        centres = numpy.stack( numpy.meshgrid( [ 0.25, 0.75 ], [ 0.25, 0.75 ], [ 0.25, 0.75 ] ), axis = -1 ).reshape( -1, 3 )
        i = numpy.arange( k ) + 0.5
        phi, th = numpy.arccos( 1 - 2 * i / k ), numpy.pi * ( 1 + 5 ** 0.5 ) * i
        unit = numpy.stack( [ numpy.cos( th ) * numpy.sin( phi ), numpy.sin( th ) * numpy.sin( phi ), numpy.cos( phi ) ], axis = 1 )
        pos = numpy.concatenate( [ centres ] + [ c + r * unit + 1e-3 * rng.normal( size = unit.shape ) for c in centres ] )
        n = len( pos )
        w = rng.uniform( -0.1, 0.1, n ) * r * r / k
        w[ :len( centres ) ] = 0
        tree = AaBsp( pos, w )
        g = rng.normal( size = n )
        original = bsp.PowerDiagram_Bsp.card_overflow_warps
        try:
            got = {}
            for warps in ( 4, 64 ):
                bsp.PowerDiagram_Bsp.card_overflow_warps = warps
                if warps == 4:
                    _check3( pos, w, label = "8 spheres of 150, 4 slots", plain = True )
                for kernel in ( "FP64", "FP32" ):
                    m = _m( _pd( pos, w, kernel, True, *_BOX3, tree = tree ) )
                    _, pb = loom.vjp( _measures_fn3( pos, w, tree, kernel, True, "weights" ), w )
                    got[ warps, kernel ] = ( m, numpy.asarray( pb( g )[ 0 ] ) )
        finally:
            bsp.PowerDiagram_Bsp.card_overflow_warps = original
        for kernel in ( "FP64", "FP32" ):
            for a, b in zip( got[ 4, kernel ], got[ 64, kernel ] ):
                assert numpy.isfinite( a ).all() and numpy.array_equal( a, b ), kernel

if test( "the_card_3d_raises_past_the_vertex_limit" ):
    import numpy
    import loom
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from sdot.PowerDiagram_Bsp import card_variant_for
    _GPU = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    if not _GPU:
        skip( _NO_GPU )
    else:
        from loom.drivers.CallArg_Errors import KernelFailure
        import jax
        rng = numpy.random.default_rng( 42 )
        pos = _sphere3( 600, rng )                   # 1196 vertices ( the cells of the sphere's seeds reach ~1000 on the way )
        tree = AaBsp( pos, None )
        # ( the reference is the card's own double kernel: the generic path's scratch, grown for a cell of 1196 vertices on
        # every work item, takes gigabytes -- `the_card_cells_3d_overflow_into_the_later_passes` checks spheres against the
        # plain storage )
        ref = _m( _pd( pos, None, "FP64", True, *_BOX3, tree = tree ) )
        assert abs( ref.sum() - 1 ) < 1e-12 and 4.1e-3 < ref[ 0 ] < 4.4e-3, ( ref.sum(), ref[ 0 ] )
        for kernel in ( "FP64", "FP32" ):
            m = _m( _pd( pos, None, kernel, True, *_BOX3, tree = tree ) )
            r = _rel( m, ref )
            assert not numpy.isnan( m ).any() and r.max() < ( 1e-14 if kernel == "FP64" else 1e-5 ), ( kernel, r.max() )
            pd = _pd( pos, None, kernel, True, *_BOX3, tree = tree )
            pd.card_max_vertices = 1024
            try:
                pd.measures.value
                raise AssertionError( "a cell past the vertex limit went through" )
            except KernelFailure as e:
                assert "more than 1024 vertices" in str( e ) and "seed 0" in str( e ), str( e )
            pd = _pd( pos, numpy.zeros( len( pos ) ), kernel, True, *_BOX3, tree = AaBsp( pos, numpy.zeros( len( pos ) ) ) )
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
