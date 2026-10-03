"""THE 2D CELLS OF THE CARD ( `include/sdot/gpu/Cell2D.cuh`, `PowerDiagram_Bsp._measures_on_card` ).

On a CUDA device, `PowerDiagram.measures` in 2D goes through a dedicated kernel ( the cell in registers, two
overflow passes, the tree as records in the kernel's float, the float accuracy fixes of the old GPU campaign ).
These tests compare it with the GENERIC path of the same diagram ( `use_card_cells = False`, double kernel ):
the same cells, to the rounding of the kernel's float -- which, for the float kernel, is the DOUBLE's rounding
on the measure ( the vertices are re-solved in double ), except where the float decides a different topology.

Every test skips itself without a CUDA device ( the path does not exist on a CPU ).
"""
import numpy

from loom import driver
from errand import test, skip

from sdot import AaBsp, PowerDiagram, box_half_spaces

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
    ok = ref > 0
    floor = 1e-3 * ref.sum() / len( ref )
    return numpy.abs( m[ ok ] - ref[ ok ] ) / numpy.maximum( ref[ ok ], floor )


def _check( pos, w = None, mi = ( 0, 0 ), ma = ( 1, 1 ), tol64 = 1e-9, tol32_med = 1e-11, tol32_max = 1e-6, label = "" ):
    """the card ( float and double kernels ) against the generic double path, on the same tree"""
    tree = AaBsp( pos, w )
    ref = _m( _pd( pos, w, "FP64", False, mi, ma, tree ) )
    for kernel in ( "FP64", "FP32" ):
        pd = _pd( pos, w, kernel, True, mi, ma, tree )
        assert pd._measures_on_card() is not None, ( label, "the dedicated path did not take the call" )
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

if test( "the_card_cells_overflow_into_the_second_and_third_passes" ):
    if not _GPU:
        skip( _NO_GPU )
    else:
        # a seed in the middle of a ring of `k` seeds: its cell is a k-gon ( k > 8: second pass, k > 16: third pass,
        # k > 64 and k > 256: the third pass grows its capacity ), plus a uniform background
        rng = numpy.random.default_rng( 2 )
        for k in ( 12, 40, 100, 300 ):
            a = 2 * numpy.pi * ( numpy.arange( k ) + rng.uniform( -0.1, 0.1, k ) ) / k
            ring = 0.5 + 0.2 * numpy.stack( [ numpy.cos( a ), numpy.sin( a ) ], axis = 1 )
            back = rng.uniform( 0.001, 0.999, size = ( 2000, 2 ) )
            back = back[ numpy.linalg.norm( back - 0.5, axis = 1 ) > 0.3 ]
            pos = numpy.concatenate( [ [ [ 0.5, 0.5 ] ], ring, back ] )
            ref = _check( pos, label = f"ring of { k }" )
            assert abs( ref[ 0 ] - k * 0.1 ** 2 * numpy.tan( numpy.pi / k ) ) < 0.2 * ref[ 0 ]   # ~ the k-gon of apothem 0.1
        # and weighted: the central seed heavier, its k-gon wider
        w = numpy.zeros( len( pos ) )
        w[ 0 ] = 0.03
        _check( pos, w, label = "heavy centre" )

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
