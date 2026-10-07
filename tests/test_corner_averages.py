"""The corner averages of a DG1 density ( `sdot.corner_averages`, `include/sdot/CornerAverages.h` ): the spreading on its own mesh
for the width continuation. The witnesses: the original at `k = 0`, the mass and the positivity at every `k`, the powers of the two
averages against a plain numpy version, and the Chebyshev series close to them."""

from errand import test


def _case( d, seed ):
    """a Delaunay mesh of the unit square / cube, and a positive DG1 density, discontinuous between the simplices"""
    import numpy
    from scipy.spatial import Delaunay
    rng = numpy.random.default_rng( seed )
    corners = numpy.array( numpy.meshgrid( *[ [ 0.0, 1.0 ] ] * d, indexing = "ij" ) ).reshape( d, -1 ).T
    nodes = numpy.concatenate( [ corners, rng.uniform( 0, 1, ( 300, d ) ) ] )
    sx = Delaunay( nodes ).simplices
    cv = rng.uniform( 0, 1, sx.shape ) * ( rng.random( len( sx ) ) < 0.3 )[ :, None ]
    return nodes, sx, cv


def _numpy_powers( nodes, sx, cv, ks ):
    """the reference: `A` then `N`, by numpy, `k` integer"""
    import numpy
    d = nodes.shape[ 1 ]
    P = nodes[ sx ]
    vol = numpy.abs( numpy.linalg.det( P[ :, 1: ] - P[ :, :1 ] ) )
    w = numpy.repeat( vol, d + 1 )
    m = numpy.bincount( sx.ravel(), weights = w, minlength = len( nodes ) )
    out, v = {}, cv
    for k in range( max( ks ) + 1 ):
        if k in ks:
            out[ k ] = v
        mean = numpy.repeat( v.mean( 1, keepdims = True ), d + 1, axis = 1 )
        v = ( numpy.bincount( sx.ravel(), weights = w * mean.ravel(), minlength = len( nodes ) ) / m )[ sx ]
    return [ out[ k ] for k in ks ], vol


def _mass( vol, v ):
    return ( vol * v.mean( -1 ) ).sum( -1 )


if test( "the_powers_of_the_corner_averages" ):
    import numpy
    from sdot.corner_averages import corner_averages
    for d in ( 2, 3 ):
        nodes, sx, cv = _case( d, d )
        ks = [ 0, 1, 2, 7, 40 ]
        want, vol = _numpy_powers( nodes, sx, cv, ks )
        got = corner_averages( nodes, sx, cv, ks, method = "iter" )
        for k, g, w in zip( ks, got, want ):
            assert numpy.abs( g - w ).max() < 1e-12 * w.max(), ( d, k, numpy.abs( g - w ).max() )
        assert numpy.abs( _mass( vol, got ) / _mass( vol, cv ) - 1 ).max() < 1e-12
        # between two integers: the linear mix
        mid = corner_averages( nodes, sx, cv, [ 0.25, 7.5 ], method = "iter" )
        assert numpy.abs( mid[ 0 ] - ( 0.75 * want[ 0 ] + 0.25 * want[ 1 ] ) ).max() < 1e-12 * want[ 1 ].max()
        w8 = _numpy_powers( nodes, sx, cv, [ 8 ] )[ 0 ][ 0 ]
        assert numpy.abs( mid[ 1 ] - 0.5 * ( want[ 3 ] + w8 ) ).max() < 1e-12 * w8.max()


if test( "the_chebyshev_series_is_the_heat_of_the_averages" ):
    import numpy
    from sdot.corner_averages import corner_averages
    for d in ( 2, 3 ):
        nodes, sx, cv = _case( d, 10 + d )
        ks = [ 0, 0.5, 1, 50, 400 ]
        st = {}
        ch = corner_averages( nodes, sx, cv, ks, method = "cheb", stats = st )
        it = corner_averages( nodes, sx, cv, ks, method = "iter" )
        _, vol = _numpy_powers( nodes, sx, cv, [ 0 ] )
        assert numpy.abs( ch[ :3 ] - it[ :3 ] ).max() == 0                      # ( up to one iteration: the same states )
        assert ( ch >= 0 ).all()
        assert numpy.abs( _mass( vol, ch ) / _mass( vol, cv ) - 1 ).max() < 1e-10
        # `exp( -t ( I - W ) )` and `W^t` share the smooth modes: close once t is large
        assert numpy.abs( ch[ 4 ] - it[ 4 ] ).max() < 0.05 * it[ 4 ].max(), numpy.abs( ch[ 4 ] - it[ 4 ] ).max() / it[ 4 ].max()
        assert st[ "nb_products" ] < 400, st                                   # ( far fewer products than 50 + 400 )
