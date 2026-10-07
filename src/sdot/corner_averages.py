"""The corner averages of a DG1 density: its spreading on its own mesh, for the width continuation ( `include/sdot/CornerAverages.h` ).

A DG1 density is a value per node of each simplex: `cv [ nb_elems, d + 1 ]` on `nodes [ nb_nodes, d ]`, `simplices [ nb_elems, d + 1 ]`.
`corner_averages( ..., ks )` gives the states `k` of the path ( `[ len( ks ), nb_elems, d + 1 ]` ): `k = 0` the original, `k = 1` one
iteration of the two averages ( the corners of a simplex take its mean, then each corner the mean of those touching its node ), and
beyond, `W^( k - 1 )` ( `method = "iter"` ) or its Chebyshev approximation `exp( -( k - 1 ) ( I - W ) )` ( `"cheb"`, about
`sqrt( k ln( 1 / eps ) )` products instead of `k` ). Mass exact, positive, continuous in `k`.
"""

import numpy

from loom.tensor import Axis, IntTensor, RealTensor, ShapeVar

_METHODS = { "iter": 0, "cheb": 1 }


def corner_averages( nodes, simplices, cv, ks, method = "cheb", eps = 1e-8, stats = None ):
    """see the module docstring. `stats`, a dict: receives `nb_products`"""
    import loom
    from loom.compilation.FfiCode import FfiCode
    nd = numpy.ascontiguousarray( nodes, dtype = float )
    sx = numpy.ascontiguousarray( simplices, dtype = numpy.int32 )
    cv = numpy.ascontiguousarray( cv, dtype = float )
    ks = numpy.atleast_1d( numpy.asarray( ks, dtype = float ) )
    d = nd.shape[ 1 ]
    if sx.shape[ 1 ] != d + 1 or cv.shape != sx.shape:
        raise ValueError( f"corner_averages: `simplices` and `cv` have to be [ nb_elems, { d + 1 } ] ( got { sx.shape } and { cv.shape } )" )
    num_node = Axis( ShapeVar( len( nd ) ), name = "num_node" )
    num_elem = Axis( ShapeVar( len( sx ) ), name = "num_elem" )
    num_corner = Axis( ShapeVar( d + 1 ), name = "num_corner" )
    dim = Axis( ShapeVar( d ), name = "dim" )
    num_k = Axis( ShapeVar( len( ks ) ), name = "num_k" )
    out = RealTensor[ num_k, num_elem, num_corner ]()
    count = IntTensor[ Axis( ShapeVar( 1 ), name = "num_stat" ) ]()
    loom.ffi_call(
        "corner_averages",
        FfiCode.inline( f"sdot::corner_averages::run<{ d }>( queue, args.inputs.nodes, args.inputs.simplices, args.inputs.cv, "
                        "args.inputs.ks, args.inputs.params, args.outputs.out, args.outputs.count );",
                        includes = [ "sdot/CornerAverages.h" ] ),
        nodes = RealTensor[ num_node, dim ]( nd ),
        simplices = IntTensor[ num_elem, num_corner, dict( size = 32 ) ]( sx ),
        cv = RealTensor[ num_elem, num_corner ]( cv ),
        ks = RealTensor[ num_k ]( ks ),
        params = RealTensor[ Axis( ShapeVar( 2 ), name = "num_param" ) ]( numpy.array( [ _METHODS[ method ], eps ], dtype = float ) ),
        out = loom.out( out ), count = loom.out( count ),
        has_dynamic_capacity = False,
    )
    if stats is not None:
        stats[ "nb_products" ] = int( numpy.asarray( count ).reshape( -1 )[ 0 ] )
    return numpy.asarray( out ).reshape( len( ks ), len( sx ), d + 1 )
