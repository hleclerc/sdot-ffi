"""What the DISPLAY of the cells restricted to the support of a density shares ( `PowerDiagram.support_pieces` ).

A distribution that knows where it has mass describes it as BLOCKS -- convex polytopes, each wall carrying `SEAM` when
the region on the other side has mass too and `SUPPORT` otherwise ( `cell/Ids.h` ) -- and says which blocks each cell
must be cut to. The cutting itself is one C++ call over the pieces, the same for every distribution
( `diagram::build_support_piece` ). Here: the packing of the walls, and the boxes of the cells.
"""

import numpy


def pack_blocks( dirs, offs, ids ):
    """`( dirs [ B, F, d ], offs [ B, F ], ids [ B, F ] )`, from lists of walls of varying lengths. A missing wall has a zero
    direction ( the kernel skips it ). At least one block, so that the arrays are never empty."""
    d = next( ( len( w[ 0 ] ) for w in dirs if len( w ) ), 1 )
    B = max( len( dirs ), 1 )
    F = max( max( ( len( w ) for w in dirs ), default = 1 ), 1 )
    rd, ro, ri = numpy.zeros( ( B, F, d ) ), numpy.zeros( ( B, F ) ), numpy.zeros( ( B, F ), numpy.int32 )
    for b, ( dr, of, ii ) in enumerate( zip( dirs, offs, ids ) ):
        n = len( dr )
        rd[ b, :n ], ro[ b, :n ], ri[ b, :n ] = dr, of, ii
    return rd, ro, ri


def cell_vertices( cells ):
    """`( vp [ n, cap, d ], live [ n, cap ], nv [ n ] )`: the vertices of each item of a batched `Cell`, read once from its
    padded storage -- `live` says which rows are vertices"""
    n = cells.nb_items
    vp = numpy.asarray( cells.vertex_positions.raw, dtype = float )
    vp = vp.reshape( ( -1, ) + vp.shape[ -2: ] )[ : n ]
    nv = numpy.atleast_1d( numpy.asarray( cells.nb_vertices.value ) ).reshape( -1 ).astype( int )
    nv = nv[ : n ] if nv.size >= n else numpy.full( n, int( nv[ 0 ] ) )
    live = numpy.arange( vp.shape[ 1 ] )[ None, : ] < nv[ :, None ]
    return vp, live, nv


def boxes_of( points, live ):
    """`( lo [ n, d ], hi [ n, d ] )`: the box of the live rows of `points [ n, cap, d ]`"""
    lo = numpy.where( live[ ..., None ], points,  numpy.inf ).min( axis = 1 )
    hi = numpy.where( live[ ..., None ], points, -numpy.inf ).max( axis = 1 )
    return lo, hi


def overlapping_pairs( alo, ahi, blo, bhi ):
    """`( i, j )`: the pairs of boxes `a[ i ]`, `b[ j ]` that meet -- a uniform grid whose step is the median size of the
    `b` boxes, each box registered in the grid cells it covers, then the exact test. For a display: sizes in the
    millions are fine, no tree needed."""
    if not len( alo ) or not len( blo ):
        return numpy.zeros( 0, int ), numpy.zeros( 0, int )
    d = alo.shape[ 1 ]
    origin = numpy.minimum( alo.min( axis = 0 ), blo.min( axis = 0 ) )
    step = float( numpy.median( ( bhi - blo ).max( axis = 1 ) ) )
    ext = numpy.maximum( ahi.max( axis = 0 ), bhi.max( axis = 0 ) ) - origin
    step = max( step, float( ext.max() ) / 1024, 1e-300 )
    dims = numpy.floor( ext / step ).astype( int ) + 1

    def registered( lo, hi ):
        k0 = numpy.clip( numpy.floor( ( lo - origin ) / step ).astype( int ), 0, dims - 1 )
        k1 = numpy.clip( numpy.floor( ( hi - origin ) / step ).astype( int ), 0, dims - 1 ) + 1
        cnt = numpy.prod( k1 - k0, axis = 1 )
        owner = numpy.repeat( numpy.arange( len( lo ) ), cnt )
        local = numpy.arange( cnt.sum() ) - numpy.repeat( numpy.cumsum( cnt ) - cnt, cnt )
        key = numpy.zeros( len( owner ), numpy.int64 )
        for a in range( d ):                                        # the cell of rank `local` in the box, axis by axis
            span = ( k1 - k0 )[ owner, a ]
            key = key * dims[ a ] + k0[ owner, a ] + local % span
            local = local // span
        return key, owner

    ka, ia = registered( alo, ahi )
    kb, ib = registered( blo, bhi )
    ob = numpy.argsort( kb, kind = "stable" )
    kb, ib = kb[ ob ], ib[ ob ]
    start = numpy.searchsorted( kb, ka, side = "left" )
    stop = numpy.searchsorted( kb, ka, side = "right" )
    cnt = stop - start
    i = numpy.repeat( ia, cnt )
    j = ib[ numpy.repeat( start, cnt ) + numpy.arange( cnt.sum() ) - numpy.repeat( numpy.cumsum( cnt ) - cnt, cnt ) ]
    pair = numpy.unique( i * len( blo ) + j )
    i, j = pair // len( blo ), pair % len( blo )
    meet = ( ( alo[ i ] <= bhi[ j ] ) & ( blo[ j ] <= ahi[ i ] ) ).all( axis = 1 )
    return i[ meet ], j[ meet ]
