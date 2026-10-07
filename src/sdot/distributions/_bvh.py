import numpy as np


def build_bvh( lo, hi, leaf = 4 ):
    """A bounding volume hierarchy over `n` boxes `[ lo, hi ]` ( `[ n, d ]` ), on the host, in numpy.

    Returns `( perm, node_lo, node_hi, links )`: `perm` the order of the boxes that makes every leaf a RANGE; the node boxes
    `[ m, d ]`; `links` `[ m, 4 ]` = `left, right, begin, end` ( `left = right = -1` for a leaf; `begin, end` the range of
    `perm` under the node ). The root is node 0, and a child's number is above its parent's.

    Built level by level -- every node of a level split at the median of its centers, along the widest axis -- so that the
    cost is a few vectorized sorts and not a Python loop over the nodes."""
    n, d = lo.shape
    ctr = 0.5 * ( lo + hi )
    perm = np.arange( n )
    cap = 4 * max( n // max( leaf, 1 ), 1 ) + 16
    left = np.full( cap, -1, dtype = np.int64 )
    right = np.full( cap, -1, dtype = np.int64 )
    begin = np.zeros( cap, dtype = np.int64 )
    end = np.zeros( cap, dtype = np.int64 )
    level = np.zeros( cap, dtype = np.int64 )
    end[ 0 ] = n
    nb = 1
    active = np.array( [ 0 ] )
    depth = 0
    while active.size:
        act = active[ ( end[ active ] - begin[ active ] ) > leaf ]
        if not act.size:
            break
        b, e = begin[ act ], end[ act ]
        counts = e - b
        total = int( counts.sum() )
        starts = np.cumsum( counts ) - counts
        idx = np.repeat( b - starts, counts ) + np.arange( total )          # the positions of `perm` under the nodes of the level
        seg = np.repeat( np.arange( len( act ) ), counts )
        c = ctr[ perm[ idx ] ]
        extent = np.maximum.reduceat( c, starts, axis = 0 ) - np.minimum.reduceat( c, starts, axis = 0 )
        axis = np.argmax( extent, axis = 1 )
        key = c[ np.arange( total ), axis[ seg ] ]
        order = np.lexsort( ( key, seg ) )
        perm[ idx ] = perm[ idx ][ order ]
        mid = b + counts // 2
        L = nb + 2 * np.arange( len( act ) )
        R = L + 1
        nb += 2 * len( act )
        left[ act ], right[ act ] = L, R
        begin[ L ], end[ L ], begin[ R ], end[ R ] = b, mid, mid, e
        depth += 1
        level[ L ] = level[ R ] = depth
        active = np.stack( [ L, R ], axis = 1 ).ravel()

    left, right, begin, end, level = left[ :nb ], right[ :nb ], begin[ :nb ], end[ :nb ], level[ :nb ]
    slo, shi = lo[ perm ], hi[ perm ]
    node_lo = np.zeros( ( nb, d ) )
    node_hi = np.zeros( ( nb, d ) )
    leaves = np.flatnonzero( left < 0 )
    lb = begin[ leaves ]                                                   # the leaves tile `perm`, in order of position
    pos = np.argsort( lb )
    leaves, lb = leaves[ pos ], lb[ pos ]
    node_lo[ leaves ] = np.minimum.reduceat( slo, lb, axis = 0 )
    node_hi[ leaves ] = np.maximum.reduceat( shi, lb, axis = 0 )
    for lv in range( int( level.max() ) - 1, -1, -1 ):
        inner = np.flatnonzero( ( level == lv ) & ( left >= 0 ) )
        node_lo[ inner ] = np.minimum( node_lo[ left[ inner ] ], node_lo[ right[ inner ] ] )
        node_hi[ inner ] = np.maximum( node_hi[ left[ inner ] ], node_hi[ right[ inner ] ] )
    return perm, node_lo, node_hi, np.stack( [ left, right, begin, end ], axis = 1 )
