"""Change of representation of a convex polytope: from half-spaces to vertices.

`{ x : dir_i . x <= off_i }` (H-representation) is what a cell can always give, but
it cannot be displayed as is: one needs VERTICES, EDGES and FACES. The HTML page does this
work in the browser, at every cut; here is the Python version, which the file outputs
(VTK) need.

Written in ANY dimension, not only in 3D:
- a VERTEX is the intersection of `d` planes, feasible for all the others;
- an EDGE joins two vertices that share `d-1` active planes;
- a FACE (in the sense of a polygon, what VTK can draw) is carried by `d-2` common planes.
  In 3D this gives back "one face per cut"; in 4D these are the squares of a tesseract; in 2D only
  a single polygon remains, the whole polytope.

This is exact for a SIMPLE polytope (no vertex carrying more than `d` planes), which is what cells
in generic position are. A degenerate vertex (more than `d` concurrent planes) can make
extra edges appear.

The cost is C(nb_planes, d) small systems -- negligible for a cell (a few hundred
in 3D, a few thousand in 4D), but this is indeed a HOST computation: the day it becomes the bottleneck,
a kernel is what is needed, not an optimization from here.
"""
from itertools import combinations

import numpy as np


def clip_planes( bounds ):
    """The 2d half-spaces of a box `[ [lo, hi], ... ]`.

    To be added to an UNBOUNDED polytope: without it, it has no vertex, hence nothing to show.
    On a bounded polytope that fits in the box, these planes stay inactive and change nothing.
    """
    d = len( bounds )
    dirs, offs = [], []
    for k, ( lo, hi ) in enumerate( bounds ):
        e = np.zeros( d ); e[ k ] = 1.0
        dirs.append(  e ); offs.append(  hi )
        dirs.append( -e ); offs.append( -lo )
    return np.array( dirs, np.float64 ), np.array( offs, np.float64 )


def vertices_of( dirs, offs, tol = 1e-9 ):
    """The vertices of the polytope, and for each the set of planes that carry it.

    Returns `( verts [n, d], active [n] )`, `active[ i ]` being a `frozenset` of plane indices.
    """
    A = np.asarray( dirs, np.float64 )
    b = np.asarray( offs, np.float64 ).reshape( -1 )
    nrm = np.linalg.norm( A, axis = 1 )
    keep = nrm > 1e-12                       # a degenerate plane carries no vertex
    A, b = A[ keep ] / nrm[ keep, None ], b[ keep ] / nrm[ keep ]
    n, d = A.shape
    if n < d:
        return np.zeros( ( 0, d ) ), []

    # all the d-tuples of planes at once: `det` discards the dependent systems, then a single
    # grouped `solve`. This is what makes the enumeration tractable in Python.
    combos = np.array( list( combinations( range( n ), d ) ), dtype = np.int64 )
    M = A[ combos ]                                        # [ k, d, d ]
    rhs = b[ combos ]                                      # [ k, d ]
    ok = np.abs( np.linalg.det( M ) ) > 1e-10
    if not ok.any():
        return np.zeros( ( 0, d ) ), []
    # `rhs[ ..., None ]`: since numpy 2, a rank-2 right-hand side is read as ONE
    # matrix, not as a stack of vectors -- hence the explicit axis.
    X = np.linalg.solve( M[ ok ], rhs[ ok ][ ..., None ] )[ ..., 0 ]     # [ k', d ]

    scale = max( 1.0, float( np.abs( b ).max() ) )
    eps   = tol * scale
    inside = ( X @ A.T <= b + eps ).all( axis = 1 )
    X = X[ inside ]
    if len( X ) == 0:
        return np.zeros( ( 0, d ) ), []

    # coincident vertices merged: two different d-tuples designate the same corner as soon as more
    # than d planes meet there.
    keys = np.round( X / ( 10 * eps ) ).astype( np.int64 )
    _, first = np.unique( keys, axis = 0, return_index = True )
    X = X[ np.sort( first ) ]

    on = np.abs( X @ A.T - b ) < 10 * eps                  # [ n_verts, n_planes ]
    active = [ frozenset( np.flatnonzero( row ).tolist() ) for row in on ]
    return X, active


def edges_of( active, nb_dims, nb_real = None ):
    """The edges: two vertices that share at least `d-1` active planes.

    `nb_real` marks the boundary between the planes of the polytope (the first ones) and those of the CLIPPING
    BOX (see `clip_planes`): an edge lying on the box is not an edge of the
    polytope, it is the border of the field, and drawing it amounts to drawing the box.
    """
    res = []
    for i in range( len( active ) ):
        for j in range( i + 1, len( active ) ):
            shared = active[ i ] & active[ j ]
            if len( shared ) < nb_dims - 1:
                continue
            if nb_real is not None and any( p >= nb_real for p in shared ):
                continue
            res.append( ( i, j ) )
    return np.array( res, np.int64 ).reshape( -1, 2 )


def faces_of( verts, active, edges, nb_dims ):
    """The POLYGONAL faces, in cyclic order -- what VTK (or a renderer) can draw.

    A face is carried by `d-2` common planes. We do not look for them among all the sub-
    sets: each EDGE already carries `d-1` of them, so the faces that contain it
    are obtained by removing one from it. This avoids useless combinatorics.
    """
    groups = {}
    for i, j in edges:
        shared = active[ i ] & active[ j ]
        for sub in combinations( sorted( shared ), nb_dims - 2 ):
            groups.setdefault( sub, set() ).update( ( int( i ), int( j ) ) )

    res = []
    for ids in groups.values():
        ids = sorted( ids )
        if len( ids ) < 3:
            continue
        order = _cyclic_order( verts[ ids ] )
        if order is not None:
            res.append( [ ids[ k ] for k in order ] )
    return res


def _cyclic_order( pts, tol = 1e-12 ):
    """Sorts COPLANAR points by rotating around their center.

    The plane of the face is found in place (two independent directions taken among the offsets
    from the center, orthonormalized): this holds in any dimension, there is no "normal"
    to invoke beyond 3D.
    """
    ctr = pts.mean( axis = 0 )
    rel = pts - ctr
    scale = float( np.linalg.norm( rel, axis = 1 ).max() )
    if scale < tol:
        return None

    u = rel[ int( np.argmax( np.linalg.norm( rel, axis = 1 ) ) ) ]
    u = u / np.linalg.norm( u )
    perp = rel - np.outer( rel @ u, u )
    k = int( np.argmax( np.linalg.norm( perp, axis = 1 ) ) )
    if np.linalg.norm( perp[ k ] ) < tol * scale:
        return None                                        # aligned points: not a polygon
    w = perp[ k ] / np.linalg.norm( perp[ k ] )
    return list( np.argsort( np.arctan2( rel @ w, rel @ u ) ) )


def polytope_mesh( dirs, offs, bounds = None ):
    """`( verts [n, d], edges [m, 2], faces )` of a polytope given as half-spaces.

    `bounds` adds a clipping box (see `clip_planes`): indispensable if the polytope can
    be unbounded, without effect otherwise.
    """
    dirs = np.asarray( dirs, np.float64 )
    offs = np.asarray( offs, np.float64 ).reshape( -1 )

    # degenerate planes are discarded HERE and not in `vertices_of`: the latter renumbers what
    # it keeps, and the `nb_real` boundary would no longer line up.
    nrm = np.linalg.norm( dirs, axis = 1 )
    dirs, offs = dirs[ nrm > 1e-12 ], offs[ nrm > 1e-12 ]
    nb_real = len( dirs )

    if bounds is not None:
        cd, co = clip_planes( bounds )
        dirs, offs = np.concatenate( [ dirs, cd ] ), np.concatenate( [ offs, co ] )

    verts, active = vertices_of( dirs, offs )
    if len( verts ) == 0:
        return verts, np.zeros( ( 0, 2 ), np.int64 ), []
    d = dirs.shape[ 1 ]

    # two edge lists, and this is not an inelegance: the FACES need all of them (a
    # clipping face is made of clipping edges, and without it we would see the inside of the
    # polytope), the DRAWING only needs those of the polytope.
    edges = edges_of( active, d )
    return verts, edges_of( active, d, nb_real ), faces_of( verts, active, edges, d )
