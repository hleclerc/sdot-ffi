"""The cell a power diagram's cells START from, when the domain is a polytope and not a box.

A cell of a power diagram is its domain cut by the bisectors of the seeds. The domain is therefore the cell
that the cuts start from -- a box, or a convex polytope -- and the card's kernels lay it down in one go instead of cutting
a box by the facets of the polytope for every seed ( `gpu/Cell2D.cuh`, `gpu/Cell3D.cuh`: `start_vertices` ).

`start_cell( directions, offsets )` describes that cell from the half-spaces `direction . x <= offset`: the facets that
matter ( the redundant ones leave ), the vertices, and the topology the kernels need.

  * 2D: the vertices counterclockwise; facet `i` ( the cut `-5 - i` ) is the edge from vertex `i` to vertex `i + 1`;
  * 3D: facet `f` is the cut `-7 - f`; vertex `v` carries three facets `k( v, 0..2 )` and has three neighbours,
    `nn( v, j )` the one OPPOSITE `k( v, j )` -- the neighbour along the edge that is not on facet `k( v, j )`. This is the
    form of a vertex in the kernels' cells, and it is that of a SIMPLE polyhedron ( three facets per vertex ) --
    a pyramid's apex is not: it has no start cell, and the facets cut a box instead.

`None` when there is none: not 2D or 3D, not bounded, too many facets ( the kernels' registers: `MAX_FACETS` ) or not simple.
`bounding_box` is the box of the vertices, always when the polytope is bounded: the SCALE of the domain.
"""

import numpy as np

#: the facets and vertices a start cell holds: the cuts and vertex indices of the kernels' cells are bytes
MAX_FACETS = 255


def bounded_polytope( directions, offsets ):
    """`( vertices, unit directions, offsets )` of the polytope, or `None` if it is empty, flat, unbounded or not readable
    on the host. The vertices are deduplicated."""
    try:
        dirs = np.asarray( directions, dtype = float )
        offs = np.asarray( offsets, dtype = float ).reshape( -1 )
    except ( TypeError, ValueError, RuntimeError ):      # a tracer, or a torch tensor that requires grad
        return None
    if dirs.ndim != 2 or len( dirs ) != len( offs ) or len( offs ) <= dirs.shape[ 1 ] or dirs.shape[ 1 ] not in ( 2, 3 ):
        return None
    d = dirs.shape[ 1 ]
    norm = np.sqrt( ( dirs ** 2 ).sum( axis = 1 ) )
    if not np.all( norm > 0 ):
        return None
    dirs, offs = dirs / norm[ :, None ], offs / norm

    from scipy.optimize import linprog
    from scipy.spatial import HalfspaceIntersection
    # an interior point: the center of the largest ball
    res = linprog( c = np.r_[ np.zeros( d ), -1.0 ], A_ub = np.c_[ dirs, np.ones( len( offs ) ) ], b_ub = offs,
                   bounds = [ ( None, None ) ] * d + [ ( 0, None ) ] )
    if res.status != 0 or not res.x[ d ] > 0:
        return None
    try:
        hs = HalfspaceIntersection( np.c_[ dirs, -offs ], res.x[ :d ] )
    except Exception:                                    # ( qhull: unbounded, degenerate )
        return None
    pts = hs.intersections
    if not np.all( np.isfinite( pts ) ):
        return None
    scale = max( float( np.abs( pts ).max() ), 1.0 )
    key = np.round( pts / ( 1e-9 * scale ) ).astype( np.int64 )
    _, first = np.unique( key, axis = 0, return_index = True )
    return pts[ np.sort( first ) ], dirs, offs


def start_cell( directions, offsets ):
    """see the module docstring: a dict `bounding_box = ( lo, hi )` and, if the polytope can start the cells, `directions`,
    `offsets` ( the facets that matter, in the order of the cuts' identifiers ), `vertices`, `topology` ( 3D: `[ nv, 6 ]`,
    `k` then `nn` ); or `None` if it is not a bounded polytope"""
    poly = bounded_polytope( directions, offsets )
    if poly is None:
        return None
    verts, dirs, offs = poly
    d = dirs.shape[ 1 ]
    out = dict( bounding_box = ( verts.min( axis = 0 ), verts.max( axis = 0 ) ) )
    scale = max( float( np.abs( verts ).max() ), 1.0 )
    tol = 1e-9 * scale

    # the facets that matter, the equal ones merged: those with `d` vertices on them at least ( a facet )
    on = np.abs( verts @ dirs.T - offs[ None, : ] ) < tol            # [ nv, nf ]
    faces = []
    for f in np.flatnonzero( on.sum( axis = 0 ) >= d ):
        if not any( np.abs( dirs[ f ] - dirs[ g ] ).max() < 1e-9 and abs( offs[ f ] - offs[ g ] ) < tol for g in faces ):
            faces.append( f )
    nv, nf = len( verts ), len( faces )
    if nf > MAX_FACETS or nv > MAX_FACETS:
        return out
    fdir, foff = dirs[ faces ], offs[ faces ]
    fon = np.abs( verts @ fdir.T - foff[ None, : ] ) < tol            # [ nv, nf ]
    if not np.all( fon.sum( axis = 1 ) == d ):                       # not simple
        return out

    if d == 2:
        c = verts.mean( axis = 0 )
        order = np.argsort( np.arctan2( verts[ :, 1 ] - c[ 1 ], verts[ :, 0 ] - c[ 0 ] ) )
        verts, fon = verts[ order ], fon[ order ]
        edge = []                                                    # edge i: vertex i -> i + 1, the facet they share
        for i in range( nv ):
            both = np.flatnonzero( fon[ i ] & fon[ ( i + 1 ) % nv ] )
            if len( both ) != 1:
                return out
            edge.append( both[ 0 ] )
        out.update( directions = fdir[ edge ], offsets = foff[ edge ], vertices = verts, topology = None )
        return out

    topo = np.zeros( ( nv, 6 ), dtype = np.int32 )
    triples = [ np.flatnonzero( fon[ v ] ) for v in range( nv ) ]
    for v in range( nv ):
        topo[ v, :3 ] = triples[ v ]
        for j in range( 3 ):
            shared = [ triples[ v ][ a ] for a in range( 3 ) if a != j ]      # the edge is on the two other facets
            others = [ u for u in range( nv ) if u != v and fon[ u, shared[ 0 ] ] and fon[ u, shared[ 1 ] ] ]
            if len( others ) != 1:
                return out
            topo[ v, 3 + j ] = others[ 0 ]
    out.update( directions = fdir, offsets = foff, vertices = verts, topology = topo )
    return out
