"""Draw a cell in a `Visualizer` -- and what an UNBOUNDED cell drops.

It is the cell that chooses its primitives, not the visualizer: in 2D the polygon, in 3D the
faces read back from the lattice, beyond that the H-representation ( the page shows a 3D slice of it ) with the
projected wireframe. Everything is read from the derived tensors that each regime knows how to provide
( `Cell._edges_of`, `_faces_of`, `_planes_of`, ... ): nothing here depends on the dimension.

= What is FAKE in an unbounded cell

The `INFINITE` walls do not exist, and their vertices are placed where the cuts had to push
them back, at a distance that means nothing. Three rules, read from `cut_ids`:

  * an `INFINITE` wall is not sent -- what remains is the true, unbounded polytope, which the
    page closes itself on the scene's box;
  * an edge LYING on fake walls ( all its planes are ) is not an edge of the
    cell: it is not drawn;
  * an edge that touches one at one end only is a real edge, truncated somewhere:
    it is drawn DOTTED, brought back to a length taken from the REAL extent of the scene.

A `SEAM` ( the wall between two pieces of a cell that both have mass, `PowerDiagram.support_pieces` )
is not drawn either: neither its face, nor an edge that lies on it -- except, in 3D, the edge where the
border of the support FOLDS on the seam ( a concave crease: `_show_creases` ).

The framing follows the same logic: only the real vertices set it.
"""

import numpy as np


def add_to_viz( cell, viz, color, opacity, faces, edges, points, owners = None, nb_colors = None ):
    from .Cell import INFINITE, SEAM

    nb_dims = cell.dim
    items = cell._items()
    # one per item, empty or not ( see `Cell.add_to_viz` ) -- or one per OWNER, for the pieces of cells
    if owners is None:
        first_color = viz.reserve_colors( len( items ) )
    else:
        owners = np.asarray( owners ).reshape( -1 )
        first_color = viz.reserve_colors( nb_colors if nb_colors is not None else int( owners.max( initial = -1 ) ) + 1 )

    if max( ( it.nb_vertices for it in items ), default = 0 ) == 0:
        return viz

    # FIRST PASS: what, in each item, belongs to the fake simplex -- we need the overall
    # view before drawing, to give the truncated rays a length taken from the scene
    parts = { b: _infinite_parts( cell, it ) for b, it in enumerate( items ) if it.nb_vertices }
    if owners is not None and nb_dims == 3:
        _show_creases( cell, items, owners, parts )
    stub = _ray_length( [ it.vp for it in items ], parts )

    for b, it in enumerate( items ):
        if it.nb_vertices == 0:
            continue
        col = color if color is not None else viz.color_at( first_color + ( b if owners is None else int( owners[ b ] ) ) )
        edge_col = viz.darker( col )

        fake, ev, hide, dash = parts[ b ]
        vp = _truncated_rays( it.vp, fake, ev, stub )
        real = vp[ ~fake ] if fake.any() else vp
        bounded = not fake.any()
        if not bounded:
            # this pool no longer FRAMES the scene: the end of a ray is at a chosen distance
            viz.note_bounds( real if len( real ) else vp )

        if nb_dims > 3 or not bounded:
            dirs, offs = cell._planes_of( it )
            keep = ( it.cid != INFINITE ) & ( np.abs( dirs ).sum( axis = 1 ) > 0 )
            if keep.any():
                # `faces = False` -> zero opacity rather than no faces: they stay in the
                # z-buffer, so the hidden edges stay hidden. `edges`: beyond 3D this polytope
                # is a SLICE, and its edges are those of the slice; in 2D / 3D it is the cell
                # that draws its own below, knowing what is truncated.
                viz.add_polytope( dirs[ keep ], offs[ keep ], nb_dims = nb_dims, color = col,
                                  opacity = opacity if faces else 0, edges = nb_dims > 3 )
        elif faces and nb_dims >= 2:
            # a `SEAM` is inside the cell: its face is not drawn ( in 2D the face is the piece itself )
            fcs = cell._faces_of( it )
            if nb_dims == 3 and ( it.cid == SEAM ).any():
                fcs = [ f for f, k in zip( fcs, cell._face_cuts_of( it ) ) if it.cid[ k ] != SEAM ]
            viz.add_faces( vp, fcs, color = col, opacity = opacity, frames = bounded )

        if edges and len( ev ):
            # in dimension > 3 the wireframe is a PROJECTION: it fades behind the slice
            op = 0.55 if nb_dims > 3 else 1.0
            solid = ~hide & ~dash
            if solid.any():
                viz.add_edges( vp, ev[ solid ], color = edge_col, opacity = op, frames = bounded )
            if dash.any():
                viz.add_edges( vp, ev[ dash ], color = edge_col, opacity = op, dashed = True, frames = False )

        if points:
            viz.add_points( real if len( real ) else vp, color = edge_col )
    return viz


def _infinite_parts( cell, it ):
    """`( fake, ev, hide, dash )`: which vertices are fake, the list of edges, and which ones
    not to draw / to draw dotted. A vertex is fake as soon as ONE of its planes is. An edge that
    lies on a `SEAM` ( the inner wall of a piece, see `PowerDiagram.support_pieces` ) is hidden too."""
    from .Cell import INFINITE, SEAM

    infinite = it.cid == INFINITE
    vc = cell._vertex_cut_indices_of( it )
    ev = cell._edges_of( it )
    ec = cell._edge_cuts_of( it )
    fake = infinite[ vc ].any( axis = 1 ) if it.nb_vertices else np.zeros( 0, bool )
    hide = infinite[ ec ].all( axis = 1 ) if len( ev ) and ec.shape[ 1 ] else np.zeros( len( ev ), bool )
    if len( ev ) and ec.shape[ 1 ]:
        hide = hide | ( it.cid[ ec ] == SEAM ).any( axis = 1 )
    dash = ( fake[ ev[ :, 0 ] ] | fake[ ev[ :, 1 ] ] ) & ~hide if len( ev ) else np.zeros( 0, bool )
    return fake, ev, hide, dash


def _show_creases( cell, items, owners, parts ):
    """The edges of the pieces of a cell that lie on a `SUPPORT` wall and a `SEAM` are hidden by `_infinite_parts`: most are
    the seam crossing a flat border of the support, which goes on, coplanar, in the piece behind the seam. But where the border
    FOLDS on the seam -- a concave crease of the support, the piece behind the seam rising above the wall -- the edge is real,
    and is drawn again here.

    The piece behind the seam belongs to the SAME cell ( the edge is inside the cell ), so it is enough to look, among the
    `SUPPORT` faces of the other pieces of the cell, for one that lies on the same plane and holds the middle of the edge:
    found, the border goes on flat; not found, it folds."""
    from .Cell import SEAM, SUPPORT

    # the `SUPPORT` faces of each cell: ( piece, cut, unit normal, offset, vertices of the cycle )
    faces, scale = {}, 0.0
    for b, it in enumerate( items ):
        if b not in parts or not ( it.cid == SUPPORT ).any():
            continue
        scale = max( scale, float( np.ptp( it.vp, axis = 0 ).max() ) )
        for cyc, k in cell._faces_and_cuts_of( it ):
            if it.cid[ k ] != SUPPORT:
                continue
            pts = it.vp[ cyc ]
            nrm = np.cross( pts - pts.mean( axis = 0 ), np.roll( pts, -1, axis = 0 ) - pts.mean( axis = 0 ) ).sum( axis = 0 )
            nn = float( np.linalg.norm( nrm ) )
            if nn > 0:
                nrm = nrm / nn
                if nrm @ ( pts.mean( axis = 0 ) - it.vp.mean( axis = 0 ) ) < 0:     # outward ( the walk has no fixed orientation )
                    nrm = -nrm
                faces.setdefault( int( owners[ b ] ), [] ).append( ( b, k, nrm, float( nrm @ pts.mean( axis = 0 ) ), pts ) )
    tol = 1e-5 * max( scale, 1e-300 )

    def holds( pts, nrm, m ):
        """`m`, on the plane of the convex polygon `pts`, is in it ( up to `tol` )"""
        e = np.roll( pts, -1, axis = 0 ) - pts
        side = np.cross( e, m - pts ) @ nrm / np.maximum( np.linalg.norm( e, axis = 1 ), 1e-300 )
        return ( side >= -tol ).all() or ( side <= tol ).all()

    for b, it in enumerate( items ):
        if b not in parts:
            continue
        fake, ev, hide, dash = parts[ b ]
        if not len( ev ) or not hide.any():
            continue
        ec = cell._edge_cuts_of( it )
        ids = it.cid[ ec ]
        mine = { k: ( nrm, off ) for bb, k, nrm, off, _ in faces.get( int( owners[ b ] ), [] ) if bb == b }
        for e in np.nonzero( hide & ( ids == SEAM ).any( axis = 1 ) & ( ids == SUPPORT ).any( axis = 1 ) & ~fake[ ev ].any( axis = 1 ) )[ 0 ]:
            k = int( ec[ e ][ ids[ e ] == SUPPORT ][ 0 ] )
            if k not in mine:
                continue
            nrm, off = mine[ k ]
            m = it.vp[ ev[ e ] ].mean( axis = 0 )
            flat = any( bb != b and nrm @ n2 > 1 - 1e-6 and abs( off - o2 ) <= tol and holds( pts, n2, m )
                        for bb, _, n2, o2, pts in faces[ int( owners[ b ] ) ] )
            if not flat:
                hide[ e ] = False


def _ray_length( vps, parts ):
    """The length to give a truncated ray: a fraction of the REAL extent of the scene, taken
    over all the items at once ( a diagram must have the same stub everywhere ). The percentiles
    and not the min / max: a Voronoi vertex of three nearly aligned seeds is real but says
    nothing about the scale at which we are looking."""
    real = [ vps[ b ][ ~parts[ b ][ 0 ] ] for b in parts ]
    real = [ r for r in real if len( r ) ]
    if not real:
        return None                      # no real vertex anywhere: we keep the simplex as it is
    pts = np.concatenate( real, axis = 0 )
    lo, hi = np.percentile( pts, 5, axis = 0 ), np.percentile( pts, 95, axis = 0 )
    diag = float( np.linalg.norm( hi - lo ) )
    return 0.2 * diag if diag > 0 else None


def _truncated_rays( vp, fake, ev, stub ):
    """`vp` with its FAKE vertices brought back to `stub` from the real vertex they start from -- read from the
    edges: the one that joins a real vertex to a fake one IS the ray. A fake vertex without an
    identifiable ray is brought back relative to the center of the real vertices."""
    if stub is None or not fake.any():
        return vp

    src = np.full( len( vp ), -1 )
    for a, b in ev:
        if fake[ a ] and not fake[ b ]: src[ a ] = b
        if fake[ b ] and not fake[ a ]: src[ b ] = a

    real = vp[ ~fake ]
    centre = real.mean( axis = 0 ) if len( real ) else vp.mean( axis = 0 )

    res = np.array( vp, dtype = float )
    for f in np.nonzero( fake )[ 0 ]:
        origin = vp[ src[ f ] ] if src[ f ] >= 0 else centre
        ray = vp[ f ] - origin
        n = float( np.linalg.norm( ray ) )
        if n > 1e-12:
            res[ f ] = origin + ray / n * stub
    return res
