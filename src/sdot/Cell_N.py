"""A SIMPLE polytope in dimension >= 3: the connectivity is carried by the vertices.

Each vertex is on EXACTLY `d` planes ( general position ), and names its `d` cuts
( `vertex_cuts`, indices into `cut_ids`, increasing ) and its `d` neighbors ( `vertex_nbrs` ) -- the
neighbor `r` being "opposite" cut `r`: on the other side of the edge carried by the `d - 1`
other cuts. This bijection is what makes the faces and edges readable without searching for anything.
The planes are read back off the vertices of each face. Kernel side: `cell/LocalN.h`.
"""

import numpy as np
from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar

from .Cell import Cell, Item
from .CellScratch import words_of


class Cell_N( Cell ):
    vertex_positions : RealTensor[ "num_vertex", "dim_axis" ]
    vertex_cuts      : IntTensor [ "num_vertex", "dim_axis", dict( size = 32 ) ]
    vertex_nbrs      : IntTensor [ "num_vertex", "dim_axis", dict( size = 32 ) ]
    cut_ids          : IntTensor [ "num_cut", dict( size = 32 ) ]

    num_vertex       : Axis[ "nb_vertices" ]
    num_cut          : Axis[ "nb_cuts" ]
    dim_axis         : Axis[ "nb_dims" ]
    nb_vertices      : ShapeVar
    nb_cuts          : ShapeVar
    nb_dims          : CtShapeVar

    default_nb_dims = 3
    _GEOMETRY = ( "vertex_positions", "vertex_cuts", "vertex_nbrs", "cut_ids" )

    # ---- what the kernel asks for --------------------------------------------------------------

    def scratch_words( self, cap, fp_size ):
        """`LocalN::words_for( cap )`: `cap` vertices AND `cap` cuts, with the temporaries of the
        cut, of the compaction and of the triangulation"""
        d = self.dim
        return ( 4 * d + 5 ) * words_of( fp_size // 8, cap ) + ( 5 * d + 10 ) * words_of( 4, cap )

    def init_capacity( self ):
        return 2 ** self.dim

    def _cut_capacities( self ):
        """The most a cut can produce: one more cut, and for the vertices, in 3D,
        Euler on a simple polytope ( `V = 2 F - 4` ), hence `2 ( nc + 1 ) - 4`. Beyond that there is
        no linear bound in `F`; each new vertex is on a crossing edge, and a
        simple polytope has `nv d / 2` edges."""
        d = self.dim
        cap_c = self._cap_c() + 1
        cap_v = 2 * cap_c - 4 if d == 3 else self._cap_v() * ( 1 + d // 2 )
        return max( cap_v, self.init_capacity() ), cap_c

    # ---- reading an item, and what derives from it -------------------------------------------------

    def _item( self, b ):
        d = self.dim
        if self.vertex_positions.raw is None:              # never written: not a vertex
            return Item( np.zeros( ( 0, d ) ), np.zeros( 0, int ), vc = np.zeros( ( 0, d ), int ), vn = np.zeros( ( 0, d ), int ) )
        nv, nc = self._count( self.nb_vertices, b ), self._count( self.nb_cuts, b )
        return Item( self._rows( self.vertex_positions, b, nv ).astype( float ),
                     self._rows( self.cut_ids, b, nc ).astype( int ),
                     vc = self._rows( self.vertex_cuts, b, nv ).astype( int ),
                     vn = self._rows( self.vertex_nbrs, b, nv ).astype( int ) )

    def _vertex_cut_indices_of( self, it ):
        return it.vc

    def _edges_of( self, it ):
        """one edge per pair `( a, vn[ a, r ] )`, each seen once"""
        nv, d = it.vn.shape
        a = np.repeat( np.arange( nv ), d )
        b = it.vn.reshape( -1 )
        keep = a < b
        return np.stack( [ a[ keep ], b[ keep ] ], axis = 1 )

    def _edge_cuts_of( self, it ):
        """the edge opposite cut `r` of vertex `a` is carried by its other `d - 1` cuts"""
        nv, d = it.vn.shape
        res = []
        for a in range( nv ):
            for r in range( d ):
                if a < int( it.vn[ a, r ] ):
                    res.append( [ int( it.vc[ a, q ] ) for q in range( d ) if q != r ] )
        return np.asarray( res, int ).reshape( -1, d - 1 )

    def _faces_of( self, it ):
        """In 3D only: for each cut, the walk from neighbor to neighbor on its face -- from `v`
        we go to `vn[ v, r ]` for the two `r` whose cut `vc[ v, r ]` is NOT the face."""
        return [ cycle for cycle, _ in self._faces_and_cuts_of( it ) ]

    def _face_cuts_of( self, it ):
        """the cut ( index into `cut_ids` ) of each face of `_faces_of`"""
        return [ k for _, k in self._faces_and_cuts_of( it ) ]

    def _faces_and_cuts_of( self, it ):
        nv, d = it.vc.shape
        if d != 3:
            return []
        res = []
        for k in range( len( it.cid ) ):
            on = np.nonzero( ( it.vc == k ).any( axis = 1 ) )[ 0 ]
            if len( on ) < 3:
                continue
            start = int( on[ 0 ] )
            cycle, prev, cur = [ start ], -1, start
            while True:
                nxt = [ int( it.vn[ cur, r ] ) for r in range( d ) if it.vc[ cur, r ] != k ]
                nxt = [ w for w in nxt if w != prev ]
                if not nxt or nxt[ 0 ] == start or len( cycle ) > nv:
                    break
                cycle.append( nxt[ 0 ] )
                prev, cur = cur, nxt[ 0 ]
            if len( cycle ) >= 3:
                res.append( ( cycle, k ) )
        return res

    def _planes_of( self, it ):
        """The null vector of the space spanned by the vertices of each face -- by SVD, which depends
        neither on the order nor on coincident vertices -- oriented outward by the vertex FARTHEST
        from the plane ( not the first one that comes, which may coincide with the face to 1e-16 )."""
        nv, d = it.vp.shape
        dirs, offs = np.zeros( ( len( it.cid ), d ) ), np.zeros( len( it.cid ) )
        for k in range( len( it.cid ) ):
            on = np.nonzero( ( it.vc == k ).any( axis = 1 ) )[ 0 ]
            if len( on ) < d:
                continue
            pts = it.vp[ on ]
            _, sv, vt = np.linalg.svd( pts - pts.mean( axis = 0 ), full_matrices = True )
            if len( sv ) >= d and sv[ d - 2 ] <= 1e-9 * max( sv[ 0 ], 1e-300 ):
                continue                                    # not a hyperplane: degenerate face
            n = vt[ -1 ]
            off = float( n @ pts[ 0 ] )
            others = np.setdiff1d( np.arange( nv ), on )
            if len( others ):
                sd = it.vp[ others ] @ n - off
                if sd[ np.argmax( np.abs( sd ) ) ] > 0:
                    n, off = -n, -off
            dirs[ k ], offs[ k ] = n, off
        return dirs, offs
