"""A POLYGON: the vertices in cyclic order are the whole geometry.

The invariant: CUT `i` CARRIES THE EDGE `[ v_i, v_i+1 ]`, so vertex `i` is the corner of cuts
`i-1` and `i`, and `nb_cuts == nb_vertices`. Nothing else is stored -- the plane of a cut is read
back from its edge ( `_planes_of` ). Kernel side: `cell/Local2.h`, and on CPU inside a diagram
the register kernel `cell/Engine2Reg.h`.
"""

import numpy as np
from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar

from .Cell import Cell, Item
from .CellScratch import words_of


class Cell_2( Cell ):
    vertex_positions : RealTensor[ "num_vertex", "dim_axis" ]
    cut_ids          : IntTensor [ "num_cut", dict( size = 32 ) ]

    num_vertex       : Axis[ "nb_vertices" ]
    num_cut          : Axis[ "nb_cuts" ]
    dim_axis         : Axis[ "nb_dims" ]
    nb_vertices      : ShapeVar
    nb_cuts          : ShapeVar
    nb_dims          : CtShapeVar

    default_nb_dims = 2
    _GEOMETRY = ( "vertex_positions", "cut_ids" )

    # ---- what the kernel asks for --------------------------------------------------------------

    @staticmethod
    def scratch_words( cap, fp_size ):
        """`Local2::words_for( cap )`: the vertices, the identifiers, the planes, the temporaries"""
        return 8 * words_of( fp_size // 8, cap ) + words_of( 4, cap )

    def init_capacity( self ):
        return 4

    def _cut_capacities( self ):
        """a half-plane removes a piece of the polygon and adds an edge: at most ONE more vertex,
        and as many cuts as vertices"""
        cap = self._cap_v() + 1
        return cap, cap

    # ---- reading an item, and what derives from it -------------------------------------------------

    def _item( self, b ):
        if self.vertex_positions.raw is None:              # never written: not a vertex
            return Item( np.zeros( ( 0, 2 ) ), np.zeros( 0, int ) )
        nv, nc = self._count( self.nb_vertices, b ), self._count( self.nb_cuts, b )
        return Item( self._rows( self.vertex_positions, b, nv ).astype( float ),
                     self._rows( self.cut_ids, b, nc ).astype( int ) )

    def _vertex_cut_indices_of( self, it ):
        """vertex `i` is the corner of cuts `i-1` and `i`"""
        nv = it.nb_vertices
        i = np.arange( nv )
        return np.stack( [ ( i - 1 ) % nv, i ], axis = 1 ) if nv else np.zeros( ( 0, 2 ), int )

    def _edges_of( self, it ):
        nv = it.nb_vertices
        i = np.arange( nv )
        return np.stack( [ i, ( i + 1 ) % nv ], axis = 1 ) if nv else np.zeros( ( 0, 2 ), int )

    def _edge_cuts_of( self, it ):
        """edge `i` is carried by cut `i`"""
        return np.arange( it.nb_vertices )[ :, None ]

    def _faces_of( self, it ):
        """a single face: the polygon itself"""
        return [ list( range( it.nb_vertices ) ) ] if it.nb_vertices >= 3 else []

    def _planes_of( self, it ):
        """the unit outward normal of the edge `[ v_i, v_i+1 ]` -- `( dy, -dx )` for a counter-clockwise
        polygon -- and its offset read on `v_i`"""
        nv = it.nb_vertices
        dirs, offs = np.zeros( ( len( it.cid ), 2 ) ), np.zeros( len( it.cid ) )
        if nv == 0:
            return dirs, offs
        e = it.vp[ ( np.arange( nv ) + 1 ) % nv ] - it.vp
        n = np.stack( [ e[ :, 1 ], - e[ :, 0 ] ], axis = 1 )
        lng = np.linalg.norm( n, axis = 1 )
        ok = lng > 0
        dirs[ ok ] = n[ ok ] / lng[ ok, None ]
        offs = ( dirs * it.vp ).sum( axis = 1 )
        return dirs, offs
