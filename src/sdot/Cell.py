"""A convex polytope, and what all the dimension regimes have in common.

`Cell( nb_dims, ... )` builds the class that suits the dimension -- `Cell_1` ( a segment ),
`Cell_2` ( a polygon ), `Cell_N` ( a simple polytope in dimension >= 3 ) -- each in its own
file, with exactly its tensors. This file only contains the CONTRACT and the common trunk :

  * what is stored is the CONVENIENT format ( `vertex_positions [ nv, d ]` in the caller's
    float, `cut_ids [ nc ]` ), transferred to the kernel's form at each call
    ( `cell/Local*.h`, in the kernel's float -- `kernel_dtype`, `float32` by default ) ;
  * the operations are kernels on a scratch sized by the host ( `CellScratch` ) ;
  * the "convenient" tensors that are not stored -- planes, edges, faces, bounding -- are
    DERIVED host side, each regime saying how to read them off its own tensors.

`cut_ids` carries the identity of the cuts : the index of the facing seed for a bisector, a
negative integer otherwise ( `cell/Ids.h` ). An `INFINITE` wall that still carries a vertex says that the
cell is not bounded.
"""

import os

import numpy as np
import loom
from loom.compilation.FfiCode import FfiCode
from loom.drivers.driver import driver
from loom.tensor import Axis, RealTensor, ShapeVar, Tensor
from loom.util import Aggregate

from .CellScratch import CellScratch, fp_size
from . import cell_viz

# the cut identifiers that do not designate a seed -- see `cell/Ids.h`, which is authoritative
INFINITE = -2 ** 31          # a wall of the replacement simplex of an unbounded cell
PIECE    = -2 ** 31 + 1      # a cutting plane added by a distribution
BOUNDARY = -1                # "not a seed", with no further precision ( = `domain_id( 0 )` )

# THE KERNEL FLOAT. The geometry is cut in `float32` by default -- this is what fits eight
# vertices in a register ( `cell/Engine2Reg.h` ) -- and everything derived from it ( a measure, a
# gradient ) is computed in the caller's float. `SDOT_KTYPE=FP64` changes the default ;
# `kernel_dtype = ...` changes it for one cell ( or one diagram ).
DEFAULT_KERNEL_DTYPE = os.environ.get( "SDOT_KTYPE", "FP32" )


def set_kernel_dtype( dtype ):
    """The kernel float for cells ( and diagrams ) built FROM NOW ON without an explicit
    `kernel_dtype` : `"FP32"` ( the default ) or `"FP64"`. Returns the previous setting."""
    global DEFAULT_KERNEL_DTYPE
    previous = DEFAULT_KERNEL_DTYPE
    DEFAULT_KERNEL_DTYPE = dtype
    return previous


def kernel_dtype():
    return DEFAULT_KERNEL_DTYPE


def cell_class_for( nb_dims ):
    from .Cell_1 import Cell_1
    from .Cell_2 import Cell_2
    from .Cell_N import Cell_N
    return { 1: Cell_1, 2: Cell_2 }.get( int( nb_dims ), Cell_N )


class Item:
    """The geometry of ONE item of a cell ( batched or not ), in numpy : `vp [ nv, d ]`,
    `cid [ nc ]`, and what the regime adds to it ( `vc` / `vn` for `Cell_N` )."""
    def __init__( self, vp, cid, **extra ):
        self.vp = vp
        self.cid = cid
        self.__dict__.update( extra )

    @property
    def nb_vertices( self ):
        return len( self.vp )


class Cell( Aggregate ):
    # ---- what a regime must provide ---------------------------------------------------------------
    #
    #   default_nb_dims      the dimension when the class is built without being given one
    #   _GEOMETRY            the tensors that a cut rewrites
    #   scratch_words( cap, fp_size )   the same formula as `Local*::words_for`
    #   init_capacity()      the room of a hypercube
    #   _cut_capacities()    `( vertices, cuts )` : at most what a cut can produce
    #   _item( b )           the geometry of item `b`, read off its tensors
    #   _vertex_cut_indices_of( it ), _edges_of( it ), _edge_cuts_of( it ), _faces_of( it ),
    #   _planes_of( it )     the derived tensors, read off an `Item`

    def __new__( cls, nb_dims = None, *args, **kwargs ):
        if cls is Cell:
            if nb_dims is None:
                raise TypeError( "Cell( nb_dims, ... ) : the dimension decides the class" )
            cls = cell_class_for( nb_dims )
        return super().__new__( cls )

    def __init__( self, nb_dims = None, init_as_unbounded = True, batch_axes = None, kernel_dtype = None ):
        nb_dims = int( nb_dims if nb_dims is not None else self.default_nb_dims )
        self._kernel_dtype = kernel_dtype or DEFAULT_KERNEL_DTYPE
        self.__base_init__( nb_dims = nb_dims, batch_axes = batch_axes )
        if init_as_unbounded:
            self.init_as_unbounded()

    @property
    def kernel_dtype( self ):
        return self._kernel_dtype

    @property
    def dim( self ):
        return int( self.nb_dims.value )

    @classmethod
    def make_hypercube( cls, nb_dims, origin = None, axes = None, cut_id = BOUNDARY, batch_axes = None, **kwargs ):
        res = cls( nb_dims, init_as_unbounded = False, batch_axes = batch_axes, **kwargs )
        res.init_as_hypercube( origin, axes, cut_id )
        return res

    @classmethod
    def make_unbounded( cls, nb_dims, batch_axes = None, **kwargs ):
        return cls( nb_dims, batch_axes = batch_axes, **kwargs )

    def _empty_like_me( self ):
        """A cell of the same regime, same batch, same kernel, without geometry."""
        return type( self )( self.dim, init_as_unbounded = False, batch_axes = self.batch_axes or None,
                             kernel_dtype = self._kernel_dtype )

    # ---- the capacities, and the scratch of a call ---------------------------------------------

    def _cap_v( self ):
        """The capacity in vertices we start from : the count when the host knows it, the allocated
        capacity otherwise ( a count written by a kernel is a device value under a trace )."""
        n = self.nb_vertices.static_count()
        if n is not None:
            return max( int( n ), 1 )
        return int( self.nb_vertices.allocated_capacity() or self.init_capacity() )

    def _cap_c( self ):
        n = self.nb_cuts.static_count()
        if n is not None:
            return max( int( n ), 1 )
        return int( self.nb_cuts.allocated_capacity() or self.init_capacity() )

    def _call_scratch( self, cap ):
        """the scratch of a call on THIS cell : one row per item, `cap` vertices.
        Already marked `loom.scratch`, so it is passed under its name and nothing else."""
        return CellScratch.for_call( self.scratch_words( cap, fp_size( self._kernel_dtype ) ),
                                     self._kernel_dtype, batch_axes = self.batch_axes or None )

    # ---- the kernels ------------------------------------------------------------------------------

    def init_as_unbounded( self, batch_axes = None ):
        """"All of space", represented by a SIMPLEX whose walls are marked `INFINITE`.

        These planes are not real cuts : they are stopgaps, and their offsets are
        made up. It is `cut` that pushes them back as it goes, until they no longer change
        anything about the current cut ( `Local2::grow_for` ) ; the cell becomes bounded again the day
        no vertex carries one any more.
        """
        if batch_axes is not None:
            self.apply_batch_axes( batch_axes )
        cap = self.dim + 1
        loom.ffi_call(
            "init_as_unbounded",
            FfiCode.per_item( code = "outputs.cell( batch_index ).init_as_unbounded( scratch.pool( batch_index ) );" ),
            cell = loom.out( self, capacities = { "nb_vertices": cap, "nb_cuts": cap } ),
            pool = self._call_scratch( cap ),
        )

    def init_as_hypercube( self, origin = None, axes = None, cut_id = BOUNDARY, batch_axes = None ):
        """the parallelotope `origin + sum_j t_j axes[ j ]`, `t` in `[ 0, 1 ]^d` -- the unit cube
        by default. All the cuts carry `cut_id`."""
        if batch_axes is not None:
            self.apply_batch_axes( batch_axes )

        d = self.dim
        origin = RealTensor[ self.dim_axis ]( np.zeros( d ) if origin is None else origin )
        axes = RealTensor[ self.dim_axis, Axis( ShapeVar( d ), name = "num_axis" ) ]( np.eye( d ) if axes is None else axes )

        cap = self.init_capacity()
        loom.ffi_call(
            "init_as_hypercube",
            FfiCode.per_item( code = "outputs.cell( batch_index ).init_as_hypercube( scratch.pool( batch_index ), inputs.origin, inputs.axes, inputs.cut_id );" ),
            cell = loom.out( self, capacities = { "nb_vertices": cap, "nb_cuts": cap } ),
            pool = self._call_scratch( cap ),
            cut_id = cut_id,
            origin = origin,
            axes = axes,
        )

    def cut( self, direction, offset, cut_id = BOUNDARY ):
        """Intersects the cell with the half-space `direction . x <= offset`, IN PLACE.

        `direction` need not be normalized : `offset` is the dot product it is
        compared against as is. Since the inputs and outputs of a `driver.call` are disjoint, the
        kernel writes into a NEW cell and the in-place update is only a rebinding. The
        room the cut requires is BOUNDED in advance ( `_cut_capacities` ) : no second pass.
        """
        direction = RealTensor[ self.dim_axis ]( direction )
        offset = RealTensor[ () ]( offset )

        cap_v, cap_c = self._cut_capacities()
        res = self._empty_like_me()
        loom.ffi_call(
            "cut",
            FfiCode.per_item( code = "inputs.cell( batch_index ).cut( outputs.res( batch_index ), scratch.pool( batch_index ), "
                                     "inputs.direction, inputs.offset, inputs.cut_id );" ),
            cut_id = cut_id,
            direction = direction,
            offset = offset,
            cell = self,
            res = loom.out( res, capacities = { "nb_vertices": cap_v, "nb_cuts": cap_c } ),
            pool = self._call_scratch( max( cap_v, cap_c ) ),
        )
        self._adopt_geometry( res )
        return self

    def _adopt_geometry( self, other ):
        """Takes over on `self` what the kernel has just written into `other` : the VALUES ( the
        tensor storage, the counts ), not the `Attribute` objects, whose identity must
        survive the cut."""
        for name in self._GEOMETRY:
            getattr( self, name ).set( getattr( other, name ) )
        # a count written by a kernel is a DEVICE value : `set_count`, not `set`
        for name in ( "nb_vertices", "nb_cuts" ):
            getattr( self, name ).set_count( getattr( other, name ).raw )

    @property
    def measure( self ) -> Tensor:
        """The measure of the cell : length, area, volume -- in the caller's float, whatever
        the kernel's is. `TF::max` for an unbounded cell. Differentiable with respect to
        `vertex_positions`."""
        res = RealTensor[ tuple( self.batch_axes ) ]()
        loom.ffi_call(
            "measure",
            FfiCode.per_item( code = "inputs.cell( batch_index ).measure( outputs.res( batch_index ), scratch.pool( batch_index ) );" ),
            FfiCode.per_item( "inputs.cell( batch_index ).measure_bwd( outputs.res( batch_index ), grad_of_outputs.res( batch_index ), "
                              "grad_of_inputs.cell( batch_index ).vertex_positions, scratch.pool( batch_index ) );" ),
            cell = self,
            res = loom.out( res ),
            pool = self._call_scratch( max( self._cap_v(), self._cap_c() ) ),
        )
        return res

    # ---- reading an item ---------------------------------------------------------------------------

    @property
    def nb_items( self ):
        return int( np.prod( [ int( ax.max ) for ax in self.batch_axes ] ) ) if self.batch_axes else 1

    def _count( self, shape_var, b ):
        """the count of item `b` -- a count written by a kernel has one per item, a count
        known to the host is the same for all"""
        v = np.atleast_1d( np.asarray( shape_var.value ) ).reshape( -1 ).astype( int )
        return int( v[ b ] if v.size == self.nb_items else v[ 0 ] )

    def _rows( self, tensor, b, count ):
        """the first `count` rows of item `b` of tensor `tensor` ( `[ items..., cap, ... ]` ).
        The batch axes are flattened, and NOTHING else : on a GPU the batch is padded to
        the device alignment (`Device.batch_alignment`), item `b` is at row `b` of a
        buffer that has more than `nb_items`."""
        raw = np.asarray( tensor.raw )
        return raw.reshape( ( -1, ) + raw.shape[ len( self.batch_axes ) : ] )[ b ][ : count ]

    def _items( self ):
        return [ self._item( b ) for b in range( self.nb_items ) ]

    def _per_item( self, of_item ):
        """`of_item( item )` for each item ; the value itself if the cell is not batched."""
        if not self.batch_axes:
            return of_item( self._item( 0 ) )
        return [ of_item( it ) for it in self._items() ]

    def vertices( self, item = 0 ):
        """`vertex_positions` of item `item`, in numpy `[ nb_vertices, d ]` -- without the padding
        of a batch."""
        return self._item( item ).vp

    # ---- the "convenient" tensors, derived --------------------------------------------------------

    def _infinite_cuts_of( self, it ):
        """the mask of the `INFINITE` cuts that still carry a vertex"""
        alive = np.zeros( len( it.cid ), bool )
        vc = self._vertex_cut_indices_of( it )
        if len( vc ):
            alive[ np.unique( vc.reshape( -1 ) ) ] = True
        return ( it.cid == INFINITE ) & alive

    @property
    def is_bounded( self ):
        """True if no `INFINITE` wall carries a vertex any more."""
        return self._per_item( lambda it: not self._infinite_cuts_of( it ).any() )

    @property
    def vertex_cut_indices( self ):
        """`[ nb_vertices, d ]` : the `d` cuts ( indices into `cut_ids` ) of which each vertex is the corner."""
        return self._per_item( self._vertex_cut_indices_of )

    @property
    def edges( self ):
        """`[ nb_edges, 2 ]` : the edges, as vertex indices."""
        return self._per_item( self._edges_of )

    @property
    def faces( self ):
        """A list of cycles of vertex indices, one face per cut that carries one ( 2D and 3D )."""
        return self._per_item( self._faces_of )

    @property
    def cut_planes( self ):
        """`( directions[ nb_cuts, d ], offsets[ nb_cuts ] )`, re-read off the geometry -- the UNIT
        outward normal of each face and its offset. A cut without a vertex has a zero direction."""
        return self._per_item( self._planes_of )

    @property
    def cut_directions( self ):
        return self._per_item( lambda it: self._planes_of( it )[ 0 ] )

    @property
    def cut_offsets( self ):
        return self._per_item( lambda it: self._planes_of( it )[ 1 ] )

    # ---- display ----------------------------------------------------------------------------------

    def add_to_viz( self, viz, color = None, opacity = 1.0, faces = True, edges = True, points = False ):
        """Draws itself into a `Visualizer` ( see `sdot.viz.Visualizer`, and `cell_viz` for what an
        unbounded cell drops ). Each item takes its color from its RANK IN THE BATCH,
        on a block reserved in advance : the color of a cell of a diagram says WHICH seed."""
        return cell_viz.add_to_viz( self, viz, color, opacity, faces, edges, points )
