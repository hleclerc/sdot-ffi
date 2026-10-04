"""The seeds in BSP TREE ORDER: `sorted_positions` / `sorted_weights` are the seeds
permuted the way `tree.seed_indices` arranges them, so that a leaf reads in one piece, and
each cell is cut only by the seeds the tree could not rule out ( `AaBsp.py`,
`cell/Providers.h::ProviderBsp` ).

Everything the kernel handles is in that order -- ranks, cut identifiers,
gradients on the seeds -- and this is where we translate: `positions` / `weights` give the seeds back
in the user's order, measures and cells leave the kernel already at their index
( `user_id( k )` ), and the gathering `sorted = positions[ seed_indices ]` is an operation of the
backend, DIFFERENTIABLE: a derivative with respect to `sorted_positions` comes back to `positions` on its own.

On a CUDA card, in 2D, the tree is built by the card ( `AaBsp._init_on_card` ): the order and its inverse are then the
card's arrays ( tracers under a `jit`, positions traced or not ), and the gathers run there.

Changing the WEIGHTS does not change the tree, only the affine majorant each node carries
( `refresh_weight_majorants` ): this is what makes a diagram reusable from one step to the next
of a fit ( `SdotPlanNd` ). Changing the POSITIONS rebuilds it.

THE MEMORY ( `memo_nbrs [ n, K ]`, `memo_counts [ n ]`, in tree ranks ): the neighbors of
each cell at the last `measures`, which the provider proposes first to the next one
( `cell/Providers.h`, `MEMO` ). Written by the `measures` kernel into two fresh tensors,
taken back here after the call ( the inputs and outputs of a call are disjoint ) -- except under a
trace, where what comes out is a tracer: the previous memory stays, it is always valid. Erased
with the tree, when the positions change. `memory = 0` does not name it: `NoneTensor` on the C++ side, and the
ordinary path at compile time.
"""

import numpy as np

import loom
from loom.drivers.driver import driver
from loom.compilation.FfiCode import FfiCode
from loom.tensor import Axis, IntTensor, RealTensor, ShapeVar
from loom.util import Aggregate

from .AaBsp import AaBsp, take_rows
from .PowerDiagram import PowerDiagram


class PowerDiagram_Bsp( PowerDiagram ):
    tree             : AaBsp

    sorted_positions : RealTensor[ "num_point", "dim" ]
    sorted_weights   : RealTensor[ "num_point" ]

    memo_nbrs        : IntTensor[ "num_point", "num_memo", dict( size = 32 ) ]
    memo_counts      : IntTensor[ "num_point", dict( size = 32 ) ]
    num_memo         : Axis[ "nb_memo" ]
    nb_memo          : ShapeVar

    def _init_seeds( self, positions, weights, accelerator ):
        tree = accelerator if isinstance( accelerator, AaBsp ) else AaBsp( positions, weights )
        n = int( tree.nb_bsp_seeds.value )
        if n != int( positions.shape[ 0 ] ):
            raise ValueError( f"the accelerator was built on { n } seeds, this diagram has { int( positions.shape[ 0 ] ) }" )
        # the order and its inverse: host arrays for a tree built on the host, the card's ( maybe traced ) for a tree built
        # there ( `AaBsp._init_on_card` ) -- the gathers below then run on the card too
        self._rank_of = tree.rank_of_seeds()
        if isinstance( self._rank_of, np.ndarray ):
            self._order = np.asarray( tree.seed_indices ).reshape( -1 ).astype( np.int64 )
        else:
            self._order = getattr( tree.seed_indices, "raw", tree.seed_indices ).reshape( -1 )
        res = { "tree": tree, "sorted_positions": self._gather( positions ) }
        if weights is not None:
            res[ "sorted_weights" ] = self._gather( weights )
            # a tree coming from outside may have been built on other weights, or none: its majorant
            # is redone on THESE ( built here, it already has them; so has a tree built on these very weights )
            if tree is accelerator and getattr( tree, "_majorant_weights", None ) is not weights:
                tree.refresh_weight_majorants( res[ "sorted_positions" ], res[ "sorted_weights" ] )
        # the memory starts empty ( no memories: the ordinary path, until the first `measures` )
        K = int( getattr( self, "_memory", 0 ) )
        res[ "nb_memo" ] = K
        if K > 0:
            res[ "memo_nbrs" ] = np.zeros( ( n, K ), dtype = np.int32 )
            res[ "memo_counts" ] = np.zeros( n, dtype = np.int32 )
        return res

    def _memo_for_call( self ):
        if not self.memo_counts.is_defined:
            return "0, 0", {}, None
        nbrs = IntTensor[ self.num_point, self.num_memo, dict( size = 32 ) ]()
        counts = IntTensor[ self.num_point, dict( size = 32 ) ]()
        # the role is carried by the value, so there is no list of paths to merge into
        # those of the call any more: they are arguments like the others, marked.
        return "outputs.memo_nbrs_out, outputs.memo_counts_out", dict(
            memo_nbrs_out = loom.out( nbrs ), memo_counts_out = loom.out( counts ) ), ( nbrs, counts )

    def _memo_after_call( self, produced ):
        if produced is None:
            return
        nbrs, counts = produced
        if driver.is_traced( counts.raw ):          # under a trace: keep the previous memories
            return
        self.memo_nbrs = nbrs.raw
        self.memo_counts = counts.raw

    # ---- the 2D cells of the card ( `include/sdot/gpu/Cell2D.cuh`, `Laplacian2D.cuh` ) ------------------------
    #
    # `measures` on a CUDA device, in 2D: one thread per cell, the cell in registers, the overflow redone by later
    # passes ( a warp per cell in shared, then in global memory ), the tree as aligned records in the kernel's float,
    # the float accuracy fixes of the old GPU campaign. It takes the call only where it computes the SAME thing: a
    # constant density ( none, or an `Image` whose values are all equal on exactly the box ), a box domain, no
    # neighbour memory -- and it is an ffi call like the others, with its own ADJOINT ( `measures_vjp`: the facets of
    # each cell, gathered ), so a traced or differentiated call takes it too. `use_card_cells = False` on a diagram,
    # or `SDOT_CARD_CELLS=0`, sends every call to the generic path ( the tests compare the two ).
    #
    # NOTHING IS READ BACK, NOTHING RUNS AGAIN: the fourth pass ( global memory ) works within a FIXED budget taken once
    # per call, `card_overflow_warps_for` slots of `card_max_vertices` vertices ( the per-cell limit ), the cells that
    # reach it going through the slots in successive batches ( `Cell2D.cuh::Overflow` ); the forward, the backward and
    # the card's Newton use the same scheme. Past `card_max_vertices` a cell is a `KernelFailure` naming its seed ( eager
    # and traced ), never a NaN.

    use_card_cells = True
    #: the hard limit of vertices per cell ( the size of a slot of the fourth pass )
    card_max_vertices = 1 << 15
    #: how many cells the fourth pass holds at once ( its slots; fewer when the slots would exceed `card_overflow_bytes` )
    card_overflow_warps = 64
    #: the most memory the fourth pass's slots take from the call's pool
    card_overflow_bytes = 256 << 20

    def _card_overflow_warps( self, variant ):
        """the slots of the fourth pass for this diagram and variant ( `card_overflow_warps_for` )"""
        return card_overflow_warps_for( variant, int( self.nb_points.value ), int( self.card_max_vertices ),
                                        int( self.card_overflow_warps ), int( self.card_overflow_bytes ) )

    def _card_failures( self ):
        """what the kernel's failure codes mean ( `Cell2D.cuh::Failure` )"""
        return { 1: ( "a cell of the power diagram has more than " + str( self.card_max_vertices ) + " vertices ( seed {value} ): "
                      "raise `card_max_vertices` on the diagram, or look at the input ( coincident seeds? )" ) }

    def _card_density( self ):
        """the constant density the card integrates, or `None` when the distribution is not a constant on the box"""
        if self.distribution is None:
            return 1.0
        from .distributions.Image import Image
        dist = self.distribution
        if not isinstance( dist, Image ):
            return None
        try:
            if int( np.prod( np.asarray( dist.shape.value ) ) ) > 64:     # reading the values is only cheap when they are few
                return None
            vals = np.asarray( dist.values, dtype = float ).reshape( -1 )
            geo = dist._grid_geometry()
            mi = np.asarray( self.box_min, dtype = float ).reshape( -1 )
            ma = np.asarray( self.box_max, dtype = float ).reshape( -1 )
        except Exception:                                  # a traced or differentiated image: the generic path
            return None
        if geo is None or vals.size == 0 or not np.all( vals == vals[ 0 ] ):
            return None
        d, frame, origin, lo, hi = geo
        if not np.array_equal( frame, np.eye( d ) ):
            return None
        scale = max( float( np.abs( np.concatenate( [ mi, ma ] ) ).max() ), 1.0 )
        if np.abs( origin + lo - mi ).max() > 1e-12 * scale or np.abs( origin + hi - ma ).max() > 1e-12 * scale:
            return None
        return float( vals[ 0 ] )

    def _card_variant( self ):
        """`( C++ variant, density )` of the dedicated kernel for this diagram, or `None` ( the generic path ) --
        see `card_variant_for` for what is chosen from the inputs"""
        import os
        if not self.use_card_cells or os.environ.get( "SDOT_CARD_CELLS", "1" ).lower() in ( "0", "no", "false", "off" ):
            return None
        if self.dim_count != 2 or self.memo_counts.is_defined:
            return None
        if not getattr( driver.device, "is_cuda_gpu", False ):
            return None
        if not self.box_min.is_defined or self.bnd_directions.is_defined:
            return None
        rho = self._card_density()
        if rho is None:
            return None
        from .CellScratch import fp_size
        variant = card_variant_for( fp_size( self._domain_cell().kernel_dtype ), int( self.nb_points.value ),
                                    int( self.tree.nb_bsp_nodes.value ) )
        return variant, rho

    def _card_call( self, name, variant, rho, facets = False, moments = False, with_vjp = True ):
        """the measures ( and per `facets` / `moments` the laplacian's CSR, the barycentres and costs ) on the card"""
        n = int( self.nb_points.value )
        res = RealTensor[ self.num_point ]()
        kwargs = dict( power_diagram = self, density = RealTensor( np.float64( rho ) ), res = loom.out( res ) )
        lap = mom = None
        if facets:
            lap = _CardLaplacian( nb_rows = n + 1, nb_points = n )
            kwargs[ "lap" ] = loom.out( lap, capacities = { "nb_nnz": card_nnz_capacity( n ) } )
        if moments:
            mom = _CardMoments( nb_points = n, nb_dims = 2 )
            kwargs[ "mom" ] = loom.out( mom )
        limits = f"{ int( self.card_max_vertices ) }, { self._card_overflow_warps( variant ) }"
        includes = [ "sdot/gpu/Laplacian2D.cuh" if ( facets or moments ) else "sdot/gpu/Cell2D.cuh" ]
        if facets or moments:
            out = " | ".join( [ "sdot::gpu2d::MEASURES" ] + [ "sdot::gpu2d::FACETS" ] * facets + [ "sdot::gpu2d::MOMENTS" ] * moments )
            fwd = ( f"sdot::gpu2d::cells<{ variant }, { out }>( queue, args.inputs.power_diagram, args.outputs.res, "
                    f"{ 'args.outputs.lap' if facets else '0' }, { 'args.outputs.mom' if moments else '0' }, args.errors, args.allocator, "
                    f"args.inputs.density, { limits } );" )
        else:
            fwd = ( f"sdot::gpu2d::measures<{ variant }>( queue, args.inputs.power_diagram, args.outputs.res, "
                    f"args.errors, args.allocator, args.inputs.density, { limits } );" )
        kernels = [ FfiCode.inline( fwd, includes = includes, allocator = True ) ]
        if with_vjp:
            kernels.append( FfiCode.inline(
                f"sdot::gpu2d::measures_vjp<{ variant }>( queue, args.inputs.power_diagram, args.grad_of_outputs.res, "
                "args.grad_of_inputs.power_diagram.sorted_positions, args.grad_of_inputs.power_diagram.sorted_weights, "
                f"args.errors, args.allocator, args.inputs.density, { limits } );",
                includes = includes, allocator = True ) )
        loom.ffi_call( name, *kernels, failures = self._card_failures(), **kwargs )
        return res, lap, mom

    def _measures_on_card( self ):
        v = self._card_variant()
        if v is None:
            return None
        return self._card_call( "power_diagram_measures_card_2d", *v )[ 0 ]

    def _card_cells( self, facets = True, moments = False ):
        """THE CELLS ON THE CARD, and what Newton makes of them ( a test and bench hook; `SdotPlanNd` calls the C++ of
        `Laplacian2D.cuh` directly ): a dict with `measures` ( user order ), and per the flags `row`, `col`, `val`, `dia`
        -- the laplacian's CSR IN RANKS ( the tree's order, `tree.seed_indices` maps a rank to a seed ), `nnz` --,
        `bary` and `cost` ( user order ). Not differentiable: the facets are what the adjoint is made of. `None` if the
        card does not take this diagram."""
        v = self._card_variant()
        if v is None:
            return None
        res, lap, mom = self._card_call( "power_diagram_cells_card_2d", *v, facets = facets, moments = moments, with_vjp = False )
        out = dict( measures = res )
        if lap is not None:
            nnz = int( np.asarray( lap.nb_nnz.value ).reshape( -1 )[ 0 ] )
            out.update( row = lap.row, col = lap.col, val = lap.val, dia = lap.dia, nnz = nnz )
        if mom is not None:
            out.update( bary = mom.bary, cost = mom.cost )
        return out

    def _gather( self, seeds ):
        """`seeds[ seed_indices ]`, through the backend: a tracer stays a tracer, and the derivative
        with respect to what we gather comes back through here on its own"""
        return take_rows( seeds, self._order )

    def _scatter( self, sorted_values ):
        """the inverse: what is stored in tree order, put back in the user's order"""
        return sorted_values[ self._rank_of ]

    # ---- what the user reads and writes ------------------------------------------------------------

    @property
    def positions( self ):
        """`[ n, d ]`, in the user's order -- a NEW tensor, gathered from the storage"""
        return RealTensor[ self.num_point, self.dim ]( self._scatter( self.sorted_positions.raw ) )

    @positions.setter
    def positions( self, positions ):
        """new positions: the tree is rebuilt on them ( so concrete values are needed )"""
        pos = positions if hasattr( positions, "shape" ) else np.asarray( positions, dtype = float )
        w = self.sorted_weights.raw[ self._rank_of ] if self.sorted_weights.is_defined else None
        for name, value in self._init_seeds( pos, w, None ).items():
            setattr( self, name, value )
        self._dom_cell = None

    @property
    def weights( self ):
        """`[ n ]` in the user's order, or an `Unbound` tensor ( `is_defined == False` )
        when the diagram carries none -- the same thing `PowerDiagram_Plain` stores"""
        res = RealTensor[ self.num_point ]()
        if self.sorted_weights.is_defined:
            res.set( self._scatter( self.sorted_weights.raw ) )
        return res

    @weights.setter
    def weights( self, weights ):
        """new weights: the seeds do not move, nor does the tree -- only the affine majorant
        of the weights of each node is redone, in one kernel over the nodes"""
        self.sorted_weights = self._gather( weights )
        self.tree.refresh_weight_majorants( self.sorted_positions, self.sorted_weights )

    def _ranks_of_items( self ):
        return self._rank_of

    # ---- what the `SdotPlanNd` solver writes -------------------------------------------------------------

    def _solver_weights_call( self ):
        """`( C++ expression of the diagram with WRITABLE weights, kwargs of the call, what has to be
        taken back afterwards )`: the sorted weights and the tree majorants are OUTPUTS of the call
        ( `PowerDiagram_Bsp.h::with_weights` ), which `_solver_weights_after` adopts"""
        sw = RealTensor[ self.num_point ]()
        wa = RealTensor[ self.tree.num_bsp_node, self.tree.dim ]()
        wb = RealTensor[ self.tree.num_bsp_node ]()
        args = dict( sorted_weights_out = sw, node_wa_out = wa, node_wb_out = wb )
        expr = ( "inputs.power_diagram.with_weights( outputs.sorted_weights_out, outputs.node_wa_out, "
                 "outputs.node_wb_out" )
        # the memory too ( `memory > 0` ): the solver redoes it at each sweep, in two fresh
        # outputs that it initializes from the previous memories
        if self.memo_counts.is_defined:
            nbrs = IntTensor[ self.num_point, self.num_memo, dict( size = 32 ) ]()
            counts = IntTensor[ self.num_point, dict( size = 32 ) ]()
            args.update( memo_nbrs_out = nbrs, memo_counts_out = counts )
            expr += ", outputs.memo_nbrs_out, outputs.memo_counts_out"
        # the markers carry the role: there is no side list of paths any more.
        return ( expr + " )", { n: loom.out( t ) for n, t in args.items() }, args )

    def _solver_weights_after( self, produced ):
        # the TENSORS, whose storage is adopted as it is ( a buffer would go through `driver.array`, which brings a concrete
        # device array back to the host )
        self.sorted_weights = produced[ "sorted_weights_out" ]
        self.tree.node_wa = produced[ "node_wa_out" ]
        self.tree.node_wb = produced[ "node_wb_out" ]
        if "memo_counts_out" in produced:
            self.memo_nbrs = produced[ "memo_nbrs_out" ].raw
            self.memo_counts = produced[ "memo_counts_out" ].raw

    def _grad_seeds_expr( self ):
        return "grad_of_inputs.power_diagram.sorted_positions, grad_of_inputs.power_diagram.sorted_weights"


# ---- the card's variants and work buffers ( see `PowerDiagram_Bsp._card_variant` ) ---------------------------------

def card_variant_for( kernel_fp_size, nb_seeds, nb_nodes ):
    """THE C++ VARIANT of the 2D card kernel, chosen from the inputs ( `sdot::gpu2d::Variant< TK, TR, TN, MAX_HEIGHT >` ):

      * `TK` the kernel's float ( `kernel_dtype` );
      * `TR` a rank / cut identifier: `int` while the seeds and the four box sides fit in it, `long long` beyond;
      * `TN` a node index: `int` while the nodes do ( a perfect tree of depth 31 ), `long long` beyond;
      * `MAX_HEIGHT` the walk's stack, one slot per height ( and a bit per height in its mask ): 32, or 64 past a depth
        of 32 -- no depth limit below the 2^64 nodes that no memory holds anyway.

    The depth is that of the perfect binary tree of `nb_nodes = 2^depth - 1` nodes ( `AaBsp`, preorder )."""
    depth = int( nb_nodes ).bit_length()
    if depth > 64:
        raise ValueError( f"a BSP tree of depth { depth }: past what a 64-bit node index addresses" )
    tk = "float" if int( kernel_fp_size ) == 32 else "double"
    tr = "int" if int( nb_seeds ) <= 2 ** 31 - 9 else "long long"
    tn = "int" if int( nb_nodes ) <= 2 ** 31 - 1 else "long long"
    height = 32 if depth <= 32 else 64
    return f"sdot::gpu2d::Variant<{ tk }, { tr }, { tn }, { height }>"


def card_overflow_warps_for( variant, nb_seeds, max_vertices, warps = 64, max_bytes = 256 << 20 ):
    """THE SLOTS OF THE FOURTH PASS ( `Cell2D.cuh::Overflow` ): `warps` cells at once, fewer when `warps` slots of the
    per-cell limit would take more than `max_bytes` of the call's pool, at least one block of four. A slot holds
    `min( max_vertices, n + 4 )` vertices ( a cell of `n` seeds in a box has at most `n + 3` ), each five reals of the
    kernel and two cut identifiers ( `WarpCell::bytes_for` ); the cells past the slots wait for one to be free, so the
    number of slots is a matter of speed on the rare cells that get there, never of correctness."""
    tk, tr = variant.split( "<" )[ 1 ].split( "," )[ :2 ]
    per_vertex = 5 * ( 4 if tk.strip() == "float" else 8 ) + 2 * ( 4 if tr.strip() == "int" else 8 )
    cap = max( 4, min( int( max_vertices ), int( nb_seeds ) + 4 ) )
    fit = int( max_bytes ) // ( cap * per_vertex )
    return int( max( 4, min( int( warps ), fit ) ) )


def card_nnz_capacity( nb_seeds ):
    """the first guess of the laplacian's entries: a planar graph has at most `3 n - 6` edges, each in two rows ( a
    float topology may add a few slivers: the margin, and loom grows it if it was not enough )"""
    return 6 * int( nb_seeds ) + 1024


class _CardLaplacian( Aggregate ):
    """the laplacian of the Laguerre graph in CSR, in RANKS ( `Laplacian2D.cuh` ): `val` are the `c_ij > 0`, `dia` the
    row sums; `nb_nnz` is written by the kernel ( capacity: `card_nnz_capacity` )"""
    row       : IntTensor[ "num_row", dict( size = 64 ) ]
    col       : IntTensor[ "num_nnz", dict( size = 64 ) ]
    val       : RealTensor[ "num_nnz" ]
    dia       : RealTensor[ "num_point" ]
    num_row   : Axis[ "nb_rows" ]
    num_nnz   : Axis[ "nb_nnz" ]
    num_point : Axis[ "nb_points" ]
    nb_rows   : ShapeVar
    nb_nnz    : ShapeVar
    nb_points : ShapeVar


class _CardMoments( Aggregate ):
    """the barycentre of each cell ( its seed if empty ) and `rho int_cell |x - p|^2`, user order"""
    bary      : RealTensor[ "num_point", "dim" ]
    cost      : RealTensor[ "num_point" ]
    num_point : Axis[ "nb_points" ]
    dim       : Axis[ "nb_dims" ]
    nb_points : ShapeVar
    nb_dims   : ShapeVar
