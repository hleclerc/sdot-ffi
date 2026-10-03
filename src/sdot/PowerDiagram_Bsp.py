"""The seeds in BSP TREE ORDER: `sorted_positions` / `sorted_weights` are the seeds
permuted the way `tree.seed_indices` arranges them, so that a leaf reads in one piece, and
each cell is cut only by the seeds the tree could not rule out ( `AaBsp.py`,
`cell/Providers.h::ProviderBsp` ).

Everything the kernel handles is in that order -- ranks, cut identifiers,
gradients on the seeds -- and this is where we translate: `positions` / `weights` give the seeds back
in the user's order, measures and cells leave the kernel already at their index
( `user_id( k )` ), and the gathering `sorted = positions[ seed_indices ]` is an operation of the
backend, DIFFERENTIABLE: a derivative with respect to `sorted_positions` comes back to `positions` on its own.

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
from loom.tensor import Axis, IntTensor, RealTensor, ShapeVar

from .AaBsp import AaBsp
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
        self._order = np.asarray( tree.seed_indices ).reshape( -1 ).astype( np.int64 )
        self._rank_of = tree.rank_of_seeds()
        res = { "tree": tree, "sorted_positions": self._gather( positions ) }
        if weights is not None:
            res[ "sorted_weights" ] = self._gather( weights )
            # a tree coming from outside may have been built on other weights, or none: its majorant
            # is redone on THESE ( built here, it already has them )
            if tree is accelerator:
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

    def _gather( self, seeds ):
        """`seeds[ seed_indices ]`, through the backend: a tracer stays a tracer, and the derivative
        with respect to what we gather comes back through here on its own"""
        raw = getattr( seeds, "raw", seeds )
        return raw[ self._order ]

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
        self.sorted_weights = produced[ "sorted_weights_out" ].raw
        self.tree.node_wa = produced[ "node_wa_out" ].raw
        self.tree.node_wb = produced[ "node_wb_out" ].raw
        if "memo_counts_out" in produced:
            self.memo_nbrs = produced[ "memo_nbrs_out" ].raw
            self.memo_counts = produced[ "memo_counts_out" ].raw

    def _grad_seeds_expr( self ):
        return "grad_of_inputs.power_diagram.sorted_positions, grad_of_inputs.power_diagram.sorted_weights"
