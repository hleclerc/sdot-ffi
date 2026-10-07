import math

import numpy as np

# `loom.tensor` FIRST: `AaBsp` is the first module that `sdot/__init__.py` imports, and
# `loom.drivers.driver` imported before it cuts the `driver <-> tensor` cycle on the wrong side.
import loom
from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar, new_batch_axis
from loom.compilation.FfiCode import FfiCode
import loom
from loom.util import Aggregate

from .SpatialAccelerator import SpatialAccelerator


class AaBsp( SpatialAccelerator ):
    """An AXIS-ALIGNED BSP: a binary tree of boxes, each leaf holding a
    handful of seeds.

    The tree is built by median cuts on the longest axis, until a leaf
    has no more than `max_seeds_per_leaf` seeds: smaller, the tree costs more in descents than it
    saves in avoided cuts; larger, we pay for bisectors that we already knew
    would be of no use.

    The balance is TEN, and it is measured (Xeon W-2145 + RTX 2080 Ti, 1e6 seeds in 2D): the
    plateau goes from 6 to 12, and 30 -- the old default, measured on another processor -- costs 15 % there.
    It is a setting that follows the MACHINE and not the problem: it arbitrates between the cost of a cut
    and that of a box eviction, and the two do not move together from one processor to the
    other. To be reopened as soon as one of the two changes (see the `pd accelerated` bench, which sweeps it:
    `./run bench "test_PowerDiagram::pd accelerated" --leaf-size=6,10,16,30`).

    = What each node carries, and why

    The BOX (`node_box`, `lo` then `hi`) contains all the seeds of the subtree, and an AFFINE
    MAJORANT of their weights: `w( y ) <= node_wa . y + node_wb` for every seed `y` of the subtree.
    The two together suffice to answer "nothing in there can cut this cell", and the
    affine majorant is what makes the answer sharp: the classic bound is a CONSTANT majorant
    (the node's max weight), which treats the whole box as if the heaviest seed were
    everywhere. A weight that varies smoothly in space -- which is exactly the regime of
    semi-discrete optimal transport, where the weights are a potential -- is then very poorly bounded.

    The other reason, less obvious and decisive: degree 1 costs NOTHING more to test. The
    minimum of `|p - y|² - wa . y` over a box is SEPARABLE per axis, its free minimum is at
    `y = p + wa / 2`, and a per-axis `clamp` gives the exact answer. The constant bound does the
    same work with `wa = 0`. A degree-2 majorant would break this separability.

    The majorant is chosen AT CONSTRUCTION, node by node, between the least-squares
    fitted affine and the constant: the affine is only kept if it clearly tightens the SPREAD of the
    residuals, which is precisely what makes the slack of the bound (see `_weight_majorant`). With no
    weights at all, `node_wa` / `node_wb` are not named: they stay `Unbound`, arrive as
    `NoneTensor`, and the term disappears from the kernel at COMPILATION.

    = The walk

    A depth-first descent, the nearest child first (see `AaBsp.cxx`). The first node
    reached is therefore the leaf of the seed itself: the cell shrinks right away on
    its immediate neighbors, and everything that follows is pruned against an already small cell. That is
    what makes the stack SUFFICIENT where a priority queue would otherwise be needed: at each level
    we pop one node and push two, so the stack never exceeds the DEPTH of
    the tree -- a capacity known at construction (`max_depth`), and not a capacity to
    guess and then double.

    = Where it is built

    In a KERNEL, one call per LEVEL of the tree (`_build_in_kernel` + `bsp_build_level.h`), and one
    work-item per node of the level. What makes it possible is that the SHAPE of the tree does not depend on the
    data: the cut is MEDIAN, so the depth is `ceil( log2( n / leaf_size ) ) + 1` and
    the nodes are those of a PERFECT binary tree of that depth, both functions of
    `n` alone (see `max_depth_for` / `max_nb_nodes_for`, and the test
    `the_tree_shape_does_not_depend_on_the_data`, which checks it down to entirely
    degenerate clouds). So there is NO capacity to guess -- neither for the descent's stack, nor for
    the node arrays -- and the NUMBER OF CALLS itself is known before looking at a point.

    The loop over the levels stays on the host side, as does the index arithmetic between two
    levels: arrays the size of a level, never of the cloud. That is what still prevents
    this construction from going under a `jit` -- but an `AaBsp` is a CONSTANT of the trace (see
    below), so that is not what is asked of it.

    ON A CUDA CARD, IN 2D AND 3D, the whole tree is ONE ffi call instead ( `gpu/Bsp2D.cuh`, `_init_on_card` ): the seeds
    sorted once per axis, then per level the boxes read off the sorted lists and a stable partition of the lists by
    a scan -- nothing read back, so it runs under a `jit` on TRACED positions too, and a solve on the card builds its
    tree inside its jitted program ( `SdotPlanNd` ). The same tree, but for which half of the seeds TIED at a median
    goes left ( the lowest indices on the card ) and the order inside a leaf. Its tensors stay on the card ( tracers
    under a `jit` ), and so do the order and its inverse ( `rank_of_seeds` ) that `PowerDiagram_Bsp` gathers with.

    The CLOUD, for its part, never comes back down: `positions` / `weights` are passed to the kernel as they
    arrive, without `np.asarray`. Seeds that live on the GPU stay there -- which matters for
    whoever rebuilds the tree at each Newton step, where a host round trip would cost twice
    the cloud per step and serve no purpose. What comes back down is the size of a LEVEL
    (the boxes, the `mid`), plus the final permutation: measured, 0.11 s out of the 4.5 s of a tree
    at 1e6 seeds, the rest being the kernels themselves -- and mostly the very first
    levels, where two or four work-items sweep the whole cloud.

    = DIFFERENTIATION, and why there is none

    The tree is a COMBINATORIAL object, and the true gradient through it is exactly ZERO --
    pruning does not change the set of surviving cuts, only which ones we try. Its floating-point
    outputs (boxes, majorants) are therefore not differentiable, and the host construction
    gets this for free by making them constants of the trace.
    """

    # the seeds, REORDERED: the seed indices grouped by leaf, each leaf
    # occupying the slice `[ node_begin, node_end )`. It is this grouping that makes reading
    # a leaf a contiguous read and not a gather of scattered indices.
    seed_indices : IntTensor[ "num_bsp_seed" ]


    # the tree, numbered AS A HEAP: node `k` has its children at `2k+1` / `2k+2`, the root is 0, and
    # level `L` occupies `[ 2^L - 1, 2^(L+1) - 1 )`. `node_left < 0` MEANS leaf, and only occurs
    # at the LAST level -- a node that has nothing left to cut passes its whole slice to its left
    # child and nothing to the right (see `_build_in_kernel`), so that an EMPTY child (`begin == end`)
    # is the
    # only other thing to distinguish, which `ProviderBsp::next` does in two integer reads.
    node_left    : IntTensor[ "num_bsp_node" ]
    node_right   : IntTensor[ "num_bsp_node" ]
    node_begin   : IntTensor[ "num_bsp_node" ]
    node_end     : IntTensor[ "num_bsp_node" ]

    # the bounding box of the subtree -- `lo` THEN `hi`, IN THE SAME ARRAY, and that is the point:
    # the walk is pointer-chasing, so what costs is not the number of bytes read but the
    # number of CACHE LINES touched. Two separate arrays are two lines at two places
    # in memory for a single box; interleaved, a node's box fits in one contiguous
    # read (32 bytes in 2D FP64).
    #
    # It matters because the tree fits in no cache: ~16 MB at 1e6 seeds, against 1 MB of L2
    # per core and 11 MB of L3 for all. Measured: L2 misses PER CELL go from 15 on one
    # thread to 163 on eight, with identical per-core locality -- the Skylake-SP L3 is a
    # non-inclusive VICTIM cache, so with eight cores each has only an eighth of the
    # catch-up. Dividing the number of lines touched is the only handle on this.
    node_box     : RealTensor[ "num_bsp_node", "num_lohi", "dim" ]
    node_wa      : RealTensor[ "num_bsp_node", "dim" ]
    node_wb      : RealTensor[ "num_bsp_node" ]

    num_bsp_seed : Axis[ "nb_bsp_seeds" ]
    num_bsp_node : Axis[ "nb_bsp_nodes" ]
    num_lohi     : Axis[ "nb_lohi" ]
    dim          : Axis[ "nb_dims" ]

    nb_bsp_seeds : ShapeVar
    nb_bsp_nodes : ShapeVar
    nb_lohi      : CtShapeVar
    nb_dims      : CtShapeVar


    def __init__( self, positions, weights = None, max_seeds_per_leaf = 10 ):
        """`positions`: `[ n, d ]`. `weights`: `[ n ]`, or nothing (the Euclidean case).

        `max_seeds_per_leaf` is the grain of the tree -- see the class docstring.

        The cloud is NOT converted to numpy: it goes to the kernel as it arrives (`Tensor.set`
        reads its shape without touching its data), so seeds that live on the GPU stay there.
        Only a SHAPE is read here, and a shape is not data.
        """
        pos = positions if hasattr( positions, "shape" ) else np.asarray( positions, dtype = float )
        # ON A CUDA CARD, in 2D and 3D: the whole tree in ONE ffi call ( `gpu/Bsp2D.cuh` ), nothing read back -- traced positions
        # included ( under a `jit` the tree is then built in the jitted program )
        if builds_on_card( pos ):
            self._init_on_card( pos, weights, max_seeds_per_leaf )
            return
        # under a `jit`, `positions` is a tracer: its shape can be read, but the per-level loop
        # reads the `mid` back on the host side and has nothing to read on a tracer. Say so HERE rather than
        # letting the backend's error surface fifteen lines later: it is not an accident,
        # it is the accepted limit of the host-side construction (see the class docstring).
        if loom.is_traced( pos ):
            raise TypeError( "`AaBsp` is built on the HOST, from concrete positions: it cannot be "
                             "built from a traced array (inside a `jit`). Build it outside, and "
                             "pass it in -- the tree is a constant of the trace, which is also what "
                             "makes it invisible to the gradients." )
        if len( pos.shape ) != 2:
            raise ValueError( f"`positions` has to be [ n, d ] ( got { tuple( pos.shape ) } )" )
        w = None if weights is None else ( weights if hasattr( weights, "shape" ) else np.asarray( weights, dtype = float ) )
        if w is not None and int( np.prod( w.shape ) ) != int( pos.shape[ 0 ] ):
            raise ValueError( "`weights` has to hold one weight per position" )
        if int( pos.shape[ 0 ] ) == 0:
            raise ValueError( "an accelerator over no seed at all has nothing to accelerate" )

        # TRACED weights, on the other hand, are fine: they only enter the majorant, and the SHAPE of
        # the tree depends only on the positions. The tree is therefore built without them ( its `mid` remain
        # readable on the host side, even under a `jit` ), and the majorant is redone afterwards, in a kernel
        # that accepts tracers ( `refresh_weight_majorants` )
        traced_w = w is not None and loom.is_traced( w )
        # the positions are concrete: the construction is EVALUATED even under a trace ( `loom.concrete_eval`: its calls
        # read each level back, which a trace cannot do ), so a solve inside `jax.jit` still gets its tree
        with loom.concrete_eval():
            tree = _build_in_kernel( pos, None if traced_w else w, int( max_seeds_per_leaf ) )

        # the depth, which is EXACTLY `max_depth_for( n, leaf )`: the tree now has the
        # fixed shape that this majorant described (see `_build_in_kernel`). It is what sizes
        # the stack
        # of the descent, and a stack that is too short would be a walk that skips seeds.
        self.max_depth = tree[ "max_depth" ]
        self.max_seeds_per_leaf = int( max_seeds_per_leaf )
        self._nb_leaves = tree[ "nb_leaves" ]
        self._rank = None

        kwargs = dict(
            seed_indices = tree[ "seed_indices" ],
            node_left    = tree[ "node_left"    ],
            node_right   = tree[ "node_right"   ],
            node_begin   = tree[ "node_begin"   ],
            node_end     = tree[ "node_end"     ],
            node_box     = tree[ "node_box"     ],
        )
        # no weights -> we do not NAME the two majorant tensors: leaving them `Unbound`
        # (never allocated, `NoneTensor` on the C++ side) removes the term from the kernel, where zeros
        # would be an array to read. Same rule as `PowerDiagram.weights`, and for the same
        # reason: "no weights" is a STATE, not a value.
        if w is not None:
            kwargs[ "node_wa" ] = tree[ "node_wa" ]
            kwargs[ "node_wb" ] = tree[ "node_wb" ]

        self.__base_init__( nb_dims = int( pos.shape[ 1 ] ), nb_lohi = 2, **kwargs )

        self._majorant_weights = weights
        if traced_w:
            order = tree[ "seed_indices" ]
            self.refresh_weight_majorants( getattr( pos, "raw", pos )[ order ], w[ order ] )
            self._majorant_weights = weights


    def _init_on_card( self, pos, weights, max_seeds_per_leaf ):
        """THE CARD'S BUILD ( `gpu/Bsp2D.cuh::build_tree` ): the same tree as `_build_in_kernel` ( slices, boxes, median
        cuts on the longest axis, preorder ), but for the side of a tie at a median ( the lowest indices go left ) and
        the order inside a leaf. Every output stays on the card -- a tracer under a `jit` --, and the weights, if any,
        enter the majorants by `refresh_weight_majorants` ( one more call )."""
        if len( pos.shape ) != 2:
            raise ValueError( f"`positions` has to be [ n, d ] ( got { tuple( pos.shape ) } )" )
        n, d = int( pos.shape[ 0 ] ), int( pos.shape[ 1 ] )
        if n == 0:
            raise ValueError( "an accelerator over no seed at all has nothing to accelerate" )
        w = None if weights is None else ( weights if hasattr( weights, "shape" ) else np.asarray( weights, dtype = float ) )
        if w is not None and int( np.prod( w.shape ) ) != n:
            raise ValueError( "`weights` has to hold one weight per position" )
        leaf = int( max_seeds_per_leaf )
        # ( the call's TENSORS, not their buffers: a buffer given to a field goes through `loom.array`, which brings a
        # concrete device array back to the host -- the tensor's storage is adopted as it is )
        tree = _build_on_card( pos, leaf )
        self.max_depth = AaBsp.max_depth_for( n, leaf )
        self.max_seeds_per_leaf = leaf
        self._nb_leaves = None                           # ( a host read: counted when asked, see `nb_leaves` )
        self._rank = tree.pop( "rank_of" ).raw
        # ( an adopted tensor does not make the counts pull from it: they are prescribed, both functions of `n` and `leaf` )
        self.__base_init__( nb_dims = d, nb_lohi = 2, nb_bsp_seeds = n, nb_bsp_nodes = 2 ** self.max_depth - 1, **tree )
        if w is not None:
            order = tree[ "seed_indices" ].raw
            self.refresh_weight_majorants( take_rows( getattr( pos, "raw", pos ), order ), take_rows( getattr( w, "raw", w ), order ) )
            self._majorant_weights = weights

    def set_zero_weight_majorants( self ):
        """the majorants of ZERO weights ( a cold start ): zero, without a call -- on the card, and traced under a `jit`"""
        self.node_wa = RealTensor[ self.num_bsp_node, self.dim ].zeros()
        self.node_wb = RealTensor[ self.num_bsp_node ].zeros()

    @property
    def nb_leaves( self ):
        """the non-empty leaves ( a host read of the slices when the tree was built on the card )"""
        if self._nb_leaves is None:
            beg = np.asarray( self.node_begin ).reshape( -1 )
            end = np.asarray( self.node_end ).reshape( -1 )
            self._nb_leaves = int( ( ( np.asarray( self.node_left ).reshape( -1 ) < 0 ) & ( end > beg ) ).sum() )
        return self._nb_leaves

    @staticmethod
    def max_depth_for( nb_seeds, max_seeds_per_leaf = 10 ):
        """A MAJORANT of the depth, WITHOUT seeing the points -- and it is REACHED as soon as the
        seeds are distinct: the cut is median, so the tree is balanced and its shape
        depends only on `n`. A degenerate cloud (coincident seeds) closes leaves earlier,
        so it only shrinks the tree, never the opposite. That is what says that a
        kernel construction has no capacity to guess.
        """
        n, leaf = int( nb_seeds ), max( int( max_seeds_per_leaf ), 1 )
        return 1 if n <= leaf else math.ceil( math.log2( n / leaf ) ) + 1

    @staticmethod
    def max_nb_nodes_for( nb_seeds, max_seeds_per_leaf = 10 ):
        """A majorant of the number of nodes, `n` alone -- see `max_depth_for`. A binary tree whose
        leaves are all at the same level has `2 * leaves - 1` of them, and the leaves number at
        most `2 ** ( depth - 1 )`."""
        return 2 * 2 ** ( AaBsp.max_depth_for( nb_seeds, max_seeds_per_leaf ) - 1 ) - 1

    @classmethod
    def of( cls, power_diagram, max_seeds_per_leaf = 10 ):
        """The accelerator of the seeds of `power_diagram` -- its positions AND its weights.

        The shortcut one almost always wants: a BSP built on other weights than those
        of the diagram would remain CORRECT (the majorant would only prune less well) but
        would have no reason to be good.
        """
        # the diagram's BUFFERS, not their host copy: `Tensor.raw` is the backend's array, and
        # `__init__` passes it to the kernel without touching it. A diagram whose seeds are on the GPU
        # therefore builds its tree there without the cloud coming back down -- and it used to come down twice,
        # once through `np.asarray` and once through the re-upload.
        pos = power_diagram.positions.raw
        w = power_diagram.weights
        w = w.raw if w.is_defined else None
        return cls( pos, w, max_seeds_per_leaf = max_seeds_per_leaf )


    # -- what the caller needs to know ----------------------------------------------------------

    def nb_seeds( self ):
        return int( self.nb_bsp_seeds.value )

    def rank_of_seeds( self ):
        """the inverse of `seed_indices`: the RANK ( in the tree's order ) of seed `i` ( on the card, and possibly traced,
        for a tree built there )"""
        if self._rank is not None:
            return self._rank
        order = np.asarray( self.seed_indices ).reshape( -1 ).astype( np.int64 )
        rank = np.empty_like( order )
        rank[ order ] = np.arange( len( order ) )
        return rank

    # -- what is redone without rebuilding ------------------------------------------------------

    def refresh_weight_majorants( self, sorted_positions, sorted_weights ):
        """The affine majorant of the weights of EACH node, redone on new weights -- the seeds
        being given IN THE TREE'S ORDER ( what `PowerDiagram_Bsp` stores ), one slice per
        node. One kernel, one work-item per node ( `AaBsp.h::refresh_majorant` ).

        It is what makes a tree REUSABLE when only the weights change ( a step of `SdotPlanNd`,
        where the positions are the constants of the problem ): the shape of the tree depends only on the
        positions, and the only thing that speaks of weights is this majorant. It accepts
        TRACED weights ( the nodes then come out traced too ) and cuts the gradient: the majorant is
        an object of PRUNING, which does not change the result -- its true derivative is zero.
        """
        nb_nodes = int( self.nb_bsp_nodes.value )
        self._majorant_weights = None
        if builds_on_card( self.node_box ):
            # ON THE CARD: `Majorant2D.cuh`'s ( 3D: `Majorant3D.cuh`'s ) launches over the seeds and the levels ( the per-node kernel below gives the
            # root's million seeds to one thread ); the tree's tensors are read where they are, traced or not
            num_seed = self.num_bsp_seed
            sp = RealTensor[ num_seed, self.dim ]( loom.ops().stop_gradient( getattr( sorted_positions, "raw", sorted_positions ) ) )
            sw = RealTensor[ num_seed ]( loom.ops().stop_gradient( getattr( sorted_weights, "raw", sorted_weights ) ).reshape( -1 ) )
            wa = RealTensor[ self.num_bsp_node, self.dim ]()
            wb = RealTensor[ self.num_bsp_node ]()
            tn = "int" if nb_nodes <= 2 ** 31 - 1 else "long long"
            d = int( self.nb_dims.value )
            fn, inc = ( f"sdot::gpu2d::refresh_majorants<{ tn }>", "sdot/gpu/Bsp2D.cuh" ) if d == 2 else \
                      ( f"sdot::gpu3d::refresh_majorants<{ d }, { tn }>", "sdot/gpu/Majorant3D.cuh" )
            loom.ffi_call(
                f"bsp_refresh_majorants_card_{ d }d",
                FfiCode.inline( f"{ fn }( queue, args.inputs.node_box, args.inputs.node_begin, "
                                "args.inputs.node_end, args.inputs.sorted_positions, args.inputs.sorted_weights, args.outputs.node_wa, "
                                "args.outputs.node_wb, args.allocator );",
                                includes = [ inc ], allocator = True ),
                node_box = self.node_box, node_begin = self.node_begin, node_end = self.node_end,
                sorted_positions = sp, sorted_weights = sw,
                node_wa = loom.out( wa ), node_wb = loom.out( wb ),
                has_dynamic_capacity = False,
            )
            self.node_wa = wa                            # ( the tensors: their buffers stay on the card )
            self.node_wb = wb
            return
        num_node = new_batch_axis( nb_nodes, prefix = "bspnode" )
        majorant = _NodeMajorant( nb_dims = int( self.nb_dims.value ), batch_axes = [ num_node ] )

        # the gradient is cut AT THE INPUT: a kernel without an adjoint under `loom.grad` is an
        # error, and this one has nothing to propagate ( see above )
        cloud = _BspCloud( nb_dims = int( self.nb_dims.value ),
                           positions = loom.ops().stop_gradient( getattr( sorted_positions, "raw", sorted_positions ) ),
                           weights = loom.ops().stop_gradient( getattr( sorted_weights, "raw", sorted_weights ) ) )

        # the tree is NOT an argument: its current majorants are what we replace, and under
        # a trace they may be tracers of a closed trace ( see `SdotPlanNd` ). Only the
        # slices go in.
        loom.ffi_call(
            "bsp_refresh_majorants",
            FfiCode.per_item( includes = [ "sdot/bsp_build_level.h" ],
                code = "bsp_refresh_majorant( inputs.cloud, inputs.node_begin( batch_index ), inputs.node_end( batch_index ), "
                           "outputs.majorant.wa( batch_index ), outputs.majorant.wb( batch_index ) );" ),
            cloud = cloud,
            majorant = loom.out( majorant ),
            node_begin = IntTensor[ num_node ]( np.asarray( self.node_begin ).reshape( -1 ) ),
            node_end   = IntTensor[ num_node ]( np.asarray( self.node_end ).reshape( -1 ) ),
            has_dynamic_capacity = False,
        )
        self.node_wa = majorant.wa.raw
        self.node_wb = majorant.wb.raw


def _weight_majorant( pos, w ):
    """`( a, b )` such that `w_i <= a . pos_i + b` for every seed of the node, the tightest we
    know how to do quickly.

    Two candidates: the CONSTANT (`a = 0`, `b = max w`), and the AFFINE fitted by least squares
    then raised until it majorizes. We keep the affine only if it clearly tightens
    the SPREAD of the residuals -- that is, `max( w - a.y ) - min( w - a.y )`, which is exactly
    the slack of the bound: a majorant is worth the gap between it and the real weight of the seed
    that attains it. The two are not comparable in the absolute (the affine credits the "low weight"
    side of the box less, but the opposite side more), and the spread is the honest way
    to decide without depending on where the box is looked at from.

    Compared to WHAT, however, requires a precaution: a fit with `d + 1` parameters on
    `m` points tightens the spread even when there is nothing to fit, all the more strongly as
    `m` is small -- and a leaf is small by construction. The threshold is therefore the
    tightening that CHANCE already gives, `sqrt( 1 - d / ( m - 1 ) )` (measured: 0.93 for
    `d = 2, m = 13`, 0.76 for `d = 3, m = 8`), and the affine must do clearly better than it.
    Without this correction, a node of purely random weights kept the affine one time in
    three -- always VALID (the raising takes care of it), but one more vector to read per node
    for a bound that is no better.

    A second safeguard, learned on a real cloud (`solvers_des_familles`, § 7.5): seeds
    clamped at the edge, `x` equal to within 1e-8, and the normal matrix is nearly singular -- the
    ratio of singular values, 3e-8, passes the `rcond` of `lstsq` -- so the slope comes out at
    1e13 and `b` is computed at 1e9 by a cancellation that eats everything. Once raised, the majorant stays
    true, but it no longer majorizes ANYTHING useful. A slope is therefore only admitted if, over the extent
    of the node, it remains of the order of the spread of the weights: beyond that it explains nothing, it
    amplifies rounding. And since the rounding margin of `b` is relative to `|a . y|` (the kernel
    computes in `float`), `|a . y|` must not exceed a hundred times the spread, otherwise this
    margin ceases to be negligible compared to what is being majorized.
    """
    d = pos.shape[ 1 ]
    if w is None:
        return np.zeros( d ), 0.0

    m = len( w )
    spread = float( w.max() - w.min() )
    a = np.zeros( d )
    if m >= 2 * ( d + 1 ) and spread > 0:
        # centered: least squares on raw `pos` would be badly conditioned as soon as the node
        # is far from the origin. The constant does not change the spread, it is taken up by `b`.
        q = pos - pos.mean( axis = 0 )
        fit = np.linalg.lstsq( q, w - w.mean(), rcond = None )[ 0 ]
        r = w - pos @ fit
        by_chance = np.sqrt( max( 1.0 - d / ( m - 1 ), 0.0 ) )
        extent = pos.max( axis = 0 ) - pos.min( axis = 0 )
        reach = np.abs( pos ).max( axis = 0 )
        well_behaved = bool( ( np.abs( fit ) * extent <= 8 * spread ).all() and ( np.abs( fit ) * reach <= 100 * spread ).all() )
        if well_behaved and float( r.max() - r.min() ) < 0.85 * by_chance * spread:
            a = fit

    b = float( ( w - pos @ a ).max() )

    # a rounding MARGIN on the constant, and on it alone. The box does not need one:
    # `float32( min( y ) ) == min( float32( y ) )` (a rounding is monotone), so `node_box`
    # remains exact once converted. `b`, on the contrary, is the only term that the host
    # and the kernel compute DIFFERENTLY -- here `w - a . y` in double, there in `TF` -- and a `b`
    # rounded downward would cease to majorize. Enlarging `b` can only prune less, never lie.
    scale = abs( b ) + float( w.max() - w.min() ) + float( np.abs( pos @ a ).max() )
    return a, b + 1e-6 * scale


# -- the construction, LEVEL BY LEVEL -----------------------------------------------------------


class _BspCloud( Aggregate ):
    """The cloud WHILE BEING SORTED: the seeds arranged in the order in which the tree groups them, plus
    the original index of each.

    The positions (and the weights) are kept PERMUTED next to the indices, and not re-read through
    them: a node then reads its points in one piece, where an indirection through `order` would make
    it a scattered gather. It matters everywhere, and above all at the first levels, where very few
    work-items sweep the whole cloud.

    TWO are needed per level, one read and the other written: the inputs and outputs of a call are
    disjoint (see `loom.ffi_call`), and the sort of a level is a permutation, so each cell of
    the output is written by the work-item of the node that contains it -- once and only once, with no
    atomic or barrier, because the slices of a level PARTITION `[ 0, n )`.
    """

    positions : RealTensor[ "num_point", "dim" ]
    weights   : RealTensor[ "num_point" ]
    order     : IntTensor[ "num_point" ]

    num_point : Axis[ "nb_points" ]
    dim       : Axis[ "nb_dims" ]

    nb_points : ShapeVar
    nb_dims   : CtShapeVar


class _NodeMajorant( Aggregate ):
    """the affine majorant of the weights of ONE node, `w( y ) <= wa . y + wb` -- batched over the nodes
    when all of them are redone ( `AaBsp.refresh_weight_majorants` )"""

    wa    : RealTensor[ "dim" ]
    wb    : RealTensor

    dim      : Axis[ "nb_dims" ]
    nb_dims  : CtShapeVar


class _BspLevel( Aggregate ):
    """What ONE level of the tree carries, PER NODE -- batched over the nodes of the level, hence one
    work-item per node.

    `begin` / `end` are the INPUT (the node's slice, decided by the level above); everything
    else is output. `mid` says where to cut: the left child receives `[ begin, mid )`, the right
    `[ mid, end )`, and `mid == end` is a node that had nothing left to cut and propagates everything
    to the left (see `bsp_build_level.h`).
    """

    begin : IntTensor
    end   : IntTensor
    mid   : IntTensor

    box   : RealTensor[ "num_lohi", "dim" ]
    wa    : RealTensor[ "dim" ]
    wb    : RealTensor

    num_lohi : Axis[ "nb_lohi" ]
    dim      : Axis[ "nb_dims" ]
    nb_lohi  : CtShapeVar
    nb_dims  : CtShapeVar


def _preorder_of_heap( depth ):
    """`p[ i ]` = where the node of heap rank `i` lands in PREORDER (DFS).

    = Why change the numbering

    In a HEAP, the children of node `i` are at `2i+1` / `2i+2`: a root -> leaf path jumps to
    addresses that DIVERGE exponentially, and the worst is at the bottom of the tree, where the
    levels are biggest -- at 1e6 seeds, level 17 has 65 536 nodes, so the children of a deep
    node are two megabytes away from it. Yet it is a root -> leaf path that the walk
    traverses FOR EACH CELL.

    In PREORDER, the left child is at `i+1` and the right at `i + 2^(h-1)`, where `h` is the height of the
    subtree: the jumps SHRINK as we descend, and the last levels -- the most numerous
    and the most visited -- sit within a few nodes of each other. The troublesome property is
    exactly reversed.

    = What it does not change

    Neither the shape of the tree, nor the slices, nor the walk: it is a PERMUTATION of the same nodes.
    What changes is the address at which each is written -- and that is measured to be what
    matters: at equal instruction counts (17 500 per cell against 26 950 for pysdot, so FEWER), we
    generated 199 cache misses per cell on eight cores where pysdot -- a quadtree in Z order,
    hence with contiguous subtrees -- generates 5.9.
    """
    nb_nodes = 2 ** depth - 1
    pre = np.zeros( nb_nodes, dtype = np.int64 )
    for level in range( depth - 1 ):
        h = depth - level                       # height of the subtree of a node of this level
        idx = np.arange( 2 ** level - 1, 2 ** ( level + 1 ) - 1 )
        pre[ 2 * idx + 1 ] = pre[ idx ] + 1
        pre[ 2 * idx + 2 ] = pre[ idx ] + 2 ** ( h - 1 )
    return pre


def _build_in_kernel( pos, w, leaf_size ):
    """The same tree as `_build`, built by `bsp_build_level.h` instead of numpy.

    = Why one call PER LEVEL

    A level reads the slices that the previous one produced, and there is no GLOBAL barrier
    in a kernel -- only within a work-group. The barrier is therefore the end of the
    launch, and the host chains `depth` calls. It is not a stopgap: `depth` is
    `max_depth_for( n, leaf_size )`, a function of `n` ALONE (median cut), so the number
    of calls is known in advance and depends on no data -- about fifteen at 1e6 seeds, against
    the ~130 000 Python loop iterations that the host version does per node.

    = What stays on the host side, and what it costs

    The index arithmetic between two levels (`[ begin, mid )` / `[ mid, end )`) and the
    gluing of the levels into a single node array. Arrays the size of a LEVEL,
    never of the cloud. It is also what still prevents this construction from going under a `jit`
    -- but an `AaBsp` is a CONSTANT of the trace anyway (see the class docstring),
    so that is not what is asked of it.

    = The name of the batch axis

    A fresh axis per level would give `depth` different C++ sources, hence `depth` compilations
    (see `loom.tensor.batch`). The tensors of a level are therefore COPIED into numpy and the level
    released before the next: the name goes back to the pool, the `depth` calls share a single
    source, and only the first one compiles.
    """
    n, d = pos.shape

    depth = AaBsp.max_depth_for( n, leaf_size )
    num_param = Axis( ShapeVar( 1 ), name = "num_bsp_param" )

    src = _BspCloud( nb_dims = d, positions = pos, order = np.arange( n, dtype = np.int64 ),
                     **( {} if w is None else { "weights": w } ) )

    beg = np.zeros( 1, dtype = np.int64 )
    end = np.full( 1, n, dtype = np.int64 )

    begs, ends, boxes, was, wbs = [], [], [], [], []

    # ON A CARD, THE TOP LEVELS ON THE HOST. A level runs one work item per node: the first levels are a handful of work
    # items each sweeping ( and selecting in ) a large part of the cloud -- on a GPU thread that is seconds at 1e6 seeds
    # ( 4.5 to 5.7 s measured on the RTX 2080 Ti, against 0.15 s for the whole tree on the CPU ). Those levels are done here
    # by numpy, with the same rule ( the box, the longest axis, the median by rank, propagation when nothing can be cut, the
    # majorant of `_weight_majorant` ); the levels where the nodes are many and small stay in the kernel. The halves of a
    # cut are not ordered the same way as the kernel's selection would order them, which no property of the tree depends on.
    host_levels = 0
    if getattr( loom.resolved_device(), "is_cuda_gpu", False ) and not loom.is_traced( pos ) and n > 64 * leaf_size:
        host_levels = min( depth, 10 )
    if host_levels:
        P = np.array( np.asarray( pos ), dtype = np.float64 )
        W = None if w is None else np.array( np.asarray( w ), dtype = np.float64 ).reshape( -1 )
        order = np.arange( n, dtype = np.int64 )
        for level in range( host_levels ):
            nb = len( beg )
            mid = end.copy()
            box = np.zeros( ( nb, 2, d ) )
            wa = np.zeros( ( nb, d ) )
            wb = np.zeros( nb )
            for j in range( nb ):
                b, e = int( beg[ j ] ), int( end[ j ] )
                if e <= b:
                    continue
                p = P[ b:e ]
                lo, hi = p.min( axis = 0 ), p.max( axis = 0 )
                box[ j, 0 ], box[ j, 1 ] = lo, hi
                if W is not None:
                    wa[ j ], wb[ j ] = _weight_majorant( p, W[ b:e ] )
                ax = int( np.argmax( hi - lo ) )
                if e - b > leaf_size and hi[ ax ] > lo[ ax ]:
                    m = b + ( e - b ) // 2
                    idx = np.argpartition( p[ :, ax ], m - b )
                    P[ b:e ] = p[ idx ]
                    order[ b:e ] = order[ b:e ][ idx ]
                    if W is not None:
                        W[ b:e ] = W[ b:e ][ idx ]
                    mid[ j ] = m
            begs.append( beg )
            ends.append( end )
            boxes.append( box )
            if w is not None:
                was.append( wa )
                wbs.append( wb )
            nb_ = np.empty( 2 * nb, dtype = np.int64 )
            ne_ = np.empty( 2 * nb, dtype = np.int64 )
            nb_[ 0::2 ], nb_[ 1::2 ] = beg, mid
            ne_[ 0::2 ], ne_[ 1::2 ] = mid, end
            beg, end = nb_, ne_
        src = _BspCloud( nb_dims = d, positions = P, order = order, **( {} if W is None else { "weights": W } ) )

    for level in range( host_levels, depth ):
        # the output cloud SHARES the points axis (hence its count) with the input: it is the
        # same permutation, rearranged.
        dst = _BspCloud( nb_dims = d, num_point = src.num_point )

        num_node = new_batch_axis( 2 ** level, prefix = "bspnode" )
        lvl = _BspLevel( nb_dims = d, nb_lohi = 2, batch_axes = [ num_node ], begin = beg, end = end )

        perm = IntTensor[ src.num_point ]()
        leaf = IntTensor[ num_param ]()
        leaf.set( np.array( [ leaf_size ], dtype = np.int64 ) )

        # WHAT THE LEVEL WRITES, named POSITIVELY. `begin` / `end` are its INPUT ( the slice
        # decided by the level above ): not naming them is enough to leave them read and not
        # allocated. Without weights, neither the cloud nor the majorant has a tensor: not named either,
        # they stay `Unbound`, arrive as `NoneTensor`, and the two corresponding blocks of the
        # kernel disappear at compilation ( same rule as `PowerDiagram.weights` ).
        written_dst = [ "positions", "order" ] + ( [ "weights" ] if w is not None else [] )
        written_lvl = [ "mid", "box" ] + ( [ "wa", "wb" ] if w is not None else [] )

        loom.ffi_call(
            "bsp_build_level",
            FfiCode.per_item( includes = [ "sdot/bsp_build_level.h" ],
                code = "bsp_build_level( inputs.src, outputs.dst, scratch.perm, "
                           "outputs.lvl.begin( batch_index ), outputs.lvl.end( batch_index ), "
                           "outputs.lvl.box( batch_index ), "
                           "outputs.lvl.wa( batch_index ), outputs.lvl.wb( batch_index ), outputs.lvl.mid( batch_index ), "
                           "SI( inputs.leaf_size( 0 ) ) );" ),
            src = src,
            dst = loom.out( dst, writes = written_dst ),
            lvl = loom.out( lvl, writes = written_lvl ),
            perm = loom.scratch( perm ),
            leaf_size = leaf,
            # all sizes are prescribed upstream (they depend only on `n` and on the
            # level): no count is decided by the kernel, so nothing can overflow and the
            # runtime check -- a device -> host sync per call -- has nothing to watch.
            has_dynamic_capacity = False,
        )

        mid = np.asarray( lvl.mid ).reshape( -1 )
        begs.append( beg )
        ends.append( end )
        boxes.append( np.asarray( lvl.box ).reshape( -1, 2, d ).copy() )
        if w is not None:
            was.append( np.asarray( lvl.wa ).reshape( -1, d ).copy() )
            wbs.append( np.asarray( lvl.wb ).reshape( -1 ).copy() )

        if level + 1 < depth:
            nb = np.empty( 2 * len( beg ), dtype = np.int64 )
            ne = np.empty( 2 * len( beg ), dtype = np.int64 )
            nb[ 0::2 ], nb[ 1::2 ] = beg, mid
            ne[ 0::2 ], ne[ 1::2 ] = mid, end
            beg, end = nb, ne

        order = np.asarray( dst.order ).reshape( -1 ).copy()
        src = dst

        # GIVE BACK the axis name before borrowing another: the next level takes it at the
        # START of its turn, so as long as this one is alive a new one is needed -- and a new name
        # is one more C++ source, hence one more compilation (see `loom.tensor.batch`).
        del lvl, num_node, perm, dst, leaf

    nb_nodes = 2 ** depth - 1

    # the levels concatenate in order, which gives the HEAP numbering: node `k` of
    # level `L` is global `2^L - 1 + k`. It is the form in which they COME OUT of the kernel.
    node_begin = np.concatenate( begs )
    node_end   = np.concatenate( ends )
    node_box   = np.concatenate( boxes )
    node_wa    = np.concatenate( was ) if w is not None else np.zeros( ( nb_nodes, d ) )
    node_wb    = np.concatenate( wbs ) if w is not None else np.zeros( nb_nodes )

    is_leaf = np.zeros( nb_nodes, dtype = bool )
    is_leaf[ 2 ** ( depth - 1 ) - 1: ] = True              # the last level: ONLY leaves

    # ... then we ARRANGE them in preorder, which is a pure permutation: same tree, same
    # slices, same walk, only the ADDRESSES change (see `_preorder_of_heap`).
    pre = _preorder_of_heap( depth )
    def in_preorder( a ):
        r = np.empty_like( a )
        r[ pre ] = a
        return r

    heap = np.arange( nb_nodes, dtype = np.int64 )
    node_left = np.where( is_leaf, -1, pre[ np.where( is_leaf, 0, 2 * heap + 1 ) ] )
    node_right = np.where( is_leaf, -1, pre[ np.where( is_leaf, 0, 2 * heap + 2 ) ] )

    node_end_pre = in_preorder( node_end )
    node_begin_pre = in_preorder( node_begin )

    return dict(
        seed_indices = order,
        node_left    = in_preorder( node_left ),
        node_right   = in_preorder( node_right ),
        node_begin   = node_begin_pre,
        node_end     = node_end_pre,
        node_box     = in_preorder( node_box ),
        node_wa      = in_preorder( node_wa ),
        node_wb      = in_preorder( node_wb ),
        max_depth    = depth,
        nb_leaves    = int( ( in_preorder( is_leaf ) & ( node_end_pre > node_begin_pre ) ).sum() ),
    )


# -- the construction ON THE CARD ( `gpu/Bsp2D.cuh` ) --------------------------------------------------------------------


def builds_on_card( x ):
    """whether a tree on `x` ( positions `[ n, d ]`, or any `[ ..., d ]` tensor of the tree ) is built and refreshed by the
    card's kernels: a CUDA device, 2D or 3D -- the card's cells' scope ( `PowerDiagram_Bsp._card_variant` ). `SDOT_CARD_TREE=0`
    sends every tree to the host-driven build ( `_build_in_kernel`: under a trace, a constant evaluated while tracing ) -- the
    comparison point of the benches"""
    import os
    if os.environ.get( "SDOT_CARD_TREE", "1" ).lower() in ( "0", "no", "false", "off" ):
        return False
    return bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) ) and len( x.shape ) >= 2 and int( x.shape[ -1 ] ) in ( 2, 3 )


def take_rows( a, idx ):
    """`a[ idx ]` along the first axis, through the backend of `idx`: a numpy order indexes as numpy, a device one ( a
    tree built on the card, a tracer under a `jit` ) brings `a` to the device first -- a gather the backend differentiates"""
    a = getattr( a, "raw", a )
    if isinstance( idx, np.ndarray ):
        return a[ idx ]
    if "torch" in type( idx ).__module__:
        import torch
        return torch.as_tensor( a, device = idx.device )[ idx ]
    import jax.numpy as jnp
    return jnp.asarray( a )[ idx ]


def _build_on_card( pos, leaf ):
    """the tree's tensors ( loom tensors, their buffers on the card ) from ONE ffi call ( `sdot::gpu2d::build_tree` ): `seed_indices`, `rank_of` ( its inverse ),
    `node_begin` / `node_end` / `node_left` / `node_right` / `node_box` in preorder. Every size is a function of `n` and
    `leaf`; the work's index type is chosen here ( `int` while `d n` fits in it ). The positions enter WITHOUT their
    gradient: the tree is combinatorial ( see the class docstring )."""
    n, d = int( pos.shape[ 0 ] ), int( pos.shape[ 1 ] )
    depth = AaBsp.max_depth_for( n, leaf )
    nb_nodes = 2 ** depth - 1
    num_seed = Axis( ShapeVar( n ), name = "num_bsp_seed" )
    num_node = Axis( ShapeVar( nb_nodes ), name = "num_bsp_node" )
    num_lohi = Axis( ShapeVar( 2 ), name = "num_lohi" )
    dim = Axis( ShapeVar( d ), name = "dim" )
    positions = RealTensor[ num_seed, dim ]( loom.ops().stop_gradient( getattr( pos, "raw", pos ) ) )
    out = dict( seed_indices = IntTensor[ num_seed ](), rank_of = IntTensor[ num_seed ](),
                node_begin = IntTensor[ num_node ](), node_end = IntTensor[ num_node ](),
                node_left = IntTensor[ num_node ](), node_right = IntTensor[ num_node ](),
                node_box = RealTensor[ num_node, num_lohi, dim ]() )
    ti = "int" if d * n <= 2 ** 31 - 1 else "long long"
    loom.ffi_call(
        "bsp_build_card",
        FfiCode.inline( f"sdot::gpu2d::build_tree<{ d }, { ti }>( queue, args.inputs.positions, { int( leaf ) }, "
                        "args.outputs.seed_indices, args.outputs.rank_of, args.outputs.node_begin, args.outputs.node_end, "
                        "args.outputs.node_left, args.outputs.node_right, args.outputs.node_box, args.allocator );",
                        includes = [ "sdot/gpu/Bsp2D.cuh" ], allocator = True ),
        positions = positions,
        has_dynamic_capacity = False,
        **{ k: loom.out( v ) for k, v in out.items() },
    )
    return out
