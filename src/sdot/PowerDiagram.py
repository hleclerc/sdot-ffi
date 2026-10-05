"""Power diagram ( Laguerre ) -- THE VIEW, and what all the storages have in common.

The cell of seed `i` is where its POWER DISTANCE wins:

    |x - d_i|² - w_i  <=  |x - d_j|² - w_j   for every other j

Expanded, the inequality loses its `|x|²` on both sides and becomes a half-space: a power
diagram costs exactly what a Voronoi diagram costs, one plane per rival and the same cut. Only the
DIFFERENCES of weights reach the planes: "all equal" and "no weights at all" are the same
object, and the Euclidean case is called `Voronoi` ( see `Voronoi.py` ).

`PowerDiagram( positions, weights, ... )` does not carry a diagram: it carries its SEEDS and the
convex domain that bounds them, and rebuilds what it is asked for, cell by cell, in the
scratch of a work-item ( `diagram/Ops.h` ). This file is the CONTRACT -- what a user reads and
writes: `positions`, `weights`, `measures`, `cells`, `cell( i )` -- and the common trunk: the domain,
the distribution, the scratch, the three kernels. HOW the seeds are laid out is up to a
specialization, chosen at construction:

  * `PowerDiagram_Plain` -- the seeds as they came, each cell cut by the `n - 1`
    bisectors. The floor, and what remains when the positions are a tracer;
  * `PowerDiagram_Bsp`   -- the seeds in the order of a BSP tree ( `AaBsp` ), a leaf being
    read in one piece, and each cell cut only by the seeds the tree could not
    rule out. The default as soon as the positions are concrete.

The neighborhood is ACCELERABLE, not the result: an accelerator can only silence cuts that
would have removed nothing, so the cells are the SAME, up to rounding errors.
"""

import numpy as np

import loom
from loom.compilation.FfiCode import FfiCode
from loom.drivers.driver import driver
from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar, Tensor, new_batch_axis
from loom.util import Aggregate

from .Cell import BOUNDARY, Cell
from .CellScratch import CellScratch, fp_size


def diagram_class_for( positions, weights, accelerator ):
    """The specialization that lays out these seeds: the BSP tree unless it is not wanted
    ( `accelerator = "plain"` ) or one cannot be built here -- TRACED seeds, positions
    or weights: the tree is built host-side, and under a `jit` even a constant comes out of a
    kernel traced. Whoever wants the tree under a trace builds it outside and passes it ( `accelerator = tree` )."""
    from .PowerDiagram_Bsp import PowerDiagram_Bsp
    from .PowerDiagram_Plain import PowerDiagram_Plain
    if accelerator == "plain":
        return PowerDiagram_Plain
    if accelerator is None and any( driver.is_traced( getattr( x, "raw", x ) ) for x in ( positions, weights ) if x is not None ):
        return PowerDiagram_Plain
    return PowerDiagram_Bsp


class PowerDiagram( Aggregate ):
    # ---- what a specialization provides -------------------------------------------------------------
    #
    #   _init_seeds( positions, weights, accelerator )   lays out the seeds in its tensors
    #   positions / weights            properties, readable and writable, in the USER's ORDER
    #   _ranks_of_items()              for `cells`: the rank ( storage order ) of seed `i`
    #
    # and C++ side ( `diagram/Ops.h` ): `point( k )`, `weight( k )`, `user_id( k )`, `provider( k0 )`.

    # THE STARTING CELL, when a better one than "all of space" is known: the box
    # `box_min <= x <= box_max`, laid down in one go by `Cell.init_as_hypercube`. Absent ( `Unbound` ),
    # each cell is born as an unbounded REPLACEMENT SIMPLEX whose every cut must first
    # push back the infinite planes. This is not a second domain: it is the same one, expressed in
    # the form that can be laid down directly; what a box does not say stays in `bnd_*`.
    box_min        : RealTensor[ "dim" ]
    box_max        : RealTensor[ "dim" ]

    # the domain: a list of half-spaces, hence any polyhedral convex set. Absent, the
    # cells that run off to infinity stay there -- and are measured as such ( `TF::max` ).
    bnd_directions : RealTensor[ "num_boundary", "dim" ]
    bnd_offsets    : RealTensor[ "num_boundary" ]

    num_point      : Axis[ "nb_points" ]
    num_boundary   : Axis[ "nb_boundaries" ]
    dim            : Axis[ "nb_dims" ]

    nb_points      : ShapeVar
    nb_boundaries  : ShapeVar
    nb_dims        : CtShapeVar

    def __new__( cls, positions = None, weights = None, *args, accelerator = None, **kwargs ):
        if cls is PowerDiagram:
            cls = diagram_class_for( positions, weights, accelerator )
        return super().__new__( cls )

    def __init__( self, positions, weights = None, boundaries = None, accelerator = None,
                  distribution = None, kernel_dtype = None, scratch_capacity = None, memory = None ):
        """`positions`: `[ n, d ]`. `weights`: `[ n ]`, or nothing ( the Euclidean case ). The domain:

        - `boundaries = ( directions, offsets )` -- the half-spaces `direction . x <= offset`.
          A box is written `box_half_spaces( mi, ma )`, which is there for that;
        - nothing -- the boundary cells stay infinite.

        `accelerator`: `None` ( a BSP tree, built here ), an `AaBsp` already built on these positions
        ( what is needed to differentiate with respect to traced positions ), or `"plain"`. Without effect
        on the RESULT -- only on what it costs.

        `distribution`: WHAT to integrate against ( `Image`, `SumOfGaussians`, ... ). Absent,
        `measures` returns the volume of the cells; present, the integral of its density over them,
        NORMALIZED here once and for all. If it has a bounded SUPPORT, it is added to the domain.

        `kernel_dtype`: the float in which the geometry is cut ( `FP32` by default,
        `SDOT_KTYPE` to change the default ). `scratch_capacity`: for how many vertices per
        cell the scratch of a work-item is sized at the start -- a guess, which loom doubles
        on overflow.

        `memory`: with the BSP tree, the diagram REMEMBERS, per seed, the neighbors of its cell
        at the last `measures` and proposes them first to the next one ( `ProviderBsp`, `MEMO` ) --
        which spares the transient cuts, a quarter to a third of the diagram in 3D, and remains
        exact whatever weights have moved in the meantime. `memory` is the capacity per
        seed ( 32 by default in 3D and above, 0 in 2D where it yields nothing measurable, and 0 in 3D on a CUDA card
        whose dedicated cells take the calls: they do not use it ); a seed that has more neighbors than that simply has
        no memory. `0` to turn it off.
        """
        # the SUPPORT of the distribution bounds the domain, for free and without changing the
        # result: what lies beyond brings no mass. The caller's domain is INTERSECTED
        # with it, not replaced.
        if distribution is not None:
            support = distribution.bounding_half_spaces()
            if support is not None:
                if boundaries is None:
                    boundaries = support
                else:
                    boundaries = ( np.concatenate( [ np.asarray( boundaries[ 0 ], dtype = float ), support[ 0 ] ] ),
                                   np.concatenate( [ np.asarray( boundaries[ 1 ], dtype = float ), support[ 1 ] ] ) )

        # no domain -> we do NOT name the two tensors: leaving them `Unbound` ( never
        # allocated, `NoneTensor` C++ side ) is not the same thing as passing them `None`.
        kwargs = {}
        if boundaries is not None:
            # WHERE TO START FROM, read off the half-spaces themselves: those a box already expresses leave
            # the list -- they are `2d` of them and would come back on every cell ( 25 %, measured ).
            start_box = axis_aligned_box( *boundaries )
            if start_box is not None:
                mi, ma, kept = start_box
                kwargs[ "box_min" ], kwargs[ "box_max" ] = mi, ma
                boundaries = ( np.asarray( boundaries[ 0 ], dtype = float )[ kept ],
                               np.asarray( boundaries[ 1 ], dtype = float )[ kept ] )
            if len( boundaries[ 1 ] ):
                kwargs[ "bnd_directions" ], kwargs[ "bnd_offsets" ] = boundaries

        pos = positions if hasattr( positions, "shape" ) else np.asarray( positions, dtype = float )
        if len( pos.shape ) != 2:
            raise ValueError( f"`positions` has to be [ n, d ] ( got { tuple( pos.shape ) } )" )
        n, d = int( pos.shape[ 0 ] ), int( pos.shape[ 1 ] )

        self._kernel_dtype = kernel_dtype
        self._scratch_capacity = int( scratch_capacity or { 2: 64, 3: 128 }.get( d, 256 ) )
        self._memory = int( self._default_memory( d ) if memory is None else memory )
        self.__base_init__( nb_dims = d, nb_points = n, **self._init_seeds( pos, weights, accelerator ), **kwargs )

        # NOT a field: the distribution is a CALL argument, normalized RIGHT HERE rather than at
        # each `measures` -- "the diagram integrates THIS measure" is a property of the object
        self.distribution = None
        if distribution is not None:
            dd = int( distribution.nb_dims.value )
            if dd != d:
                raise ValueError( f"the distribution lives in { dd }D, this diagram in { d }D" )
            self.distribution = distribution.normalized_version()

    def _default_memory( self, d ):
        """the neighbour memories per seed when `memory` is not given ( see `__init__` )"""
        return 0 if d <= 2 else 32

    @property
    def dim_count( self ):
        return int( self.nb_dims.value )

    @property
    def kernel_dtype( self ):
        return self._domain_cell().kernel_dtype

    # ---- the domain, and the distribution ----------------------------------------------------------

    def _domain_cell( self ):
        """The domain as a POLYTOPE, computed ONCE -- what each cell starts from. Built with
        the SAME code as the `cell( i )` oracle, which keeps the two descriptions of the domain
        literally identical. It is the type of THIS cell that decides, C++ side, the local
        shape in which each cell is built."""
        if getattr( self, "_dom_cell", None ) is None:
            self._dom_cell = self._start_cell()
            if self.bnd_directions.is_defined:
                # the backend's VALUES, not numpy's: under a `jit` the half-spaces may be
                # traced. `stop_gradient`: the domain is a constant of the problem ( its cuts
                # carry `BOUNDARY`, "not a seed", and have nowhere to send a derivative ).
                dirs = driver.stop_gradient( self.bnd_directions.raw )
                offs = driver.stop_gradient( self.bnd_offsets.raw )
                for b in range( int( self.bnd_directions.shape[ 0 ] ) ):
                    self._dom_cell.cut( dirs[ b ], offs[ b ], BOUNDARY )
        return self._dom_cell

    def _start_cell( self ):
        """the starting box, or all of space"""
        d = self.dim_count
        kw = dict( kernel_dtype = self._kernel_dtype )
        if self.box_min.is_defined:
            mi = np.asarray( self.box_min ).reshape( -1 )
            ma = np.asarray( self.box_max ).reshape( -1 )
            return Cell.make_hypercube( d, mi, np.diag( ma - mi ), **kw )
        return Cell.make_unbounded( d, **kw )

    def _dist_for( self ):
        """How a call names its distribution: `( C++ expression, expression of its cotangent,
        kwargs of the call )`. Without a distribution, `unit_density()` -- a value that the C++ makes
        itself -- and a cotangent `0` that `UnitDensity` ignores."""
        if self.distribution is None:
            return "inputs.power_diagram.unit_density()", "0", {}
        return "inputs.distribution", "grad_of_inputs.distribution", { "distribution": self.distribution }

    # ---- the memory ( `PowerDiagram_Bsp` only ) -------------------------------------------------------

    def _memo_for_call( self ):
        """`( C++ expression of the two output tensors, arguments of the call, what has to be
        taken back afterwards )` -- nothing for a storage that has no memory"""
        return "0, 0", {}, None

    def _memo_after_call( self, produced ):
        pass

    # ---- a dedicated kernel for the card ( `PowerDiagram_Bsp` only ) ---------------------------------

    def _measures_on_card( self ):
        """the measures by a kernel written for the device, or `None`: the generic path takes the call"""
        return None

    # ---- the scratch -------------------------------------------------------------------------------

    def _scratch_words( self, cap, nb_cells, with_grad ):
        """What a work-item holds, in words: `nb_cells` local cells of `cap` vertices, and
        for the adjoint one cotangent per vertex -- THE SAME FORMULA as `diagram::words_for`."""
        dom = self._domain_cell()
        words = nb_cells * dom.scratch_words( cap, fp_size( dom.kernel_dtype ) )
        if with_grad:
            words += -( -self.dim_count * cap * 8 // 32 ) * 8
        return words

    def _nb_work_cells( self ):
        """a distribution that CUTS the cell asks for a second local cell"""
        return 2 if self.distribution is not None and getattr( self.distribution, "cuts_pieces", False ) else 1

    # ---- what is read ------------------------------------------------------------------------------

    @property
    def measures( self ) -> Tensor:
        """The measure of each cell: `[ n ]`, indexed like `positions`.

        One call, one sweep: each work-item builds a cell in its scratch,
        writes its volume, and starts over with the next seed. Nothing of the diagram is kept.
        With a `distribution`, it is the INTEGRAL of its density over the cell ( same sweep ).

        DIFFERENTIABLE with respect to the seeds, `positions` as well as `weights`, and with respect to the VALUES of
        the distribution ( `diagram::measures_bwd` redoes the same sweep ). The DOMAIN is a
        constant: a cut that comes from it carries a negative identifier, so its share goes nowhere.

        On a CUDA card, in 2D, a dedicated kernel may take the call instead ( `_measures_on_card`, see
        `PowerDiagram_Bsp` ): same cells, cell in registers.
        """
        res = self._measures_on_card()
        if res is not None:
            return res

        dom = self._domain_cell()
        # the budget that decides parallelism: what ONE work-item holds -- its scratch, sized
        # for the backward from the forward on ( it redoes the sweep on a scratch of the same shape )
        nb_words = self._scratch_words( self._scratch_capacity, self._nb_work_cells(), True )
        nt = driver.device.nb_threads( nb_local_bytes_per_thread = 4 * nb_words, batch_axes = [ self.num_point ] )

        # the work-item axis is a BATCH axis carried by the scratch: `thread_index` /
        # `nb_threads` are the rank of this work-item and their number, the strided loop reads off them
        num_thread = new_batch_axis( nt, prefix = "thread" )
        res = RealTensor[ self.num_point ]()
        dist_expr, grad_dist_expr, dist_kwargs = self._dist_for()
        memo_expr, memo_args, memo_produced = self._memo_for_call()

        loom.ffi_call(
            "power_diagram_measures",
            FfiCode.per_item( code = "inputs.power_diagram.measures( outputs.res, inputs.dom_cell, scratch.pool( batch_index ), "
                           f"{ dist_expr }, { memo_expr }, thread_index, nb_threads );",
                # the gradients on the seeds are SHARED by all items: each work-item
                # accumulates into them ( `atomic_add` C++ side ), and the platform zeroes them before the body
                ),
            FfiCode.per_item( "inputs.power_diagram.measures_bwd( outputs.res, inputs.dom_cell, grad_of_outputs.res, "
                           f"{ self._grad_seeds_expr() }, "
                           f"scratch.pool( batch_index ), { dist_expr }, { grad_dist_expr }, "
                           "thread_index, nb_threads );" ),
            power_diagram = self,
            dom_cell = dom,
            res = loom.out( res ),
            pool = CellScratch.for_call( nb_words, dom.kernel_dtype, batch_axes = [ num_thread ] ),
            **dist_kwargs,
            **memo_args,
        )
        self._memo_after_call( memo_produced )
        return res

    @property
    def moments( self ):
        """`( masses, first, second )`: for each cell, `int rho`, `int x rho` ( `[ n, d ]` ) and
        `int |x|^2 rho` -- enough to write a TRANSPORT COST, `sum_i int_{cell_i} |x - p_i|^2 rho
        = second - 2 p . first + |p|^2 mass`, and the barycenters `first / mass`. Same sweep as
        `measures`, on a piecewise-constant distribution ( `Image`, or nothing ) only.
        NOT differentiable: a transport cost is differentiated by the envelope theorem, at the
        adjusted weights -- `2 mass_i ( p_i - b_i )` -- which `SdotPlanNd` does by itself."""
        dom = self._domain_cell()
        nb_words = self._scratch_words( self._scratch_capacity, self._nb_work_cells(), False )
        nt = driver.device.nb_threads( nb_local_bytes_per_thread = 4 * nb_words, batch_axes = [ self.num_point ] )
        num_thread = new_batch_axis( nt, prefix = "thread" )
        mass = RealTensor[ self.num_point ]()
        first = RealTensor[ self.num_point, self.dim ]()
        second = RealTensor[ self.num_point ]()
        dist_expr, _, dist_kwargs = self._dist_for()

        loom.ffi_call(
            "power_diagram_moments",
            FfiCode.per_item( code = "inputs.power_diagram.moments( outputs.mass, outputs.first, outputs.second, "
                           f"inputs.dom_cell, scratch.pool( batch_index ), { dist_expr }, thread_index, nb_threads );" ),
            power_diagram = self,
            dom_cell = dom,
            mass = loom.out( mass ), first = loom.out( first ), second = loom.out( second ),
            pool = CellScratch.for_call( nb_words, dom.kernel_dtype, batch_axes = [ num_thread ] ),
            **dist_kwargs,
        )
        return mass, first, second

    def hessian_rows( self ):
        """`( nb_nbrs, ids, vals )`: for each cell `i`, its neighbors `j` ( `ids[ i, :nb_nbrs[ i ] ]`,
        indexed like `positions`; negative for the domain, to be ignored ) and `vals[ i, r ] =
        int_{facet ij} rho / ( 2 | p_i - p_j | )` -- enough to assemble the JACOBIAN of the measures with respect
        to the weights, `d m_i / d w_j = - vals`, `d m_i / d w_i = + sum_j vals`, which is also the
        Hessian of the dual functional of a transport ( `SdotPlanNd`, `objective = "newton"` ).
        Host arrays. A piecewise-constant distribution only. One call batched over the
        cells, like `cells` ( the number of neighbors per cell has a capacity that loom doubles )."""
        n, d = int( self.nb_points.value ), self.dim_count
        num_cell = new_batch_axis( n, prefix = "cell" )
        ranks = IntTensor[ num_cell ]( self._ranks_of_items() )

        dom = self._domain_cell()
        cap = self._scratch_capacity
        nbrs = Neighbors( batch_axes = [ num_cell ] )

        nb_words = self._scratch_words( cap, self._nb_work_cells(), False )
        nt = driver.device.nb_threads( nb_local_bytes_per_thread = 4 * nb_words, batch_axes = [ num_cell ] )
        dist_expr, _, dist_kwargs = self._dist_for()

        loom.ffi_call(
            "power_diagram_hessian_rows",
            FfiCode.per_item( code = "inputs.power_diagram.hessian_row( SI( inputs.ranks( batch_index ) ), inputs.dom_cell, "
                           f"outputs.nbrs( batch_index ), scratch.pool, thread_index, { dist_expr } );",
                max_nb_threads = "return scratch.pool.words.shape( 0 );" ),
            power_diagram = self,
            dom_cell = dom,
            ranks = ranks,
            pool = CellScratch.for_call( nb_words, dom.kernel_dtype, nb_threads = nt ),
            nbrs = loom.out( nbrs, capacities = { "nb_nbrs": 16 } ),
            **dist_kwargs,
        )
        counts = np.asarray( nbrs.nb_nbrs.value ).reshape( -1 ).astype( int )
        ids = np.asarray( nbrs.ids ).reshape( n, -1 )
        vals = np.asarray( nbrs.vals ).reshape( n, -1 )
        return counts, ids, vals

    @property
    def cells( self ) -> Cell:
        """ALL the cells, in ONE call: a `Cell` batched over the seeds, in the order of
        `positions`, its `cut_ids` designating the seeds in that same order.

        The query that does not reduce a cell to a number, hence the only one whose memory is
        a function of the number of seeds -- which is what a DISPLAY is. The scratch, for its part, stays PER
        WORK-ITEM ( `max_nb_threads` ). The returned `Cell` is drawn as is.
        """
        n, d = int( self.nb_points.value ), self.dim_count
        num_cell = new_batch_axis( n, prefix = "cell" )
        ranks = IntTensor[ num_cell ]( self._ranks_of_items() )

        dom = self._domain_cell()
        cap = self._scratch_capacity
        cells = Cell( d, init_as_unbounded = False, batch_axes = [ num_cell ], kernel_dtype = dom.kernel_dtype )

        nb_words = self._scratch_words( cap, 1, False )
        nt = driver.device.nb_threads( nb_local_bytes_per_thread = 4 * nb_words, batch_axes = [ num_cell ] )
        loom.ffi_call(
            "power_diagram_cells",
            FfiCode.per_item( code = "inputs.power_diagram.build_cell( SI( inputs.ranks( batch_index ) ), inputs.dom_cell, "
                           "outputs.cells( batch_index ), scratch.pool, thread_index );",
                max_nb_threads = "return scratch.pool.words.shape( 0 );" ),
            power_diagram = self,
            dom_cell = dom,
            ranks = ranks,
            pool = CellScratch.for_call( nb_words, dom.kernel_dtype, nb_threads = nt ),
            cells = loom.out( cells, capacities = { "nb_vertices": cap, "nb_cuts": cap } ),
        )
        return cells

    def cell( self, i ) -> Cell:
        """The cell of seed `i`, built PYTHON SIDE -- one `driver.call` per cut.

        The slow path, and deliberately so: the same geometry obtained by an orchestration
        entirely different from the kernel's, hence the tests' ORACLE. It is NOT the display
        path ( `n²` round trips ), see `cells`.
        """
        d = self.dim_count
        pos = np.asarray( self.positions ).reshape( -1, d )
        w = np.asarray( self.weights ).reshape( -1 ) if self.weights.is_defined else None

        res = self._start_cell()
        if self.bnd_directions.is_defined:
            bds = np.asarray( self.bnd_directions ).reshape( -1, d )
            bos = np.asarray( self.bnd_offsets ).reshape( -1 )
            for b in range( len( bds ) ):
                res.cut( bds[ b ], float( bos[ b ] ), BOUNDARY )

        p0 = pos[ i ]
        for j in range( len( pos ) ):
            if j == i:
                continue
            direction = pos[ j ] - p0
            offset = float( direction @ ( p0 + pos[ j ] ) / 2 )
            if w is not None:
                offset += float( w[ i ] - w[ j ] ) / 2
            res.cut( direction, offset, j )
        return res

    def add_to_viz( self, viz, **kwargs ):
        """Draws itself into a `Visualizer`: all the cells, in one call ( see `cells` )."""
        return self.cells.add_to_viz( viz, **kwargs )


class Neighbors( Aggregate ):
    """the neighbors of ONE cell and the weight of each facet ( see `PowerDiagram.hessian_rows` ) --
    batched over the cells, `nb_nbrs` per cell"""
    ids     : IntTensor [ "num_nbr", dict( size = 32 ) ]
    vals    : RealTensor[ "num_nbr" ]
    num_nbr : Axis[ "nb_nbrs" ]
    nb_nbrs : ShapeVar


# ---- the domain, read off half-spaces ------------------------------------------------------------

def axis_aligned_box( directions, offsets ):
    """`( mi, ma, kept )`: the box that these half-spaces bound, and which of them it does
    NOT replace. `None` if they do not bound a box.

    We do not try to recognize a box "written properly": we look, axis by axis, for the tightest
    bound that the half-spaces ALIGNED ON THAT AXIS give. A domain that is not
    a box but that contains one ( an octagon ) therefore still provides a starting point, and
    what sticks out is removed by the remaining planes. `None` as soon as an axis is not bounded on both
    sides: the starting cell must be a BOUNDED polytope.
    """
    try:
        dirs = np.asarray( directions, dtype = float )
        offs = np.asarray( offsets, dtype = float ).reshape( -1 )
    except ( TypeError, ValueError, RuntimeError ):      # RuntimeError: a torch tensor that requires grad
        return None                              # geometry not readable here ( a tracer ): too bad
    if dirs.ndim != 2 or len( dirs ) != len( offs ):
        return None

    d = dirs.shape[ 1 ]
    mi, ma = np.full( d, -np.inf ), np.full( d, np.inf )
    for k in range( len( dirs ) ):
        nz = np.flatnonzero( dirs[ k ] )
        if nz.size != 1:
            continue                             # not aligned on an axis: it will be cut, that's all
        a = int( nz[ 0 ] )
        c = dirs[ k, a ]
        if c > 0:
            ma[ a ] = min( ma[ a ], offs[ k ] / c )
        else:
            mi[ a ] = max( mi[ a ], offs[ k ] / c )

    if not np.isfinite( mi ).all() or not np.isfinite( ma ).all():
        return None
    if not ( mi < ma ).all():                    # an empty box cannot be laid down: the general path will empty it
        return None

    # WHICH planes the box already expresses: those, aligned, that reach the retained bound on their axis
    kept = np.ones( len( dirs ), dtype = bool )
    for k in range( len( dirs ) ):
        nz = np.flatnonzero( dirs[ k ] )
        if nz.size != 1:
            continue
        a = int( nz[ 0 ] )
        c = dirs[ k, a ]
        bound = ma[ a ] if c > 0 else mi[ a ]
        kept[ k ] = offs[ k ] / c != bound
    return mi, ma, kept


def box_half_spaces( mi, ma ):
    """The box `mi <= x <= ma` as `2d` half-spaces `direction . x <= offset`."""
    mi = np.asarray( mi, dtype = float ).reshape( -1 )
    ma = np.asarray( ma, dtype = float ).reshape( -1 )
    if mi.size != ma.size:
        raise ValueError( "`box_half_spaces( mi, ma )` wants two corners of the same dimension" )
    d = mi.size
    return np.concatenate( [ np.eye( d ), -np.eye( d ) ] ), np.concatenate( [ ma, -mi ] )
