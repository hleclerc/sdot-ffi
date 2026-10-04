"""THE SOLUTION of a semi-discrete transport in dimension `d >= 2`: the weights of a power
diagram such that the mass of each cell against the target density equals the mass of the
matching dirac.

It is requested from an `OtProblem`, and in no other way:

    sol = OtProblem( SumOfDiracs( pos ), image ).solve()
    sol = ot_solve( SumOfDiracs( pos ), image )        # the same, for a transport solved ONCE

THE WHOLE FIT IS IN C++ ( `sdot/sdotplan/` ), in ONE `driver.call`: the starting point, the
damped Newton, its diagrams, the laplacian, the linear solver, the damping. Python only sets up
the problem and reads what comes out. That is what the `solvers_des_familles` bench concluded
( README § 3, § 7, § 9, § 10 ):

  * the DAMPED NEWTON of Kitagawa-Mérigot-Thibert wins everywhere, by 2x to 4x in diagrams as in
    time, against L-BFGS and the conjugate gradient on the dual, whatever their
    preconditioning -- the difficulty of semi-discrete transport is not the non-linearity of the
    dual, it is its NON-REGULARITY ( cells that empty out ), and a gradient step is crushed
    there just like a Newton step, only worse ( `sdotplan/Newton.h` );
  * the hessian is the laplacian of the Laguerre graph, assembled WITHOUT SORTING from the facets
    of the diagram that measured the residual -- one sweep delivers both ( `sdotplan/Sweep.h`,
    `sdotplan/Laplacian.h` ); sparse Cholesky in 2D, algebraic multigrid beyond and at large
    sizes ( `sdotplan/Linear.h` );
  * the trial step restarts from the last accepted step, never from 1 ( ten diagrams per step
    saved in the linear phase ); in 2D, the LIMITS of the cells being crushed along the
    direction replace blind trials ( `sdotplan/Bounds.h` ).

= On a CUDA card

The same solve, in ONE ffi call too, whose handler drives the Newton loop on the call's stream ( `gpu/Newton2D.cuh`, see
`_build_card` ): the tree's majorants, the cells, the laplacian, the linear solver ( the card's multigrid ) and the step all
run on the card, and the host only reads back the scalars it decides on -- so the solve runs under `jax.jit` as well. It
takes 2D problems in a box against a CONSTANT density; anything else raises on a card ( the CPU solves it ).

= The starting point

Newton needs an ADMISSIBLE start ( no empty cell ). Voronoi is one as soon as the diracs are in
the domain; otherwise, or if the given weights empty a cell, the C++ itself picks the best of
the three -- the given weights, Voronoi, the SIMILARITY that brings the cloud back into the
domain -- and reports it ( `stats[ "start" ]` ). See `sdotplan/Solve.h`.

= What the solution CARRIES, and why it is not just a weight vector

`weights` is the answer when the seeds are distinct. When two seeds are MERGED to `1e-8`,
there is no pair of `double`s that encodes the plane separating them to within `1e-9`: the bench
measured and computed it ( README § 23.11 ), and that is why a solver whose interface is `w`
plateaus around `1e-6` on a degenerate cloud. The solution therefore ALSO carries the aggregate
-- `clusters`, `cluster_nu` -- even when there is no cluster, so that the rule is readable once
and for all: if the consumer wants CELLS, it takes the reduced diagram plus the planes, and
everything is exact; if it wants WEIGHTS, it accepts the floor `eps |w| / ( 2 delta h )`.

What it does not cost: the TRANSPORT COST is blind to this degeneracy ( § 23.12, relative gap
`1.2e-17` ), so anything that is a sum weighted by the masses -- the cost, the total mass, a
global moment -- need know nothing about it.

= A single diagram

The positions are the CONSTANTS of the fit: the diagram is built ONCE ( its tree included ), and
the solver only SETS the trial weights on it -- which, for BSP storage, recomputes the weight
upper bound of each node and nothing else. What it writes ( the sorted weights, the upper
bounds ) are OUTPUTS of the call, taken back by the diagram afterwards: the inputs of a call
are read-only.

`PowerDiagram` and `Cell` are ON-DEMAND TOOLS: `sol.power_diagram()` builds them for drawing or
for a functional that was not anticipated. Nothing in the solve goes through them on the
Python side.
"""

import warnings

import loom
from loom.compilation.FfiCode import FfiCode
from loom.drivers.driver import driver
from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar, Tensor
from loom.util import Aggregate

from .CellScratch import fp_size
from .PowerDiagram import PowerDiagram


# what `stats` carries, in the order of `sdotplan/Solve.h::Stat`
_STATS = [ "status", "residual", "residual0", "nb_iter", "nb_diag", "nb_backtracks", "t_majorant", "t_diag", "t_asm", "t_lin", "t_lim", "eps",
           "domain_mass", "nb_overflowed", "nb_cell_lim", "nb_limit_rounds", "lin_nb_hierarchies", "lin_nb_iter", "lin_worst", "start", "t_total",
           "nb_continuation_steps", "min_start_mass", "it_switch", "it_double" ]
# one row of `history`, in the order of `sdotplan/Solve.h::Hist`
_HISTORY = [ "step", "t", "residual_l2", "min_measure", "max_abs_residual", "nb_diag", "nb_evals", "s" ]
_STATUS = { 0: "running", 1: "converged", 2: "max iterations", 3: "stagnation", 4: "linear solver failure", 5: "card capacity",
            6: "failure" }
_START = { 0: "weights0", 1: "voronoi", 2: "similarity" }
_LIN = { "auto": 0, "cholesky": 1, "amg": 2, "cg": 3, "mg": 4 }
_AMG_VARIANT = { "auto": -1, "sa_spai0": 0, "sa_gs": 1, "rs_gs": 2 }
_STEP = { "trials": 0, "limits": 1 }
_RESIDUAL = { "lin": 0, "log": 1, "power": 2 }
_CONTINUATION = { "never": 0, "auto": 1, "always": 2 }

#: the old arguments of `SdotPlanNd( src, dst, ... )`, and where they live now -- read by the
#: deprecated path ( see `__init__` )
_DEPRECATED_TO_TUNING = ( "accelerator", "memory", "step", "linear_solver", "mass_rtol", "t_min",
                          "max_backtracks", "restart_factor", "conv_start", "conv_ratio", "conv_min",
                          "conv_threshold" )


class _Options( Aggregate ):
    """the solver settings, as `sdotplan/Solve.h` reads them"""
    mass_tol       : RealTensor
    mass_rtol      : RealTensor
    t_min          : RealTensor
    mult_ok        : RealTensor
    factor         : RealTensor
    beta0          : RealTensor
    mult_lim       : RealTensor
    confidence     : RealTensor
    conv_s0        : RealTensor
    conv_ratio     : RealTensor
    conv_min       : RealTensor
    conv_threshold : RealTensor
    residual_power : RealTensor
    lin_tol        : RealTensor
    residual_switch : RealTensor
    max_iter       : IntTensor
    max_backtracks : IntTensor
    lin            : IntTensor
    amg_variant    : IntTensor
    mg_pack        : IntTensor
    mg_recycle     : IntTensor
    mg_rebuild     : IntTensor
    mg_stop        : IntTensor
    mg_nu          : IntTensor
    step           : IntTensor
    residual       : IntTensor
    trace          : IntTensor
    continuation   : IntTensor
    cap0           : IntTensor
    kernel_fp_size : CtShapeVar


#: the card solver's options, ONE real tensor ( `gpu/Newton2D.cuh::Opt`, same order )
_CARD_OPTIONS = [ "tol_abs", "tol_rel", "t_min", "mult_ok", "factor", "maxit", "max_backtracks", "step", "residual", "power", "switch",
                  "lin", "host_method", "lin_tol", "amg_variant", "mg_shift", "mg_recycle", "mg_rebuild", "mg_stop", "mg_nu", "mg_kcycle",
                  "lin_maxit", "trace", "mg_float", "mg_smoothed" ]


def _card_linear_solves( row, col, val, dia, rhs, method = "mg", tol = 1e-6, smoothed = None, precision = "float", recycle = 2, rebuild = 1,
                         kcycle = -1, nu = 0 ):
    """THE CARD'S LINEAR SOLVER ALONE ( a test hook, `gpu/Newton2D.cuh::linear_solves` ): the systems `L d_j = rhs[ j ]` ( each
    `rhs[ j ]` of zero sum ) with ONE laplacian `L` given as the CSR of `Laplacian2D.cuh` ( `row`, `col`, `val > 0` the
    off-diagonals, `dia` ), solved in sequence by ONE solver, as along a Newton solve ( the recycled start: `recycle`
    solutions; a hierarchy every `rebuild` solves ). `method`: `"mg"` or `"cg"`; `smoothed`: the levels of the smoothed
    aggregation ( `None`: the default ); `precision`: of the multigrid's levels. Returns `( d, its )`: the solutions ( zero
    mean, `k x n` ) and the iterations of each solve ( -1: failed )."""
    import numpy as np
    rhs = np.atleast_2d( np.asarray( rhs, dtype = np.float64 ) )
    k, n = rhs.shape
    row = np.asarray( row, dtype = np.int64 ).reshape( -1 )[ :n + 1 ]
    nnz = int( row[ n ] )
    opts = [ n, k, 0 if method == "cg" else 1, tol, -1 if smoothed is None else smoothed, int( precision == "float" ), recycle, rebuild, kcycle, nu ]
    def axis( m, name ):
        return Axis( ShapeVar( int( m ) ), name = name )
    solution = RealTensor[ axis( k * n, "num_lin_all" ) ]()
    iterations = RealTensor[ axis( k, "num_lin_rhs" ) ]()
    loom.ffi_call(
        "sdotplan_card_linear_2d",
        FfiCode.inline( "sdot::gpu2d::linear_solves( queue, args.inputs.row, args.inputs.col, args.inputs.val, args.inputs.dia, args.inputs.rhs, "
                        "args.inputs.options, args.outputs.solution, args.outputs.iterations, args.allocator );",
                        includes = [ "sdot/gpu/Newton2D.cuh" ], sources = [ "sdot/sdotplan/Linear.cpp" ], allocator = True ),
        row = IntTensor[ axis( n + 1, "num_lin_row" ), dict( size = 64 ) ]( row ),
        col = IntTensor[ axis( max( nnz, 1 ), "num_lin_nnz" ), dict( size = 64 ) ]( np.asarray( col, dtype = np.int64 ).reshape( -1 )[ :max( nnz, 1 ) ] ),
        val = RealTensor[ axis( max( nnz, 1 ), "num_lin_val" ) ]( np.asarray( val, dtype = np.float64 ).reshape( -1 )[ :max( nnz, 1 ) ] ),
        dia = RealTensor[ axis( n, "num_lin_dia" ) ]( np.asarray( dia, dtype = np.float64 ).reshape( -1 )[ :n ] ),
        rhs = RealTensor[ axis( k * n, "num_lin_b" ) ]( rhs.reshape( -1 ) ),
        options = RealTensor[ axis( len( opts ), "num_lin_opt" ) ]( np.asarray( opts, dtype = np.float64 ) ),
        solution = loom.out( solution ),               # ( not `d`: a name loom's generated templates use )
        iterations = loom.out( iterations ),
    )
    return np.asarray( solution.raw ).reshape( k, n ), np.asarray( iterations.raw ).reshape( -1 ).astype( int )


#: the precision of the card multigrid's levels by default ( the outer flexible CG is in double either way ): float, the
#: same iteration counts as double and 1.15-1.4x faster ( `calibration_lmo_today.md`, GPU step 5 )
_CARD_MG_PRECISION = "float"


def card_facet_capacity( nb_seeds ):
    """the first guess of the upper facets of a 2D diagram ( a planar graph: at most `3 n - 6` edges, and a margin for the
    slivers of a float topology ); loom grows it if a diagram wants more"""
    return 3 * int( nb_seeds ) + 512


class _CardSolveWork( Aggregate ):
    """what the card's solve writes besides its results: the STATUS of each cell at the last diagram ( 0 = done,
    `Cell2D.cuh::Status` ), and two capacities that a diagram may ask loom to grow ( `nb_spill`: vertices per cell of the
    fourth pass; `nb_facets`: the upper facets of a diagram )"""
    status    : IntTensor[ "num_point", dict( size = 32 ) ]
    num_point : Axis[ "nb_points" ]
    nb_points : ShapeVar
    nb_spill  : ShapeVar
    nb_facets : ShapeVar


class _History( Aggregate ):
    """one ACCEPTED step per row ( `step = 0`: the start ), and the weights of each step if they
    were requested ( `weights`, otherwise `Unbound` -- a `[ step, n ]` array that is not always wanted )"""
    rows      : RealTensor[ "num_step", "num_hist" ]
    weights   : RealTensor[ "num_step", "num_point" ]
    num_step  : Axis[ "nb_steps" ]
    num_hist  : Axis[ "nb_hist" ]
    num_point : Axis[ "nb_points" ]
    nb_steps  : ShapeVar
    nb_hist   : ShapeVar
    nb_points : ShapeVar


class SdotPlanNd:
    """see the module docstring"""

    # -- how to get one -----------------------------------------------------------------------

    @classmethod
    def _solve( cls, problem, settings, verbose, warm = None ):
        """THE path: `OtProblem.solve()` and only it goes through here. `warm` is the last SOLUTION
        the problem returned -- the start when the settings do not impose one ( see
        `_start_from_plan` )."""
        self = cls.__new__( cls )
        self._build( problem, settings, verbose, warm )
        return self

    def __init__( self, src_dist, dst_dist, verbose = False, **kwargs ):
        """DEPRECATED -- `OtProblem( src_dist, dst_dist ).solve( Iterative( ... ) )`.

        There is now a single entry point: a problem is posed ( `OtProblem` ), then a solution is
        requested from it. This constructor translates the old arguments and will be removed."""
        warnings.warn( "SdotPlanNd( src, dst, ... ) is deprecated: pose the problem then "
                       "solve it -- OtProblem( src, dst ).solve( Iterative( ... ) ). See the "
                       "docstring of `OtProblem`.", DeprecationWarning, stacklevel = 2 )
        from .OtProblem import Iterative, OtProblem, Tuning

        kw = dict( kwargs )
        if "kernel_dtype" in kw:
            kd = kw.pop( "kernel_dtype" )
            kw[ "precision" ] = { None: "auto", "FP64": "fp64", "FP32": "fp32" }.get( kd, kd )
        if "mass_tol" in kw:
            kw[ "tol" ] = kw.pop( "mass_tol" )
        tuning = Tuning( **{ k: kw.pop( k ) for k in _DEPRECATED_TO_TUNING if k in kw } )
        self._build( OtProblem( src_dist, dst_dist ), Iterative( tuning = tuning, **kw ), verbose )

    # -- what the call does -------------------------------------------------------------------

    def _build( self, problem, settings, verbose, warm = None ):
        # two solvers: the CPU's ( `sdotplan/Solve.h`, host code on the CPU queue ) and, on a CUDA device, the card's
        # ( `gpu/Newton2D.cuh`: 2D, a box, a constant density -- see `_build_card` ); any other device is refused
        on_card = bool( getattr( driver.device, "is_cuda_gpu", False ) )
        if not driver.device.is_cpu and not on_card:
            raise NotImplementedError( f"SdotPlanNd: no solver for the device { driver.device } ( the CPU, or a CUDA card in 2D )" )
        tun = settings.tuning
        if settings.precision == "mixed" and not on_card:
            raise ValueError( "precision = 'mixed' ( the float kernel, then the double one ) is the card's: on the CPU, 'fp64' or 'fp32'" )
        #: the problem this is the solution of
        self.problem = problem
        src_dist, dst_dist = problem.source, problem.target
        d = problem.nb_dims

        # the domain: the support of the density, which must bound it ( raises otherwise ).
        # `PowerDiagram` adds it itself from the distribution -- we ASK for it here so that the
        # refusal is reported before anything has been allocated.
        problem.domain

        # the limits-based step only exists in 2D ( `sdotplan/Bounds.h` ); elsewhere, trials
        step = { "auto": "limits" if d == 2 else "trials" }.get( tun.step, tun.step )
        if step == "limits" and d != 2:
            raise ValueError( "step = 'limits': 2D only for now ( see `sdotplan/Bounds.h` )" )
        if step not in _STEP:
            raise ValueError( f"unknown step: { tun.step !r } ( 'auto', 'trials' or 'limits' )" )
        if tun.amg_variant not in _AMG_VARIANT:
            raise ValueError( f"unknown amg_variant: { tun.amg_variant !r } ( { ', '.join( _AMG_VARIANT ) } )" )
        if tun.residual not in _RESIDUAL:
            raise ValueError( f"unknown residual: { tun.residual !r } ( { ', '.join( _RESIDUAL ) } )" )
        if tun.linear_solver not in _LIN:
            raise ValueError( f"unknown linear_solver: { tun.linear_solver !r } ( { ', '.join( _LIN ) } )" )

        # THE START. Three sources, in this order: the given plan, the given bare weights, the last
        # plan of the problem. `_start_from_plan` says what it could draw from it, and
        # `stats[ "warm_start" ]` reports it -- a silently discarded warm start is exactly what loses
        # an afternoon.
        plan = settings.ot_plan if settings.ot_plan is not None else warm
        impose = settings.ot_plan is not None
        if settings.weights0 is not None:
            w0_given, self._warm_start = settings.weights0, "weights0"
        elif plan is not None:
            w0_given, self._warm_start = self._start_from_plan( plan, src_dist, impose )
        else:
            w0_given, self._warm_start = None, "none"

        # THE diagram, built once on the positions ( see the module docstring ); the weights it
        # carries at a given moment are the last ones set
        # Under a trace ( `jax.jit` ) the weights may be tracers, and a diagram with traced weights falls back on the
        # plain storage. The zeros of a cold start are made on the host ( a constant ); given traced weights, the TREE is
        # built on the positions alone ( it only depends on them ) and the weights enter its majorants ( a kernel )
        accelerator = tun.accelerator
        pos_raw = getattr( src_dist.positions, "raw", src_dist.positions )
        if w0_given is not None and accelerator is None and not driver.is_traced( pos_raw ) \
                and driver.is_traced( getattr( w0_given, "raw", w0_given ) ):
            from .AaBsp import AaBsp
            accelerator = AaBsp( pos_raw )
        # ( and what is concrete is EVALUATED under the trace: the domain, the tree, the density's values stay readable )
        import numpy as np
        with driver.concrete_eval():
            self._pd = PowerDiagram( src_dist.positions,
                                     np.zeros( int( src_dist.nb_diracs.value ) ) if w0_given is None else w0_given,
                                     accelerator = accelerator, kernel_dtype = settings.kernel_dtype,
                                     distribution = dst_dist, memory = tun.memory,
                                     scratch_capacity = tun.scratch_capacity )
        pd = self._pd
        n = int( pd.nb_points.value )

        #: the target masses, indexed like the cells
        self._masses = RealTensor[ pd.num_point ]( src_dist.weights.raw )

        if on_card:
            return self._build_card( pd, settings, step, verbose )

        options = _Options(
            mass_tol = float( settings.tol ), mass_rtol = float( tun.mass_rtol ), t_min = float( tun.t_min ),
            mult_ok = float( tun.restart_factor ), factor = 0.9, beta0 = 0.25, mult_lim = 2.0, confidence = 0.0,
            conv_s0 = float( tun.conv_start or 0.0 ), conv_ratio = float( tun.conv_ratio ),
            conv_min = float( tun.conv_min or 0.0 ), conv_threshold = float( tun.conv_threshold ),
            residual_power = float( tun.residual_power ), lin_tol = float( tun.linear_tol or 0.0 ), residual_switch = float( tun.residual_switch ),
            max_iter = int( settings.max_iter ), max_backtracks = int( tun.max_backtracks ), lin = _LIN[ tun.linear_solver ], amg_variant = _AMG_VARIANT[ tun.amg_variant ],
            mg_pack = int( tun.mg_pack or 0 ), mg_recycle = -1 if tun.mg_recycle is None else int( tun.mg_recycle ), mg_rebuild = int( tun.mg_rebuild or 0 ), mg_stop = int( tun.mg_stop or 0 ), mg_nu = int( tun.mg_nu or 0 ),
            step = _STEP[ step ], residual = _RESIDUAL[ tun.residual ], trace = int( bool( verbose ) ), continuation = _CONTINUATION[ settings.continuation ],
            cap0 = int( pd._scratch_capacity ),
            kernel_fp_size = fp_size( pd.kernel_dtype ),
        )

        weights = RealTensor[ pd.num_point ]()
        history = _History( nb_hist = len( _HISTORY ), nb_points = n )
        # THE AXES ARE THOSE OF THE DIAGRAM, and it is the one thing not to miss here: a `num_point`
        # axis built separately carries the same NAME without being the same, and
        # `cell_masses * ( positions - barycenters )` then becomes an outer product `[ n, n, d, d ]`
        # instead of the gradient. They are therefore declared from `pd`.
        cell_masses = RealTensor[ pd.num_point ]()
        barycenters = RealTensor[ pd.num_point, pd.dim ]()
        cost        = RealTensor()
        stats = RealTensor[ Axis( ShapeVar( len( _STATS ) ), name = "num_stat" ) ]()
        w0 = RealTensor[ pd.num_point ]( pd.weights.raw )

        dom = pd._domain_cell()
        dist_expr, _, dist_kwargs = pd._dist_for()
        pd_expr, pd_kwargs, pd_produced = pd._solver_weights_call()

        loom.ffi_call(
            "sdotplan_solve",
            # `handler` and not the default scaffolded kernel: this body IS the handler. It is
            # HOST code -- it needs the `queue`, and it drives its own parallelism ( a hundred
            # diagrams in a single call ), so there is no per-item functor nor `run_parallel`
            # to generate around it. See the docstring of `FfiCode`.
            # `inline`: the body IS that of the handler, loom only writes its wrapper
            # ( `void kernel( queue, batch_axes, args )` ). It is HOST code -- it needs the
            # `queue`, and it drives its own parallelism ( a hundred diagrams in a single call ),
            # so there is no per-item functor nor `run_parallel` to generate around it.
            FfiCode.inline( includes = [ "sdot/sdotplan/Solve.h" ],
                sources = [ "sdot/sdotplan/Linear.cpp" ],
                code = "\n".join( [
                    # the expressions that `PowerDiagram` generates name `inputs` / `outputs` ( the shape
                    # of a per-item kernel ): we FIND them again here under their name, and nothing needs to know it
                    "auto &inputs = args.inputs; auto &outputs = args.outputs;",
                    "using TK_sdotplan = std::conditional_t<CT_VALUE( inputs.options.kernel_fp_size ) == 64, double, float>;",
                    f"auto pd_sdotplan = { pd_expr };",
                    "sdotplan::SolverOptions os;",
                    "sdotplan::NewtonOptions &no = os.newton;",
                    "no.tol_abs = double( inputs.options.mass_tol ); no.tol_rel = double( inputs.options.mass_rtol ); no.t_min = double( inputs.options.t_min );",
                    "no.mult_ok = double( inputs.options.mult_ok ); no.factor = double( inputs.options.factor ); no.beta0 = double( inputs.options.beta0 );",
                    "no.mult_lim = double( inputs.options.mult_lim ); no.confidence = double( inputs.options.confidence );",
                    "no.maxit = int( SI( inputs.options.max_iter ) ); no.max_backtracks = int( SI( inputs.options.max_backtracks ) );",
                    "no.step = int( SI( inputs.options.step ) ); no.trace = SI( inputs.options.trace ) != 0;",
                    "no.residual = int( SI( inputs.options.residual ) ); no.power = double( inputs.options.residual_power ); no.switch_residual = double( inputs.options.residual_switch );",
                    "os.lin = sdotplan::Lin( int( SI( inputs.options.lin ) ) ); os.cap0 = SI( inputs.options.cap0 );",
                    "os.lin_options.tol = double( inputs.options.lin_tol ); os.lin_options.amg_variant = int( SI( inputs.options.amg_variant ) );",
                    "os.lin_options.mg_pack = int( SI( inputs.options.mg_pack ) ); os.lin_options.mg_recycle = int( SI( inputs.options.mg_recycle ) ); os.lin_options.mg_rebuild = int( SI( inputs.options.mg_rebuild ) ); os.lin_options.mg_stop = int( SI( inputs.options.mg_stop ) ); os.lin_options.mg_nu = int( SI( inputs.options.mg_nu ) );",
                    "os.continuation = int( SI( inputs.options.continuation ) ); os.continuation_threshold = double( inputs.options.conv_threshold );",
                    "os.conv_s0 = double( inputs.options.conv_s0 ); os.conv_ratio = double( inputs.options.conv_ratio ); os.conv_min = double( inputs.options.conv_min );",
                    f"sdotplan::solve<TK_sdotplan>( queue, pd_sdotplan, inputs.power_diagram, inputs.dom_cell, { dist_expr }, inputs.nu, inputs.w0, os, "
                    "outputs.weights, outputs.history, outputs.stats, outputs.cell_masses, outputs.barycenters, outputs.cost );",
                ] ) ),
            power_diagram = pd,
            dom_cell = dom,
            nu = self._masses,
            w0 = w0,
            options = options,
            weights = loom.out( weights ),
            # `nb_steps` is WRITTEN by the kernel ( the number of accepted steps ), so it must be
            # named: a `ShapeVar` that is not declared stays unbound, and the C++ only sees a null
            # view. The weights of each step, for their part, are only written if they were requested --
            # an unnamed `weights` stays observed, hence neither allocated nor read.
            history = loom.out( history, writes = ( [ "rows", "nb_steps", "weights" ] if settings.keep_weights
                                                    else [ "rows", "nb_steps" ] ),
                                capacities = { "nb_steps": int( settings.max_iter ) + 1 } ),
            stats = loom.out( stats ),
            cell_masses = loom.out( cell_masses ),
            barycenters = loom.out( barycenters ),
            cost = loom.out( cost ),
            has_dynamic_capacity = False,
            **pd_kwargs,
            **dist_kwargs,
        )
        pd._solver_weights_after( pd_produced )

        #: the FITTED weights, `[ n ]`, indexed like `positions` ( `weights[ 0 ] == 0`: the gauge )
        self.weights = weights
        #: the measure of each cell at the FITTED weights, `[ n ]` -- the one Newton measured
        self.cell_masses = cell_masses
        #: the barycenter of each cell, `[ n, d ]` ( its seed if it is empty )
        self.barycenters = barycenters
        #: the transport cost `W_2^2` ( a scalar `Tensor`: `float( sol.cost )` for the number )
        self.cost = cost
        self._read_stats( stats, settings )
        self._read_history( history, settings, pd )

    def _build_card( self, pd, settings, step, verbose ):
        """THE CARD'S SOLVE ( `include/sdot/gpu/Newton2D.cuh` ): the same Newton, the same options and outputs as the CPU's, in
        ONE ffi call whose handler drives the loop on the call's stream -- so it runs under `jax.jit` too. Taken in 2D, with
        the BSP tree, a box domain and a CONSTANT density ( `PowerDiagram_Bsp._card_variant` ); anything else is refused
        here rather than solved on another path: an `Image` or gaussians need the card's cells to integrate a density,
        which they do not do yet.

        What differs from the CPU ( said in `Newton2D.cuh` ): `step = "limits"` is the exact step of the area polynomials
        ( all the cells, from the accepted diagram's edges ) checked by the trial diagram, not the CPU's local limits with a
        first trial `beta`; the linear solver is the card's multigrid ( `"auto"`, `"mg"` ) or CG ( `"cg"` ), or the CPU's
        through a copy of the laplacian ( `"cholesky"`, `"amg"`, or `Tuning( linear_host = True )` ); there is no width
        continuation ( `"always"` is refused, `"auto"` proceeds without )."""
        import numpy as np
        tun = settings.tuning
        reason = None
        if pd.dim_count != 2:
            reason = "the card solves in 2D only ( 3D: the CPU )"
        elif settings.continuation == "always":
            reason = "the width continuation needs a convolved density, which the card does not integrate yet"
        variant = pd._card_variant() if reason is None else None
        if reason is None and variant is None:
            reason = ( "the card's cells take a box domain, a CONSTANT density ( no density, or an `Image` with all its "
                       "values equal on exactly the box ), the BSP tree and no neighbour memory -- an `Image` / gaussian "
                       "density on the card is not there yet" )
        if reason is not None:
            raise NotImplementedError( f"SdotPlanNd on a CUDA device: { reason }. Use the CPU device ( LOOM_DEVICE=cpu ) "
                                       "for this problem." )
        variant, rho = variant
        n = int( pd.nb_points.value )
        # THE KERNEL'S FLOAT, chosen here: `fp64` / `fp32` one kernel; `auto` / `mixed` the float kernel, then the double one
        # from the iteration where a float step stagnates ( `Newton2D.cuh`: a cut decided in float on a degenerate cloud stops
        # the merit from decreasing -- the lines with equal areas -- while elsewhere the float kernel's measures, re-solved in
        # double, reach the double's residual for a fifth of its time )
        from .PowerDiagram_Bsp import card_variant_for
        nodes = int( pd.tree.nb_bsp_nodes.value )
        if settings.precision in ( "auto", "mixed" ):
            variant = f"{ card_variant_for( 32, n, nodes ) }, { card_variant_for( 64, n, nodes ) }"
        else:
            variant = card_variant_for( 32 if settings.precision == "fp32" else 64, n, nodes )

        # the linear solver: the card's ( `Linear2D.cuh` ), or a CPU one of `Linear.cpp` on a copy of the laplacian
        lin_name = tun.linear_solver
        if getattr( tun, "linear_host", False ) or lin_name in ( "cholesky", "amg" ):
            lin_kind = 2
        else:
            lin_kind = 0 if lin_name == "cg" else 1
        pack = int( tun.mg_pack or 0 )
        if pack and ( pack & ( pack - 1 ) ):
            raise ValueError( f"mg_pack must be a power of two ( got { pack } )" )
        opts = [ 0.0 ] * len( _CARD_OPTIONS )
        def put( name, v ):
            opts[ _CARD_OPTIONS.index( name ) ] = float( v )
        put( "tol_abs", settings.tol ); put( "tol_rel", tun.mass_rtol ); put( "t_min", tun.t_min ); put( "mult_ok", tun.restart_factor )
        put( "factor", 0.9 ); put( "maxit", settings.max_iter ); put( "max_backtracks", tun.max_backtracks ); put( "step", _STEP[ step ] )
        put( "residual", _RESIDUAL[ tun.residual ] ); put( "power", tun.residual_power ); put( "switch", tun.residual_switch )
        put( "lin", lin_kind ); put( "host_method", _LIN[ lin_name ] ); put( "lin_tol", tun.linear_tol or 0.0 )
        put( "amg_variant", _AMG_VARIANT[ tun.amg_variant ] ); put( "mg_shift", pack.bit_length() - 1 if pack else 0 )
        put( "mg_recycle", -1 if tun.mg_recycle is None else tun.mg_recycle ); put( "mg_rebuild", tun.mg_rebuild or 0 )
        put( "mg_stop", tun.mg_stop or 0 ); put( "mg_nu", tun.mg_nu or 0 )
        put( "mg_kcycle", -1 if getattr( tun, "mg_kcycle", None ) is None else tun.mg_kcycle )
        put( "lin_maxit", 0 ); put( "trace", int( bool( verbose ) ) )
        mg_precision = getattr( tun, "mg_precision", None ) or _CARD_MG_PRECISION
        if mg_precision not in ( "float", "double" ):
            raise ValueError( f"mg_precision: 'float' or 'double' ( got { mg_precision !r } )" )
        put( "mg_float", int( mg_precision == "float" ) )
        put( "mg_smoothed", -1 if getattr( tun, "mg_smoothed", None ) is None else tun.mg_smoothed )
        options = RealTensor[ Axis( ShapeVar( len( _CARD_OPTIONS ) ), name = "num_card_opt" ) ]( np.asarray( opts, dtype = np.float64 ) )

        weights = RealTensor[ pd.num_point ]()
        history = _History( nb_hist = len( _HISTORY ), nb_points = n )
        cell_masses = RealTensor[ pd.num_point ]()
        barycenters = RealTensor[ pd.num_point, pd.dim ]()
        cost        = RealTensor()
        stats = RealTensor[ Axis( ShapeVar( len( _STATS ) ), name = "num_stat" ) ]()
        w0 = RealTensor[ pd.num_point ]( pd.weights.raw )
        work = _CardSolveWork( nb_points = n )
        pd_expr, pd_kwargs, pd_produced = pd._solver_weights_call()
        maxv = int( pd.card_max_vertices )

        loom.ffi_call(
            "sdotplan_solve_card_2d",
            FfiCode.inline(
                f"sdot::gpu2d::solve<{ variant }>( queue, args.inputs.power_diagram, args.inputs.nu, args.inputs.w0, args.inputs.options, "
                "args.outputs.weights, args.outputs.history, args.outputs.stats, args.outputs.cell_masses, args.outputs.barycenters, "
                "args.outputs.cost, args.outputs.sorted_weights_out, args.outputs.node_wa_out, args.outputs.node_wb_out, args.outputs.work, "
                f"args.errors, args.allocator, args.inputs.density, { maxv } );",
                includes = [ "sdot/gpu/Newton2D.cuh" ], sources = [ "sdot/sdotplan/Linear.cpp" ], allocator = True ),
            failures = pd._card_failures(),
            power_diagram = pd,
            nu = self._masses,
            w0 = w0,
            options = options,
            density = RealTensor( np.float64( rho ) ),
            weights = loom.out( weights ),
            history = loom.out( history, writes = ( [ "rows", "nb_steps", "weights" ] if settings.keep_weights
                                                    else [ "rows", "nb_steps" ] ),
                                capacities = { "nb_steps": int( settings.max_iter ) + 1 } ),
            stats = loom.out( stats ),
            cell_masses = loom.out( cell_masses ),
            barycenters = loom.out( barycenters ),
            cost = loom.out( cost ),
            work = loom.out( work, capacities = { "nb_spill": int( pd.card_spill_capacity ), "nb_facets": card_facet_capacity( n ) } ),
            **pd_kwargs,
        )
        pd._solver_weights_after( pd_produced )
        if not driver.is_traced( work.status.raw ):
            #: the per-cell status of the last diagram of the card's solve ( 0: done, `Cell2D.cuh::Status` )
            self.card_status = work.status
        self.weights = weights
        self.cell_masses = cell_masses
        self.barycenters = barycenters
        self.cost = cost
        self._read_stats( stats, settings )
        self._read_history( history, settings, pd )

    @staticmethod
    def _start_from_plan( plan, src_dist, impose ):
        """What a previous PLAN provides as a start: `( weights0, what_was_taken_back )`.

        A plan whose number of diracs no longer matches is worthless ( the common case: a stage
        of multi-scale ). It is dropped, but REPORTED -- and if the caller had imposed it
        explicitly through `ot_plan`, we raise, because they believe they are warm-starting and
        are not.

        The CLUSTERS are not yet taken back: there are none ( aggregation is step 7 of
        `notes/2026-10-02-sdotplan.md` ). When they arrive, this is where they go through -- and
        that is why the start is a plan and not a weight vector: `plan.clusters` and the
        positions it was solved for are what allows not re-detecting the clusters when the seeds
        have not moved ( README § 23.11, § 23.8 )."""
        n_plan = int( plan.weights.shape[ 0 ] )
        n = int( src_dist.nb_diracs.value )
        if n_plan != n:
            if impose:
                raise ValueError( f"Iterative( ot_plan = ... ): this plan carries { n_plan } weights and the "
                                  f"source has { n } -- it cannot serve as a start. Remove it, "
                                  "or keep an `OtProblem` ( it expires its solution on its own )" )
            return None, f"none ( the kept plan carries { n_plan } weights, the source has { n } )"
        return plan.weights, "ot_plan"

    def _read_stats( self, stats, settings ):
        #: what the solver reports ( see `sdotplan/Solve.h::Stat` ), plus `status` and `start` spelled out
        st = stats.raw
        if driver.is_traced( st ):
            # under a trace ( `jax.jit` ): the numbers are tracers, kept as such -- nothing is read on the host
            self.stats = { name: st[ k ] for k, name in enumerate( _STATS ) }
            self.stats[ "aggregation" ] = "requested, not wired yet ( step 7 )" if settings.aggregate else "not requested"
            self.stats[ "warm_start" ] = self._warm_start
            return
        self.stats = { name: float( st[ k ] ) for k, name in enumerate( _STATS ) }
        self.stats[ "status" ] = _STATUS.get( int( self.stats[ "status" ] ), "?" )
        self.stats[ "it_switch" ] = int( self.stats[ "it_switch" ] )
        self.stats[ "it_double" ] = int( self.stats[ "it_double" ] )
        self.stats[ "start" ] = _START.get( int( self.stats[ "start" ] ), "?" )
        for name in ( "nb_iter", "nb_diag", "nb_backtracks", "nb_overflowed", "nb_cell_lim", "nb_limit_rounds",
                      "lin_nb_hierarchies", "lin_nb_iter", "nb_continuation_steps" ):
            self.stats[ name ] = int( self.stats[ name ] )
        # aggregation: REQUESTED here, not yet done by the C++ ( step 7 of
        # `notes/2026-10-02-sdotplan.md` ) -- and that is reported rather than kept quiet, because a
        # degenerate cloud then plateaus around `1e-6` without the slightest message ( README § 23.11 ).
        self.stats[ "aggregation" ] = ( "requested, not wired yet ( step 7 )" if settings.aggregate
                                       else "not requested" )
        #: what the warm start provided: `"ot_plan"`, `"weights0"`, or `"none"` ( and why )
        self.stats[ "warm_start" ] = self._warm_start

    def _read_history( self, history, settings, pd ):
        #: one dict per ACCEPTED step -- `step = 0` is the starting point: `t`, `residual_l2`,
        #: `min_measure`, `max_abs_residual`, `nb_diag` ( cumulative ), `nb_evals` ( the diagrams of
        #: this step ), and `weights` if `keep_weights`
        if driver.is_traced( history.rows.raw ):
            self.history = None                      # under a trace: not read ( it would be a host read of a tracer )
            return
        nb_steps = int( history.nb_steps.value )
        rows = history.rows.raw[ :nb_steps ]
        self.history = []
        for s in range( nb_steps ):
            entry = { name: float( rows[ s, k ] ) for k, name in enumerate( _HISTORY ) }
            entry[ "step" ] = int( entry[ "step" ] )
            if settings.keep_weights:
                entry[ "weights" ] = RealTensor[ pd.num_point ]( history.weights.raw[ s ] )
            self.history.append( entry )

    # -- what the solution says ----------------------------------------------------------------

    @property
    def converged( self ):
        return self.stats[ "status" ] == "converged"

    @property
    def clusters( self ):
        """Which CLUSTER each dirac belongs to, or `None` when none was merged.

        `None` is the common case and it means "the weights suffice". As soon as there are
        clusters, it is the pair ( reduced diagram, cutting planes ) that carries the accuracy and
        not `weights` -- see the module docstring and README § 23.11."""
        return None

    @property
    def target_masses( self ) -> Tensor:
        """`nu`: the target mass of each cell ( the masses of the diracs, normalized, then rescaled
        to what the domain contains -- see `sdotplan/Solve.h` )"""
        return self._masses * ( self.stats[ "domain_mass" ] / float( self._masses.sum() ) )

    def residual( self, weights = None ) -> Tensor:
        """`measure_i( weights ) - nu_i` -- ZERO at the sought point, DIFFERENTIABLE with respect to
        `weights` ( see `PowerDiagram.measures` ). One more diagram: it is a diagnostic tool, not
        an output of the solve ( `cell_masses` is )."""
        w = self.weights if weights is None else weights
        return self.power_diagram( w ).measures - self.target_masses

    def power_diagram( self, weights = None ) -> PowerDiagram:
        """THE ON-DEMAND TOOL: the `PowerDiagram` for `weights` ( by default: the FITTED weights )
        -- always the SAME object, on which those weights are set. For drawing, for inspecting, for
        a functional that was not anticipated; nothing in the solve goes through it."""
        self._pd.weights = self.weights if weights is None else weights
        return self._pd

    # -- what the plan is WORTH -------------------------------------------------------------------

    def transport( self ):
        """`( cost, barycenters, masses )` at the FITTED weights: the cost `W_2^2 = sum_i int_{cell_i}
        |x - p_i|^2 rho` ( a scalar `Tensor` ), the barycenter of each cell ( `[ n, d ]` ) and
        its mass ( `[ n ]` ).

        All three are OUTPUTS of the call -- the solver measures them at the fitted weights, in the
        same C++ and with the same scratch. An empty cell keeps its seed as barycenter."""
        return self.cost, self.barycenters, self.cell_masses

    def cost_and_position_grad( self ):
        """`( cost, grad )`: the cost, and its derivative with respect to the dirac POSITIONS
        ( `[ n, d ]` ) -- by the envelope theorem: at the optimal weights, the derivative of the cost
        with respect to `p_i` does not go through the cells, and equals `2 m_i ( p_i - b_i )`, `b_i` the
        barycenter of the cell and `m_i` its mass ( which is the target mass of the dirac ). It is the
        same formula as `SdotPlan1d`, and what a reconstruction consumes ( `otrec` )."""
        return self.cost, 2 * self.cell_masses * ( self._pd.positions - self.barycenters )
