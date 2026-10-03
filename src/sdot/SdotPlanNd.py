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
           "nb_continuation_steps", "min_start_mass", "it_switch" ]
# one row of `history`, in the order of `sdotplan/Solve.h::Hist`
_HISTORY = [ "step", "t", "residual_l2", "min_measure", "max_abs_residual", "nb_diag", "nb_evals", "s" ]
_STATUS = { 0: "running", 1: "converged", 2: "max iterations", 3: "stagnation", 4: "linear solver failure" }
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
        # the solver is HOST code on the CPU queue ( `sdotplan/Sweep.h` ): a driver whose device
        # is a GPU cannot call it today -- `LOOM_DEVICE=cpu`, or a CPU driver
        if not driver.device.is_cpu:
            raise NotImplementedError( "SdotPlanNd: the solver runs on the CPU for now ( the sweeps and the "
                                       "linear solver are host code ); choose the CPU device ( LOOM_DEVICE=cpu )" )
        tun = settings.tuning
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
        self._pd = PowerDiagram( src_dist.positions,
                                 RealTensor[ src_dist.num_dirac ].full( 0.0 ) if w0_given is None else w0_given,
                                 accelerator = tun.accelerator, kernel_dtype = settings.kernel_dtype,
                                 distribution = dst_dist, memory = tun.memory,
                                 scratch_capacity = tun.scratch_capacity )
        pd = self._pd
        n = int( pd.nb_points.value )

        #: the target masses, indexed like the cells
        self._masses = RealTensor[ pd.num_point ]( src_dist.weights.raw )

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
        self.stats = { name: float( st[ k ] ) for k, name in enumerate( _STATS ) }
        self.stats[ "status" ] = _STATUS.get( int( self.stats[ "status" ] ), "?" )
        self.stats[ "it_switch" ] = int( self.stats[ "it_switch" ] )
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
