"""`OtProblem` : WHAT WE SOLVE -- and the only entry point to semi-discrete transport plans.

= Two objects, and not one more

`OtProblem` carries the INPUTS, and nothing else : two distributions. No solver setting,
no allocated tensor, no tree, no scratch capacity. Building it triggers no
call. An input can be rewired and the solution asked for again :

    pb = OtProblem( SumOfDiracs( pos ), image )
    sol = pb.solve()
    pb.source = SumOfDiracs( next_pos )           # an input changes
    sol = pb.solve()                              # and the previous SOLUTION is the starting point

`solve()` returns a SOLUTION, a separate object -- `SdotPlan1d` or `SdotPlanNd`. A problem can
have several solutions ( two tolerances, two starting points ) and must therefore not carry one.

For a transport that is solved ONLY ONCE, `ot_solve( source, target, ... )` does both in
one expression. It is only a shortcut in that case : as soon as one solves several times, it is the
kept `OtProblem` that carries the SOLUTION of the previous time, and it is what makes a
neighbouring transport cost a few Newton steps instead of a few dozen. A plan and not weights, because
weights alone do not describe a solution as soon as there are coincident seeds ( § 23.11 ).

= Direct or iterative : the problem has already chosen

    d = 1, diracs against a density  ->  SORT plus inversion of the cumulative distribution function.  `SdotPlan1d`
    d >= 2                           ->  FIXED POINT of a damped Newton.                                `SdotPlanNd`

The first is exact, loop-free, tolerance-free and needs no starting point ; it is differentiable
end to end and batched over thousands of angles. The second is a loop, with a tolerance, a
starting point, a history, and a gradient by the envelope theorem.

**The parameters therefore do not differ because the caller chooses a method : they differ
because the REGIME differs, and the regime is read off the inputs.** Hence the rule : there is one
settings object PER REGIME, `Direct` and `Iterative`, and **its type IS the regime**. Giving one that does not
match is an error, not an argument silently ignored :

    sol = pb.solve()                              # the regime's defaults, which are the right ones
    sol = pb.solve( Iterative( tol = 1e-10 ) )    # d >= 2
    sol = pb.solve( Direct( with_barycenters = True ) )   # d = 1

= The bench settings are not user settings

Everything that `solvers_des_familles` varies to COMPARE algorithms -- the choice of step, the linear
solver, the continuation scale, the spatial accelerator, the neighbour memory -- has a MEASURED
default, and a user who changes it chooses badly. These settings exist, behind a service door :
`Iterative( tuning = Tuning( ... ) )`. `Tuning` is explicitly UNSTABLE --
it is the surface through which the bench keeps existing, and nothing else must depend on it.

And the lesson not to relearn ( README § 24.5 ) : **a default belongs to the regime where it was
measured.** `residu = log` wins on direct solves and breaks continuation in density.
"""

import numpy as np                  # the only host arrays in here : the few planes of the domain


_PRECISIONS = { "auto": "FP64", "fp64": "FP64", "fp32": "FP32", "mixed": "FP32" }


class Tuning:
    """THE BENCH SETTINGS. Unstable, unsupported, and nothing in `sdot` must depend on it.

    Exactly the parameters removed from the public surface ( see the module ) : they keep their
    measured default, and the bench section that measured it is written next to it.
    """

    def __init__( self,
                  # the step, and the linear solver -- `auto` = what § 24.4 concludes
                  step = "auto", linear_solver = "auto",
                  # the AMG: `auto` = aggregation + spai0 ( the old `newton` default ), `rs_gs`, `sa_gs`; the relative
                  # tolerance of the iterative solvers ( `None`: 1e-6 for AMG, README § 17.2 )
                  amg_variant = "auto", linear_tol = None,
                  # the in-house multigrid ( `linear_solver = "mg"`, `sdotplan/Multigrid.h` ): seeds per aggregate ( a power of 2,
                  # `None`: 8 ), solutions kept for the start by projection ( `None`: 2 ), solves per hierarchy ( `None`: 4 ),
                  # coarsening stops under that many unknowns ( `None`: 1000 )
                  mg_pack = None, mg_recycle = None, mg_rebuild = None, mg_stop = None, mg_nu = None,
                  # ON A CUDA CARD ( `gpu/Linear2D.cuh` ): the levels accelerated by the K-cycle ( `None`: 2 ), and
                  # `linear_host = True` to solve on the host with the CPU solver named by `linear_solver` ( a copy of the
                  # laplacian each way ) instead of the card's multigrid / CG; the card's multigrid: packets of 4, the first
                  # `mg_smoothed` levels by the smoothed aggregation ( `None`: 1; 0: the plain aggregation everywhere, the
                  # multigrid of the old GPU campaign ), Chebyshev degree 1, K-cycle on the 2 levels after the smoothed ones,
                  # recycling 2, a hierarchy per solve; `mg_precision` ( `"float"` / `"double"`, `None`: float ) the
                  # precision of its levels, the outer iteration being in double ( `calibration_lmo_today.md`, GPU step 5 )
                  mg_kcycle = None, linear_host = False, mg_precision = None, mg_smoothed = None,
                  # the scale of the width continuation ( § 9.2 : the ratio sqrt( 2 ) is measured )
                  conv_start = None, conv_ratio = 2 ** 0.5, conv_min = None, conv_threshold = 1e-2,
                  # the damping safeguards ( § 3 : `restart_factor = 4` is measured )
                  t_min = 1e-10, max_backtracks = 60, restart_factor = 4.0, mass_rtol = 0.0,
                  # the residual of the direction and of the merit ( § 24.5 ): `log` then `lin` once
                  # `max|a-nu|/nu <= residual_switch` ( the old default, -50 % of the diagrams on the hard
                  # cases ), `lin` ( the previous behaviour ), or `power` ( `g_p`, `p = residual_power` )
                  residual = "log", residual_power = 0.5, residual_switch = 2.0,
                  # the aggregation of near-coincident seeds ( `sdotplan/Aggregation.h` ): a pair is merged when one ulp of its
                  # weights moves more than `tol / aggregation_margin` of mass between them ( `None`: 4 )
                  aggregation_margin = None,
                  # the machine : the spatial accelerator, the neighbour memory ( § 11 ), the scratch ( § 18.2 )
                  accelerator = None, memory = None, scratch_capacity = None ):
        self.step              = step
        self.linear_solver     = linear_solver
        self.amg_variant       = amg_variant
        self.linear_tol        = linear_tol
        self.mg_pack           = mg_pack
        self.mg_recycle        = mg_recycle
        self.mg_rebuild        = mg_rebuild
        self.mg_stop           = mg_stop
        self.mg_nu             = mg_nu
        self.mg_kcycle         = mg_kcycle
        self.linear_host       = bool( linear_host )
        self.mg_precision      = mg_precision
        self.mg_smoothed       = mg_smoothed
        self.conv_start        = conv_start
        self.conv_ratio        = conv_ratio
        self.conv_min          = conv_min
        self.conv_threshold    = conv_threshold
        self.t_min             = t_min
        self.max_backtracks    = max_backtracks
        self.restart_factor    = restart_factor
        self.mass_rtol         = mass_rtol
        self.residual          = residual
        self.residual_power    = residual_power
        self.residual_switch   = residual_switch
        self.aggregation_margin = aggregation_margin
        self.accelerator       = accelerator
        self.memory            = memory
        self.scratch_capacity  = scratch_capacity


class Iterative:
    """The settings of the ITERATIVE REGIME ( `d >= 2`, `SdotPlanNd` ) -- and its type says the regime.

    `tol` : we stop as soon as `max_i | m_i - nu_i | <= tol`, ABSOLUTE, in the unit of the normalized
    masses ( the target mass of a dirac is `1 / n` ). `max_iter` : Newton steps, at most.

    `ot_plan` : THE STARTING POINT, in the form of a previous SOLUTION -- what a reconstruction lives
    on, where a neighbouring transport costs a few Newton steps instead of a few dozen. It is a plan and
    not a weight vector, and that is structural : as soon as there are coincident seeds, the weights alone
    DO NOT DESCRIBE the solution ( § 23.11 -- it is the pair ( reduced problem, cutting planes ) that
    carries the precision ), so a starting point that is only `w` loses the aggregate. A plan carries it, and
    it also carries the POSITIONS for which it was solved : when they have not changed, the
    clusters need not be detected again ( which step 7 will exploit ).

    `weights0` : the BARE weights, when that is all we have ( a file, a deliberate trial ).
    `ot_plan` is the right way ; giving both raises. With neither, `OtProblem` proposes the
    last solution it returned ( see `OtProblem.solve` ). A starting point that empties a cell is
    not an error : the C++ itself compares the starting points it knows and keeps the best
    ( `stats[ "start" ]`, and `stats[ "warm_start" ]` says what the warm start supplied ).

    `continuation` : the WIDTH CONTINUATION -- first solve for the density convolved with a wide
    gaussian, then a narrower and narrower one, each step starting from the weights of the previous one.
    This is what a concentrating density needs ( narrow bumps, deserts where cells have no mass, and where
    direct Newton STAGNATES -- § 9.1 ). `"auto"` triggers it when the starting point leaves a cell
    without mass ; `"always"` / `"never"`.

    `precision` : the float in which the geometry is cut. `"mixed"` ( and `"auto"` ON A CUDA CARD ) : the float kernel, then the
    double one from the first float step that stagnates -- the card's float kernel re-solves its vertices in double, so it
    reaches the double's residual on regular clouds for a fraction of its cost; the CPU has no such switch ( `"mixed"` is
    refused there ). On the CPU, `"auto"` is `FP64` : the bench measured it
    ( § 4 ), damping requires a strict decrease of the residual that the noise of an area in `float`
    refuses well before the tolerance. The `fp32 -> fp64` switch is STRUCTURAL
    ( § 19.10 ) and therefore belongs to the C++, not to the caller.

    `aggregate` : MERGE the diracs that the doubles cannot separate ( § 23.6 - § 23.12, `sdotplan/Aggregation.h` ). Two
    seeds `delta` apart are split by a plane whose offset is `( w_i - w_j ) / 2 delta`: when one ulp of their weights moves
    more than a quarter of the tolerance between them, no weight vector can reach `tol` there, and the pair is merged --
    the tests then read the mass of the cluster, the full Newton direction still places the plane between its members as
    well as the doubles let it, and once the aggregated problem has converged each merged cell is re-split by its local
    problem. Exact duplicates ( equal positions ) are merged too, their cells kept empty, their representative carrying
    their mass. On, because the alternative is a SILENT floor : `lines_equal` ( a pair `1e-8` apart ) stagnated at
    `2.35e-6` relative; with it, `stats[ "status" ]` says `converged (aggregated)` and `stats[ "residual_full" ]` gives
    what is left of the full problem ( the floor of the doubles ). Nothing changes, and nothing is paid, on a cloud whose
    pairs the doubles separate. `SdotPlanNd.clusters` says which seeds were merged.

    `keep_weights` : keep the weights of EVERY step in `history`, to replay the descent ( an
    array `[ step, n ]`, which one does not always want ).
    """

    regime = "iterative"

    def __init__( self, tol = 1e-8, max_iter = 100, ot_plan = None, weights0 = None,
                  continuation = "auto", precision = "auto", aggregate = True, keep_weights = False,
                  tuning = None ):
        if precision not in _PRECISIONS:
            raise ValueError( f"unknown precision : { precision !r } ( { ', '.join( _PRECISIONS ) } )" )
        if continuation not in ( "auto", "always", "never" ):
            raise ValueError( f"unknown continuation : { continuation !r } ( 'auto', 'always' or 'never' )" )
        if ot_plan is not None and weights0 is not None:
            raise ValueError( "Iterative : `ot_plan` AND `weights0` -- there is only one starting point. `ot_plan` "
                              "is the one to keep ( it carries the aggregate and the positions, see the docstring )" )
        self.ot_plan      = ot_plan
        self.tol          = float( tol )
        self.max_iter     = int( max_iter )
        self.weights0     = weights0
        self.continuation = continuation
        self.precision    = precision
        self.aggregate    = bool( aggregate )
        self.keep_weights = bool( keep_weights )
        self.tuning       = tuning or Tuning()

    @property
    def kernel_dtype( self ):
        return _PRECISIONS[ self.precision ]


class Direct:
    """The settings of the DIRECT REGIME ( `d = 1`, `SdotPlan1d` ) -- and its type says the regime.

    There is neither a tolerance nor a number of iterations : the solution is a sort plus an inversion of the
    cumulative distribution function, hence exact. What remains is to say what we want as OUTPUT.

    `with_barycenters` : produce AND store the barycenters ( `[ n, d ]` per batch element,
    `80 GB` at the scale of a reconstruction ). Off by default : the adjoint knows how to recompute them.
    """

    regime = "direct"

    def __init__( self, with_barycenters = False ):
        self.with_barycenters = bool( with_barycenters )


class OtProblem:
    """see the module docstring"""

    def __init__( self, source, target ):
        """`source` : the DISCRETE distribution -- a `SumOfDiracs` ( or a `ProjectedSumOfDiracs` ) ;
        its `weights`, normalized, are the target masses of the cells.

        `target` : the CONTINUOUS distribution to integrate against ( `Image`, `SumOfGaussians`, ... ).

        THE DOMAIN COMES FROM `target`, AND FROM IT ALONE : the support it declares
        ( `bounding_half_spaces` -- the box of an image, `centers +- 6 sigma` for gaussians ),
        which must be BOUNDED. The diracs have nothing to do with it : their cells can be far from them.
        Their envelope is only used for the STARTING POINT ( the reframing ). This is why `domain` is
        READ-ONLY -- there is no second place to state it."""
        if target is None:
            raise ValueError( "OtProblem : a target is needed -- it is what gives the domain" )
        self._last = None                                # the last solution returned ( not its weights )
        self._source = None
        self._target = None
        self.source = source
        self.target = target

    # -- the inputs, adjustable ---------------------------------------------------------------

    @property
    def source( self ):
        """the discrete distribution, normalized"""
        return self._source

    @source.setter
    def source( self, source ):
        if source is None:
            raise ValueError( "OtProblem : a source is needed" )
        source = source.normalized_version()
        # a cloud whose SIZE changed invalidates the kept solution ( a multi-scale stage )
        if self._source is not None and self._nb_diracs_of( source ) != self._nb_diracs_of( self._source ):
            self._last = None
        self._source = source

    @property
    def target( self ):
        """the target density, normalized -- and what gives the domain"""
        return self._target

    @target.setter
    def target( self, target ):
        if target is None:
            raise ValueError( "OtProblem : a target is needed -- it is what gives the domain" )
        # evaluated when the target is concrete, even under a trace ( `jax.jit` ): its values stay readable on the host,
        # which is how a solver knows a constant density ( `PowerDiagram_Bsp._card_density` ); traced values stay traced
        from loom.drivers.driver import driver
        with driver.concrete_eval():
            self._target = target.normalized_version()

    # -- what can be READ off the inputs ------------------------------------------------------

    @property
    def nb_dims( self ):
        return int( self._source.nb_dims.value )

    @property
    def nb_diracs( self ):
        return self._nb_diracs_of( self._source )

    @property
    def regime( self ):
        """`"direct"` ( a sort, `d = 1` ) or `"iterative"` ( a damped Newton, `d >= 2` ) -- DEDUCED,
        never chosen. See the module docstring."""
        return "direct" if self.nb_dims == 1 else "iterative"

    @property
    def domain( self ):
        """The domain, as half-spaces `( directions, offsets )` : `direction . x <= offset`. It comes
        from the support that the TARGET declares, and from it alone. Raises if this support does not bound it."""
        d = self.nb_dims
        support = self._target.bounding_half_spaces()
        if support is not None:
            dirs = np.asarray( support[ 0 ], dtype = float ).reshape( -1, d )
            offs = np.asarray( support[ 1 ], dtype = float ).reshape( -1 )
            if self._bounds( dirs, offs, d ):
                return dirs, offs
        raise ValueError( "OtProblem : the support of the target does not bound the domain "
                          "( `bounding_half_spaces` ) -- it is up to the target to declare it "
                          "( `SumOfGaussians( support_sigmas = ... )` )" )

    # -- solving -------------------------------------------------------------------------------

    def solve( self, settings = None, verbose = False ):
        """The SOLUTION : a `SdotPlan1d` if the regime is direct, a `SdotPlanNd` if it is iterative.

        `settings` : `None` for the regime's defaults, or a `Direct` / `Iterative` -- and its TYPE
        must be that of the regime, otherwise this raises ( see the module docstring ).

        An `Iterative` that does not impose a starting point ( neither `ot_plan` nor `weights0` ) restarts from the
        last SOLUTION of THIS problem, when there is one and the number of diracs has not
        changed : this is what a reconstruction lives on, and it no longer has to be copied by hand
        by the caller."""
        regime = self.regime
        if settings is None:
            settings = { "direct": Direct, "iterative": Iterative }[ regime ]()
        elif getattr( settings, "regime", None ) != regime:
            want = { "direct": "Direct", "iterative": "Iterative" }[ regime ]
            raise TypeError( f"OtProblem.solve : this problem is solved in regime { regime !r } "
                             f"( nb_dims = { self.nb_dims } ), so its settings are a `{ want }` "
                             f"and not a `{ type( settings ).__name__ }` -- see the docstring of `OtProblem`" )

        if regime == "direct":
            from .SdotPlan1d import SdotPlan1d
            return SdotPlan1d._solve( self, settings, verbose )

        from .SdotPlanNd import SdotPlanNd
        # the WARM RE-START : the last SOLUTION, when the caller does not impose a starting point.
        # A plan and not weights -- it carries the aggregate, which `w` alone cannot describe
        # ( README § 23.11 ). Passed separately, and not written into `settings` : a settings object that
        # the caller keeps must not start carrying the state of the problem.
        sol = SdotPlanNd._solve( self, settings, verbose, warm = self._last )
        self._last = sol
        return sol

    # -- the details -----------------------------------------------------------------------------

    @staticmethod
    def _nb_diracs_of( dist ):
        if not getattr( dist, "_is_dirac_source", False ):
            raise TypeError( f"OtProblem : { type( dist ).__name__ } is not a discrete source "
                             "-- it is the source that carries the target masses of the cells" )
        return int( dist.nb_diracs.value )

    def _bounds( self, dirs, offs, d ):
        """Do these half-spaces bound ? A box is recognized without building anything ; otherwise we
        build the polytope to find out ( a few planes, host side )."""
        from .PowerDiagram import axis_aligned_box
        if axis_aligned_box( dirs, offs ) is not None:
            return True
        from .Cell import Cell
        dom = Cell.make_unbounded( d, kernel_dtype = "FP64" )
        for k in range( len( offs ) ):
            dom.cut( dirs[ k ], float( offs[ k ] ) )
        return bool( dom.is_bounded )


# -- THE SHORTCUT ------------------------------------------------------------------------------

def ot_solve( source, target, *, verbose = False, **settings ):
    """Pose the problem and solve it, in one expression -- for a transport that is solved
    ONLY ONCE.

        sol = ot_solve( SumOfDiracs( pos ), image )
        sol = ot_solve( SumOfDiracs( pos ), image, tol = 1e-10 )     # d >= 2 : an `Iterative`
        sol = ot_solve( SumOfDiracs( pos2 ), image, ot_plan = sol )   # restarting from the previous plan
        sol = ot_solve( diracs_1d, image_1d, with_barycenters = True )   # d = 1 : a `Direct`

    The settings are given here as KEYWORDS, and not as an object : the regime is deduced from the
    inputs (see the module docstring), so the settings class that receives them is too.
    A name that does not belong to the regime raises, saying which one it is and what it accepts --
    never silently ignored.

    To restart from a previous solve, `ot_plan = the_previous_solution` -- a PLAN and not
    weights, because weights alone do not describe a solution as soon as there are coincident seeds
    ( see `Iterative` ).

    **But in a loop, keeping the problem is better.** It proposes the last solution by itself,
    it invalidates it by itself when the cloud changes size, and it is one fewer `ot_plan` to
    pass around. A reconstruction, a descent, a parameter sweep therefore keep the problem and
    rewire its inputs :

        pb = OtProblem( SumOfDiracs( pos ), image )
        for _ in range( nb_steps ):
            sol = pb.solve()                      # restarts from the previous weights
            pb.source = SumOfDiracs( move( pos, sol ) )

    `ot_solve` builds a fresh `OtProblem` at each call : without `ot_plan`, it restarts from zero."""
    pb = OtProblem( source, target )
    cls = { "direct": Direct, "iterative": Iterative }[ pb.regime ]
    try:
        solver_settings = cls( **settings )
    except TypeError as e:
        import inspect
        names = [ n for n in inspect.signature( cls ).parameters if n != "self" ]
        raise TypeError( f"ot_solve : this problem is solved in regime { pb.regime !r } "
                         f"( nb_dims = { pb.nb_dims } ), so its settings are those of `{ cls.__name__ }` "
                         f"( { ', '.join( names ) } ) -- { e }" ) from None
    return pb.solve( solver_settings, verbose )
