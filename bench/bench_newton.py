"""THE NEWTON: a whole semi-discrete transport solve, on the cases of the old C++ campaign
( `nsdot/solvers_des_familles`, README § 3, § 24 ) -- to compare the python `sdot` with it.

    errand --env jax "bench_newton"                                         # 2D uniform, 1e5 seeds, KMT damping
    errand --env jax "bench_newton" --case=lines_voronoi --step=trials,limits
    errand --env lmo-jax "bench_newton" --case=planes_voronoi --threads=8 --pin=yes

THE PROBLEM, as the old bench poses it: the seeds are the cloud; the TARGET is Lebesgue on the unit
square / cube ( an `Image` of ones ) and the target mass of every seed is the same, `1 / n` ( the
`equal areas` of the campaign ); we start from weights 0 ( the Voronoi diagram ), and stop when
`max_i | a_i - nu_i | / nu_i <= rtol`, `rtol = 1e-6` ( `--newton-tol` of the old `newton` binary ). The weights of the
`equal` files are the SOLUTION ( they were the witness of the old campaign ): they are not used to start, they
are compared with what the solver finds ( `witness`, max |w - w_file| after fixing the gauge ).

WHAT IS TIMED: `OtProblem( ... ).solve( ... )` from a cold start -- the diagram ( tree included ) is built
inside the call, like the old `TOTAL` ( tree + majorants + diagrams + assembly + linear solves + limits + rest ).
After a warm-up solve on a small prefix ( the first call compiles ), we keep the repetition with the smallest
wall time out of `reps` and print ITS statistics ( `SdotPlanNd._STATS` ).

HOW THE OPTIONS MAP onto the old ones ( more in `bench/README.md` ):

  old                                               here
  KMT, `--pas essais --residu lin` ( the old base ) --step=trials   ( the library's default for d = 3 )
  `--pas essai-limites --facteur 0.9`                --step=limits   ( 2D only; the library's default for d = 2: `auto` )
  `--residu log` + switch to lin ( the old default ) --residual=log ( the default ), --residual-switch=2
  `--residu lin`                                     --residual=lin
  `--residu puissance --puis p`                      --residual=power --residual-power=p
  `--solver chol | amg | cg | mg`                    --linear-solver=cholesky | amg | cg | mg  ( `auto`: see Linear.cpp ); --amg-variant, --linear-tol
  `--newton-tol 1e-6` ( relative )                   --rtol=1e-6 --tol=0
  `--newton-max`                                     --max-iter
  `--kernel double | float`                          --kernel=double | float  ( no `mixte` )
"""

import numpy
from errand import Param, bench

import benchlib
import cases
import reference_lmo

CASE_CHOICES = [ "uniform", "lines_voronoi", "lines_equal", "planes_voronoi", "planes_equal" ]

if p := bench( "newton",
               case          = Param( "uniform", choices = CASE_CHOICES, help = "the case ( see cases.py ); `uniform` uses --dim" ),
               dim           = Param( 2, help = "dimension of the `uniform` case ( 2 or 3 )" ),
               n             = Param( 0, help = "number of seeds ( 0: the case of the campaign: 1e5 for the uniform ones, the file's size otherwise )" ),
               threads       = Param( 0, help = "workers ( 0: all the cores; the campaign: 8 )" ),
               pin           = Param( "yes", choices = [ "yes", "no", "env" ], help = "pin worker w to core w ( SDOT_PIN_THREADS ); `env`: leave the variable alone" ),
               kernel        = Param( "double", choices = [ "double", "float" ], help = "the kernel float type ( the campaign: double )" ),
               step          = Param( "trials", choices = [ "trials", "limits", "auto" ], help = "trials = KMT damping ( the old base ), limits = the limits step ( 2D ), auto = the library's choice" ),
               linear_solver = Param( "auto", choices = [ "auto", "cholesky", "amg", "cg", "mg" ], help = "the linear solver ( mg: the in-house multigrid, `sdotplan/Multigrid.h` )" ),
               residual      = Param( "log", choices = [ "log", "lin", "power" ], help = "the residual of the direction and of the merit: log + switch to lin ( the old default ), lin ( KMT ), power ( g_p )" ),
               residual_power = Param( 0.5, help = "the exponent p of `--residual=power` ( the old `--puis` )" ),
               residual_switch = Param( 2.0, help = "back to lin once max|a-nu|/nu <= this ( the old `--bascule-residu`; 0: never )" ),
               restart_factor = Param( 4.0, help = "TRIALS: the next trial starts from this times the last step ( the old KMT restarts from 1: use 1e9 )" ),
               amg_variant   = Param( "auto", choices = [ "auto", "sa_spai0", "sa_gs", "rs_gs" ], help = "AMGCL: auto = aggregation + spai0 ( the old default )" ),
               linear_tol    = Param( 0.0, help = "relative tolerance of the AMG / CG ( 0: 1e-6 for the AMG; the old `newton` default was 1e-10 )" ),
               mg_pack       = Param( 0, help = "MG: seeds per aggregate, a power of two ( 0: 8 )" ),
               mg_recycle    = Param( -1, help = "MG: solutions kept for the start by projection ( -1: 2, 0: off )" ),
               mg_rebuild    = Param( 0, help = "MG: solves per hierarchy ( 0: 4 )" ),
               mg_stop       = Param( 0, help = "MG: coarsening stops under this many unknowns ( 0: 1000 )" ),
               mg_nu         = Param( 0, help = "MG: Chebyshev smoothing steps per level ( 0: 1 in 3D, 3 in 2D )" ),
               openmp        = Param( -1, help = "EXPERIMENT: compile with -fopenmp for the AMG ( -1: no; 0: OMP_NUM_THREADS left alone, all the hardware threads like the old binary; k: OMP_NUM_THREADS=k )" ),
               rtol          = Param( 1e-6, help = "stop on max|a-nu|/nu <= rtol ( the campaign: 1e-6 )" ),
               tol           = Param( 0.0, help = "or on max|a-nu| <= tol, in normalized masses ( 0: off ); the library's default is 1e-8" ),
               max_iter      = Param( 100, help = "Newton steps, at most" ),
               continuation  = Param( "never", choices = [ "never", "auto", "always" ], help = "width continuation ( the old Newton has none )" ),
               memory        = Param( -1, help = "neighbour memories per seed ( -1: the default of the dimension, 0: none )" ),
               start         = Param( "zero", choices = [ "zero", "file" ], help = "weights 0 ( Voronoi ) or the weights of the file ( the `equal` ones are the solution: 0 iterations )" ),
               ref           = Param( "", help = "variant of reference_lmo.NEWTON to compare with ( kmt, kmt_log, best, model; default: kmt for trials, best for limits )" ),
               reps          = Param( 3, help = "repetitions ( we keep the fastest )" ),
               seed          = Param( 0, help = "seed of the uniform draw" ) ):
    benchlib.set_threads( p.threads, p.pin )          # BEFORE the first kernel: the pool reads them once
    if p.openmp >= 0:
        benchlib.set_openmp( p.openmp )
    import time
    from sdot import Image, Iterative, OtProblem, SumOfDiracs, Tuning

    name = f"uniform{ p.dim }d" if p.case == "uniform" else p.case
    dim = cases.dimension( name )
    n_want = p.n or ( 100_000 if p.case == "uniform" else None )
    pos, w_file, d = cases.case( name, n = n_want, seed = p.seed )
    n = len( pos )
    print( "  " + benchlib.env_report() + "; " + benchlib.driver_line() )

    # Lebesgue on the unit domain: ONE tile of density 1 -- the domain is the support of the target
    target = lambda: Image( values = numpy.ones( ( 1, ) * d ), origin = [ 0.0 ] * d, frame = numpy.eye( d ) )

    def settings( n_points, max_iter ):
        return Iterative( tol = p.tol, max_iter = max_iter, continuation = p.continuation,
                          precision = "fp64" if p.kernel == "double" else "fp32",
                          weights0 = w_file[ :n_points ] if p.start == "file" else None,
                          aggregate = False,            # not wired yet in the C++ ( see README, gap list )
                          tuning = Tuning( step = p.step, linear_solver = p.linear_solver, mass_rtol = p.rtol,
                                           residual = p.residual, residual_power = p.residual_power, residual_switch = p.residual_switch,
                                           restart_factor = p.restart_factor, amg_variant = p.amg_variant, linear_tol = p.linear_tol or None,
                                           mg_pack = p.mg_pack or None, mg_recycle = None if p.mg_recycle < 0 else p.mg_recycle,
                                           mg_rebuild = p.mg_rebuild or None, mg_stop = p.mg_stop or None, mg_nu = p.mg_nu or None,
                                           memory = None if p.memory < 0 else p.memory ) )

    # warm-up on a small prefix: the first solve of the process compiles the kernels
    m = min( n, 2000 )
    OtProblem( SumOfDiracs( pos[ :m ] ), target() ).solve( settings( m, 3 ) )

    best = None
    for _ in range( p.reps ):
        t = time.perf_counter()
        sol = OtProblem( SumOfDiracs( pos ), target() ).solve( settings( n, p.max_iter ) )
        wall = time.perf_counter() - t
        if best is None or wall < best[ 0 ]:
            best = ( wall, sol )
    wall, sol = best
    st = sol.stats

    # the witness: the file's weights ( the `equal` ones ), up to the gauge ( ours: w[ 0 ] = 0 )
    witness = None
    if p.case.endswith( "_equal" ) and n == len( w_file ):
        got = numpy.asarray( sol.weights ).reshape( -1 )
        witness = float( numpy.abs( ( got - got[ 0 ] ) - ( w_file - w_file[ 0 ] ) ).max() )

    # -- the old numbers to compare with
    variant = p.ref or reference_lmo.variant_of( p.step if p.step != "auto" else ( "limits" if dim == 2 else "trials" ), p.residual )
    ref = reference_lmo.newton_ref( name, variant ) if abs( n - 100_000 ) <= 1000 and p.start == "zero" else None
    if ref and ref.get( "seconds" ) is None:       # an old row with iterations but no total time: nothing to divide by
        ref = None
    others = [ r for r in reference_lmo.newton_variants( name ) if abs( n - 100_000 ) <= 1000 and p.start == "zero" ]

    res_rel = st[ "residual" ] * n                    # max|a - nu| / nu, with nu = 1 / n
    p.results.update( n = n, dim = d, threads = benchlib.nb_threads(), kernel = p.kernel, step = p.step,
                      residual = p.residual, linear_solver = p.linear_solver, amg_variant = p.amg_variant, linear_tol = p.linear_tol,
                      it_switch = st[ "it_switch" ], lin_nb_hierarchies = st[ "lin_nb_hierarchies" ], status = st[ "status" ], converged = int( sol.converged ),
                      iterations = st[ "nb_iter" ], diagrams = st[ "nb_diag" ], backtracks = st[ "nb_backtracks" ],
                      t_total_wall = wall, t_total_cpp = st[ "t_total" ], t_diag = st[ "t_diag" ], t_majorant = st[ "t_majorant" ],
                      t_asm = st[ "t_asm" ], t_lin = st[ "t_lin" ], t_lim = st[ "t_lim" ],
                      residual_rel = res_rel, start = st[ "start" ], lin_nb_iter = st[ "lin_nb_iter" ] )
    if witness is not None:
        p.results[ "witness_max_abs_weights" ] = witness
    if ref:
        p.results[ "ref_variant" ] = variant
        p.results[ "ref_seconds" ] = ref[ "seconds" ]
        p.results[ "ratio_new_old" ] = wall / ref[ "seconds" ]

    f = lambda x: f"{ x :.3f}"
    header = [ "case", "n", "thr", "kernel", "step", "res", "lin", "status", "it", "diag", "t_diag", "t_asm", "t_lin", "t_lim", "t_maj", "total", "old", "new/old" ]
    row = [ name, n, benchlib.nb_threads(), p.kernel, p.step, p.residual, p.linear_solver, st[ "status" ], st[ "nb_iter" ], st[ "nb_diag" ], f( st[ "t_diag" ] ), f( st[ "t_asm" ] ),
            f( st[ "t_lin" ] ), f( st[ "t_lim" ] ), f( st[ "t_majorant" ] ), f( wall ),
            "-" if not ref else f"{ ref[ 'seconds' ] :.2f}", benchlib.ratio( wall, ref and ref[ "seconds" ] ) ]
    print( benchlib.table( header, [ row ] ) )
    print( f"  min of { p.reps } ( warm-up apart ); residual max|a-nu|/nu = { res_rel :.2e}; start { st[ 'start' ] }; switch at it { st[ 'it_switch' ] }; backtracks { st[ 'nb_backtracks' ] }; "
           f"linear iterations { st[ 'lin_nb_iter' ] }; C++ total { st[ 't_total' ] :.3f} s"
           + ( f"; witness { witness :.1e}" if witness is not None else "" ) )
    if ref:
        print( f"  old ( { variant } ): { ref[ 'iterations' ] } it, { ref[ 'diagrams' ] } diag, { ref[ 'seconds' ] } s -- { ref[ 'source' ] }" )
    elif others:
        print( f"  old rows for this case: " + "; ".join( f"{ r[ 'variant' ] } { r[ 'seconds' ] } s ( { r[ 'iterations' ] } it, { r[ 'diagrams' ] } diag )" for r in others ) )
    else:
        print( "  ( no old number at this n / for this case )" )
