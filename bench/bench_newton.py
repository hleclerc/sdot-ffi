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

ON THE CARD ( `lmo-jax`, a CUDA device: `SdotPlanNd` solves on the card, `gpu/Newton2D.cuh` ): the same solve, in ONE ffi call.
`--jit=no|yes|both` times it eager and / or under `jax.jit` ( the compiled function called again: no tracing, no Python
per call ); the KERNEL-ONLY time of the solve ( `LOOM_KERNEL_TIMING=1`, every launch of the call between CUDA events ) is
printed next to the wall time, and the stages are the card's: `t_maj` ( the weight majorants ), `t_diag` ( the cells ),
`t_asm` ( right-hand side + laplacian CSR ), `t_lin` ( the linear solve, wall: it reads residuals back ), `t_lim` ( the
polynomial passes of `step = limits` ). `t_tree` is the BSP tree alone ( on the card: one call of `gpu/Bsp2D.cuh`, also part of
every timed solve, eager or jitted ), wall and kernel-only.
`--linear-host=yes` solves the linear systems with the CPU solver named by `--linear-solver` on a copy of the laplacian.
`--save-weights=f.npy` / `--compare=f.npy`: the weights of the solve, saved / compared ( after the gauge ) with another
run's -- the CPU's plan against the card's.
"""

import numpy
from errand import Param, bench

import benchlib
import cases
import reference_lmo

CASE_CHOICES = [ "uniform", "lines_voronoi", "lines_equal", "planes_voronoi", "planes_equal", *cases.DENSITY_CASES ]

if p := bench( "newton",
               case          = Param( "uniform", choices = CASE_CHOICES, help = "the case ( see cases.py ); `uniform` uses --dim" ),
               dim           = Param( 2, help = "dimension of the `uniform` case ( 2 or 3 )" ),
               n             = Param( 0, help = "number of seeds ( 0: the case of the campaign: 1e5 for the uniform ones, the file's size otherwise )" ),
               threads       = Param( 0, help = "workers ( 0: all the cores; the campaign: 8 )" ),
               pin           = Param( "yes", choices = [ "yes", "no", "env" ], help = "pin worker w to core w ( SDOT_PIN_THREADS ); `env`: leave the variable alone" ),
               kernel        = Param( "double", choices = [ "double", "float", "mixed" ], help = "the kernel float type ( the campaign: double; mixed: the card's float then double )" ),
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
               aggregate     = Param( "yes", choices = [ "yes", "no" ], help = "merge the seeds the doubles cannot separate ( `sdotplan/Aggregation.h`, the library's default: yes )" ),
               continuation  = Param( "default", choices = [ "default", "never", "auto", "always" ], help = "width continuation ( default: never on Lebesgue -- the old Newton has none --, always on a density, as § 9 / § 12 )" ),
               conv_start    = Param( 0.5, help = "DENSITY: the first width of the continuation ( § 9: 0.5; 0: half the diameter of the domain )" ),
               sigma         = Param( 0.02, help = "DENSITY gauss4: the scale of the widths ( § 9: 0.05, 0.02, 0.01 )" ),
               image_size    = Param( 512, help = "DENSITY image / image_hole: pixels per side ( § 12: 512 )" ),
               memory        = Param( -1, help = "neighbour memories per seed ( -1: the default of the dimension, 0: none )" ),
               start         = Param( "zero", choices = [ "zero", "file" ], help = "weights 0 ( Voronoi ) or the weights of the file ( the `equal` ones are the solution: 0 iterations )" ),
               ref           = Param( "", help = "variant of reference_lmo.NEWTON to compare with ( kmt, kmt_log, best, model; default: kmt for trials, best for limits )" ),
               reps          = Param( 3, help = "repetitions ( we keep the fastest )" ),
               seed          = Param( 0, help = "seed of the uniform draw" ),
               jit           = Param( "both", choices = [ "no", "yes", "both" ], help = "ON THE CARD: time the eager solve, the jitted one, or both" ),
               linear_host   = Param( "no", choices = [ "no", "yes" ], help = "ON THE CARD: the linear solver on the host ( the CPU's, through a copy )" ),
               mg_precision  = Param( "auto", choices = [ "auto", "float", "double" ], help = "ON THE CARD: the precision of the multigrid's levels" ),
               mg_kcycle     = Param( -1, help = "ON THE CARD: levels accelerated by the K-cycle ( -1: the default, 2 )" ),
               mg_smoothed   = Param( -1, help = "ON THE CARD: levels passed on by the smoothed aggregation ( -1: the default, 1; 0: plain aggregation )" ),
               trace         = Param( "no", choices = [ "no", "yes" ], help = "print the solver's trace ( per iteration, per linear solve )" ),
               card_graphs   = Param( "yes", choices = [ "yes", "no" ], help = "ON THE CARD: the linear solver's iterations replayed from CUDA graphs ( no: plain launches, every kernel timed )" ),
               card_tree     = Param( "yes", choices = [ "yes", "no" ], help = "ON THE CARD: the tree built by the card in one call ( no: the host-driven build, `SDOT_CARD_TREE=0`, evaluated while tracing )" ),
               lin_dump      = Param( "", help = "ON THE CARD: write every linear system to `<prefix>_<k>.bin` ( `SDOT_CARD_LIN_DUMP`, `Linear2D.cuh::dump_system` )" ),
               all_slots     = Param( "no", choices = [ "no", "yes" ], help = "ON THE CARD: print every timed kernel slot ( index, ms, launches, registers )" ),
               save_weights  = Param( "", help = "save the final weights ( .npy )" ),
               compare       = Param( "", help = "compare the final weights with a saved .npy ( max |w - w_ref| after the gauge, relative to 1 / n )" ) ):
    benchlib.set_threads( p.threads, p.pin )          # BEFORE the first kernel: the pool reads them once
    benchlib.set_kernel_timing()                      # ( the card only: events around each launch, read once per library )
    if p.card_graphs == "no":
        import os
        os.environ[ "SDOT_CARD_GRAPHS" ] = "0"
    if p.card_tree == "no":
        import os
        os.environ[ "SDOT_CARD_TREE" ] = "0"
    if p.lin_dump:
        import os
        os.environ[ "SDOT_CARD_LIN_DUMP" ] = p.lin_dump
    if p.openmp >= 0:
        benchlib.set_openmp( p.openmp )
    import time
    from sdot import Image, Iterative, OtProblem, SumOfDiracs, Tuning

    density = p.case in cases.DENSITY_CASES
    if density:
        # THE DENSITY CASES ( `cases.py` ): the seeds uniform in the unit square, the target the old campaign's density
        name = p.case + ( f"_s{ p.sigma :g}" if p.case == "gauss4" else f"_{ p.image_size }" )
        dim = d = 2
        pos, w_file = cases.uniform( p.n or cases.DENSITY_N[ p.case ], 2, p.seed )
    else:
        name = f"uniform{ p.dim }d" if p.case == "uniform" else p.case
        dim = cases.dimension( name )
        n_want = p.n or ( 100_000 if p.case == "uniform" else None )
        pos, w_file, d = cases.case( name, n = n_want, seed = p.seed )
    n = len( pos )
    print( "  " + benchlib.env_report() + "; " + benchlib.driver_line() )
    continuation = p.continuation if p.continuation != "default" else ( "always" if density else "never" )

    # Lebesgue on the unit domain: ONE tile of density 1 -- the domain is the support of the target
    if density:
        target = lambda: cases.density_target( p.case, sigma = p.sigma, size = p.image_size )
    else:
        target = lambda: Image( values = numpy.ones( ( 1, ) * d ), origin = [ 0.0 ] * d, frame = numpy.eye( d ) )

    def settings( n_points, max_iter ):
        return Iterative( tol = p.tol, max_iter = max_iter, continuation = continuation,
                          precision = { "double": "fp64", "float": "fp32", "mixed": "mixed" }[ p.kernel ],
                          weights0 = w_file[ :n_points ] if p.start == "file" else None,
                          aggregate = p.aggregate == "yes",
                          tuning = Tuning( step = p.step, linear_solver = p.linear_solver, mass_rtol = p.rtol,
                                           residual = p.residual, residual_power = p.residual_power, residual_switch = p.residual_switch,
                                           restart_factor = p.restart_factor, amg_variant = p.amg_variant, linear_tol = p.linear_tol or None,
                                           conv_start = ( p.conv_start or None ) if density else None,
                                           mg_pack = p.mg_pack or None, mg_recycle = None if p.mg_recycle < 0 else p.mg_recycle,
                                           mg_rebuild = p.mg_rebuild or None, mg_stop = p.mg_stop or None, mg_nu = p.mg_nu or None,
                                           mg_kcycle = None if p.mg_kcycle < 0 else p.mg_kcycle,
                                           mg_smoothed = None if p.mg_smoothed < 0 else p.mg_smoothed,
                                           mg_precision = None if p.mg_precision == "auto" else p.mg_precision, linear_host = p.linear_host == "yes",
                                           memory = None if p.memory < 0 else p.memory ) )

    # warm-up on a small prefix: the first solve of the process compiles the kernels
    m = min( n, 2000 )
    OtProblem( SumOfDiracs( pos[ :m ] ), target() ).solve( settings( m, 3 ) )

    import loom
    on_card = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    kt = benchlib.KernelTiming() if on_card else None
    # THE TREE alone ( on the card: one call, `gpu/Bsp2D.cuh`; waited for, min of `reps` ), kernel-only next to the wall
    from sdot import AaBsp
    t_tree = t_tree_kern = None
    for _ in range( max( p.reps, 1 ) ):
        if kt: kt.reset()
        t = time.perf_counter()
        tree = AaBsp( pos )
        benchlib.block_until_ready( getattr( tree.seed_indices, "raw", None ) )
        tw = time.perf_counter() - t
        tk = kt.read() if kt else None
        if t_tree is None or tw < t_tree:
            t_tree, t_tree_kern = tw, ( tk[ "ms" ] / 1e3 if tk else None )
    del tree

    best = None
    if not on_card or p.jit in ( "no", "both" ):
        for _ in range( p.reps ):
            if kt: kt.reset()
            t = time.perf_counter()
            sol = OtProblem( SumOfDiracs( pos ), target() ).solve( settings( n, p.max_iter ), verbose = p.trace == "yes" )
            benchlib.block_until_ready( sol.weights.raw )
            wall = time.perf_counter() - t
            k = kt.read() if kt else None
            if best is None or wall < best[ 0 ]:
                best = ( wall, sol, k )
    wall, sol, kern = best if best else ( None, None, None )
    st = sol.stats if sol else None

    # UNDER `jax.jit`: the solve of a function of the target masses ( the positions are constants of the trace; on the card
    # the tree is built INSIDE the jitted program, at every call ); the first call traces and compiles, the next ones only run
    jit_wall = jit_kern = None
    if on_card and p.jit in ( "yes", "both" ):
        import jax
        def solve_masses( masses ):
            plan = OtProblem( SumOfDiracs( pos, masses ), target() ).solve( settings( n, p.max_iter ), verbose = p.trace == "yes" )
            from sdot.SdotPlanNd import _STATS
            return plan.weights.raw, { k: plan.stats[ k ] for k in _STATS }
        f = jax.jit( solve_masses )
        masses = numpy.ones( n )
        out = f( masses )
        benchlib.block_until_ready( out[ 0 ] )
        for _ in range( p.reps ):
            kt.reset()
            t = time.perf_counter()
            out = f( masses )
            benchlib.block_until_ready( out[ 0 ] )
            wj = time.perf_counter() - t
            kj = kt.read()
            if jit_wall is None or wj < jit_wall:
                jit_wall, jit_kern, jit_out = wj, kj, out
        if sol is None:                                # the stats of the jitted solve, read now that it ran
            from sdot.SdotPlanNd import _STATUS, _START
            raw = { k: float( numpy.asarray( v ) ) for k, v in jit_out[ 1 ].items() if not isinstance( v, str ) }
            st = dict( raw )
            st[ "status" ] = _STATUS.get( int( raw[ "status" ] ), "?" )
            st[ "start" ] = _START.get( int( raw[ "start" ] ), "?" )
            for key in ( "nb_iter", "nb_diag", "nb_backtracks", "lin_nb_iter", "lin_nb_hierarchies", "it_switch", "nb_limit_rounds", "it_double",
                         "nb_clusters", "nb_aggregated", "nb_duplicates", "nb_polish" ):
                st[ key ] = int( raw[ key ] )
            wall = jit_wall
        final_w = numpy.asarray( jit_out[ 0 ] ).reshape( -1 )
    else:
        final_w = numpy.asarray( sol.weights ).reshape( -1 )

    if p.save_weights:
        numpy.save( p.save_weights, final_w )
    w_gap = None
    if p.compare:
        ref_w = numpy.load( p.compare ).reshape( -1 )
        w_gap = float( numpy.abs( ( final_w - final_w[ 0 ] ) - ( ref_w - ref_w[ 0 ] ) ).max() * n )

    # the witness: the file's weights ( the `equal` ones ), up to the gauge ( ours: w[ 0 ] = 0 )
    witness = None
    if p.case.endswith( "_equal" ) and n == len( w_file ):
        got = final_w
        witness = float( numpy.abs( ( got - got[ 0 ] ) - ( w_file - w_file[ 0 ] ) ).max() )

    # -- the old numbers to compare with
    variant = p.ref or reference_lmo.variant_of( p.step if p.step != "auto" else ( "limits" if dim == 2 else "trials" ), p.residual )
    if density:
        # the density rows ( `reference_lmo.DENSITY` ): `kmt` for trials, `limits` for limits ( the old § 9 / § 12 paths )
        dvar = p.ref or ( "limits" if p.step in ( "limits", "auto" ) else "kmt" )
        rows = reference_lmo.density_rows( p.case, n, sigma = p.sigma, size = p.image_size )
        ref = next( ( r for r in rows if r[ "variant" ] == dvar and r.get( "seconds" ) ), None )
        others = rows
        variant = dvar
    else:
        ref = reference_lmo.newton_ref( name, variant ) if abs( n - 100_000 ) <= 1000 and p.start == "zero" else None
        if ref and ref.get( "seconds" ) is None:       # an old row with iterations but no total time: nothing to divide by
            ref = None
        others = [ r for r in reference_lmo.newton_variants( name ) if abs( n - 100_000 ) <= 1000 and p.start == "zero" ]

    res_rel = st[ "residual" ] * n                    # max|a - nu| / nu, with nu = 1 / n
    p.results.update( n = n, dim = d, threads = benchlib.nb_threads(), kernel = p.kernel, step = p.step,
                      residual = p.residual, linear_solver = p.linear_solver, amg_variant = p.amg_variant, linear_tol = p.linear_tol,
                      it_switch = st[ "it_switch" ], lin_nb_hierarchies = st[ "lin_nb_hierarchies" ], status = st[ "status" ],
                      converged = int( st[ "status" ] in ( "converged", "converged (aggregated)" ) ),
                      aggregate = p.aggregate, nb_clusters = st[ "nb_clusters" ], nb_aggregated = st[ "nb_aggregated" ], nb_duplicates = st[ "nb_duplicates" ],
                      nb_polish = st[ "nb_polish" ], residual_full_rel = st[ "residual_full" ] * n,
                      iterations = st[ "nb_iter" ], diagrams = st[ "nb_diag" ], backtracks = st[ "nb_backtracks" ],
                      t_total_wall = wall, t_total_cpp = st[ "t_total" ], t_diag = st[ "t_diag" ], t_majorant = st[ "t_majorant" ],
                      t_asm = st[ "t_asm" ], t_lin = st[ "t_lin" ], t_lim = st[ "t_lim" ],
                      residual_rel = res_rel, start = st[ "start" ], lin_nb_iter = st[ "lin_nb_iter" ],
                      on_card = int( on_card ), t_tree = t_tree, t_tree_kernels = t_tree_kern, nb_limit_rounds = st[ "nb_limit_rounds" ],
                      continuation = continuation, nb_continuation_steps = st[ "nb_continuation_steps" ] )
    if on_card:
        p.results.update( linear_host = p.linear_host, jit = p.jit,
                          t_eager_wall = best[ 0 ] if best else None, t_eager_kernels = kern[ "ms" ] / 1e3 if kern else None,
                          t_jit_wall = jit_wall, t_jit_kernels = jit_kern[ "ms" ] / 1e3 if jit_kern else None )
    if w_gap is not None:
        p.results[ "weights_gap_vs_ref" ] = w_gap
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
    if on_card:
        f3 = lambda x: "-" if x is None else f"{ x :.3f}"
        print( f"  card: eager wall { f3( best[ 0 ] if best else None ) } s ( kernels { f3( kern[ 'ms' ] / 1e3 if kern else None ) } s ), "
               f"jit wall { f3( jit_wall ) } s ( kernels { f3( jit_kern[ 'ms' ] / 1e3 if jit_kern else None ) } s ); tree alone { t_tree * 1e3 :.2f} ms "
               f"( kernels { '-' if t_tree_kern is None else f'{ t_tree_kern * 1e3 :.2f}' } ms; inside the solve's totals ); limit rounds { st[ 'nb_limit_rounds' ] }; double kernel from it { st[ 'it_double' ] }" )
        k = jit_kern or kern
        if k:
            top = sorted( k[ "all" ], key = lambda r: -r[ "ms" ] )[ :8 ]
            if p.all_slots == "yes":
                for r in sorted( k[ "all" ], key = lambda r: r[ "slot" ] ):
                    print( f"    slot { r[ 'slot' ] :3d}  { r[ 'ms' ] :9.2f} ms  x{ r[ 'count' ] :6d}  regs { r[ 'regs' ] }  grid { r[ 'grid' ] }  block { r[ 'block' ] }" )
            print( "  heaviest kernel slots: " + "; ".join( f"{ r[ 'code_name' ] }#{ r[ 'slot' ] } { r[ 'ms' ] :.1f} ms x{ r[ 'count' ] }" for r in top ) )
    if w_gap is not None:
        print( f"  weights vs { p.compare }: max gap { w_gap :.2e} ( x n, after the gauge )" )
    if density:
        print( f"  density { name }, continuation { continuation }: { st[ 'nb_continuation_steps' ] } stages; cost { float( numpy.asarray( sol.cost ) ) if sol else float( 'nan' ) :.12e}" )
    print( f"  min of { p.reps } ( warm-up apart ); residual max|a-nu|/nu = { res_rel :.2e}; start { st[ 'start' ] }; switch at it { st[ 'it_switch' ] }; backtracks { st[ 'nb_backtracks' ] }; "
           f"linear iterations { st[ 'lin_nb_iter' ] }; C++ total { st[ 't_total' ] :.3f} s; "
           f"aggregation { p.aggregate }: { st[ 'nb_aggregated' ] } seeds in { st[ 'nb_clusters' ] } clusters ( { st[ 'nb_duplicates' ] } duplicates ), "
           f"re-split { st[ 'nb_polish' ] } diag, full residual { st[ 'residual_full' ] * n :.2e}"
           + ( f"; witness { witness :.1e}" if witness is not None else "" ) )
    if ref:
        print( f"  old ( { variant } ): { ref[ 'iterations' ] } it, { ref[ 'diagrams' ] } diag, { ref[ 'seconds' ] } s -- { ref[ 'source' ] }" )
    if density and others:
        print( "  old rows for this case: " + "; ".join( f"{ r[ 'variant' ] }: { r.get( 'iterations' ) } it, { r.get( 'diagrams' ) } diag, "
                                                         f"{ r.get( 'seconds' ) } s ( { r[ 'path' ] } )" for r in others ) )
    elif others and not ref:
        print( f"  old rows for this case: " + "; ".join( f"{ r[ 'variant' ] } { r[ 'seconds' ] } s ( { r[ 'iterations' ] } it, { r[ 'diagrams' ] } diag )" for r in others ) )
    elif not ref:
        print( "  ( no old number at this n / for this case )" )
