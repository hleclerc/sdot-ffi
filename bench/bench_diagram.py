"""ONE DIAGRAM: the cost of a power diagram at fixed weights, on the cases of the old C++ campaigns
( `nsdot/solvers_des_familles`, README § 3, § 11, § 19, on the CPU; `nsdot/gpu_des_familles` on the GPU ) --
to compare the python `sdot` with them.

    errand --env jax "bench_diagram"                                        # uniform 2D, 1e6 seeds
    errand --env jax "bench_diagram" --case=lines_voronoi,lines_equal       # a matrix: one run each
    errand --env lmo-numpy "bench_diagram" --case=uniform --dim=3 --threads=8 --pin=yes --kernel=double
    errand --env lmo-jax "bench_diagram" --case=uniform --dim=2 --kernel=float  # on the card

How it is timed ON THE CPU: the tree and the diagram are built ONCE, after two warm-up rounds ( the first compiles,
the second fills the neighbour memory ), and we keep the MINIMUM of `reps` ( 3 by default ). Three durations are
taken apart: `t_weights` ( `pd.weights = w`: the sort and the refresh of the weight majorants -- the old bench does
not time it, its Newton reports it as its own `majorants` column ), `t_measures` ( `pd.measures`, the cells: THIS
is what the old `diagramme` times, and the ratio `new / old` compares it ) and `t_convert` ( the copy to numpy ).

ON A CUDA CARD ( the loom device is a `CudaGpu`: `lmo-jax` ), the protocol of `gpu_des_familles`
( `doc/07-methode.md` ): after the two rounds above, `pd.measures` IN A LOOP for at least `--warmup` seconds
( 0.3: a card left idle while the CPU built the tree runs its first kernels at a reduced clock -- the old bench saw
17 instead of 13 ns/seed ), then the MINIMUM of `reps` ( 10 by default ) of two times:

  * KERNEL ONLY: the time the card spent in the kernels of the `measures` call, from CUDA events recorded around
    each launch ( `LOOM_KERNEL_TIMING=1`, `loom/devices/kernel_timing.py`, `CudaQueue.h` ). This is what the old
    GPU bench reports, and what the ratio `new / old` compares.
  * WALL: the whole call, from python to a finished result on the card ( the dispatch, the handler, the scratch,
    the overflow check ) -- what a user pays per diagram.

and, once, what the compiler made of the main kernel ( the longest one ): registers, local memory per thread,
resident blocks per SM and the occupancy they give. With `--kernel=float`, the ACCURACY against a `double` kernel
on the same cloud and tree ( `--accuracy` ): median, p99.99 and max over the cells of `| m_float - m_double | / m_double`.

A Voronoi cloud ( weights all zero ) is run WITHOUT weights by default ( `--weights=auto` ): the old engine
compiles its `POIDS = false` variant for it, with no majorant terms. `--weights=zeros` hands the zeros over,
which runs the weighted kernel: that is the cost of a Laguerre diagram that happens to have zero weights.
The old CPU campaign: 8 pinned threads, kernel `double`, minimum of the repetitions, the machine alone: see
`bench/README.md` for the protocol.

The reference numbers are those of `reference_lmo.py` ( CPU ) and `reference_lmo_gpu.py` ( GPU ); the ratio
`new / old` is > 1 where we are slower.
"""

import numpy
from errand import Param, bench

import benchlib
import cases
import reference_lmo
import reference_lmo_gpu

CASE_CHOICES = [ "uniform", "lines_voronoi", "lines_equal", "planes_voronoi", "planes_equal" ]

if p := bench( "diagram",
               case      = Param( "uniform", choices = CASE_CHOICES, help = "the case ( see cases.py ); `uniform` uses --dim" ),
               dim       = Param( 2, help = "dimension of the `uniform` case ( 2 or 3 )" ),
               n         = Param( 0, help = "number of seeds ( 0: the case of the campaign: 1e6 for the uniform ones, the file's size otherwise )" ),
               wscale    = Param( 0.0, help = "uniform case: weights in [ -w h^2, w h^2 ] ( 0: Voronoi; the old bench's `--weights`; 0.3 is a Laguerre one )" ),
               threads   = Param( 0, help = "workers ( 0: all the cores; the campaign: 8 )" ),
               pin       = Param( "yes", choices = [ "yes", "no", "env" ], help = "pin worker w to core w ( SDOT_PIN_THREADS ); `env`: leave the variable alone" ),
               kernel    = Param( "double", choices = [ "double", "float" ], help = "the kernel float type ( the CPU campaign: double )" ),
               leaf_size = Param( 10, help = "seeds per BSP leaf ( the campaign: 10 )" ),
               weights   = Param( "auto", choices = [ "auto", "zeros" ], help = "`auto`: no weights at all when they are all zero ( Voronoi, like the old engine ); `zeros`: the weighted kernel with zero weights" ),
               memory    = Param( -1, help = "neighbour memories per seed ( -1: the default of the dimension, 0: none )" ),
               reps      = Param( 0, help = "timed repetitions, we keep the minimum ( 0: 3 on the CPU, 10 on a GPU )" ),
               warmup    = Param( 0.3, help = "GPU: seconds of `measures` in a loop before the timed runs ( the old GPU bench: 0.3 )" ),
               accuracy  = Param( "auto", choices = [ "auto", "yes", "no" ], help = "compare with a `double` kernel run of the same diagram ( auto: on a GPU, for --kernel=float )" ),
               seed      = Param( 0, help = "seed of the uniform draw" ),
               output    = Param( "measures", choices = [ "measures", "facets", "vjp", "moments" ],
                                  help = "GPU, 2D card kernel: what one call computes -- `measures`; `facets`: the measures AND the laplacian's CSR "
                                         "( Newton's turn: `_card_cells`, cells + COO + assembly ); `vjp`: the adjoint of the measures alone "
                                         "( the pullback, wrt the weights, or the positions without weights ); `moments`: measures + barycentres + costs" ) ):
    benchlib.set_threads( p.threads, p.pin )          # BEFORE the first kernel: the pool reads them once
    benchlib.set_kernel_timing()                      # BEFORE the first kernel too: each library reads it once
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    from loom.drivers.driver import driver
    import time

    on_gpu = bool( getattr( driver.device, "is_cuda_gpu", False ) )
    reps = p.reps or ( 10 if on_gpu else 3 )
    name = f"uniform{ p.dim }d" if p.case == "uniform" else p.case
    pos, w, d = cases.case( name, n = p.n or None, seed = p.seed, wscale = p.wscale )
    n = len( pos )
    kernel = "FP64" if p.kernel == "double" else "FP32"
    memory = None if p.memory < 0 else p.memory
    print( "  " + benchlib.env_report() + "; " + benchlib.driver_line() )

    unweighted = p.weights == "auto" and not numpy.any( w )
    w_arg = None if unweighted else w
    print( f"  weights: { 'none ( Euclidean kernel, like the old POIDS = false )' if unweighted else 'given ( weighted kernel )' }" )

    # the tree is built once, and warmed up on a small prefix so that `t_build` does not measure a compilation
    AaBsp( pos[ :1000 ], None if unweighted else w[ :1000 ], max_seeds_per_leaf = p.leaf_size )
    t = time.perf_counter()
    bsp = AaBsp( pos, w_arg, max_seeds_per_leaf = p.leaf_size )
    t_build = time.perf_counter() - t

    t = time.perf_counter()
    pd = PowerDiagram( pos, weights = w_arg, boundaries = box_half_spaces( [ 0 ] * d, [ 1 ] * d ), accelerator = bsp,
                       kernel_dtype = kernel, memory = memory )
    t_ctor = time.perf_counter() - t
    print( f"  positions as { benchlib.dtype_of( pd.sorted_positions ) } ( TF ), kernel { kernel }" )

    timing = benchlib.KernelTiming() if on_gpu else None
    if p.output != "measures" and not ( on_gpu and d == 2 and pd._card_variant() is not None ):
        raise ValueError( f"--output={ p.output }: the 2D card kernel only ( a CUDA device, 2D )" )
    pullback = None
    if p.output == "vjp":                             # the forward once: what is timed is the pullback
        import jax
        if unweighted:
            def fwd( x ):
                return PowerDiagram( x, boundaries = box_half_spaces( [ 0 ] * d, [ 1 ] * d ), accelerator = bsp,
                                     kernel_dtype = kernel ).measures.value
            _, pullback = jax.vjp( fwd, pos )
        else:
            def fwd( x ):
                pd.weights = x
                return pd.measures.value
            _, pullback = jax.vjp( fwd, w )
        cot = numpy.random.default_rng( 1 ).normal( size = n )

    def call():
        if p.output == "measures":
            return pd.measures.value                  # the cells: what the old `diagramme` times
        if p.output == "vjp":
            return pullback( cot )[ 0 ]
        out = pd._card_cells( facets = p.output == "facets", moments = p.output == "moments" )
        benchlib.block_until_ready( out[ "val" ].raw if p.output == "facets" else out[ "cost" ].raw )
        return out[ "measures" ].value

    def once():
        t0 = time.perf_counter()
        if not unweighted and p.output != "vjp":
            pd.weights = w
        t1 = time.perf_counter()
        if timing:
            timing.reset()                            # waits for what is in flight ( the weights ) and zeroes
            t1 = time.perf_counter()
        meas = call()
        benchlib.block_until_ready( meas )            # a GPU result is asynchronous: the wall time waits for it
        t2 = time.perf_counter()
        k = timing.read() if timing else None
        m = numpy.asarray( meas ).reshape( -1 )
        t3 = time.perf_counter()
        return ( t1 - t0, t2 - t1, t3 - t2 ), m, k

    once()                                            # warm-up: this one compiles
    once()                                            # and this one fills the memory
    nb_warm, t_warm = 0, time.perf_counter()
    if on_gpu:                                        # and the card goes back to its clock
        while nb_warm < 2 or time.perf_counter() - t_warm < p.warmup:
            once()
            nb_warm += 1
    t_warm = time.perf_counter() - t_warm

    ts, m, k = once()
    t_weights, best, t_convert = ts
    best_kernel = k and k[ "ms" ] * 1e-3
    kernels = k
    for _ in range( reps - 1 ):
        ts, _, k = once()
        t_weights, best, t_convert = min( t_weights, ts[ 0 ] ), min( best, ts[ 1 ] ), min( t_convert, ts[ 2 ] )
        if k and k[ "ms" ] * 1e-3 < best_kernel:
            best_kernel, kernels = k[ "ms" ] * 1e-3, k

    # -- the accuracy, against a double kernel on the same tree
    acc = None
    if p.output == "measures" and ( p.accuracy == "yes" or ( p.accuracy == "auto" and on_gpu and p.kernel == "float" ) ):
        if p.kernel == "double":
            print( "  accuracy: the kernel is already double ( it is the reference )" )
        else:
            pd64 = PowerDiagram( pos, weights = w_arg, boundaries = box_half_spaces( [ 0 ] * d, [ 1 ] * d ), accelerator = bsp,
                                 kernel_dtype = "FP64", memory = memory )
            pd64.measures                             # the memory, as for the timed one
            m64 = numpy.asarray( pd64.measures.value ).reshape( -1 )
            acc = benchlib.accuracy( m, m64 )

    # -- the old number to compare with
    ns = best / n * 1e9
    p.results[ "n" ] = n
    p.results[ "dim" ] = d
    p.results[ "threads" ] = benchlib.nb_threads()
    p.results[ "kernel" ] = p.kernel
    p.results[ "t_diagram" ] = best
    p.results[ "t_measures" ] = best
    p.results[ "t_weights" ] = t_weights
    p.results[ "t_convert" ] = t_convert
    p.results[ "weights_given" ] = int( not unweighted )
    p.results[ "ns_per_seed" ] = ns
    p.results[ "t_bsp_build" ] = t_build
    p.results[ "t_ctor" ] = t_ctor
    p.results[ "sum_of_measures" ] = float( m.sum() )
    p.results[ "reps" ] = reps
    p.results[ "output" ] = p.output
    if acc:
        for key, v in acc.items():
            p.results[ f"acc_{ key }" ] = v

    if not on_gpu:
        mem_kind = "none" if p.memory == 0 else "memo"
        wkind = "laguerre" if p.wscale else None
        ref = reference_lmo.diagram_ref( name, n, mem_kind, wkind ) or reference_lmo.diagram_ref( name, n, None, wkind )
        if ref:
            p.results[ "ref_seconds" ] = ref[ "seconds" ]
            p.results[ "ref_ns_per_seed" ] = ref[ "ns_per_seed" ]
            p.results[ "ratio_new_old" ] = ns / ref[ "ns_per_seed" ]

        header = [ "case", "n", "threads", "kernel", "time s", "ns/seed", "old ns/seed", "new/old", "old = " ]
        row = [ name + ( f" w~{ p.wscale }h2" if p.wscale else "" ), n, benchlib.nb_threads(), p.kernel, f"{ best :.4f}", f"{ ns :.0f}",
                "-" if not ref else f"{ ref[ 'ns_per_seed' ] :.0f}",
                benchlib.ratio( ns, ref and ref[ "ns_per_seed" ] ), "-" if not ref else f"{ ref[ 'stage' ] } { ref[ 'source' ] }" ]
        print( benchlib.table( header, [ row ] ) )
        print( f"  min of { reps }; apart from the cells: weights set { t_weights * 1e3 :.1f} ms ( { t_weights / n * 1e9 :.0f} ns/seed ), "
               f"copy to numpy { t_convert * 1e3 :.1f} ms; tree built in { t_build * 1e3 :.0f} ms, diagram built in { t_ctor * 1e3 :.0f} ms, "
               f"sum of the measures { m.sum() :.6f}"
               + ( "" if ref else "  ( no old number at this n )" ) )
        if acc:
            print( "  accuracy against a double kernel: " + benchlib.accuracy_line( acc ) )
    else:
        # -- THE GPU ROW: kernel only ( what the old GPU bench times ) and wall, the main kernel's resources
        ref = reference_lmo_gpu.gpu_ref( name, n, p.kernel )
        acc_ref = reference_lmo_gpu.gpu_accurate_ref( name, n, p.kernel )
        cpu_ref = reference_lmo_gpu.CPU_WITNESS.get( ( name, p.kernel ) )
        main = kernels[ "main" ] if kernels else None
        ns_k = best_kernel / n * 1e9 if best_kernel is not None else None
        p.results[ "t_kernel" ] = best_kernel if best_kernel is not None else -1
        p.results[ "ns_per_seed_kernel" ] = ns_k if ns_k is not None else -1
        p.results[ "ns_per_seed_wall" ] = ns
        p.results[ "warmup_s" ] = t_warm
        p.results[ "warmup_calls" ] = nb_warm
        if main:
            for key in ( "regs", "local_bytes", "static_shared", "block", "grid", "blocks_per_sm", "occupancy", "count" ):
                p.results[ f"main_{ key }" ] = main[ key ]
            p.results[ "nb_launches" ] = kernels[ "count" ]
        if p.output != "measures":
            ref = acc_ref = cpu_ref = None            # the old numbers are for the measures
        if ref and ns_k is not None:
            p.results[ "ref_ns_per_seed" ] = ref[ "ns_per_seed" ]
            p.results[ "ratio_new_old" ] = ns_k / ref[ "ns_per_seed" ]

        header = [ "case", "n", "kernel", "kernel ns/seed", "wall ns/seed", "regs", "occup.", "local B", "old ns/seed", "new/old", "old = " ]
        row = [ name + ( f" w~{ p.wscale }h2" if p.wscale else "" ) + ( "" if p.output == "measures" else f" [{ p.output }]" ), n, p.kernel,
                "-" if ns_k is None else f"{ ns_k :.1f}", f"{ ns :.1f}",
                main[ "regs" ] if main else "-",
                f"{ main[ 'occupancy' ] * 100 :.0f} % ( { main[ 'blocks_per_sm' ] } x { main[ 'block' ] } )" if main else "-",
                main[ "local_bytes" ] if main else "-",
                "-" if not ref else f"{ ref[ 'ns_per_seed' ] :g}",
                benchlib.ratio( ns_k, ref and ref[ "ns_per_seed" ] ),
                "-" if not ref else f"{ ref[ 'variant' ] } { ref[ 'source' ].split( ' ( ' )[ 0 ].split( ';' )[ 0 ] }" ]
        print( benchlib.table( header, [ row ] ) )
        if acc_ref is not ref and acc_ref and ns_k is not None:
            print( f"  the accurate old float kernel { acc_ref[ 'variant' ] }: { acc_ref[ 'ns_per_seed' ] } ns/seed, new/old { ns_k / acc_ref[ 'ns_per_seed' ] :.2f}" )
        if cpu_ref and ns_k is not None:
            print( f"  the old CPU witness ( 8 threads, same float ): { cpu_ref } ns/seed, i.e. x{ cpu_ref / ns_k :.1f} for this kernel" )
        if kernels:
            print( f"  kernels of the call ( best rep ): " + "; ".join(
                f"{ x[ 'code_name' ] }#{ x[ 'slot' ] } { x[ 'ms' ] :.3f} ms x{ x[ 'count' ] } ( { x[ 'regs' ] } regs, { x[ 'local_bytes' ] } B local, "
                f"grid { x[ 'grid' ] } x { x[ 'block' ] }, { x[ 'blocks_per_sm' ] } blocks/SM )" for x in kernels[ "all" ] ) )
        else:
            print( "  kernel-only time not available ( no timing entry point in the libraries: " + ", ".join( benchlib.KernelTiming.missing() ) + " )" )
        print( f"  min of { reps } after { nb_warm } warm-up calls in { t_warm * 1e3 :.0f} ms; weights set { t_weights * 1e3 :.2f} ms, "
               f"copy to numpy { t_convert * 1e3 :.2f} ms; tree built in { t_build * 1e3 :.0f} ms, diagram built in { t_ctor * 1e3 :.0f} ms, "
               f"sum of the measures { m.sum() :.9f}" + ( "" if ref else "  ( no old GPU number for this case )" ) )
        if acc:
            line = "  accuracy against a double kernel ( same tree ): " + benchlib.accuracy_line( acc )
            if acc_ref and acc_ref.get( "accuracy" ):
                a = acc_ref[ "accuracy" ]
                line += f"; old { acc_ref[ 'variant' ] }: " + ", ".join( f"{ key } { v :.1e}" for key, v in a.items() )
            print( line )
