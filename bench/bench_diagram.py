"""ONE DIAGRAM: the cost of a power diagram at fixed weights, on the cases of the old C++ campaign
( `nsdot/solvers_des_familles`, README § 3, § 11, § 19 ) -- to compare the python `sdot` with it.

    errand --env jax "bench_diagram"                                        # uniform 2D, 1e6 seeds
    errand --env jax "bench_diagram" --case=lines_voronoi,lines_equal       # a matrix: one run each
    errand --env lmo-jax "bench_diagram" --case=uniform --dim=3 --threads=8 --pin=yes --kernel=double

How it is timed: the tree and the diagram are built ONCE, after two warm-up rounds ( the first compiles, the
second fills the neighbour memory ), and we keep the MINIMUM of `reps`. Three durations are taken apart:
`t_weights` ( `pd.weights = w`: the sort and the refresh of the weight majorants -- the old bench does not
time it, its Newton reports it as its own `majorants` column ), `t_measures` ( `pd.measures`, the cells: THIS
is what the old `diagramme` times, and the ratio `new / old` compares it ) and `t_convert` ( the copy to numpy ).

A Voronoi cloud ( weights all zero ) is run WITHOUT weights by default ( `--weights=auto` ): the old engine
compiles its `POIDS = false` variant for it, with no majorant terms. `--weights=zeros` hands the zeros over,
which runs the weighted kernel: that is the cost of a Laguerre diagram that happens to have zero weights.
The old campaign: 8 pinned threads, kernel `double`, minimum of the repetitions, the machine alone: see
`bench/README.md` for the protocol.

The reference numbers are those of `reference_lmo.py` ( the ratio `new / old` is > 1 where we are slower ).
"""

import numpy
from errand import Param, bench

import benchlib
import cases
import reference_lmo

CASE_CHOICES = [ "uniform", "lines_voronoi", "lines_equal", "planes_voronoi", "planes_equal" ]

if p := bench( "diagram",
               case      = Param( "uniform", choices = CASE_CHOICES, help = "the case ( see cases.py ); `uniform` uses --dim" ),
               dim       = Param( 2, help = "dimension of the `uniform` case ( 2 or 3 )" ),
               n         = Param( 0, help = "number of seeds ( 0: the case of the campaign: 1e6 for the uniform ones, the file's size otherwise )" ),
               wscale    = Param( 0.0, help = "uniform case: weights in [ -w h^2, w h^2 ] ( 0: Voronoi; the old bench's `--weights`; 0.3 is a Laguerre one )" ),
               threads   = Param( 0, help = "workers ( 0: all the cores; the campaign: 8 )" ),
               pin       = Param( "yes", choices = [ "yes", "no", "env" ], help = "pin worker w to core w ( SDOT_PIN_THREADS ); `env`: leave the variable alone" ),
               kernel    = Param( "double", choices = [ "double", "float" ], help = "the kernel float type ( the campaign: double )" ),
               leaf_size = Param( 10, help = "seeds per BSP leaf ( the campaign: 10 )" ),
               weights   = Param( "auto", choices = [ "auto", "zeros" ], help = "`auto`: no weights at all when they are all zero ( Voronoi, like the old engine ); `zeros`: the weighted kernel with zero weights" ),
               memory    = Param( -1, help = "neighbour memories per seed ( -1: the default of the dimension, 0: none )" ),
               reps      = Param( 3, help = "timed repetitions ( we keep the minimum )" ),
               seed      = Param( 0, help = "seed of the uniform draw" ) ):
    benchlib.set_threads( p.threads, p.pin )          # BEFORE the first kernel: the pool reads them once
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    import time

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

    def once():
        t0 = time.perf_counter()
        if not unweighted:
            pd.weights = w
        t1 = time.perf_counter()
        meas = pd.measures.value                      # the cells: what the old `diagramme` times
        t2 = time.perf_counter()
        m = numpy.asarray( meas ).reshape( -1 )
        t3 = time.perf_counter()
        return ( t1 - t0, t2 - t1, t3 - t2 ), m

    once()                                            # warm-up: this one compiles
    once()                                            # and this one fills the memory
    ts, m = once()
    t_weights, best, t_convert = ts
    for _ in range( p.reps - 1 ):
        ts = once()[ 0 ]
        t_weights, best, t_convert = min( t_weights, ts[ 0 ] ), min( best, ts[ 1 ] ), min( t_convert, ts[ 2 ] )

    # -- the old number to compare with
    mem_kind = "none" if p.memory == 0 else "memo"
    wkind = "laguerre" if p.wscale else None
    ref = reference_lmo.diagram_ref( name, n, mem_kind, wkind ) or reference_lmo.diagram_ref( name, n, None, wkind )
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
    if ref:
        p.results[ "ref_seconds" ] = ref[ "seconds" ]
        p.results[ "ref_ns_per_seed" ] = ref[ "ns_per_seed" ]
        p.results[ "ratio_new_old" ] = ns / ref[ "ns_per_seed" ]

    header = [ "case", "n", "threads", "kernel", "time s", "ns/seed", "old ns/seed", "new/old", "old = " ]
    row = [ name + ( f" w~{ p.wscale }h2" if p.wscale else "" ), n, benchlib.nb_threads(), p.kernel, f"{ best :.4f}", f"{ ns :.0f}",
            "-" if not ref else f"{ ref[ 'ns_per_seed' ] :.0f}",
            benchlib.ratio( ns, ref and ref[ "ns_per_seed" ] ), "-" if not ref else f"{ ref[ 'stage' ] } { ref[ 'source' ] }" ]
    print( benchlib.table( header, [ row ] ) )
    print( f"  min of { p.reps }; apart from the cells: weights set { t_weights * 1e3 :.1f} ms ( { t_weights / n * 1e9 :.0f} ns/seed ), "
           f"copy to numpy { t_convert * 1e3 :.1f} ms; tree built in { t_build * 1e3 :.0f} ms, diagram built in { t_ctor * 1e3 :.0f} ms, "
           f"sum of the measures { m.sum() :.6f}"
           + ( "" if ref else "  ( no old number at this n )" ) )
