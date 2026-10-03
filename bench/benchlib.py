"""Small helpers shared by `bench_diagram.py` and `bench_newton.py` ( this file declares no work, and does not import
errand or sdot: it is imported by files that must be light to discover ).

THE THREADS. The kernels run on loom's process-wide pool ( `loom/cpp/runtime/cpu_thread_pool.cpp` ), which reads at
its creation, i.e. at the FIRST kernel call of the process:

  SDOT_NB_THREADS   number of workers ( default: all the cores ) -- read by the C++ pool
  LOOM_NB_THREADS   the same number, read by the python side ( `loom/devices/Cpu.py`: the per-thread scratch is sized
                    on it, 'the two must agree' ); both are set here
  SDOT_PIN_THREADS  `1`: worker `w` is pinned to core `w` ( Linux only; the calling thread, worker 0, is not pinned )

so `set_threads` must be called before anything is computed -- errand runs each entry in its own process, which is
what makes a sweep over `--threads` correct.
"""

import os
import platform
import socket

#: the variables that change what is measured, and are therefore printed with each result
ENV_VARS = ( "LOOM_NB_THREADS", "SDOT_NB_THREADS", "SDOT_PIN_THREADS", "SDOT_CPU_VARIANT", "SDOT_NO_MARCH_NATIVE", "SDOT_CXXFLAGS",
             "LOOM_CXX", "LOOM_CXXFLAGS", "LOOM_FRAMEWORK", "LOOM_DEVICE", "SDOT_KTYPE", "SDOT_CATALOGUE_DIR",
             "SDOT_CASES_DIR", "LOOM_KERNEL_TIMING", "LOOM_BUILD_DIR", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS" )


def set_threads( threads, pin ):
    """`threads`: 0 = all the cores. `pin`: `"yes"` | `"no"` | `"env"` ( leave `SDOT_PIN_THREADS` as it is )."""
    if threads > 0:
        os.environ[ "SDOT_NB_THREADS" ] = os.environ[ "LOOM_NB_THREADS" ] = str( threads )
    if pin in ( "yes", "no" ):
        os.environ[ "SDOT_PIN_THREADS" ] = "1" if pin == "yes" else "0"


def set_openmp( omp_threads ):
    """EXPERIMENT: compile with `-fopenmp` ( the builtin backend of AMGCL parallelizes through OpenMP only ) and make
    libgomp available to the kernel, which loom links WITHOUT it ( `link_shared` carries no `-fopenmp` ): the library is
    loaded GLOBAL here, before the first kernel. `omp_threads`: `OMP_NUM_THREADS` ( 0: leave it: all the hardware threads,
    which is what the old binary did -- it never called `omp_set_num_threads` ). Linux only. Call BEFORE anything is compiled."""
    import ctypes
    flags = os.environ.get( "LOOM_CXXFLAGS", os.environ.get( "SDOT_CXXFLAGS", "" ) )     # loom reads LOOM_ first
    if "-fopenmp" not in flags:
        os.environ[ "LOOM_CXXFLAGS" ] = ( flags + " -fopenmp" ).strip()
    if omp_threads > 0:
        os.environ[ "OMP_NUM_THREADS" ] = str( omp_threads )
    ctypes.CDLL( "libgomp.so.1", mode = ctypes.RTLD_GLOBAL )


def nb_threads():
    """the number of workers the pool will have"""
    try:
        n = int( os.environ.get( "LOOM_NB_THREADS", os.environ.get( "SDOT_NB_THREADS", "0" ) ) )
    except ValueError:
        n = 0
    return n if n > 0 else ( os.cpu_count() or 1 )


def env_report():
    """one line: where, how many cores, and every variable of `ENV_VARS` that is set"""
    active = [ f"{ k }={ os.environ[ k ] }" for k in ENV_VARS if k in os.environ ]
    return ( f"host { socket.gethostname() }, { platform.system() } { platform.machine() }, { os.cpu_count() } cores; "
             + ( ", ".join( active ) if active else "no sdot/loom variable set" )
             + ( "" if os.environ.get( "SDOT_PIN_THREADS", "0" ) not in ( "0", "" ) else " [threads NOT pinned]" ) )


def case_label( name, dim, n, wscale = 0.0 ):
    return f"{ name }" + ( f" ( w ~ { wscale } h^2 )" if wscale else "" ) + f" { dim }D n={ n }"


def table( header, rows ):
    """a compact table; `rows` are lists of strings"""
    width = [ max( len( str( r[ k ] ) ) for r in [ header, *rows ] ) for k in range( len( header ) ) ]
    line = lambda r: "  " + "  ".join( str( c ).ljust( w ) if k == 0 else str( c ).rjust( w ) for k, ( c, w ) in enumerate( zip( r, width ) ) )
    out = [ line( header ), "  " + "  ".join( "-" * w for w in width ) ]
    out += [ line( r ) for r in rows ]
    return "\n".join( out )


def ratio( new, old ):
    return "-" if not old or new is None else f"{ new / old :.2f}"


def driver_line():
    """the loom framework and device in use ( import loom late: call it AFTER `set_threads` )"""
    from loom.drivers.driver import driver
    return f"driver { driver.framework }, device { driver.device }"


# -- GPU ------------------------------------------------------------------------------------------------

def set_kernel_timing( on = True ):
    """`LOOM_KERNEL_TIMING=1`: CUDA events around each kernel launch ( `loom/devices/kernel_timing.py` ). Read ONCE
    per library, at its first launch: call it before anything runs. Without effect on the CPU. A value already in
    the environment is left alone."""
    os.environ.setdefault( "LOOM_KERNEL_TIMING", "1" if on else "0" )


class KernelTiming:
    """the kernel-only time of a window: `reset()`, the calls, `read()` -> `{ ms, count, main, all }`, `main` being
    the kernel that took the longest ( its registers, local bytes, occupancy... ), `all` every kernel that ran"""

    def __init__( self ):
        from loom.devices import kernel_timing
        self.kt = kernel_timing

    def reset( self ):
        self.kt.reset()

    def read( self ):
        ran = [ k for k in self.kt.read() if k[ "count" ] > 0 ]
        if not ran:
            return None
        return dict( ms = sum( k[ "ms" ] for k in ran ), count = sum( k[ "count" ] for k in ran ),
                     main = max( ran, key = lambda k: k[ "ms" ] ), all = ran )

    @staticmethod
    def missing():
        from loom.devices import kernel_timing
        return kernel_timing.missing()


def block_until_ready( x ):
    """waits for an asynchronous result ( a jax array ); nothing for the others"""
    f = getattr( x, "block_until_ready", None )
    if f is not None:
        f()
    return x


def dtype_of( tensor ):
    raw = getattr( tensor, "raw", tensor )
    return str( getattr( raw, "dtype", type( raw ).__name__ ) )


def accuracy( m, ref ):
    """`| m - ref | / ref` per cell, over the cells where `ref > 0`: median, p99.99, max ( and how many cells had none )"""
    import numpy
    m = numpy.asarray( m, dtype = float ).reshape( -1 )
    ref = numpy.asarray( ref, dtype = float ).reshape( -1 )
    ok = ref > 0
    rel = numpy.abs( m[ ok ] - ref[ ok ] ) / ref[ ok ]
    return dict( median = float( numpy.median( rel ) ), p9999 = float( numpy.quantile( rel, 0.9999 ) ), max = float( rel.max() ),
                 empty_cells = int( ( ~ok ).sum() ), sum_diff = float( abs( m.sum() - ref.sum() ) ) )


def accuracy_line( acc ):
    return ( f"median { acc[ 'median' ] :.1e}, p99.99 { acc[ 'p9999' ] :.1e}, max { acc[ 'max' ] :.1e} "
             f"( { acc[ 'empty_cells' ] } empty cells left out; | sum - sum_double | = { acc[ 'sum_diff' ] :.1e} )" )
