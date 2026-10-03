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
             "SDOT_CASES_DIR", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS" )


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
