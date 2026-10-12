"""What the cell engine as a shared library costs, against the engine inlined in every kernel.

    python bench/engine_unit.py            # on this machine ( or: errand --env lmo-jax ... )

Each variant runs in a process of its own with its flag ( `-DSDOT_NO_ENGINE_UNIT` : the engine inlined, as it was ), so that its
kernels are compiled for it. The cells are passed by reference ( they live in the work-item's scratch ): the call costs a few
cycles, to be compared with the thousands a cell takes. What is printed is the median time of `measures` and of `moments` in 2D
and 3D, FP32 and FP64, and the ratio to the inlined build.
"""
import json
import os
import subprocess
import sys

VARIANTS = { "inlined": "-DSDOT_NO_ENGINE_UNIT", "unit": "" }

CHILD = r'''
import json, statistics, time, numpy as np
import jax
from sdot import PowerDiagram, box_half_spaces
out = {}
rng = np.random.default_rng( 0 )
for d, n in ( ( 2, 200000 ), ( 3, 40000 ) ):
    pos = rng.uniform( 0.05, 0.95, size = ( n, d ) )
    box = box_half_spaces( [ 0 ] * d, [ 1 ] * d )
    for kernel in ( "FP32", "FP64" ):
        for what in ( "measures", "moments" ):
            def run():
                pd = PowerDiagram( pos, boundaries = box, accelerator = None, kernel_dtype = kernel )
                r = np.asarray( pd.measures.value ) if what == "measures" else pd.moments
                return r
            run()
            ts = []
            for _ in range( 5 ):
                t = time.perf_counter(); run(); ts.append( time.perf_counter() - t )
            out[ f"{what} {d}D {kernel}" ] = statistics.median( ts )
print( "RESULT" + json.dumps( out ) )
'''


def main():
    results = {}
    for name, flag in VARIANTS.items():
        env = dict( os.environ, PYTHONPATH = os.pathsep.join( p for p in sys.path if p ) )
        env[ "LOOM_CXXFLAGS" ] = ( env.get( "LOOM_CXXFLAGS", "" ) + " " + flag ).strip()
        r = subprocess.run( [ sys.executable, "-c", CHILD ], env = env, capture_output = True, text = True )
        line = next( ( l for l in r.stdout.splitlines() if l.startswith( "RESULT" ) ), None )
        if line is None:
            print( name, "failed:", ( r.stderr or r.stdout )[ -600: ] )
            continue
        results[ name ] = json.loads( line[ 6: ] )
    base = results.get( "inlined" )
    print( f"{'':22s}" + "".join( f"{v:>12s}" for v in results ) )
    for case in ( base or {} ):
        print( f"{case:22s}" + "".join( f"{r[ case ] * 1e3:8.1f} ms {r[ case ] / base[ case ]:4.2f}" if False else f"{r[ case ] * 1e3:7.1f}ms{r[ case ] / base[ case ]:5.2f}" for r in results.values() ) )


if __name__ == "__main__":
    main()
