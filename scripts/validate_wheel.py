#!/usr/bin/env python3
"""Validates the `sdot` wheel end to end: build -> CLEAN venv -> install -> smoke test.

  python scripts/validate_wheel.py                 # builds the wheel then tests it
  python scripts/validate_wheel.py --wheel a.whl   # tests an existing wheel
  python scripts/validate_wheel.py --keep          # keeps the venv/cache for a post-mortem

The goal is to prove that the wheel is self-contained: we install it in a FRESH venv (no
PYTHONPATH to the repo, no local `build/` reused) and we run make_hypercube(2D) +
measure -- the whole cycle generation -> compilation (host compiler) -> registration (Jax FFI) -> execution.

The key proof is that the compilation uses the C++ headers SHIPPED IN THE WHEELS
(`.../site-packages/loom/_include`, `sdot/_include`), not those of the checkout: `include_roots()`
must only name paths of the venv, and the kernel must compile with them.

Unlike `run_tests.py`, this script NEVER inserts `src/python` into sys.path: that would be
precisely the mistake that would mask a broken wheel by importing `sdot` from the checkout.
"""
from pathlib import Path
import subprocess
import argparse
import re
import tempfile
import shutil
import glob
import sys
import os

ROOT = Path( __file__ ).resolve().parents[ 1 ]

# Reproduces tests/python/test_Cell.py :: test( "basic" ), plus a numeric assertion (the original
# block only prints, useful to the eye but not usable as a pass/fail signal).
SMOKE = """
import numpy as np
from sdot import Cell

c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 2, 0 ], [ 0, 1 ] ] )
print( c.vertex_positions )
print( c.measure )
assert abs( float( np.asarray( c.measure ) ) - 2.0 ) < 1e-9, "measure != 2.0"
print( "SMOKE-OK" )
"""

PRECHECK = """
import sdot, loom.compilation as c
print( sdot.__file__ )
print( c.include_roots() )
"""


def _venv_python( venv_dir: Path ) -> Path:
    sub = "Scripts" if os.name == "nt" else "bin"
    return venv_dir / sub / ( "python.exe" if os.name == "nt" else "python" )


def _run( cmd, **kw ):
    print( "$ " + " ".join( map( str, cmd ) ), flush = True )
    return subprocess.run( [ str( c ) for c in cmd ], **kw )


# WHERE EACH PACKAGE LIVES, now that the four are separate repositories: `sdot` IS this repository, and
# `loom` is placed next to it ( by the CI, or by the work plan's `scripts/bootstrap.sh` ).
SOURCES = { "loom": ROOT / "loom", "sdot": ROOT }


def build_wheels() -> list:
    """The two wheels, `loom` then `sdot` (the second depends on the first), under `dist/`."""
    wheels = []
    for pkg in ( "loom", "sdot" ):
        src = SOURCES[ pkg ]
        if not ( src / "pyproject.toml" ).is_file():
            raise RuntimeError( f"{ pkg } not found under { src } -- `loom` must be cloned next to "
                                f"this repository ( see the workflow, or `bootstrap.sh` )" )
        _run( [ sys.executable, "-m", "pip", "wheel", "--quiet", "--no-deps", "-w", ROOT / "dist", src ], check = True )
        found = sorted( glob.glob( str( ROOT / "dist" / f"{ pkg }-*.whl" ) ), key = os.path.getmtime )
        if not found:
            raise RuntimeError( f"no { pkg } wheel produced under dist/" )
        wheels.append( Path( found[ -1 ] ) )
    return wheels


def validate( wheels: list, keep: bool ) -> int:
    scratch = Path( tempfile.mkdtemp( prefix = "sdot-validate-" ) )
    venv_dir  = scratch / "venv"
    cache_dir = scratch / "sdot-cache"   # empty -> no binary reused
    work_dir  = scratch / "run"          # cwd of the smoke test, never the repo root
    work_dir.mkdir( parents = True )
    print( f"scratch: { scratch }", flush = True )

    try:
        _run( [ sys.executable, "-m", "venv", venv_dir ], check = True )
        py = _venv_python( venv_dir )
        # `ninja` and `jax` come from PyPI; loom and sdot from the freshly built wheels
        _run( [ py, "-m", "pip", "install", "--quiet", *( f"{ w }[jax]" if w.name.startswith( "loom" ) else str( w ) for w in wheels ) ], check = True )

        # Quick pre-check: fail fast if the install is badly packaged, BEFORE compiling.
        r = _run( [ py, "-c", PRECHECK ], cwd = work_dir, capture_output = True, text = True )
        if r.returncode:
            print( r.stdout + r.stderr, flush = True )
            raise RuntimeError( "import pre-check failed" )
        # .resolve() on both sides: on macOS /var is a symlink to /private/var, and
        # __file__ is not canonicalized whereas include_roots() is.
        venv_real = str( venv_dir.resolve() )
        for line in re.findall( r"/[^\s'\]\[,]+", r.stdout ):
            if not str( Path( line ).resolve() ).startswith( venv_real ):
                raise RuntimeError(
                    f"path outside the venv (import from the checkout?): { line }\n{ r.stdout }"
                )
        print( r.stdout, flush = True )

        # Full smoke test, in an isolated env: no PYTHONPATH, fresh cache, default build.
        env = dict( os.environ )
        env.pop( "PYTHONPATH", None )
        env.pop( "SDOT_BUILD_DIR", None )
        env[ "SDOT_CACHE_DIR" ] = str( cache_dir )

        r = _run( [ py, "-c", SMOKE ], cwd = work_dir, env = env, capture_output = True, text = True )
        out = r.stdout + r.stderr
        print( out, flush = True )
        if r.returncode or "SMOKE-OK" not in r.stdout:
            raise RuntimeError( "smoke test failed" )

        # The proof that the compilation used the wheel's headers is the pre-check above:
        # `include_roots()` only names paths of the venv, and the kernel compiled with them.

        print( "\nVALIDATION OK", flush = True )
        return 0
    finally:
        if keep:
            print( f"\n--keep: scratch kept -> { scratch }", flush = True )
        else:
            shutil.rmtree( scratch, ignore_errors = True )


def main() -> int:
    p = argparse.ArgumentParser( description = "validates the sdot wheel in a clean venv" )
    p.add_argument( "--wheel", type = Path, nargs = "*", help = "existing wheels to test (loom and sdot; otherwise we build them)" )
    p.add_argument( "--keep", action = "store_true", help = "keep the scratch venv/cache" )
    args = p.parse_args()

    wheels = [ w.resolve() for w in args.wheel ] if args.wheel else build_wheels()
    return validate( wheels, args.keep )


if __name__ == "__main__":
    sys.exit( main() )
