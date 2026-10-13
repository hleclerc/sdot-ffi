#!/usr/bin/env python3
"""The wheels of loom and sdot, for the OS / CPU family of this machine: build, validate, publish.

    python scripts/wheels.py                  build dist/*.whl (the sdot one carries src/sdot/precompiled-kernels) and validate them
    python scripts/wheels.py --publish URL    ... then upload them to the index at URL (the token is in TWINE_PASSWORD)

The kernels themselves come from `python -m sdot.catalogue` ( run it first, on every machine that has hardware to cover,
and merge the directories ). The binaries do not depend on the python version: the wheel is `py3-none-<platform>`.
"""
from pathlib import Path
import argparse, shutil
import datetime
import re
import os
import subprocess
import sys
import sysconfig

ROOT = Path( __file__ ).resolve().parents[ 1 ]


def run( *cmd, **kw ):
    print( "+", " ".join( map( str, cmd ) ), flush = True )
    subprocess.run( [ str( c ) for c in cmd ], check = True, **kw )


def next_version( current, today = None ):
    """`year.month.day.number`: the version of a release made today. `number` counts the releases of the day (1, then 2, ...) -- the current
    version says how many there already were. No padding: PEP 440 would drop it, `2026.10.5.1` is what pip shows for the 5th."""
    today = today or datetime.date.today()
    stem = f"{ today.year }.{ today.month }.{ today.day }"
    m = re.fullmatch( re.escape( stem ) + r"\.(\d+)", current )
    return f"{ stem }.{ int( m.group( 1 ) ) + 1 if m else 1 }"


def set_version( version, loom ):
    """Write `version` in the `pyproject.toml` of loom and of sdot, and make sdot depend on exactly that loom: the kernels of an sdot wheel
    were compiled against the sources loom generated at that version, and are looked up by hash of them."""
    for toml in ( loom / "pyproject.toml", ROOT / "pyproject.toml" ):
        text = toml.read_text()
        text = re.sub( r'(?m)^version = "[^"]*"', f'version = "{ version }"', text, count = 1 )
        if toml.parent == ROOT:
            text = re.sub( r'(?m)^(\s*)"loom-ffi[^"]*",', rf'\1"loom-ffi=={ version }",', text, count = 1 )
        toml.write_text( text )


def current_version( loom ):
    return re.search( r'(?m)^version = "([^"]*)"', ( loom / "pyproject.toml" ).read_text() ).group( 1 )


def foreign_kernels( platform_tag ):
    """The precompiled kernels of the other operating system ( `.dylib` in a Linux wheel, `.so` in a macOS one ): the tree
    may hold both ( collected from several machines ), a wheel carries only its own."""
    keep = ".dylib" if platform_tag.startswith( "macosx" ) else ".so"
    root = ROOT / "src" / "sdot" / "precompiled-kernels"
    return [ f for f in root.rglob( "*" ) if f.is_file() and f.suffix in ( ".so", ".dylib" ) and f.suffix != keep ]


def main():
    p = argparse.ArgumentParser( description = __doc__.splitlines()[ 0 ] )
    p.add_argument( "--platform-tag", default = sysconfig.get_platform().replace( "-", "_" ).replace( ".", "_" ),
                    help = "the platform tag of the sdot wheel (default: this interpreter's; give the manylinux / macosx one for a release)" )
    p.add_argument( "--loom", default = None, help = "the loom checkout (default: ./loom or ../loom)" )
    p.add_argument( "--version", default = None, help = "write this version in loom and sdot first (`next`: year.month.day.number of today, the number counting today's releases)" )
    p.add_argument( "--no-validate", action = "store_true" )
    p.add_argument( "--no-build", action = "store_true", help = "only write the version" )
    p.add_argument( "--publish", metavar = "URL", default = None, help = "upload the wheels of dist/ to the index at this URL ( e.g. https://upload.pypi.org/legacy/ ); nothing is uploaded without it" )
    a = p.parse_args()

    loom = Path( a.loom ) if a.loom else next( d for d in ( ROOT / "loom", ROOT.parent / "loom" ) if ( d / "pyproject.toml" ).exists() )
    if a.version:
        set_version( next_version( current_version( loom ) ) if a.version == "next" else a.version, loom )
        print( "version:", current_version( loom ), flush = True )
    if a.no_build:
        return
    dist = ROOT / "dist"
    run( sys.executable, "-m", "pip", "install", "--quiet", "wheel" )
    out = dist / ".build"    # wheels of other platforms already in dist/ ( e.g. the manylinux one ) are kept
    shutil.rmtree( out, ignore_errors = True )
    out.mkdir( parents = True )
    run( sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", out, loom )
    # the other system's kernels step aside while the sdot wheel is made
    aside = ROOT / "dist" / ".aside"
    moved = []
    for f in foreign_kernels( a.platform_tag ):
        target = aside / f.relative_to( ROOT )
        target.parent.mkdir( parents = True, exist_ok = True )
        f.rename( target )
        moved.append( ( target, f ) )
    try:
        run( sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", out, ROOT )
    finally:
        for target, f in moved:
            target.rename( f )
        shutil.rmtree( aside, ignore_errors = True )
    sdot = next( out.glob( "sdot-*.whl" ) )
    run( sys.executable, "-m", "wheel", "tags", "--remove", "--python-tag", "py3", "--abi-tag", "none", "--platform-tag", a.platform_tag, sdot )

    wheels = sorted( out.glob( "*.whl" ) )
    if not a.no_validate:
        # nothing may be compiled: the kernels of the wheel are what is tested
        run( sys.executable, ROOT / "scripts" / "validate_wheel.py", "--wheel", *wheels, env = { **os.environ, "LOOM_KERNELS": "catalogue" } )
    for w in wheels:
        w.replace( dist / w.name )
    shutil.rmtree( out, ignore_errors = True )
    wheels = sorted( dist.glob( "*.whl" ) )
    if a.publish:
        run( sys.executable, "-m", "pip", "install", "--quiet", "twine" )
        run( sys.executable, "-m", "twine", "upload", "--skip-existing", "--repository-url", a.publish, *wheels,
             env = { **os.environ, "TWINE_USERNAME": "__token__" } )


if __name__ == "__main__":
    main()
