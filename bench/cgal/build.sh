#!/usr/bin/env bash
# CGAL FROM ITS GIT SOURCES, then the bench is compiled against them.
#
# From git and not from a package: CGAL moves often, and a baseline that compares against a version
# from two years ago does not tell what we want to know. CGAL 6 is a HEADER library -- so there is
# nothing to build in the clone, only a `-I` to point at it (plus gmp/mpfr to link, which
# are real libraries and come from the micromamba env).
#
#   ./sdot/bench/cgal/build.sh          # clone (or update) + compile
#   ./sdot/bench/cgal/power_2d 1000000  # run
#
# `CGAL_DIR`: where the clone lands. By default next to the toolchains cache, not in the
# checkout -- it is neither our source nor a build artifact of ours.
set -euo pipefail

here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CGAL_DIR="${CGAL_DIR:-$HOME/.cache/sdot/cgal}"
CGAL_REF="${CGAL_REF:-main}"

if [ -d "$CGAL_DIR/.git" ]; then
    echo "== CGAL: updating $CGAL_DIR ($CGAL_REF)"
    git -C "$CGAL_DIR" fetch --depth 1 origin "$CGAL_REF"
    git -C "$CGAL_DIR" checkout -q FETCH_HEAD
else
    echo "== CGAL: cloning into $CGAL_DIR ($CGAL_REF)"
    mkdir -p "$( dirname "$CGAL_DIR" )"
    git clone --depth 1 --branch "$CGAL_REF" https://github.com/CGAL/cgal.git "$CGAL_DIR"
fi
echo "   $( git -C "$CGAL_DIR" log -1 --format='%h %ad %s' --date=short )"

# the headers: CGAL in a git checkout is SPLIT into packages (Kernel_23/include, Triangulation_2/...),
# whereas a release glues them back under a single `include/`. So we pass one `-I` per useful package.
incs=()
if [ -d "$CGAL_DIR/Installation/include" ]; then
    for d in "$CGAL_DIR"/*/include; do
        [ -d "$d" ] && incs+=( -I "$d" )
    done
else
    incs+=( -I "$CGAL_DIR/include" )
fi

# gmp / mpfr / boost: from the active micromamba env (see `.envs.py`, packages of the NSDOT layer).
prefix="${CONDA_PREFIX:-${MAMBA_PREFIX:-}}"
if [ -z "$prefix" ]; then
    echo "!! no active conda/micromamba env: gmp/mpfr/boost not found." >&2
    echo "   micromamba activate nsdot   (or ./run env create)" >&2
    exit 1
fi

echo "== compiling the bench"
${CXX:-g++} -std=c++20 -O3 -march=native -DNDEBUG -DCGAL_NDEBUG \
    "${incs[@]}" -I "$prefix/include" \
    "$here/power_2d.cpp" \
    -L "$prefix/lib" -Wl,-rpath,"$prefix/lib" -lgmp -lmpfr \
    -o "$here/power_2d"

echo "== done: $here/power_2d"
