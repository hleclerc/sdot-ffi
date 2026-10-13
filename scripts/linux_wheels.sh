#!/usr/bin/env bash
# The Linux wheels, built the way a release needs them: inside the manylinux image ( glibc 2.28, a recent clang, GCC 14 for nvcc ),
# so that the kernels run on any Linux of the last years.
#
#     bash scripts/linux_wheels.sh            kernels of every framework found, then the wheels into dist/ ( tag manylinux_2_27_x86_64 )
#
# It is what `make local_wheels` does on Linux. Prerequisites ( once, see scripts/manylinux.def ):
#   $SIF       the image            ( default ~/release-images/manylinux_loom.sif )
#   $VENV      a venv of the image's python with jax numpy scipy ninja wheel twine, loom and sdot installed editable ( default ~/release-images/venv )
#   $MAMBA     micromamba root holding prj-<framework> environments, for the frameworks other than jax/numpy ( default ~/.local/share/mamba )
# The only external library of the CUDA kernels is libcudart.so.13: the machine must have a CUDA 13 runtime ( pip: nvidia-cuda-runtime ).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
SIF=${SIF:-$HOME/release-images/manylinux_loom.sif}
VENV=${VENV:-$HOME/release-images/venv}
MAMBA=${MAMBA:-$HOME/.local/share/mamba}
FRAMEWORKS=${FRAMEWORKS:-"jax numpy cupy torch"}
TAG=${TAG:-manylinux_2_27_x86_64}
[ -f "$SIF" ] || { echo "no image $SIF ( apptainer build --fakeroot, from scripts/manylinux.def )" >&2; exit 1; }

# the directories may be links to another disk ( /data ): the container only binds the real places it is told about
BINDS=()
for d in "$HERE/.." "$VENV" "$MAMBA" "$HOME/release-images" "$(dirname "$SIF")"; do
    [ -e "$d" ] && BINDS+=( --bind "$(realpath "$d")" )
done

apptainer exec --nv "${BINDS[@]}" --env HERE="$HERE" --env VENV="$VENV" --env MAMBA="$MAMBA" --env FRAMEWORKS="$FRAMEWORKS" --env TAG="$TAG" "$SIF" bash -s <<'IN'
set -euo pipefail
export LOOM_CXX=/usr/bin/clang++
GCC=/opt/rh/gcc-toolset-14/root/usr/lib/gcc/x86_64-redhat-linux/14
export LOOM_CXXFLAGS="-std=c++23 --gcc-install-dir=$GCC"
export LOOM_LDFLAGS="--gcc-install-dir=$GCC"
# nvcc reads the headers of GCC 14 through a g++ ( with a clang as host it sees C++23 features it cannot parse )
export LOOM_CUDA_HOST=/opt/rh/gcc-toolset-14/root/usr/bin/g++
export LOOM_CUDA_HOST_CXXFLAGS="-std=c++23"
export LOOM_CACHE_DIR=${LOOM_CACHE_DIR:-$HOME/release-images/cache}
cd "$HERE"
# the Linux kernels only ( the macOS ones of the same tree stay )
find src/sdot/precompiled-kernels -name '*.so' -delete
find src/sdot/precompiled-kernels -mindepth 1 -type d -empty -delete
for fw in $FRAMEWORKS; do
    py=$MAMBA/envs/prj-$fw/bin/python
    [ -x "$py" ] || py=$VENV/bin/python
    [ -x "$py" ] || { echo "=== $fw: no python, skipped"; continue; }
    echo "=== kernels for $fw ( $py )"
    # the CUDA runtime a framework brings ( pip: nvidia/*/lib ), which its own activation would have put on the path
    libs=$(ls -d "$(dirname "$py")"/../lib/python*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':' || true)
    LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:-}" LOOM_FRAMEWORK=$fw "$py" -m sdot.catalogue 2>&1 | grep -v "^\[\|^ninja" | tail -14
    rm -rf "$LOOM_CACHE_DIR/build/obj"
done
echo "=== wheels"
"$VENV/bin/python" scripts/wheels.py --platform-tag "$TAG"
echo "=== auditwheel"
for w in dist/sdot-*-"$TAG".whl; do auditwheel show "$w" | grep -A1 "constrains\|consistent" || true; done
IN
