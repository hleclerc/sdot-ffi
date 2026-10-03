#!/usr/bin/env bash
# The GPU benchmark. Self-contained : it depends on neither CGAL nor `2d_des_familles`, it reads the
# connectivity file written by `power_{2,3}d_light --dump`.
#
# `-arch=sm_75` : Turing (RTX 2080 Ti). To be changed on another card.
# No `--use_fast_math` : it would replace divisions with approximate reciprocals, which
# would distort precisely what this benchmark tries to measure -- the FP32 precision on the geometry.
set -euo pipefail
here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
nvcc -O3 -std=c++17 -arch="${ARCH:-sm_75}" -lineinfo \
     -Xptxas -v \
     "$here/cells_gpu.cu" -o "$here/cells_gpu" 2>&1 | grep -E "ptxas|error|warning" || true
ls -l "$here/cells_gpu"
