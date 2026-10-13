# Performance

This page gives the measured cost of sdot: a single power diagram, a whole Newton solve, CPU against GPU, and how both
scale with the number of diracs and with the dimension. Every number comes from the scripts in
[`bench/`](https://github.com/sdot-team/sdot/tree/main/bench). The [Reproducing](#reproducing) section gives the commands.

## Setup

| machine | hardware | used for |
|---|---|---|
| `lmo` | Xeon W-2145 (8 cores, 8 pinned threads), RTX 2080 Ti (Turing, FP64 at 1/32) | CPU, 2D GPU |
| `n22` | RTX A6000 (Ampere, 48 GB, FP64 at 1/64) | 3D GPU, 2D GPU at 1e6 |

Protocol:

* A Newton solve starts cold, from the Voronoi diagram (zero weights), tree build included. It stops when
  `max |mass - target| / target <= 1e-6`.
* GPU times are for a **jitted** call (JAX). An eager call costs about 0.1 s more.
* `mixed` (the GPU default) runs the float kernel and switches to the double kernel when float precision is no longer
  enough. The final accuracy is the double one.

## In short

| problem | n | CPU (8 threads) | GPU | speed-up |
|---|---|---|---|---|
| 2D, uniform, Lebesgue | 1e5 | 1.02 s | 0.072 s (2080 Ti) | 14x |
| 2D, clustered (5 lines) | 1e5 | 4.0 s | 0.29 s (2080 Ti) | 14x |
| 2D, uniform, Lebesgue | 1e6 | – | 0.38 s (A6000) | – |
| 3D, uniform, Lebesgue | 1e5 | 2.29 s | 0.135 s (A6000) | 17x |
| 3D, clustered (4 planes) | 1e5 | 6.67 s | 0.47 s (A6000) | 14x |
| 2D, 512² image target | 1e5 | 29.2 s | 1.88 s (2080 Ti) | 16x |
| 2D, 4 gaussians (σ = 0.02) | 1e5 | 62.7 s | 7.2 s (2080 Ti) | 9x |

The image and gaussian targets include a width continuation (16 to 22 smoothing stages), which explains why they cost
more than the Lebesgue cases.

## One power diagram

This is the cost of computing every cell's measure for fixed weights. It is the inner loop of every solver, measured in
nanoseconds per dirac (kernel time only).

| case | n | GPU float | GPU double |
|---|---|---|---|
| 2D uniform | 1e6 | **9.3** (2080 Ti) | 57 (2080 Ti) |
| 2D lines, Voronoi | 1e5 | 25 (2080 Ti) | 80 (2080 Ti) |
| 2D lines, equal areas | 1e5 | 52 (2080 Ti) | 289 (2080 Ti) |
| 3D uniform | 1e6 | **118** (A6000) | 457 (A6000) |
| 3D planes, Voronoi | 1e5 | 128 (A6000) | 451 (A6000) |
| 3D planes, equal volumes | 1e5 | 217 (A6000) | 1190 (A6000) |

So a 2D diagram with one million diracs takes about **10 ms** on a gaming GPU, and a 3D one about **120 ms**.

In float, the cell measures match the double kernel to a median relative error of 1e-14 in 2D (max 1e-8). In 3D the
median error is 0 and the maximum 7e-5. The `mixed` mode then finishes the solve in double.

## Scaling with the number of diracs

The number of Newton iterations barely depends on `n` (5 iterations and 6 diagrams for a uniform cloud at 1e5 and 1e6).
The total cost therefore follows the cost of one diagram plus one linear solve, which are both close to linear in `n`.

| dim | n | GPU, mixed | time per dirac |
|---|---|---|---|
| 2D | 1e5 | 0.073 s (A6000) | 0.73 µs |
| 2D | 1e6 | 0.383 s (A6000) | 0.38 µs |
| 2D | 1e7 | 8.4 s eager (2080 Ti, 6.6 GB) | 0.84 µs |
| 3D | 1e5 | 0.135 s (A6000) | 1.35 µs |
| 3D | 1e6 | 1.31 s (A6000) | 1.31 µs |
| 3D | 1e7 | 55 s eager (A6000) | 5.5 µs |

At 1e7 diracs in 3D, the float kernel can no longer resolve the cells (each cell is about 4.6e-3 wide in a unit box), so
the solve switches to the double kernel. On cards with slow FP64 this switch costs half of the total time.

GPU memory is about 900 bytes per dirac in 2D and 1300 in 3D at 1e6. A solve that does not fit raises a `MemoryError`
before it starts.

<!-- TODO: a plot of time vs n (1e3 .. 1e7), CPU and GPU, 2D and 3D, with the O(n log n) slope. -->

## Scaling with the dimension

Lebesgue on the unit hypercube, diracs drawn uniformly, a cold start, `rtol = 1e-6`. CPU `lmo`, 8 pinned threads,
double kernel, `step = trials`. One run per row (3 for 4D at 1e5). 6D at 1e3: 9 min (5 / 7).

| dim | total, n = 1e4 | Newton it. / diagrams | one diagram, per dirac (n = 1e4) | vertices per cell |
|---|---|---|---|---|
| 2 | 0.20 s | 5 / 6 | 0.9 µs | 6 |
| 3 | 0.40 s | 5 / 6 | 3.6 µs | ~25 |
| 4 | 5.1 s  | 5 / 6 | 80 µs | ~120 |
| 5 | 103 s  | 5 / 6 | 1.7 ms | ~670 |
| 6 | 52 min | 4 / 5 | 63 ms | ~3300 |

Newton is not affected by the dimension: the iteration count (5 iterations, 6 diagrams) and the cost of the linear
solves (under 2.5 s at 1e5) stay the same up to 6D. The whole cost is in the cells. A Laguerre cell of a uniform cloud
has a number of vertices that grows very fast with `d` (the last column, averaged over a few thousand diracs), and the
cell is cut, measured and triangulated vertex by vertex. Each added dimension multiplies the cost of a diagram by
about 20 (3D to 4D, 4D to 5D), then by about 40 from 5D to 6D.

In practice, 4D is comfortable up to a few 10⁵ diracs and 5D up to about 10⁵. 6D is limited to about 10⁴ diracs (an hour on 8 cores).
Beyond that, the cost comes from the geometry of the cells, not from the solver.

<!-- ## Comparison with other libraries

::: warning To be measured
**CGAL** (`Regular_triangulation_2/3`): the programs in `bench/cgal/` already build the same power diagrams as
`PowerDiagram.measures` (same weight convention, unit box). Their timings remain to be added here.

**Geogram** (Lévy's semi-discrete OT, `OptimalTransportMap3d`): to be added.

The previous C++ implementation of sdot (CPU, same machine, same cases) gives the following Newton totals: 2D uniform
1.08 s, 3D uniform 2.25 s, 3D planes 7.19 s. The current CPU path takes 0.9 to 1.15 times as long.
::: -->

<!-- TODO: table: n, dim, sdot CPU / sdot GPU / CGAL (triangulation only, then + cell measures) / geogram (whole OT). -->

<!-- ## Reproducing

The benchmarks run through `errand`. The complete list of options is in
[`bench/README.md`](https://github.com/sdot-team/sdot/tree/main/bench/README.md).

```bash
# one diagram, ns / dirac
errand -k bench --env lmo-jax   bench_diagram --case=uniform2d,lines_voronoi,lines_equal --kernel=float,double
# whole Newton solve, CPU then GPU
errand -k bench --env lmo-numpy bench_newton --case=uniform2d,uniform3d,planes_voronoi --threads=8
errand -k bench --env lmo-jax   bench_newton --case=uniform2d,uniform3d,planes_voronoi --kernel=mixed --jit=yes
# the dimension scan ( Lebesgue on the unit hypercube )
errand -k bench --env lmo-numpy bench_newton --case=uniform --dim=2,3,4,5,6 --n=10000 --threads=8 --reps=1
# densities (image, gaussians) with continuation
errand -k bench --env lmo-jax   bench_newton --case=image,gauss4 --sigma=0.02 --step=trials,limits
``` -->
