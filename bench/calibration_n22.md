# Calibration on n22 ( RTX A6000 ): the 3D cells of the card

n22 is node22 of the cinaps cluster ( errand env `n22-jax`, through slurm ): an RTX A6000 ( Ampere GA102, sm_86, 84 SM,
48 GB, FP64 at 1/64 of FP32 ), CUDA 13.3 ( loom[cuda]'s pip `nvcc` ). The 2D story and the lmo protocol are in
`calibration_lmo_today.md`; this file is the 3D one, measured on n22 only, with the old campaign's binary rebuilt there
for same-machine numbers.

# 3D step 1: the cells ( 2026-10-05 )

`include/sdot/gpu/Cell3D.cuh` ( + `Majorant3D.cuh` for the tree's weight majorants in 3D, the tree itself being
`Bsp2D.cuh::build_tree< 3 >` ): what the 2D card path gives, in 3D -- measures, facets ( the laplacian's CSR through
`Laplacian2D.cuh`'s assembly ), moments, the adjoint -- wired in `PowerDiagram_Bsp` for a 3D diagram on a CUDA device
( a box, a constant density; the neighbour memory defaults to 0 there, the card's cells do not use it ).

## How

* THE OLD `voies` MAPPING ( `nsdot/gpu_des_familles/src/gpu/Voies3D.cuh` ): a cell on a WARP, lane `l` holding the
  vertices `l, l + 32, ...` in `S` register slots, three cut bytes and three neighbour bytes per vertex ( the simple
  polytope of `cell/Cellule3D.h` ), the cut done by ballots ( the outside masks ), a candidate per lane for the new
  vertices, the survivors renumbered by `popc`. ( `reference_lmo_gpu.py` says "V = 8 lanes": in 3D the old `voies` is the
  whole warp, doc/02-mappages.md l.70. ) Two register passes, `S = 2` ( 64 vertices ) then `S = 4` ( 128 ), and a third
  pass in global memory for anything larger ( fixed budget of slots, batches, a `KernelFailure` past `card_max_vertices` ).
* THE NEW VERTICES PAIRED BY THE OLD CUT THEY SHARE ( exactly two per crossed face ); when that is ambiguous ( a face
  crossed four times: signs at the rounding ), the cell goes to the global pass, which pairs them by WALKING the faces.
* THE TIES DECIDED ONCE ( `widen` ): a vertex is outside a plane only beyond `c eps ( |d| ( |d| + L ) + |off| )` ( `c` = 8
  in float, 32 in double ), and the new vertices are still placed on the TRUE plane ( an inside end within the band is the
  new vertex itself ). Without the band an exact grid ( eight cells at every vertex ) came out with faces in overlapping
  cycles: cells 15-20 % too large, the sum of the measures 1.0005 in double, 12 % errors in float. With the band but the
  vertices placed on the widened plane, two seeds at one place whose weights differ by less than an ulp of the offset
  ( `test_PowerDiagram::the_generic_cell_survives_almost_coincident_seeds` ) gave two identical planes, the second one
  seeing the first one's vertices at the noise: a cell 5 % too large. Now the grids are exact to 1e-15 and that test
  passes on the card.
* THE END OF A CELL in the same kernel: the planes and the vertices to shared memory ( 4.25 KB per warp ), every vertex
  RE-SOLVED in double from its three planes ( both kernels: the widening would otherwise bias the double one by
  `tol / |d|` ), then a lane per face walks its polygon ( the fan from its lowest vertex, the tetrahedra from the vertex
  mean; a face in several cycles is summed edge by edge, oriented by `n_f x n_g` ) -- volume, moments, facet areas and
  centroids; a butterfly sum, deterministic.
* THE FLOAT FIXES of the old campaign: seed's frame, two-float positions and weights ( `x = xh + xl` ), vertices re-solved
  in double, every output taken on the re-solved vertices.

## The numbers ( `bench_diagram --dim=3`, errand queue, min of 10 after 0.3 s of warm-up; kernel-only, ns / seed )

| case | n | ours float | ours double | old `voies` float, n22 | old `voies` double, n22 | old float, lmo 2080 Ti | old double, lmo |
|---|---|---|---|---|---|---|---|
| uniform 3D | 1e6 | **118.4** | **457.1** | 107.3 ( wrong cells ) | 834.7 | 229 | 1103 |
| planes / Voronoi | 1e5 | **128.4** | **451.2** | 118.3 ( wrong cells ) | 811.9 | 231 | 1066 |
| planes / equal volumes | 1e5 | **217.2** | **1189.9** | 206.0 ( wrong cells ) | 2264.7 | 405 | 2926 |

The old binary ( `nsdot/gpu_des_familles`, `mesures --3d --variante voies -n 1000000 --cases sdot-ffi/bench/cases --threads 8
--temoin-double`, built on n22 with the pip `nvcc` 13.3, `-arch=sm_86 -O3 -lineinfo`, `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`
for its CUB, `RefAmgcl.cu` left out ) flags its float runs `FAUX`: on the A6000 its float kernel has cells off by up to 2.8
( p99.99 1.3 ) against its double witness, where ours has a median of 0 and a max of 6.8e-5 against our double kernel.
Its CPU witness there: 3192 / 2968 / 4885 ns/seed float, 2987 / 3057 / 4666 double ( 8 threads ).

So in FLOAT we are 5-10 % behind the old kernel on the same card, doing more ( two-float seeds, the ties, the re-solve, the
face walks ) and right; in DOUBLE 1.8-1.9x ahead ( not profiled: the same cut, but a different register budget, walk and
leaf reading ).

THE MAIN KERNEL ( `first_pass` ): float 95 registers, 128 B of local memory ( the walk's stack, indexed by height ), 5 blocks
of 128 per SM ( 42 % occupancy ); double 128 registers, 4 blocks ( 33 % ). The second pass ( `S = 4`, 122 / 159 registers,
2 blocks ) takes 0.8 ms of 117 on the uniform cloud, 1.5 ms of 13 on planes / Voronoi, 2.0 of 22 on planes / equal; the
third pass is empty on these clouds ( a launch, 4 us ).

WHAT NEWTON AND A GRADIENT ASK FOR ( uniform 3D 1e6, `--output=...`, kernel-only ns / seed; the measures alone 117.3 / 453.1 ):
`facets` ( the cells, the COO of the upper facets and the laplacian's CSR, `_card_cells` ) 124.4 float / 464.1 double, `vjp`
( the adjoint of the measures, the pullback alone ) 120.8 / 468.3, `moments` ( measures, barycentres, costs ) 125.8 / 481.0.

How it got there ( planes / Voronoi, planes / equal, float ):

| step | Voronoi | equal |
|---|---|---|
| first version ( 97 registers, the cell in LOCAL memory, 4 blocks / SM ) | 160.9 | 295.1 |
| the cut table's update as a select ( a conditional store became `cid[ knew >> 5 ]`, a dynamic index that sent the whole cell to local memory ) | 141.1 | 242.9 |
| `first` / `count` of the faces found by a scan instead of shared tables ( 4.25 KB per warp ), `__launch_bounds__( 128, 5 )` | **127.3** | **215.9** |

## Accuracy

* Against our double kernel after the last fixes: float median 0, p99.99 1.9e-10 / 7.8e-11 / 1.4e-10, max 2.2e-5 / 1.4e-3 /
  5.6e-9 ( uniform, planes / Voronoi, planes / equal: the max is one sliver cell of the float topology ).
* Against the GENERIC double path on the same tree ( `--witness=generic`, n = 1e5 planes, before the last fixes ): float median 6.4e-15, p99.99
  3.8e-10, max 6.7e-6 ( Voronoi ), 4.4e-15 / 5.8e-10 / 2.7e-6 ( equal volumes ); double median 6.4e-15, max 1.3e-13 /
  2.2e-13. The max of the float kernel is a sliver whose topology the float decides ( its share of the cell's volume ).
  At 1e6 the generic witness does not fit next to what the card's other users hold ( a 16 GB scratch ): the uniform row is
  against our double kernel ( median 0, p99.99 6.0e-9, max 6.8e-5 ).
* The tests ( `test_CardCells`, 3D section, n = 2e4 ): float and double kernels against the generic double path, median
  2-4e-15, max 4e-14 ( Voronoi ) / 4e-13 ( weighted ); the laplacian's entries 6e-15 median, 9e-12 max; barycentres
  1.6e-14 h ( double ) / 5.5e-12 h ( float ); exact grids 1e-12 / 1e-15.

## Findings on the way

* THE GENERIC PATH IS WRONG ON SOME 3D CELLS ( the spheres of the overflow tests: 9 cells of 1440 off by up to 9x against
  the plain storage, the exact grid: 35 of 960 ); the card's cells are right there ( 1e-14 against the plain storage ). The
  3D tests take the plain storage, or the exact value, as the reference where the generic path is wrong.
* TRANSIENT CELLS: on a sphere of 600 seeds around a centre, 24 % of the cells go past 128 vertices on the way ( a seed of
  the sphere is cut by the far side of the sphere before the centre's plane reaches it ), 220 past 512: the third pass is
  there for them, and the vertex limit of the test is 1024 ( the centre's own cell has 1196 ).
* `weights set` ( the 3D majorants, `Majorant3D.cuh` ): 7 ms at 1e5 ( planes / equal ), against 13 ms for the cells -- the
  per-level launches; not optimized.
* The tree on the card in 3D: 13-40 ms at 1e6 ( the old host build: 498 ms ).

## What remained after step 1

* The 3D Newton solve on the card: done in step 2, below.
* Faster cuts: the old doc's ideas ( survivors in place, fewer `rank` / `nth` calls ) and the walk order on clustered clouds.

# 3D step 2: the Newton solve on the card ( 2026-10-05 )

`gpu/Newton2D.cuh::solve` now takes a 3D diagram: ONE solver for both dimensions, the dimension read from the diagram. What
depends on it: the cards ( `SolveCard`: `Cell2D.cuh`'s or `Cell3D.cuh`'s `Card`, which got the 2D interface -- `prepare` with
shared last-pass slots, `lend` for the double kernel's and the moments' cards ), the majorants' records
( `Majorant3D.cuh::MajorantsN`, the walk's float nodes written by the refresh itself ), the weights packed into the kernel's
seeds ( `Cell3D.cuh::pack_seed_weights` ), the box. The laplacian ( `Laplacian2D.cuh`, its rows sorted in registers up to 32
entries in 3D ), the card multigrid ( `Linear2D.cuh` ), the damping, the aggregation, the memory check are the same code.
`SdotPlanNd._build_card` takes 3D against a constant density; `step = 'limits'` stays 2D ( refused in 3D, as on the CPU ),
a 3D density is refused with a clear `NotImplementedError`.

## The numbers ( `bench_newton --dim=3`, errand queue, `rtol = 1e-6`, cold start, min of 3; seconds, the JITTED call )

| case | n | it / diag | mixed ( default ) | float | double | t_diag / t_lin ( mixed ) | CPU lmo, 8 threads ( `calibration_lmo_today.md` ) | CPU / card |
|---|---|---|---|---|---|---|---|---|
| uniform 3D | 1e5 | 5 / 6 | **0.135** | 0.135 | 0.424 | 0.075 / 0.037 | 2.29 | **17x** |
| planes / Voronoi | 1e5 | 9 / 17 | **0.473** | 0.474 | 2.079 | 0.357 / 0.072 | 6.67 | **14x** |
| planes / equal volumes | 1e5 | 9 / 17 | **0.469** | 0.470 | 2.054 | 0.358 / 0.073 | ( the same cloud ) | - |
| uniform 3D | 1e6 | 5 / 6 | **1.308** | 1.244 ( before the float moments ) | - | 0.853 / 0.253 | - | - |
| uniform 3D | 1e7 | 9 / 16 | 55.3 ( eager ) | - | - | 39.2 / 5.4 | - | - |

At 1e7 the FLOAT KERNEL STAGNATES at iteration 6 ( its cuts decided in float, cells of 4.6e-3 in a unit box ): the mixed solve
switches to the double kernel there and converges ( to 5e-12 relative ), but the double diagrams ( 6.3 s each, 630 ns / seed )
and the double moments ( 6.9 s ) take half of it; 13 float diagrams of 1.54 s ( 154 ns / seed ). The jitted call at 1e7 failed
to load XLA's own kernels ( `CUDA_ERROR_OUT_OF_MEMORY` outside the pool: not reproduced, the other card's user may hold memory on
this one ) -- not investigated further.

The iterations and diagrams are the CPU's ( 5 / 6 and 9 / 17, the same switch of the residual at it 1 / 4, the same 7
backtracks on the planes ): the card's multigrid gives directions within its tolerance of the CPU's, the same decisions.
Eager calls cost ~0.09 s more ( Python, the tree's call ). The old campaign has no 3D Newton on the GPU.

Where the time goes ( uniform 1e6, mixed, kernel-only 1.29 s ): the cells 0.85 s ( 6 diagrams, 141 ms = 141 ns / seed: the
cells of step 1, weighted ), the linear solves 0.25 s ( 124 multigrid iterations, CUDA graphs ), the moments' last walk 0.14
s, the rest ( majorants 16 ms, assembly 21 ms, reductions ) under 0.06 s. On the planes the diagrams are 21 ms each ( 210 ns /
seed: the weighted cells of the clustered cloud ), 75 % of the solve.

THE MULTIGRID IN 3D ( float levels, smoothed first level, K-cycle, Chebyshev ): the packets and the degree swept together, as
the old campaign said to ( `t_lin` / linear iterations, planes / Voronoi 1e5, float kernel ):

| | `nu = 1` | `nu = 2` |
|---|---|---|
| packets of 4 | 0.079 s / 146 | 0.105 s / 120 |
| packets of 8 | **0.073 s / 196** | 0.089 s / 155 |

and on uniform 1e6: packets of 4 0.258 s ( 87 it ), of 8 **0.246 s** ( 124 it ). So in 3D: packets of 8, `nu = 1` -- the
old campaign's CPU optimum, and the card's 2D choice stays packets of 4 ( `SdotPlanNd._CARD_MG_PACK`, `_CARD_MG_NU`, per
dimension; `Tuning( mg_pack, mg_nu )` override them ). More iterations, cheaper cycles: the bandwidth decides.

THE MOMENTS of a mixed solve in float: the double kernel's 3D walk costs four float ones ( 56 against 12 ms at 1e5 ), and
the moments were taken by it even when the float kernel finished the solve. Now a 3D mixed solve takes them with the kernel
it finished with ( `it_double = -1`: float ): uniform 1e5 0.182 -> 0.135 s, planes 0.575 -> 0.473 s. ( 2D is unchanged: its
moments stay in double. )

## Accuracy ( `test_SdotPlanNd`, 3D section: the card's plan against the CPU's, `tol = 1e-10 / n` )

* uniform 3000 seeds, varied masses: the weights 1.7e-15 apart ( relative to max |w| ), the cost 5.3e-15, the barycentres
  1.1e-14 of a cell; the measures of the card's weights through the PLAIN storage within 1e-8 / n of the targets; the float,
  mixed, CG, host Cholesky, double-level and plain-aggregation variants within 3.3e-16 of each other;
* seeds outside the cube ( the similarity start ): 2.6e-16;
* planes 6000 seeds, equal volumes ( mixed ) and varied masses ( fp64 ): 3.8e-15 / 1.8e-15, the same iterations and diagrams
  as the CPU ( 8 / 17, 8 / 13 );
* `bench_newton --case=planes_equal`: the old campaign's equal-volume weights found again to 9.7e-13 ( mixed ) / 6.6e-14 ( double );
* the aggregation ( `_degenerate_clouds( 3 )`: exact pairs, triples, pairs 1e-9 and 1e-12 apart ): the clusters of the CPU, each
  cluster's mass within the tolerance ( fp64 and mixed );
* eager == jit to the bit ( the masses traced, then the positions too: the tree in the program ).

THE FLOAT KERNEL ALONE ( `precision = 'fp32'` ) floors at ~3e-8 relative on a 2e5 uniform cloud ( it stagnates at 1e-8 / n
there ): the mixed default switches to the double kernel when that happens, as in 2D.

## Memory ( `CardMemory.card_solve_bytes( dim = 3 )` )

The 3D parts: no finish store nor edges, 32-byte seeds, `card_facet_capacity( n, 3 ) = 10 n + 1024` upper facets ( 7.8 per
seed on a uniform cloud: room for denser clouds, an overflow being an error under `jit` ), the coarse matrices at 11 entries
per fine unknown with packets of 8. Measured at 2e5 seeds ( `stats[ "scratch_bytes" ]` ): mixed 2295.4 bytes per seed taken /
2293.5 modelled, fp64 double levels 2264.0 / 2261.5, fp32 2171.8 / 2169.9, CG 2015.6 / 2015.3 ( at 2e5 the last pass's fixed
256 MB budget is 1280 of them; ~1300 bytes per seed at 1e6 ). A solve that does not fit raises the clear `MemoryError` at once, eager and while tracing.

## What remains

* The cells are 65-75 % of a 3D solve: the faster cuts above are now the lever.
* The `limits` step in 3D: the volume of a cell along `w + t d` is a cubic while its topology holds ( each vertex is the
  crossing of three planes whose offsets are affine in `t` ), but the cells would have to keep their faces ( ~90 cut
  identifiers per cell ) for the polynomial passes; on the planes it could save some of the 7 backtracks of 17 diagrams.
* Densities in 3D ( an image, gaussians ): the cells integrate a constant only.
