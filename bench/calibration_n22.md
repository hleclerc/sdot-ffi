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

## What remains ( stage 2 )

* The 3D Newton solve on the card: `Newton2D.cuh`'s structure ( the cells with `FACETS`, the assembly, the linear solve,
  the step ) on these cells; the multigrid in 3D with `nu = 1` smoothing ( old doc/06-ce-qui-reste.md l.451: 15.1 non-zeros
  per row in 3D against 6.0 in 2D ); the `limits` step needs the volume polynomial along the direction ( `EDGES` in 2D ),
  which 3D does not have ( the CPU's 3D has `trials` only ).
* Faster cuts: the old doc's ideas ( survivors in place, fewer `rank` / `nth` calls ) and the walk order on clustered clouds.
