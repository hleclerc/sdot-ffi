# Calibration of the OLD C++ benchmark (solvers_des_familles) on lmo, 2026-10-03

Purpose: know whether the old README headline numbers still hold on this machine today, so that the old
code is a valid reference for porting its optimizations to the new python/C++ code.

## Setup

* Machine: `lmo`, Xeon W-2145 (8 cores / 16 threads), g++ 15.2, xmake `release` (`-O3 -march=native -fno-math-errno`, `double` kernel).
* Source: the **local** tree `/Users/hugo.leclerc/nsdot/solvers_des_familles` (README 8093 lines), rsynced to
  `/home/leclerc/bench-old/solvers_des_familles` (nothing under `/home/leclerc/nsdot` was touched).
  asimd headers from `sdot-ffi/include/asimd` copied to `/home/leclerc/bench-old/sdot/include/asimd`
  (xmake.lua uses `../sdot/include`). Eigen 3.4 `/usr/include/eigen3`, AMGCL and CHOLMOD from `/usr/include`.
* Binaries (kept): `/home/leclerc/bench-old/solvers_des_familles/build/linux/x86_64/release/{diagramme,newton}`.
  Build: `xmake f -m release -y && xmake -y -j4 diagramme && xmake -y -j4 newton` (one target per call; only warnings, no fix needed).
* Cases: dedup files copied from `sdot-ffi/bench/cases/` to `/home/leclerc/bench-old/cases/`
  (`lines5_n100000_s0.005_voronoi.txt` has 99 944 seeds). Referred to below as `$C`.
* Logs on lmo: `/home/leclerc/bench-old/log.txt` (every command, full per-iteration output, load before and after).
  Driver scripts: `run.sh`, `all_diag.sh`, `all_newton.sh` in the same directory.
* WARNING (found on the way): the copy of the old benchmark in lmo's `~/nsdot/solvers_des_familles` is an **older
  snapshot** (README 2582 lines, no `mg` solver, no `--solver auto`, `newton --help` says amg default). It is NOT
  the version the 8000-line README describes. I therefore built the local tree, not lmo's.
* Protocol: `--threads 8` (pinned, the default; `--no-pin` not used). `diagramme`: `--reps 3` (min inside the binary) and the whole
  command run 3 times (min of the 3 reported). `newton` has no `--reps`: each command run 3 times, the run with the smallest TOTAL is shown.
* Load: lmo is shared. Another job of the user (`tl24_LMO/N1/n1`, 4 nice-5 single-thread processes, with a
  stream of coredump handlers) ran at 4 busy cores for ~15 min and also intermittently before; runs made under
  it were 30-40 % slower (e.g. diagram 2D uniform 205 ns instead of 143, 3D uniform 2687 instead of 1875) and
  were discarded. The final driver measures the busy cores over 3 s (from /proc/stat) before every run and waits
  until < 0.6 core is busy (the 1-minute loadavg is useless for that: it includes our own previous 8-thread run).
  Every number below was taken with 0.03-0.36 busy cores before the run (all recorded in `log.txt`).
  The 3 repeats agree to 1-3 % (except one 172 vs 144 ns outlier on lines_voronoi).
  Early quiet runs of the lmo-snapshot binary (10:38, load 0.84) gave the same diagram numbers (140/138/686/1875/1733/1813/3276 ns), so the result is stable.
* The old `diagramme` has no memory option (`--memoire` of `newton` is the L-BFGS history; the 3D neighbour memory
  of the old code is `--memo`, a separate binary `memo` / `newton --memo`, not run here: README calls it "never confirmed", -19 % at best).
  Old `diagramme` therefore corresponds to the new harness "no memory" column.

## (a) Diagram (`diagramme`), ns/seed, min of 3 runs

Command: `build/linux/x86_64/release/diagramme <--2d|--3d> --load <file|uniforme -n N> --threads 8 --reps 3` (from the project dir).

| case | command (args after `diagramme`) | today (old binary) | README | today/README | new python (baseline) | new/old today |
|---|---|---|---|---|---|---|
| 2D uniform n=1e6 | `--2d --load uniforme -n 1000000` | 0.143 s, **143** | 207 (l.128) | 0.69 | 270 | 1.89 |
| 2D lines voronoi 99 944 | `--2d --load $C/lines5_n100000_s0.005_voronoi.txt` | 0.014 s, **144** | 211 (l.129) | 0.68 | 249 | 1.73 |
| 2D lines equal | `--2d --load $C/lines5_n100000_s0.005_equal.txt` | 0.068 s, **680** | 733 (l.130) | 0.93 | 713 | 1.05 |
| 3D uniform n=1e6 | `--3d --load uniforme -n 1000000` | 1.875 s, **1875** | 2768 (l.131) | 0.68 | 2268 | 1.21 |
| 3D uniform n=1e5 | `--3d --load uniforme -n 100000` | 0.173 s, **1734** | (3425 at 2e5, l.131) | - | 3295 (2248 with memory 32) | 1.90 (1.30) |
| 3D planes voronoi | `--3d --load $C/planes4_n100000_s0.02_voronoi.txt` | 0.181 s, **1808** | 2602 (l.132) | 0.69 | 3231 (2279 mem 32) | 1.79 (1.26) |
| 3D planes equal | `--3d --load $C/planes4_n100000_s0.02_equal.txt` | 0.328 s, **3281** | 4603 (l.133) | 0.71 | 6806 | 2.07 |

All sums of measures are 1.000000000 (witness OK), no cell overflow.

## (b) Newton (`newton`), 8 threads, `n = 1e5`

Command prefix: `build/linux/x86_64/release/newton <args> --threads 8 --newton-max 60`. Stage columns are the printed
`arbre | majorants | diagrammes | assemblage | resolution | limites | reste | TOTAL` line (seconds), best of 3 (TOTAL).
`resolution` = linear part (hierarchy/analysis + solve, i.e. "lin"); "maj" is the weight majorant stage; "lim" = `limites`.

| case | args | it / diag (reculs) | status, residual | arbre | maj | diag | asm | lin | lim | rest | **TOTAL** | README TOTAL (line) | today/README |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2D uniform | `--2d --load uniforme -n 100000` | 6 / 7 (0) | CONVERGE 2.69e-09 | 0.036 | 0.025 | 0.194 | 0.047 | 0.763 | 0 | 0.012 | **1.077** | 1.10 (l.6489), 1.11 (l.6569) | 0.98 |
| 2D lines voronoi, default | `--2d --load $C/lines5_..._voronoi.txt` | 14 / 39 (24) | CONVERGE 8.23e-07 | 0.018 | 0.181 | 2.762 | 0.103 | 3.013 | 0 | 0.099 | **6.177** (6.29, 6.38) | 6.44 (l.5792), 6.40 (l.6492) | 0.96 |
| 2D lines voronoi, KMT `--residu lin` | `... --residu lin` | 23 / 78 (54) | CONVERGE 2.00e-07 | 0.020 | 0.301 | 4.558 | 0.186 | 5.596 | 0 | 0.046 | **10.707** | 10.94 (l.5787) | 0.98 |
| 2D lines voronoi, README best | `... --pas essai-limites --facteur 0.9 --residu log` | 12 / 19 (0) | CONVERGE 2.69e-08 | 0.018 | 0.071 | 1.207 | 0.097 | 2.549 | 0.064 | 0.054 | **4.062** | 4.25 (l.5793) | 0.96 |
| 2D lines equal, default | `--2d --load $C/lines5_..._equal.txt` | 16 / 74 (58) | STAGNATION 2.35e-06 | 0.018 | 0.348 | 5.322 | 0.135 | 3.833 | 0 | 0.128 | **9.784** | 10.01 (l.5799) | 0.98 |
| 2D lines equal, README best | `... --pas essai-limites --facteur 0.9 --residu log` | 13 / 53 (34) | STAGNATION 2.35e-06 | 0.018 | 0.228 | 3.699 | 0.097 | 3.132 | 0.072 | 0.064 | **7.311** | 7.50 (l.5800) | 0.97 |
| 3D uniform (auto = mg) | `--3d --load uniforme -n 100000` | 5 / 6 (0) | CONVERGE 1.66e-09 | 0.035 | 0.022 | 1.312 | 0.092 | 0.772 | 0 | 0.013 | **2.246** | 1.93 (l.5810, 5 diag variant), 5.2 (l.145, early, 6/9 diag, amg era) | 1.16 |
| 3D uniform `--solver mg` | `... --solver mg` | 5 / 6 | CONVERGE 1.66e-09 | 0.034 | 0.022 | 1.305 | 0.095 | 0.775 | 0 | 0.013 | **2.243** | - | - |
| 3D uniform `--solver amg` | `... --solver amg` | 5 / 6 | CONVERGE 1.66e-09 | 0.035 | 0.029 | 1.298 | 0.096 | 1.156 | 0 | 0.013 | **2.626** | - | - |
| 3D planes voronoi, default (log+bascule, mg) | `--3d --load $C/planes4_..._voronoi.txt` | 9 / 17 (7) | CONVERGE 1.96e-09 | 0.024 | 0.072 | 5.429 | 0.185 | 1.421 | 0 | 0.054 | **7.186** | 7.50 (l.5806) | 0.96 |
| 3D planes voronoi `--residu log` | `... --residu log` | 9 / 17 (7) | CONVERGE 1.96e-09 | 0.024 | 0.076 | 5.459 | 0.207 | 1.455 | 0 | 0.063 | **7.283** | 7.50 | 0.97 |
| 3D planes voronoi, KMT `--residu lin` | `... --residu lin` | 13 / 27 (13) | CONVERGE 1.16e-10 | 0.024 | 0.121 | 7.941 | 0.309 | 1.937 | 0 | 0.031 | **10.363** | 10.48 (l.5805) | 0.99 |
| 3D planes equal, default | `--3d --load $C/planes4_..._equal.txt` | 9 / 17 (7) | CONVERGE 1.96e-09 | 0.024 | 0.074 | 5.450 | 0.191 | 1.495 | 0 | 0.056 | **7.290** | "17 diag for both planes" (l.5853) | - |

Notes:
* The default `--residu` is `log` with a switch to `lin` as soon as `max|a-nu|/nu <= 2` (`--bascule-residu`, default 2); on the 2D
  lines case that switch happens at iteration 9, so "default" = 39 diagrams, and `--residu lin` (switch -1, "never log") = the historical 78-diagram KMT.
  The `--residu log` run on 3D planes is identical to the default (the switch fires at iteration 4 in both).
* 3D uniform linear part only (hierarchy + solve, sum of the two printed lines): mg 0.24 + 0.54 = 0.78 s (173 Krylov its) vs README 0.83 s
  (l.3788); amg 0.55 + 0.51 (+0.10 shaping) = 1.16 s (115 its) vs README 1.53 s. Direct Cholesky `chol` skipped (README: > 600 s).
* Iteration/diagram counts reproduce the README exactly in every 2D and 3D planes case (78, 39, 19, 74, 53, 27, 17; residuals identical to 3 digits).
  The 3D-uniform README figures are from different eras (see Caveats); today 6 diagrams.

## (c) Linear solver, 3D uniform n=1e5 (linear part `resolution` line; whole TOTAL in (b))

| solver | linear part today | README (l.3788) | Krylov its today / README |
|---|---|---|---|
| `amg` | 1.16 s | 1.53 s | 115 / 138 |
| `mg` (default in 3D) | 0.78 s | 0.83 s | 173 / 200 |
| `chol` | not run (> 600 s in README) | > 600 s | - |

## New python harness (same day, same machine, 8 pinned threads, double kernel) vs old binary

| quantity | old binary today | new python | new/old |
|---|---|---|---|
| diagram 2D uniform 1e6 (ns/seed) | 143 | 270 | 1.89 |
| diagram lines voronoi | 144 | 249 | 1.73 |
| diagram lines equal | 680 | 713 | 1.05 |
| diagram 3D uniform 1e5 | 1734 | 3295 (2248 mem 32) | 1.90 (1.30) |
| diagram 3D uniform 1e6 | 1875 | 2268 | 1.21 |
| diagram planes voronoi | 1808 | 3231 (2279 mem 32) | 1.79 (1.26) |
| diagram planes equal | 3281 | 6806 | 2.07 |
| Newton 2D uniform total (s) / lin (s) | 1.077 / 0.763 (+ arbre, majorants 0.06) | 2.22 / 1.65 | 2.06 / 2.2 |
| Newton 2D lines voronoi total (s) | 6.18 default, 4.06 best, 10.7 KMT-lin | 11.6 (non-dedup, stagnates; to be rerun) | 1.9 vs default, 2.9 vs best (not comparable yet) |
| Newton 3D uniform total / diag / lin (s) | 2.246 / 1.31 / 0.77 (amg: 2.63 / 1.30 / 1.16) | 9.09 / 3.87 / 4.84 | 4.0 / 2.95 / 6.3 (lin vs amg: 4.2) |
| Newton 3D planes voronoi total / diag / lin (s) | 10.36 / 7.94 / 1.94 for KMT-lin (27 diag); 7.19 / 5.43 / 1.42 default (17 diag) | 28.6 (trials) / 13.1 / 14.7 | 2.76 / 1.65 / 7.6 vs KMT-lin ; 3.98 / 2.4 / 10.3 vs default |

## Analysis

1. **The old binary today is as fast as, or faster than, its README on this machine.** All Newton totals are within -4 % to +1 % of the
   README (the 3D-uniform entry compares different eras, see caveats), with identical iteration and diagram counts, and the isolated diagram
   timings are 30 % FASTER than the README (143 vs 207, 1875 vs 2768, 1808 vs 2602 ns/seed), except the degenerate 2D equal case (-7 %).
   Probable cause: compiler/CPU-governor difference (g++ 15 today); whatever it is, nothing suggests lmo is slower today. So the gap with the new
   code is NOT the machine: it is the new code. (The only way the machine matters is load: under the other job's 4 busy cores the old binary itself
   was 30-40 % slower; all comparisons above are quiet-machine.)
2. **Diagram stage, new vs old: 1.7-2.1x slower, except lines_equal (1.05x) and the cases where the new code uses the memory (1.2-1.3x).**
   The 2D uniform / lines voronoi (cheap, regular cells) are 1.7-1.9x: the new engine's per-cell overhead dominates. The degenerate lines_equal case,
   which is dominated by cell geometry (many neighbours), is on par. In 3D the new cold diagram is 1.8-2.1x slower, and with memory 32 it drops to
   1.2-1.3x, i.e. the old engine WITHOUT memory is still ~25 % faster than the new engine WITH memory. 1e6 3D (1.21x) is the closest, so the
   new code is relatively less bad when the tree/search cost is amortized (bigger n), pointing at a fixed per-cell/per-leaf overhead (candidate
   supply / pruning / 8-lane first pass) rather than memory bandwidth.
3. **Linear stage is the biggest gap: 2.2x in 2D, 4-8x in 3D.** New 3D uniform lin is 4.84 s vs 0.77 s for the old `mg` (and 1.16 s for old AMGCL):
   the new code is 4.2x slower than even AMGCL on the same problem, and 3D planes 14.7 s vs 1.4-1.9 s. The old mg (aggregation by the BSP tree, 8-seed
   blocks, nu=1 Chebyshev smoothing, recycled subspace, hierarchy reuse: README sections 17, 24) is the target to port.
4. **Newton algorithm: iteration counts.** Old default on lines voronoi: 39 diagrams (log + switch), 19 with `essai-limites --facteur 0.9`; KMT lin 78.
   On 3D planes old default needs 17 diagrams vs 27 for KMT-lin. Whatever the new "trials" strategy does (28.6 s with t_diag 13.1) should be
   compared at equal diagram counts: old KMT-lin 27 diagrams cost 7.94 s of diagrams, i.e. 0.29 s/diagram vs new 13.1/(count) (not given); the
   17-diagram default would give 7.19 s in total, 4x below the new 28.6 s.
5. Per-stage summary of what to port, by decreasing gain: (i) linear solver (old mg in 3D, AMGCL-SA+spai0 in 2D: 2-8x), (ii) Newton step strategy
   (log residual with switch, `essai-limites`: 1.7x on hard 2D, 1.5x in 3D planes, in diagram count), (iii) diagram engine constant factor (1.7-2x), with
   2D equal as the only on-par case.

## Caveats

* The 3D-uniform README reference is ambiguous: l.145 gives 5.2 s (early era, 6 it / 9 diagrams), l.5810 gives "3.05 -> 1.93 s" for the 9 -> 5 diagrams
  variant (power residual). Today's default needs 6 diagrams and 2.25 s, consistent with the latter given the diagram stage alone costs 1.3 s.
* The 3D lines "equal"/"planes" Newton README numbers (l.5806 etc.) are for the `log + bascule` line; the dedicated `--residu log` run is identical.
* Not run: `chol` (README: > 600 s), `--memo`, `--kernel float`, other binaries. No failures; the only failed attempt was the first build using lmo's older snapshot
  (superseded, see Setup), and several runs polluted by the other user job (discarded).
* `/home/leclerc/bench-old` is about 70 MB (47 MB source+objects, 24 MB cases); binaries kept.


# After the port: log residual + switch, AMG as the old default, OpenMP (2026-10-03, afternoon)

Same machine, 8 pinned threads, double kernel, `errand ... bench_newton --threads=8 --pin=yes --kernel=double`.
Timing protocol: waited for < 0.6 busy cores (/proc/stat, 5 s) before every run. Runs of the morning (iteration counts, "pre" tables) were
taken with `--no-queue`; the final table (F_*) goes through errand's exclusive queue and **after** the coordinator's change that pins the
calling thread of `cpu_thread_pool.cpp` (the iteration/diagram counts do not depend on it; the times of the "pre" tables do: compare
only within a table). Final table = min of 3, `--openmp=0` (see below), AMG tol 1e-6 for the F_* rows marked tol6.

## What was ported

* `include/sdot/sdotplan/Newton.h`: `NewtonOptions::residual` (`LIN`, `LOG`, `POWER`), `power` (p of g_p, the "exponent residual": it is
  the same 10 lines, so it is in), `switch_residual` (2); `rhs()` / `merit()` / `g_of` / `gp_of` are the old `membre_de`, `merite_de`,
  `g_de`, `gp_de` (g clamped at x >= 1e-8, centered merit, latched switch decided before the right-hand side and the merit). Default = the old
  one: LOG then LIN. `NewtonStats::it_switch`, exported as `stats[ "it_switch" ]`.
* python: `Tuning( residual = "log" | "lin" | "power", residual_power, residual_switch )` (`OtProblem.py`, `SdotPlanNd.py`).
  `residual = "lin"` is the previous behaviour.
* `Linear.cpp/.h`: `LinearOptions { tol, amg_variant }`, AMG default **smoothed aggregation + spai0** (was Ruge-Stuben+GS), tol **1e-6**
  (was 1e-10; README 17.2: 1e-4 starts to cost diagrams; counts identical to 1e-10 on the 4 cases tried). `Tuning( amg_variant = "auto" |
  "sa_spai0" | "sa_gs" | "rs_gs", linear_tol )`. AUTO now picks AMG whenever the unit is compiled with `-fopenmp`.
  The AMG sets `omp_set_num_threads` from `SDOT_NB_THREADS` (unless `OMP_NUM_THREADS` is set): 16 OpenMP threads on the 8 pinned cores
  was measured 4x SLOWER on 3D planes (12.0 s vs 2.7 s of linear algebra).
* bench: `bench_newton` options `--residual --residual-power --residual-switch --restart-factor --amg-variant --linear-tol --openmp`;
  `reference_lmo.variant_of`.
* tests: `the_log_residual_and_the_lin_residual_reach_the_same_plan` (new); `the_limits_step_..._fewer_diagrams` now pins `residual = "lin"`
  (with log the 60-seed case no longer backs off, 7 vs 9 diagrams). jax, numpy, torch, lmo-numpy: `test_SdotPlanNd` all good; numpy `test_PowerDiagram` all good.

## Finding 1: the Newton strategy, counts (reps=1, `auto` solver, pre-pin, `--no-queue`)

| case | options | new it / diag | old binary today it / diag |
|---|---|---|---|
| lines voronoi 99 944 | limits + log (best) | **12 / 19** | 12 / 19 |
| lines voronoi | trials + log, `--restart-factor=1e9` (the old KMT restarts from t = 1) | **14 / 39** | 14 / 39 (default) |
| lines voronoi | trials + log, default restart 4 | 14 / 32 | - |
| lines voronoi | trials + lin | 24 / 57 (restart 4) | 23 / 78 |
| lines equal | limits + log | **13 / 53** (stagnation 2.35e-6) | 13 / 53 |
| 3D uniform | trials + log | **5 / 6** | 5 / 6 |
| 3D uniform | power p = 0.25 | 4 / 5 | (README: 5 diag, 1.93 s) |
| 3D planes voronoi | trials + log (restart 4 or 1e9) | **9 / 17** (switch at it 4, 7 backtracks) | 9 / 17 (7) |
| 3D planes voronoi | trials + power 0.25 | 9 / 17 | (README 9 / 17) |
| 3D planes voronoi | trials + lin | 14 / 27 | 13 / 27 |

Totals with Cholesky in 2D (pre-pin): lines voronoi limits+log 4.06 s (old 4.06), lines equal 7.32 s (old 7.31), t_diag 1.64 vs 1.21 (old).

## Finding 2: the linear solver

* **Why the old 2D uniform lin is 2x faster than our Cholesky**: the old `newton` in 2D does not use Cholesky. `auto` = `mg` in 3D, **AMGCL
  (SA + spai0, tol 1e-10) in 2D** (`main_newton.cpp` l.102, `Opts::amgvar = SA_SPAI0`, `lintol = 1e-10`). It is built with `-fopenmp` (`xmake.lua`
  l.75) and never calls `omp_set_num_threads`: it runs on all 16 hardware threads. Our AMGCL was compiled WITHOUT OpenMP (loom passes no
  `-fopenmp`; its builtin backend is parallel through OpenMP only), i.e. sequential: 3.6 s for 2D uniform, 10.7 s on lines (pre-pin).
  Single-thread AMG vs OpenMP AMG (same code, `--openmp`): 2D uniform lin 3.62 -> 0.84 s; lines 10.7 -> 2.54 s; 3D uniform 4.98 -> 1.40 s;
  3D planes 8.8 -> 2.7 s (8 OMP threads; 12.0 s with 16).
* Cholesky (Eigen, sequential) vs OpenMP AMG, 8 pinned threads: 2D uniform 1.78 vs 0.84 (AMG wins), lines voronoi 2.04 vs 2.54 (Cholesky wins).
  AUTO therefore takes AMG when OpenMP is there (as the old bench did), Cholesky (2D, n <= 3e5) otherwise.
* RS_GS vs SA+spai0 (OpenMP, lines voronoi, 16 OMP threads, noisy): 2.63 s vs 2.54 s: same; SA+spai0 is the old default and parallel on both sides.

**Blocking point**: the OpenMP build is not available through the normal path. loom compiles with `LOOM_CXXFLAGS` and links `link_shared`
without `-fopenmp`. The experiment sets `LOOM_CXXFLAGS += -fopenmp` (note: loom reads `LOOM_CXXFLAGS` before `SDOT_CXXFLAGS`, and lmo's
env sets `LOOM_CXXFLAGS=-std=c++20`, so `SDOT_CXXFLAGS` is ignored there) and loads `libgomp.so.1` RTLD_GLOBAL before the first kernel
(`benchlib.set_openmp`). To ship it, loom needs: `-fopenmp` in the compile flags of the `sources` of an `FfiCode` (or a global switch) and
`-fopenmp` on the `link_shared` command (Linux g++ only; Apple clang has none, AUTO then falls back on Cholesky). Without that the
default `--linear-solver=auto` on lmo is the sequential AMG (slow) -- unless `LOOM_CXXFLAGS` carries `-fopenmp` AND libgomp is loaded.

## Final table (F_*, min of 3, post-pin, OpenMP AMG SA+spai0, tol 1e-6 where marked) vs old binary today

| case | new it / diag | t_diag | t_asm | t_lin | t_maj | total | old it / diag | old t_diag | old t_asm | old lin | old total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2D uniform (limits) | 6 / 7 | 0.34 | 0.05 | 1.11 (tol 1e-10) / **0.61** (tol6) | 0.02 | 1.72 / **1.12** | 6 / 7 | 0.194 | 0.047 | 0.763 | 1.077 |
| lines voronoi, limits+log | 12 / 19 | 1.55 | 0.10 | 2.80 (tol 1e-10) / **2.08** (tol6) | 0.09 | 4.86 / **4.16** | 12 / 19 | 1.207 | 0.097 | 2.549 | 4.062 |
| lines equal, limits+log | 13 / 53 | 4.50 | 0.11 | 3.31 | 0.23 | 8.50 | 13 / 53 | 3.699 | 0.097 | 3.132 | 7.311 |
| 3D uniform | 5 / 6 | 1.51 | 0.08 | 1.26 (1e-10) / **1.06** (tol6) | 0.02 | 3.12 / **2.90** | 5 / 6 | 1.312 | 0.092 | 0.772 (mg) / 1.156 (amg) | 2.246 / 2.626 |
| 3D planes voronoi | 9 / 17 | 5.67 | 0.12 | 2.33 (1e-10) / **1.99** (tol6) | 0.07 | 8.52 / **8.23** | 9 / 17 | 5.429 | 0.185 | 1.421 (mg) | 7.186 |
| 3D planes equal | 9 / 17 | 5.64 | 0.13 | 2.34 | 0.07 | 8.51 | 9 / 17 | 5.450 | 0.191 | 1.495 | 7.290 |

(The figures marked 1e-10 were run when the AMG default was still 1e-10; the tol6 ones with `--linear-tol=1e-6`, which is now the shipped
default. Iteration/diagram counts are identical in both.)
Before this work (previous baseline, same machine): 2D uniform total 2.22 / lin 1.65; 3D uniform 9.09 / lin 4.84; 3D planes 28.6 / lin 14.7.

## What remains

* `mg` is not ported (do not port now). It is `src/solver/Multigrille.h`, 1075 lines, OpenMP, and it needs: the seeds in TREE order (aggregation is
  `rank >> k` on the BSP / `AaBsp` order, packets of 8 in 3D, no matching); smoothed prolongation with truncation 0.2; the Galerkin triple product
  with per-thread dense accumulators; Chebyshev nu = 1 smoother (3D), recycled subspace and hierarchy reuse. In the Laplacian, `L` must be
  indexed by rank, i.e. the facets need the tree order (the new Sweep has it as `pd.ids`-like data). Old mg 3D lin: 0.77 / 1.42 s against our
  1.06-1.26 / 2.0-2.3 s with OpenMP AMG: the remaining 3D gap is about 1.4-1.6x.
* The diagram engine is now the larger gap (t_diag 1.1-1.3x in 2D, 1.4x 2D uniform, 1.04x in 3D planes): not in this task.
* OpenMP in the default build path (loom: compile + link flag), see above.

## Risks

* `residual = "log"` is the default for ALL regimes, as in the old bench; README 9.6 says log ALONE breaks the density continuation: here the switch
  fires at iteration 0 of every step after the first (it starts from a small residual), but the FIRST stage of a continuation starts from Voronoi
  with the log residual. The tests of the continuation pass; not measured on a hard density.
* AMG tol 1e-6 and SA+spai0 on clouds with incomparable edge weights (README: RS_GS wins on lines at 1e-10): counts identical on the 4 cases
  measured, not on others; `Tuning( amg_variant = "rs_gs", linear_tol = 1e-10 )` restores the previous behaviour.
* The default `restart_factor = 4` (the next trial starts at 4 t_last) gives fewer diagrams than the old restart-at-1 on lines (32 vs 39), same elsewhere.
* The OpenMP/libgomp hack only exists in the bench; local macOS builds have no OpenMP.

## Experiment log: new CELL / DIAGRAM engine on CPU (lmo, 8 pinned threads, double)

Method: one change at a time; Newton `t_diag` (bench_newton, planes_voronoi: 17 diagrams in every run) and bench_diagram ns/seed;
gate after each kept change: test_SdotPlanNd on jax (own LOOM_BUILD_DIR). lmo has other users: repeat anything within 5 %.
The coordinator's `-fopenmp` change to Compiler.py landed during the last runs; the cell kernels do not use OpenMP.

| # | change | before | after | verdict |
|---|---|---|---|---|
| 1 | `LocalN::signed_distances`: 3D first pass of `cut_impl` / `nb_outside` in asimd 8-lane blocks + popcount (partial tail); `can_cut_box` (Pruning.h) in 8-lane blocks, early exit per block | Newton planes_voronoi t_diag 9.08 s (0.534 s/diag); diagram planes_voronoi mem0 2774, planes_equal mem0 4983 | t_diag 6.51 s; 2220 / 3798 | KEPT (-28 %). Scalar fallback under `__CUDACC__` |
| 2 | `cut_impl`: neighbour matching of the new vertices branch-free (unrolled compare instead of a merge with data-dependent branches; was ~12 % of the kernel in perf) | 6.51 s | 6.00 s | KEPT (-8 %) |
| 3 | `Image::_for_each_piece`: make a tile plane only if a vertex lies beyond it (otherwise the cell is its own piece: no copy, no cuts) | 6.00 / 6.02 | 6.02 | no gain at 3D (the Image symbols in perf were the inlined measure/facets, not the splitting). Marginal in 2D (0.368 -> 0.357 s). KEPT (exact, removes a copy plus 2d first passes per cell; revert if you prefer: diff saved in the scratchpad) |
| 4 | bsp majorant margin `1e-6` -> `1e-15` on the double kernel | lines_equal 694 | 695 | REJECTED (no effect); margin is untouched |
| 5 | `diagram::for_each_seed`: CPU work-items take BLOCKS of 256 seeds dealt in turn (GPU keeps the strided loop); used by measures / moments / measures_bwd and by `sdotplan::Sweep` (measures, moments) | pure contiguous slices: lines_equal 694 -> 850 (imbalance on the clusters), lv 164; strided: 3D planes_voronoi mem32 1590 | blocks: lines_equal 615, lv 165, planes_voronoi mem32 1346 (with #2); Newton planes t_diag 6.0 -> 5.13, lines_voronoi 2.89 -> 2.12, uniform3d 1.63 -> 1.41, uniform2d 0.368 -> 0.257 | KEPT. Plain contiguous slices (the old code's choice) are 20 % worse on lines_equal; cyclic blocks keep the locality and the balance. This is the explanation of the lines_equal "noise" (strided/contiguous/pin placement of the slow slice); with pinning fixed it is stable to 1 % |
| 6 | `ProviderBsp::proximity`: branch-free distance to the box | planes_voronoi mem32 1346 | 1251 | KEPT (-7 %) |

Not done (estimates from perf, not worth the risk now): AoS tree nodes (Ptr.h + Providers ~17 % of the 3D kernel, partly inherent), caching of
`accumulate_faces_3d` between measure and facets (~4 % of the 3D kernel), per-cell fixed overheads (H8), 8-lane pruning algebra (~25 % of the
3D kernel is asimd arithmetic, mostly the pruning).

Final diagram table (min of 3 per run, ns/seed), see report: u2 143, lv 165, le 615, u3 1e5 mem0 1640, u3 1e6 mem0 1770, planes_voronoi mem0 1672,
planes_equal mem0 3153, u3 mem32 1237 (old today: 143 / 144 / 680 / 1734 / 1875 / 1808 / 3281).
Newton diagram stage (t_diag; old default/kmt numbers in (b)): planes_voronoi 5.13 s / 17 diag = 0.30 s per diagram (old 0.319), lines_voronoi 2.12 s / 32 = 66 ms (old 71),
uniform3d 1.41 s / 6 = 235 ms (old 219), uniform2d 0.257 s / 8 = 32 ms (old 28).


# The in-house multigrid `mg` ported (2026-10-03, evening)

Code: `include/sdot/sdotplan/Multigrid.h` (header comment = the algorithm and the differences with the old one), wired in `Linear.cpp`
(`Lin::MG`, `LinearSolver::order( rank_of )` called by `Solve.h` with `Sweep::rank_of` = the tree order), python `Tuning( linear_solver = "mg",
mg_pack, mg_recycle, mg_rebuild, mg_stop, mg_nu )`, bench `--linear-solver=mg --mg-pack --mg-recycle --mg-rebuild --mg-stop --mg-nu`.
`auto` = `mg` in 3D, unchanged in 2D (AMG with OpenMP, Cholesky without). Tests: `the_multigrid_reaches_the_cholesky_plan_in_2d_and_3d`,
`the_multigrid_without_a_tree_order_and_the_unknown_solver`.

Ported as is: aggregation `rank >> 3` (packets of 8, the tree order = `Sweep::rank_of`, the Laplacian stays indexed by user id: only the map
`m[ i ] = rank_of[ i ] >> 3` and the packet members `ord[ 8a + t ]` use it), smoothed prolongation (w = 0.7, truncated at 0.2, renormalized), Galerkin product
in two passes with one dense accumulator per thread, `P^t` without a transposition, Chebyshev smoother on spai0 (lmin = lmax / 10, Gershgorin lmax; `nu` = 1 in 3D, 3 in 2D),
Eigen LDLT at the bottom (<= 1000 unknowns, seed 0 struck out), outer CG in the zero-mean gauge and translation to `d[ 0 ] = 0`, recycled subspace (2 solutions, Galerkin
start), hierarchy reused for 4 solves (only the fine level is re-pointed and its relaxation coefficients recomputed). NOT ported (lost in the old README): unsmoothed
aggregation, strength filter, damped Jacobi / plain spai0 smoothers, K-cycle. Differences: the initial residual after the recycled start is the true one (the old code
assumed `|b|`); a single-level hierarchy (n <= 1000) goes through the same direct bottom (one CG iteration) instead of a Jacobi-like diagonal preconditioner; no OpenMP -> same
loops, sequential.

## t_lin, 8 pinned threads, double, n = 1e5, min of 3 (quiet runs: 0.07-0.8 busy cores before the run; lmo-numpy through the exclusive queue)

| case | new mg tol 1e-6 (default) | new mg tol 1e-10 (old protocol) | new AMG+OpenMP tol 1e-6 | old mg today (tol 1e-10) | old amg | it / diag |
|---|---|---|---|---|---|---|
| 3D uniform | **0.530** (98 CG it) | **0.780** (170) | 1.067 (68) | 0.772 (173) | 1.156 | 5 / 6, same as before |
| 3D planes Voronoi | **1.123** (188) | **1.448** (324) | 1.962 (126) | 1.421 | - | 9 / 17, same |
| 2D uniform (limits) | 0.773 (160) | - | **0.583** (131) | 0.763 (amg 1e-10) | - | 6 / 7 |
| 2D lines Voronoi (limits) | 3.274 (603) | - | **1.997** (465), Cholesky 2.12 | 2.549 | - | 12 / 19 |

Whole-solve totals with `auto` (3D): uniform 2.29 s (old mg binary 2.246; t_diag 1.42), planes 6.67 s (old 7.186). Measured later under 1 other busy core of lmo
(someone compiles there): the same runs read 0.70 / 1.33 s of t_lin, t_diag +35 %: compare only inside a run.

* mg beats the old mg at the same tolerance (planes 1.45 against 1.42 is a tie, uniform 0.78 against 0.772 a tie) and beats AMGCL+OpenMP by 2.0x / 1.75x at the shipped tol 1e-6.
  The AMG variant measured here: tol 1e-6 vs 1e-10 changes no iteration or diagram count for either solver.
* Weights agree with Cholesky to 1e-8 on the tests (`atol = 1e-8` on the weights, 1e-10 on the masses), and the runs converge to the same 1e-6 relative residual.
* Tried, mg 2D uniform 1e5 (t_lin / CG it): nu 3 (default) 0.773 / 160, nu 2 0.818 / 187, nu 1 0.762 / 262, pack 16 0.891 / 211; lines: nu 1 3.26, 2 3.36, 3 3.27. mg does NOT help in 2D at n = 1e5
  (as in the old README: -6 % at best, here -25 %), no setting reverses it: AMGCL stays `auto` in 2D.
* n = 5e5 (lmo carrying 1-2 foreign busy cores, one rep): 2D uniform limits: mg 8 it / 10 diag, t_lin 8.34 s (nu 2: 7.32), total 11.8 s; 3D uniform: mg 5.05 s (old README 4.79), 5 / 6.
  Before the fix below, AMG and Cholesky STAGNATED there (2D and 3D alike: 44 / 40 diagrams, 34 backtracks, residual 6.46e-6, whatever the AMG tolerance, 1e-8 included), mg converged.

## A defect found on the way, fixed: the right-hand side was not on the range of the Laplacian

`L d = b` is solvable only for `sum b = 0`. The solvers that strike seed 0 out (Cholesky, AMG) dump any remainder of `sum( nu - a )` as a point source on seed 0; the zero-mean gauge of
mg projects it away. At n = 5e5 that was the whole difference (Cholesky, 2D 5e5: 44 diagrams, stagnation; with `b -= mean( b )`: AMG 8 it / 10 diag, 6.8 s of t_lin, converged to 4e-10).
`Newton.h::project_on_range` now projects `b` for every solver. Counts unchanged at n = 1e5 (2D uniform 6 / 7, lines Voronoi 12 / 19 for AMG-auto and Cholesky, lines equal 13 / 53 stagnating as before,
3D uniform 5 / 6 AMG and mg, planes 9 / 17); 3D uniform 5e5 with AMG now converges in 5 / 6 (t_lin 7.15 s against mg 5.05 s).

## What remains / risks
* 2D: AMGCL is still better than mg at 1e5 (0.58 against 0.77 uniform, 2.0 against 3.3 lines); at 5e5 uniform mg 7.3-8.3 s against AMG 6.8 s (both converged since the fix): no gain, 2D `auto` unchanged.
* The mg smoothers and the pack were tuned by the old campaign (2D image density and 3D); `nu = 1` in 3D, 3 in 2D. The hard 3D cloud (planes) takes 188 CG iterations in 9 Newton iterations (324 at 1e-10).
* Without OpenMP (a compiler without it, or a link that drops it) every loop is sequential: `auto` still picks mg in 3D (AMGCL would be sequential too), not measured.
* The tree order assumes the median-cut BSP (`AaBsp`): an aligned window of 8 ranks is a subtree up to odd-size halves; `accelerator = "plain"` falls back on the identifier order (correct, weaker aggregation: tested for correctness only).
* Hierarchy reuse (4 solves) is keyed on the system size only; the graph moves by a few edges per iteration, which a Newton step with a large backtrack count may exceed: not seen on the 6 cases.
