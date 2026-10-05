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

# GPU: the baseline of the new code on the card (2026-10-03, evening)

Step 1 of the GPU work: MEASUREMENT only, nothing optimized. The question: where the new `PowerDiagram.measures` stands on
`lmo`'s RTX 2080 Ti against the old GPU campaign (`nsdot/gpu_des_familles`), with the same protocol.

## Protocol

* Machine `lmo`, RTX 2080 Ti (Turing sm_75, 68 SM, FP64 at 1/32), env `lmo-jax` (jax[cuda13], loom's pip `nvcc`, kernels
  `-O3`, no fast-math), through errand's exclusive queue (each run waited for the machine to be alone: two jobs of
  `tl24_LMO` held it before the 2D and the 3D series). Positions `float64` (`TF`, jax x64), the kernel float by `--kernel`.
* `bench/bench_diagram.py` switches to the protocol of `gpu_des_familles` (`doc/07-methode.md` l.3-5) when the loom device is a
  `CudaGpu`: two calls (compile, memory), then `pd.measures` in a loop for >= 300 ms, then the MINIMUM of 10 runs of
  * **kernel only**: CUDA events around every launch of the `measures` call, summed (`LOOM_KERNEL_TIMING=1`, new in loom:
    `CudaQueue.h::detail::CudaQueueTiming` + `loom/devices/kernel_timing.py`) -- what the old bench times;
  * **wall**: the whole call until the result is ready on the card (`block_until_ready`).
* Registers, local bytes, resident blocks per SM (`cudaFuncGetAttributes`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, at the
  launch's 128-thread block) of the MAIN kernel (the longest); occupancy = theoretical (blocks x 128 / 1024).
* Accuracy (`--kernel=float`): a second `PowerDiagram` with `kernel_dtype = FP64` on the same tree; median, p99.99, max of
  `|m_float - m_double| / m_double` per cell (the old campaign compared with the CPU double witness: same quantity).
* Old numbers: `bench/reference_lmo_gpu.py` (each with its doc line). `old` below = the old campaign's best for the case and float.
* Commands: `errand -k bench --env lmo-jax bench_diagram --case=uniform,lines_voronoi,lines_equal --dim=2 --kernel=float,double`
  and the same with `--case=uniform,planes_voronoi,planes_equal --dim=3`. Runs: `runs/bench_diagram/diagram/2026-10-03_20h01m27-lmo-jax*`
  (2D), `2026-10-03_20h04m20-lmo-jax*` (3D).

## Baseline table (min of 10, kernel only unless said)

| case | n | kernel | kernel ns/seed | wall ns/seed | regs | occupancy (blocks x 128) | local B | accuracy med / p99.99 / max | old best (variant) | new/old |
|---|---|---|---|---|---|---|---|---|---|---|
| 2D uniform | 1e6 | float | **78.1** | 86.5 | 88 | 62 % (5) | 552 | 3.7e-5 / 8.3e-4 / 2.4e-3 | 7.6 (filnrm8); 10.6 accurate (filmsk8m) | **10.3** (7.4 vs filmsk8m) |
| 2D uniform | 1e6 | double | **183.0** | 191.2 | 114 | 50 % (4) | 552 | - | 96.7 (filmsk8g) | **1.89** |
| 2D lines Voronoi | 99 944 | float | **102.1** | 166.7 | 88 | 62 % (5) | 552 | 3.5e-5 / 5.5e-3 / 1.2e-2 | 18.5 (filnrm8) | **5.52** |
| 2D lines Voronoi | 99 944 | double | **190.7** | 255.5 | 114 | 50 % (4) | 552 | - | 132 (filnrm8) | **1.45** |
| 2D lines equal | 1e5 | float | **403.7** | 467.3 | 90 | 62 % (5) | 552 | 1.5e-5 / 1.4e-3 / 1.9e-3 | 45.2 (filnrm8) | **8.93** |
| 2D lines equal | 1e5 | double | **962.7** | 1027.2 | 118 | 50 % (4) | 552 | - | 385 (filmix6) | **2.50** |
| 3D uniform | 1e6 | float | **2077** | 2216 | 96 | 62 % (5) | 952 | 3.6e-6 / 4.6e-5 / 9.3e-5 | 229 (voies) | **9.07** |
| 3D uniform | 1e6 | double | **3020** | 3168 | 128 | 50 % (4) | 1008 | - | 1103 (voies) | **2.74** |
| 3D planes Voronoi | 1e5 | float | **1991** | 2123 | 96 | 62 % (5) | 952 | 2.6e-6 / 3.3e-5 / 8.7e-5 | 231 (voies) | **8.62** |
| 3D planes Voronoi | 1e5 | double | **3374** | 3506 | 128 | 50 % (4) | 1008 | - | 1066 (voies) | **3.17** |
| 3D planes equal | 1e5 | float | **3859** | 3988 | 96 | 62 % (5) | 952 | 1.7e-6 / 1.6e-5 / 2.1e-5 | 405 (voies) | **9.53** |
| 3D planes equal | 1e5 | double | **11217** | 11359 | 128 | 50 % (4) | 1008 | - | 2926 (voies) | **3.83** |

Against the CPU: the new GPU float kernel is x1.9 (2D uniform) / x1.4 (lines V) / x1.7 (lines equal) over the old CPU witness at 8
threads in the same float, and x0.8-0.9 in 3D (SLOWER than 8 CPU threads); in double it is slower than the CPU everywhere (x0.3-0.8).
The new CPU code itself is ~143 ns/seed in 2D uniform (8 threads, double, this file above).

What the numbers say before any lever:

* **float buys almost nothing**: double / float = 2.3 (2D uniform), 1.9 (lines V), 2.4 (lines equal), 1.45 (3D uniform),
  1.7 (planes V), 2.9 (planes equal). The old kernels had 12.7 (2D: 96.7 / 7.6) and 4.8 (3D: 1103 / 229). On a card with
  FP64 at 1/32, a float kernel that is only 2x faster than its double twin is not limited by its float arithmetic.
* **the main kernel is everything**: the other launches of the call (a zero fill of the scratch, the output seeding, the error
  buffer) are 2 % of the kernel time in 2D uniform (fill: 1.6 ms of 78 ms, a 0.93 G-word scratch: 278 528 threads x 832 words),
  5.6 % on the lines (0.58 ms), 0.5 % in 3D (a 1.42 G-word scratch, 5.7 GB: 230 912 threads x 6144 words, 9.7 ms).
* **wall - kernel** is 6-8 ms per call in 2D, 13 ms for the 3D planes (1e5) and **139-148 ms for 3D uniform 1e6**: the dispatch
  and XLA's allocation of the scratch output (3.3 KB per thread in 2D, 24 KB in 3D, sized on the whole card: 0.93 G words in 2D,
  2.4 GB for the planes, 5.7 GB for 3D uniform) -- 64 ns/seed at n = 1e5 in 2D, i.e. 60 % of the lines kernel.
* **local memory** is 552 B in 2D (the provider's `SI stack[ 64 ]`, 512 B of int64, + 40) and 952 / 1008 B in 3D; no
  launch is limited by the grid (2176 blocks of 128 for 1e6 seeds, ~3.6 seeds per thread, strided in tree order).
* **accuracy**: the new float kernel is at the level of the old FAST kernels (2D uniform median 3.7e-5 against filnrm8
  3.2e-5, max 2.4e-3 against 4.8e-3), NOT of the accurate one (filmsk8m: 4.9e-14 median, 4.6e-8 max) -- although its bisector
  is already computed in double: the float error is made in the cut (the vertices interpolated in float) and in the area taken on
  float vertices, which is exactly what the old "three repairs" + "the measure on the resolved vertex" fixed.
  Lines equal and 3D are better than the old float numbers (lines equal max 1.9e-3 against 2.7).

## Gap analysis, per case (levers ranked by expected gain; to be confirmed with `ncu` before each change)

**2D uniform 1e6 (float x10.3, double x1.9).** The cell lives in the work-item's SCRATCH, i.e. in global memory (`run_memory`,
`Local2`: 8 arrays of `cap = 64` per thread, one 3.3 KB row per thread), where the old kernels kept the 8 vertices in REGISTERS
(`filnrm8` / `filmsk8`: 6 rows of 8 lanes, `NB` a state). Every cut reads and rewrites the cell, every pruning test loops over its
vertices from memory; the touched part of the 43 520 resident cells (~0.5 KB each, the arrays being 256 B apart) is ~20 MB against
5.5 MB of L2, so the cell goes back and forth to DRAM. That explains why float barely beats double. Ranking:
(1) **the cell in registers** (the `Engine2Reg` state machine has the shape; on the GPU it needs fixed-size per-thread arrays with
immediate indices, the barrel shifts of `filnrm8` or the masks of `filmsk8`), with (2) **the overflow second pass** that it
requires (a cell over 8 vertices, 1.6 % of the states, is flagged and redone by a memory kernel on the side; this also removes the
0.93 G-word scratch, its fill and most of the 8 ms of wall overhead); (3) **AoS tree nodes in the kernel type** (a node is now
read from three arrays: `node_box` in double, `node_begin` / `node_end` in int64 -- 48 B in 3 sectors instead of one 24 B float
record; `proximity` re-reads both children's boxes in double); (4) **the bisector in the kernel type with the float accuracy
fixes** (the bisector and `proximity` run in FP64 at 1/32 today, and the positions are read as 16 B doubles; the old recipe --
fixed-point or centred frame, the weight difference in double, the vertex resolved from its two planes, the measure on the
resolved vertex -- gave 4.9e-14 median for +5 %); (5) **a smaller int stack** (`SI stack[ 64 ]` in int64 = 512 B of local memory;
int32 and depth-bounded: 2 x depth <= 64 entries of 4 B, packed index + height; the old kernels had a 192 B stack). Registers
(88 -> 62 % occupancy) should fall with (3)-(5); the old `filmsk8f` ran at 63 registers and 100 %.

**2D lines Voronoi (float x5.5, double x1.45).** The same kernel, a cloud whose cells are thin and long (more cuts, more pruning
tests per cell): the smaller ratio comes from the old kernel being slower there (18.5 against 7.6), not from the new being better.
Same ranking; here (2) matters more than its share suggests because n = 1e5 makes the fixed per-call cost (scratch allocation +
fill, 6.5 ms of wall, 64 ns/seed) as large as half of the kernel.

**2D lines equal areas (float x8.9, double x2.5).** The weighted kernel (90 / 118 registers) on the degenerate cloud: 4x the
Voronoi lines in float (the old: 2.4x). The weighted pruning reads three more doubles per node (`node_wa`, `node_wb`) and loops over
the cell's vertices in memory, so (1) and (3) gain more here than on Voronoi; (4) includes the weight difference (old: in
double, the rest in float). Note also `pd.weights = w` costs 61 ms (gather + majorant refresh on the card), 1.5x the diagram, not
in the kernel time but in any Newton step.

**3D uniform 1e6 (float x9.1, double x2.7).** The cell is a polytope in a 24 KB scratch row per thread (`LocalN`, `cap = 128`),
952-1008 B of local memory, 96 / 128 registers: one thread per 3D cell has no chance of holding its cell in registers. The old
answer was **warp-per-cell (`voies`, V = 8 lanes per cell)**: the cell is spread over the lanes, each cut is a few shuffles, and
the old profile shows 27 active threads per warp in 3D (`doc/05-profils.md` l.51). That is lever (1) in 3D, and it carries the
"cell in registers" with it. Then (5) the int stack (shared by the 8 lanes: 1/8 of the local memory per thread), (3) AoS nodes in
float, (4) the bisector in the kernel type. The 5.7 GB scratch (half of the card) also caps the threads in flight (230 912
instead of the 278 528 the card could host) and goes with 139 ms of wall per call (most likely its allocation, not measured apart): it would disappear with (1).

**3D planes Voronoi (float x8.6, double x3.2).** Same kernel, same diagnosis; the planes cloud has fewer neighbours per cell than
its anisotropy suggests (231 against 229 in the old engine), and the new one is equal on both (1991 against 2077). Same ranking:
warp-per-cell first.

**3D planes equal volumes (float x9.5, double x3.8).** The weighted 3D kernel: x1.9 the Voronoi planes in float, x3.3 in
double (the old: 1.75x and 2.7x). The double weighted case is the worst ratio of the table: the majorant terms in FP64 on top of
a memory-resident polytope. Same ranking, with (4) worth more than in the Voronoi cases (the weighted pruning and the bisector
both carry the weights).

## Caveats

* Kernel-only = the sum of all the launches of the `measures` call (the main kernel is 97-99.5 % of it); the events are recorded
  on XLA's stream, so the time between them is the GPU time of the launch even when the stream was busy before.
* The occupancy is the theoretical one at the block size of the launch (128), not the achieved one (`ncu` gives that).
* The 3D runs are 1 call per rep; 3D double at 11 s per call is 2 minutes of measurement per case.
* The accuracy reference is the GPU double kernel, not the CPU witness (the old one): GPU double against CPU double differs by
  ~1e-10 (old campaign, `doc/03-chiffres.md`), far below the float errors measured here.


# GPU step 2: a dedicated 2D cell kernel for `measures` (2026-10-03 night -> 10-04)

Code: `include/sdot/gpu/Cell2D.cuh` (header comment = the design), dispatched by `PowerDiagram_Bsp._measures_on_card`
(`src/sdot/PowerDiagram_Bsp.py`; hook `PowerDiagram._measures_on_card` called first by `measures`). It takes the call only
on a CUDA device, in 2D, with the BSP tree, the unit density, a pure box domain, no neighbour memory and nothing to
differentiate (traced or `requires_grad` seeds); everything else (3D, moments, facets, hessian rows, derivatives,
distributions, other domains) keeps the generic path. `pd.use_card_cells = False` or `SDOT_CARD_CELLS=0` forces the
generic path; `SDOT_CARD_STATS=1` prints the overflow counts of each pass. loom: `CudaQueue.h` gains `launch_kernel`
(a hand-written `__global__` with its own geometry, on the call's stream, timed like loom's launches: slot kind
"custom"), `resident_grid`, `read_back`, `zero_fill`; the body is a plain `FfiCode.inline` handler with `allocator = True`.
Tests: `tests/test_CardCells.py` (5 entries: card vs generic double on uniform / clustered / off-centre box, three
weight regimes incl. empty cells, rings of 12 / 40 / 100 / 300 seeds around a centre (second, third and fourth passes),
1 seed, 7 seeds, a grid, a seed outside the domain + duplicated seeds, weights changed between calls).

## Protocol

Unchanged (`bench_diagram` on `lmo-jax` through the exclusive queue, kernel-only = sum of all the launches of the call,
min of 10). Some intermediate runs below were taken while a CPU job of the user ran next to the queue (it does not
touch the card; the GPU kernel times were stable to 1-2 % across those runs). Final runs: `runs/bench_diagram/diagram/2026-10-04_00h51m43-*`.

## Experiment log (kernel-only ns/seed; P1..P4 = the passes, ms; `fin` = the float finish kernel)

| # | change | uniform 1e6 float | lines V float | lines eq float | double (u / lV / le) | notes |
|---|---|---|---|---|---|---|
| 0 | baseline (generic path, scratch in global memory) | 78.1 | 102.1 | 403.7 | 183 / 191 / 963 | 88 regs, acc. median 3.7e-5, max 2.4e-3 |
| 1 | levers 1-5 in one dedicated kernel: cell in registers (R1 = 8, `filmsk` masks + one barrel), overflow -> second pass (R2 = 16, no read back) -> third pass in global memory; AoS nodes in the kernel float (box rounded outward), int32 stack of 48; seed frame + two-float positions/weights + double re-solve + area on re-solved vertices | 12.2 | - | - | 92.3 / - / - | P1 7.26 ms (127 regs, 304 B local, 50 %), P2 3.97 (255 regs), P3 0.84 |
| 1b | correctness: several outside runs on concurrent bisectors (seeds on a circle) -> keep the run of the farthest vertex (`main_run`); re-solved vertex trusted within 1e3 eps L / sin of the float one | = | | | | float vs double max 1.0e-3 -> 1.1e-8 (uniform 1e6) |
| 2 | `__launch_bounds__( 128, 6 )` on P1 (80 regs, spills) | 23.0 | | | | REJECTED (P1 18 ms) |
| 3 | the float re-solve in its own kernel (P1 leaves its cells in global memory, SoA) | 11.3 | | | | P1 127 -> 72 regs (88 %), P1 4.90 + fin 1.60 |
| 4 | P2 cells deferred too; third pass in LOCAL memory (cap 64), one cell per thread | 10.5 | 81.4 | 107.3 | | lines: P3 5.6 / 6.9 ms = the unrelated big cells of a warp run one after the other |
| 5 | late passes: one cell per WARP (first lane only) | 9.4 | 34.1 | 60.4 | | |
| 6 | first cut by the nearest of ranks k +- 4 (to cut the transient growth) | 9.4 | 36.2 | 64.2 | | overflow stays 10.4 %: REJECTED |
| 7 | R1 = 10 (overflow 10.5 % -> 1.0 %) | 9.2 | 42.6 | 72.7 | | P1 +1.1 ms, lines tails longer: REJECTED |
| 8 | third pass in SHARED memory (cap 512, one lane), no fourth pass on the cases | 9.4 | 29.3 | 64.1 | | |
| 9 | third pass = `WarpCell`: the 32 lanes share the cell (pruning test, signs, run, compaction by shuffles) | 9.3 | 26.0 | 50.4 | | the single-lane cell was 0.5-5 M cycles of dependent shared reads |
| 10 | `WarpCell` as the SECOND pass too (cap 64) | 15.6 | 26.7 | 53.3 | | REJECTED (uniform P2 8.5 ms); kept: a leaf read by the 32 lanes at once + shuffles (P3 0.55 -> 0.44 / 1.20 -> 0.27 ms) |
| 11 | P2 in rank order (overflow flags instead of a list) | 12.5 | 27.2 | 54.3 | | REJECTED (P2 2.3 -> 5.4 ms) |
| 12 | state after 1-11 | 9.4 | 25.7 | 51.6 | 90.9 / 133.4 / 518.2 | |
| 13 | pruning in FLOAT for both kernels (double kernel: the vertex rounded, a 1e-6 margin on the terms), nodes in float for both (32 / 48 B), descend into the nearest child without a push; plane norms carried in the finish | 9.4 | 24.9 | 51.9 | 56.9 / 82.1 / 283.1 | double P1 74 -> 45 ms (uniform) |
| 14 | the float projection fallback (near-parallel planes: project the float vertex on its outgoing plane) | = | = | = | | no measurable effect, kept (harmless) |
| 15 | `EXACT_VERTICES`: P1/P2 vertices as intersections of two planes (re-read) instead of interpolated | 10.9 | 28.2 | 57.2 | | max float/double 1.1e-8 -> 4.6e-14 (u), 2.3e-9 -> 8e-14 (lV): compile-time switch, OFF |

Overflow counts (`SDOT_CARD_STATS=1`): over 8 vertices 10.45 % (uniform) / 10.2 % (lines V) / 11.2 % (lines eq); over 16:
0.004 % / 0.13 % / 0.14 %; over the shared capacity (384 float, 256 double): 0 on the three clouds (the rings of the tests go there).

## Final table (min of 10, kernel only; old = `reference_lmo_gpu.py`)

| case | kernel | kernel ns/seed | wall ns/seed | main kernel regs / occ. | accuracy vs double kernel (median / p99.99 / max) | old best | new/old | baseline today |
|---|---|---|---|---|---|---|---|---|
| 2D uniform 1e6 | float | **9.3** | 16.9 | P1 72 / 88 % | 1.3e-14 / 7.7e-13 / 1.1e-8 | 7.6 (filnrm8); **10.6 accurate (filmsk8m: 4.9e-14 / 4.9e-12 / 4.6e-8)** | 1.22; **0.88** vs filmsk8m | 78.1 |
| 2D uniform 1e6 | double | **57.0** | 64.7 | P1 96 / 62 % | - | 96.7 (filmsk8g) | **0.59** | 183.0 |
| 2D lines Voronoi | float | **25.1** | 66.7 | P2 121 / 50 % | 1.4e-14 / 5.0e-12 / 2.3e-9 | 18.5 (filnrm8; filmsk8m 5.8e-15 / - / 1.8e-10) | 1.36 | 102.1 |
| 2D lines Voronoi | double | **80.1** | 121.6 | P1 96 / 62 % | - | 132 (filnrm8) | **0.61** | 190.7 |
| 2D lines equal | float | **52.0** | 101.8 | P1 72 / 88 % | 2.1e-14 / 1.4e-12 / **5.2e-5** (one cell) | 45.2 (filnrm8; filmsk8m 1.0e-11 / - / 2.1e-7) | 1.15 | 403.7 |
| 2D lines equal | double | **289.0** | 336.6 | P1 99 / 50 % | - | 385 (filmix6) | **0.75** | 962.7 |

Split of the float calls (ms): uniform P1 4.98, P2 2.44, P3 0.15, finish 1.58, tree + seeds + seeding 0.15; lines V
0.66 / 1.09 / 0.55 / 0.18; lines eq 1.92 / 1.91 / 1.16 / 0.17. Double: no finish kernel (area on the vertices).

* The double kernel now beats the old one everywhere (x0.59 / x0.61 / x0.75): the pruning in float (lever 4 taken past
  the float kernel) halved its first pass.
* The float kernel is between the old fast (7.6) and the old accurate (10.6) kernels on the uniform cloud, MORE accurate
  than the old accurate one (median 1.3e-14 vs 4.9e-14, p99.99 7.7e-13 vs 4.9e-12, max 1.1e-8 vs 4.6e-8; 4.6e-14 max with
  `EXACT_VERTICES` for +16 %). It costs 9.3 against 7.6 for filnrm8 mostly in the finish (1.6 ms of double arithmetic, which
  hides nothing in a kernel of its own) and in the second pass (10 % of the cells redone from scratch at 121 registers).
* The lines (n = 1e5) are TAIL-bound: the whole cloud is resident at once, so each pass costs its slowest warp, and the
  passes are chained (P1 0.66 + P2 1.09 + P3 0.55 ms). The rare hard cells (a few hundred, > 16 vertices or thousands of
  candidates) pay the walk two or three times.
* lines equal, float max 5.2e-5: ONE cell (a seed clamped at x = 1e-4, neighbours at 2e-4 with weight gaps 1000x d^2): a cut
  decided in float on planes whose offset is dominated by the weight gap (the `eps |w| / h^2` term) -- the float topology
  is wrong by a sliver along a nearly parallel edge, a first-order error that neither the re-solve nor `EXACT_VERTICES`
  removes (the old filmsk8m had 2.1e-7 there). p99.99 is 1.4e-12.
* Wall - kernel: 7.6 ms per call in 2D uniform (dispatch, output seeding, the error buffer read, and one read back of the
  third pass's count, a synchronization), 4.2-5 ms at 1e5 seeds; the 0.93 G-word scratch and its fill are gone.

CPU unchanged (`lmo-numpy`, 8 pinned threads, double, min of 3, same session): 2D uniform 1e6 141 ns/seed, lines Voronoi 164.

## What remains

* float uniform: the second pass (2.4 ms for 10 % of the cells) and the finish (1.6 ms, FP64 bound: ~32 DP ops per vertex).
  Ideas not done: continue an overflowing cell in local memory instead of restarting it; the finish in float-float.
* lines: the tails of the chained passes (send the cells that will overflow 16 straight to the warp pass; a warp per
  hard cell from the start).
* the lines-equal sliver (decide near-ties in double).
* 3D (warp per cell) is the next step, not started.

## Risks

* The dispatch condition is Python-side: a jit-traced or differentiated call goes to the generic path (correct, slow).
* `MAX_DEPTH = 27` (node index and height packed in 32 bits): ~6.7e8 seeds at 10 per leaf; deeper trees fall back.
* A cell with more than 2^18 (float) / 2^17 (double) vertices gets NaN (the fourth pass gives up after 5 growths).
* The double kernel prunes in float with a margin of 1e-6 of the terms: conservative by construction, measured equal to
  the generic double path to 1e-11-1e-9 (`test_CardCells`).
* The fourth pass and the overflow statistics read a count back (a stream synchronization per call).

# GPU step 3: what Newton needs per iteration on the card -- facets ( Hessian ), adjoint, moments ( 2026-10-04 )

Code: `include/sdot/gpu/Cell2D.cuh` ( header comment = the design ) and the new `include/sdot/gpu/Laplacian2D.cuh`;
Python `PowerDiagram_Bsp._card_variant / _card_call / _card_cells` and the module functions `card_variant_for`,
`card_nnz_capacity`; loom: `ErrorKind::failure` ( `ErrorBuffer.h` ), `KernelFailure` + `failures = { code: message }` on
`driver.call` ( `CallArg_Errors.py`, the three drivers, the traced check of `JaxDriver` ). Tests: `tests/test_CardCells.py`
( 14 entries ) and `loom/tests/test_call.py::a_failure_record_raises_with_the_message_of_the_call`.

What one walk of a cell now gives, chosen at compile time ( `Out` ): MEASURES; FACETS = the COO of the UPPER facets
( rank i < rank j, `c_ij = rho |facet| / ( 2 |p_i - p_j| )` ) that `Laplacian2D.cuh` turns into the symmetric CSR of the
Laguerre laplacian ( count / scan / fill / per-row sort / diagonal as the row sum: `L = L^T` and `L 1 = 0` to the bit, the
old campaign's `Hess2D.cuh` ); VJP = the adjoint of the measures as a GATHER over the cell's own facets
( `grad_w_i = sum_j c_ij ( g_i - g_j )`, `grad_p_i = sum_j 2 c_ij ( g_i - g_j ) ( x_ij - p_i )`, no atomic );
MOMENTS = barycentre and `rho int |x - p_i|^2` ( the cost; its position gradient is the envelope formula ). All on the
re-solved double vertices for the float kernel.

The four fixes: (1) the card path is no longer bypassed by traced / differentiated inputs: an ffi call with its own
backward ( `measures_vjp` ); (2) no read back: per-cell STATUS output, the fourth pass is a warp cell in GLOBAL memory with
a capacity `nb_spill` chosen by Python ( loom ShapeVar, 1024 vertices ), grown by loom's retry through the error buffer
( eager ) or raised ( traced: `jax.debug.callback`, ~1-2 ms per call ); (3) the walk's stack is indexed by HEIGHT ( one
pending node per height, a bit mask, the top = lowest bit ): no packing, no depth limit; rank / node index types and the
stack size are template parameters chosen by Python ( `Variant< TK, TR, TN, MAX_HEIGHT >`: int / long long, 32 / 64 );
(4) past `card_max_vertices` ( 32768 ) a cell is a `KernelFailure` with the seed's index, never a NaN.

## Protocol

As before: `bench_diagram` on `lmo-jax` through the exclusive queue, kernel only = sum of all launches of the call, min of
10. New `--output=measures|facets|vjp|moments`: `facets` = `_card_cells( facets = True )` ( cells + COO + CSR assembly:
Newton's iteration ), `vjp` = the pullback alone ( wrt the weights, or the positions without weights ). Runs
`runs/bench_diagram/diagram/2026-10-04_*` ( this session ).

## Table ( kernel-only ns/seed; in brackets: ms per call )

| case | kernel | measures ( before this step ) | measures + facets + CSR | adjoint ( pullback ) |
|---|---|---|---|---|
| 2D uniform 1e6 | float | **9.1** ( 9.3 ) | **12.4** ( 12.4 ms ) | 12.1 |
| 2D uniform 1e6 | double | **57.0** ( 57.0 ) | **60.1** ( 60.1 ms ) | 58.0 |
| lines Voronoi 1e5 | float | 24.8 ( 25.1 ) | 29.3 | 26.6 |
| lines Voronoi 1e5 | double | 81.8 ( 80.1 ) | 89.0 | 84.7 |
| lines equal 1e5 | float | 51.2 ( 52.0 ) | 55.9 | 53.5 |
| lines equal 1e5 | double | 292.7 ( 289.0 ) | 290.9 | 290.5 |

* The measures did not move ( the height stack is as fast as the packed one; float P1 64 regs / 100 %, double 96 / 62 % ).
* Facets: uniform float +3.3 ms = finish +1.2 ( the edge terms in double ) + assembly 1.8 ( fill 1.16 with its atomics,
  sort 0.41, count + scan 0.2 ); double +3.1 ms = P1 +1.2 ( 96 -> 122 regs, 62 -> 50 % ) + assembly 2.3. The first
  per-row sort was in place in global memory: 2.8 ms; in registers ( odd-even network, rows <= 16 ) 0.41 ms.
* THE OLD CAMPAIGN'S NEWTON TURN ( doc/06 l.225-239, double, 1e6: 101 ms = 96 measures + 5 assembly ): **60.1 ms** here for
  the same cells + CSR ( x0.60 ), majorants not included on either side ( see below ).
* Adjoint: a walk again ( loom's design: nothing of the forward is kept ), plus the gathered facet terms: +3 ms float,
  +1 ms double at 1e6.
* Wall ( eager, 1e6 float ): 23.5 ns/seed, 7 ms more than before -- loom's Python per call ( the custom_vjp wrapping now
  that the call has a backward: +2.3 ms, the source rendering and the analysis of the two aggregates ); under `jax.jit`
  the call is 11.7 ms wall for 9-10 ms of kernels ( weighted ), 62.6 for 57-60 in double, the error check included.
* Accuracy against the generic double path ( `test_CardCells` ): laplacian entries median 1e-14, max 2.6e-10 relative
  ( small facets, floor 1e-3 of the median entry ), same graph on uniform / lines / weighted lines / ring of 300, float and
  double kernels alike; adjoint max gap 1e-14 of the largest entry; barycentres 4e-14 h ( 6e-10 h weighted ), costs 4e-12.

CPU unchanged: `bench_newton --env lmo-numpy --case=uniform --dim=2 --threads=8`: total 1.067 s ( old 1.10 ).

## Findings on the way ( not fixed here )

* `pd.weights = w` ( `AaBsp.refresh_weight_majorants` ) costs **676 ms at 1e6 on the card** ( host copies of the slices +
  one batched loom call ): 10x a Newton turn. Step 2 must refresh the majorants in its own kernel ( e.g. in `make_nodes`,
  from the node's slice ), not through Python.
* The GENERIC GPU path gives NaN measures and facets on a few cells of the rings ( 2 of the ring of 300, 8 of the ring of
  3000 ); the card does not ( the tests exclude those rows from the comparison ).

## Risks

* The float facets come from the float topology: a sliver seen from the higher rank only is dropped ( none on the test
  clouds; the old campaign counted ~10 per million ).
* A backward cannot run again: its fourth pass is sized on `card_max_vertices` ( 8 warps x 32768 vertices, 12.6 MB in
  double ).
* `__activemask` and CUB / cooperative groups do not compile with the pinned `nvcc` ( headers mismatch ): the reservation
  of the COO and the scan are hand-written ( inline PTX `activemask` ).

# GPU step 4: SdotPlanNd ( 2026-10-04 )

`SdotPlanNd` solves on the card: on a CUDA driver, a 2D problem in a box against a CONSTANT density ( no density, or an
`Image` whose values are all equal on exactly the box ) is solved end to end on the card, in ONE ffi call whose handler (
host code ) drives the Newton loop on the call's stream -- so the same call runs eagerly and under `jax.jit`. Anything else
raises `NotImplementedError` on a card ( the CPU solves it ).

Code:

* `include/sdot/gpu/Newton2D.cuh` ( header comment = the design ): the solver -- start ( given weights / Voronoi /
  similarity, as `Solve.h` ), target rescaled to the domain mass, the log residual then lin ( `switch_residual` ), the KMT
  damping ( `t_min`, `max_backtracks`, the strict decrease ), TRIALS ( `mult_ok`, as the CPU ) and LIMITS ( see below ), the
  history, the stats, the moments at the fitted weights ( a last walk ), the diagram's weights and majorants written back.
  Two SLOTS of everything a diagram writes ( measures, COO of the facets, counters, edges ), swapped on acceptance. MIXED
  precision: the float kernel, then the double one from the first float step that stagnates ( `stats[ "it_double" ]` ).
* `include/sdot/gpu/Majorant2D.cuh`: the tree's weight majorants redone on the card ( up the tree level by level with the
  parallel-axis combination of the children's centred moments, the spread of the residuals by a root-to-leaf walk per seed
  with a segmented warp reduction and one ordered-integer atomic min / max per node and warp, then the node records ).
* `include/sdot/gpu/Linear2D.cuh`: the card's linear solvers -- Jacobi CG, and the old campaign's multigrid ( tree
  aggregation `rank >> 2`, unsmoothed Galerkin by a merge of the packet's sorted rows, Chebyshev degree 1 fused, K-cycle on
  levels 1-2, exact dense bottom <= 64 unknowns, recycling, flexible CG outside, every iteration replayed from CUDA graphs );
  and the host route ( the CSR copied to the host, a CPU solver of `Linear.cpp`, `d` copied back ).
* `include/sdot/gpu/Reduce.cuh`: deterministic reductions ( fixed grid, fixed pairing ): two runs of a solve are identical
  to the bit ( the jit test compares jitted and eager weights with `== 0` ).
* `Cell2D.cuh`: `Out::EDGES` ( the cuts of each cell in polygon order, for the step's polynomials ); `Laplacian2D.cuh`: the
  assembly with a preallocated workspace ( a solver assembles every iteration and the call's pool frees nothing before it
  returns ); `sdotplan/Report.h`: the stats / history layout shared by the CPU and card solvers ( + `IT_DOUBLE` ).
* Python: `SdotPlanNd._build_card` ( the options as ONE real tensor, the variants chosen per input: `card_variant_for`,
  float / double / mixed ), `Tuning( linear_host, mg_kcycle, mg_precision )`, `Iterative( precision = "mixed" )` ( the card's
  `auto` ), `AaBsp`: the TOP 10 LEVELS OF THE TREE ON THE HOST on a card ( numpy median splits, same rule ), the rest in the
  per-level kernel; `driver.concrete_eval()` ( loom ) so that the tree, the diagram and the normalized density are
  EVALUATED under `jax.jit` when their inputs are concrete.
* loom: `concrete_eval` on the three drivers ( jax: `ensure_compile_time_eval` ); `launch_graph` and a timed launch that
  skips its events while the stream is captured ( `CudaQueue.h` ); nvcc's link brings OpenMP when the host flags have it
  ( the `.cpp` units of `sources`, `Linear.cpp` here, are compiled with `-fopenmp`: `undefined symbol GOMP_critical_end` ).
* bench: `bench_newton` on the card ( `--jit=no|yes|both`, kernel-only time, `--linear-host`, `--mg-*`, `--kernel=mixed`,
  `--card-graphs=no --all-slots=yes`, `--save-weights` / `--compare`, `--trace` ).
* tests: `test_SdotPlanNd` ( 5 card entries: the plan of 8 variants -- step, linear solver, host solvers, kernel float,
  mixed, float levels -- against the generic double measures and against each other; the majorants bound the fitted weights;
  jit = eager to the bit; a ring of 2000 with a spill capacity too small ( loom reruns ); the refusals ), skipped without
  CUDA; `test_CardCells`: the rings compared with the PLAIN storage ( see "Findings" ).

## THE STEP ON THE CARD ( `step = limits`, the 2D default )

The old campaign's exact step ( `gpu_des_familles/src/gpu/Alpha2D.cuh`, doc/06 l.808-835 ) with the CPU's correction
rounds. Along `w + t d` the area of a cell is a polynomial of degree 2 as long as its edges do not change; its edges are
those of the accepted diagram ( `EDGES` ), so `alpha* = min_i` first root of `mass_i( t ) = eps` costs a fraction of a
diagram and no walk. The trial is `t = 1` if `alpha* >= 1`, else `factor alpha*` ( `factor = 0.9` ); if the trial diagram
still has cells under the floor, their polynomials IN THE TRIAL DIAGRAM give where they crossed it going back,
`t = factor min`, eight rounds at most; then the same KMT damping as TRIALS. It is not the CPU's `limits` ( a first trial
`beta` grown by `mult_lim`, exact local limits of the bad cells only ): the counts differ ( below ), the plan does not.
TRIALS are the CPU's exactly, and give the CPU's counts.

## Protocol

`bench_newton` on `lmo-jax` ( RTX 2080 Ti ), through the exclusive queue, min of 2-3 reps after a warm-up solve on a 2000-seed
prefix. JIT WALL: `jax.jit` of `masses -> ( weights, stats )`, called once to trace and compile, then timed ( no tracing,
no tree build: the positions are constants of the trace ). KERNELS: the kernel-only time of the call ( `LOOM_KERNEL_TIMING=1`;
the multigrid's graphs are timed as launches ). Stage times are the solver's: `t_maj`, `t_diag`, `t_asm`, `t_lim` are CUDA
events around those stages, `t_lin` the wall time of the linear solves ( they read residuals back ). CPU references:
`bench_newton` on `lmo-numpy`, 8 pinned threads, double, the same session ( AMGCL + OpenMP, `linear_solver = auto` ).
Plans: `--save-weights` on the CPU, `--compare` on the card: max | w - w_cpu | after the gauge, times n ( the weights scale
is `1 / n` ). Runs: `runs/bench_newton/newton/2026-10-04_*` ( this session ).

## Final table ( jit wall = what a jitted call costs; seconds )

| case | kernel | step | it / diag ( backtracks ) | t_maj | t_diag | t_asm | t_lin | t_lim | **jit wall** ( kernels ) | eager wall | CPU ( lmo-numpy, 8 thr ) | old GPU | plan vs CPU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform 1e6 | float | limits | 5 / 6 ( 0 ) | 0.016 | 0.069 | 0.009 | 0.652 | 0.013 | **0.777** ( 0.767 ) | 2.00 | - | 2.52 ( float, alpha*, 13 diag ) | - |
| uniform 1e6 | float | trials | 6 / 9 ( 2 ) | 0.024 | 0.105 | 0.011 | 0.753 | - | **0.913** ( 0.901 ) | 2.17 | - | 2.79 ( float, 18 diag ) | - |
| uniform 1e6 | double | limits | 5 / 6 ( 0 ) | 0.017 | 0.432 | 0.011 | 0.661 | 0.013 | **1.181** ( 1.171 ) | 2.45 | - | - | - |
| uniform 1e6 | double | trials | 6 / 9 ( 2 ) | 0.024 | 0.624 | 0.013 | 0.756 | - | **1.499** ( 1.488 ) | 2.73 | - | 4.24 ( 18 diag ) | - |
| uniform 1e6 | mixed ( auto ) | limits | 5 / 6 ( 0 ) | 0.016 | 0.070 | 0.009 | 0.656 | 0.013 | **0.843** ( 0.833 ) | 2.04 | - | - | - |
| uniform 1e5 | float | limits | 5 / 6 ( 0 ) | 0.002 | 0.011 | 0.001 | 0.116 | 0.002 | **0.119** ( 0.113 ) | 0.36 | 1.054 ( 6 / 7 ) | - | 3.2e-9 |
| uniform 1e5 | double | limits | 5 / 6 ( 0 ) | 0.002 | 0.038 | 0.001 | 0.102 | 0.001 | **0.154** ( 0.148 ) | 0.35 | 1.054 ( 6 / 7 ) | - | 3.2e-9 |
| uniform 1e5 | float | trials | 6 / 8 ( 1 ) | 0.003 | 0.012 | 0.001 | 0.116 | - | **0.137** ( 0.129 ) | 0.35 | 1.066 ( 6 / 8, 1 ) | - | 4.5e-10 |
| uniform 1e5 | double | trials | 6 / 8 ( 1 ) | 0.003 | 0.051 | 0.001 | 0.116 | - | **0.182** ( 0.175 ) | 0.39 | 1.066 ( 6 / 8, 1 ) | - | 5.1e-10 |
| lines voronoi | float | limits | 12 / 13 ( 0 ) | 0.005 | 0.061 | 0.003 | 0.437 | 0.004 | **0.517** ( 0.499 ) | 0.75 | 3.527 ( 12 / 19 ) | - | 2.4e-7 |
| lines voronoi | double | limits | 12 / 13 ( 0 ) | 0.004 | 0.311 | 0.003 | 0.438 | 0.004 | **0.793** ( 0.775 ) | 1.03 | 3.527 ( 12 / 19 ) | - | 2.4e-7 |
| lines voronoi | mixed | limits | 12 / 13 ( 0 ) | 0.005 | 0.060 | 0.003 | 0.437 | 0.004 | **0.542** ( 0.525 ) | - | 3.527 | - | - |
| lines voronoi | float | trials | 14 / 32 ( 17 ) | 0.011 | 0.158 | 0.003 | 0.508 | - | **0.686** ( 0.666 ) | 0.94 | 5.129 ( 14 / 32, 17 ) | - | 5.8e-9 |
| lines voronoi | double | trials | 14 / 32 ( 17 ) | 0.011 | 0.825 | 0.003 | 0.507 | - | **1.385** ( 1.364 ) | 1.62 | 5.129 ( 14 / 32, 17 ) | - | 5.8e-9 |
| lines equal | double | limits | 13 / 47 ( 34 ), STAGNATION 3.05e-6 | 0.016 | 1.307 | 0.003 | 0.709 | 0.004 | **2.067** ( 2.043 ) | 2.31 | 6.488 ( 13 / 53, STAGNATION 2.35e-6 ) | - | 2.1e-7 |
| lines equal | mixed ( auto ) | limits | 16 / 65 ( 49 ), STAGNATION 2.35e-6, double from it 2 | 0.022 | 1.360 | 0.004 | 0.772 | 0.005 | **2.201** ( 2.174 ) | - | 6.488 | - | - |
| lines equal | double | trials | 16 / 67 ( 51 ), STAGNATION 3.05e-6 | 0.023 | 1.843 | 0.004 | 0.858 | - | **2.764** ( 2.736 ) | 3.02 | 7.949 ( 16 / 67, 51 ) | - | 3.0e-10 |
| lines equal | float | limits / trials | 3 / 18, 2 / 38: STAGNATION at 1.5e3 | | | | | | 0.16 / 0.20 | | | | 1e4 ( wrong ) |

( lines equal: the old campaign's float alpha* converged to 7.5e-8 in 33 it / 71 diag, 3.09 s; its trials did not converge. Both
solvers here stagnate at 2.35e-6 - 3.05e-6, the floor of the double weights on this cloud -- `solvers_des_familles` § 23.11,
what the aggregation of step 7 is for. )

* **Against the CPU solve** ( jit wall ): 2D uniform 1e5 **8.9x** ( float ) / 6.8x ( double ); lines voronoi **6.8x** / 4.4x;
  lines equal 3.1x ( double ). Eager ( the tree built at each call, the Python of the call ): 3.0x / 3.4x-4.7x / 2.8x.
* **Against the old GPU campaign**, uniform 1e6: float 0.78 s against 2.52 s ( alpha* ) / 2.79 ( trials ), double 1.18
  ( limits ) / 1.50 ( trials ) against 4.24 s ( trials ). The NEWTON TURN without the linear solve ( majorants + cells +
  facets + CSR ), double 1e6: 2.8 + 72 + 2 = **77 ms** against 101 + 3.3 ms; float 1e6: **16 ms**.
* **Same counts as the CPU** for the same options with `trials`, on the three cases ( iterations, diagrams, backtracks ), and the
  same plan to 5e-10 - 6e-9 ( x n ). `limits` differs BY DESIGN ( the polynomial step over every cell, see above ): fewer
  diagrams than the CPU's `limits` ( 6 / 13 / 47 against 7 / 19 / 53 ), the same plan to the Newton tolerance ( 2.4e-7 x n on
  the lines, where both solves stop at different residuals under `rtol = 1e-6` ).
* **The linear solve is now 80-97 % of the solve** on the card ( 0.65 of 0.78 s at 1e6, 0.116 of 0.119 at 1e5 ): ~49 FCG
  iterations per solve at 1e6 and ~70 on the lines, 2.7 ms / 0.6 ms per iteration at 1e6 / 1e5.
* The weight majorants: **2.7 ms** per refresh at 1e6 ( 0.016 s for 6 ) against 676 ms through `AaBsp.refresh_weight_majorants`.
* The tree ( built once per solve, before it ): 4.5 - 5.7 s at 1e6 on the card -> **0.85 s** with the top 10 levels on the host
  ( 0.12 s at 1e5 ); the CPU builds it in 0.15 s. It is most of the eager wall at 1e6 ( 2.0 s eager vs 0.78 jitted ).
* Float vs double kernel: the same iterations, diagrams and final residual on uniform and lines voronoi ( the float kernel's
  measures are re-solved in double ); the cells cost 5-6x less. On lines equal the float kernel stagnates at once ( the float
  sliver of step 2 ) -- hence `mixed`, the card's `auto`: float, then double from the first stagnating float step ( uniform and
  lines voronoi never switch; lines equal switches at iteration 2 and ends like the double kernel ).

## The linear solver ( 2D uniform, float kernel, limits, jit wall unless said )

| # | change | 1e5 | 1e6 | notes |
|---|---|---|---|---|
| 0 | the chain complete: card multigrid as the old one ( one-block Jacobi bottom, 60 sweeps, coarsest <= 1000 ), plain launches | 0.36 s ( lin 0.34 ) | - | bottom x772 launches = 184 ms |
| 0b | the alternatives at 1e5: card CG ( Jacobi, tol 1e-6 ) | 0.53 s, 6432 it | | |
| 0c | host route: CPU multigrid / AMGCL / Cholesky on a copy of the CSR | 0.42 / 0.42 / 1.30 s | | 138 / 95 / - it; copies included |
| 1 | exact dense bottom ( <= 64 unknowns, `( A + c 1 1^T )^-1` in one block ) | 0.242 | | |
| 2 | every iteration replayed from CUDA graphs ( ~150 launches each ) | 0.167 | 1.318 ( double 1.722 ) | old GPU: 2.52 / 4.24 |
| 3 | restriction one thread per fine row + shuffles; coarse assembly as a merge of sorted rows ( no sort ) | | 0.859 ( 1.260 ) | restriction 455 -> 172 ms, assembly 196 -> 35 ms |
| 4 | degree-1 smoother fused ( pre: `x = M b / theta`; post out of place ) | | 0.807 | |
| 5 | levels in float ( outer FCG in double ) | | 0.80 = | REJECTED as default ( no gain; 1.04 with 6 ) |
| 6 | four lanes per CSR row | | 0.869 | REJECTED ( slower: the coarse levels pay 4x the threads ) |
| - | sweeps at 1e6 ( trials ): coarsest 64 / 256 / 1000, K on 2 / 3 levels, Chebyshev degree 1 / 2 | | 1.52 - 2.61 | best: 64, K on 2, degree 1 ( the defaults ); iterations 204 - 281 whatever the setting |
| - | recycling 0 / 2 | same iterations | | the successive Newton directions share little |

What remains in it: ~49 iterations per solve with unsmoothed aggregation + K-cycle ( the CPU's smoothed aggregation needs ~28,
AMGCL ~19 ) and a cycle bound by its ~150 small dependent kernels, not by bytes.

## Findings on the way

* The GENERIC GPU path ( BSP ) gets a few cells of the rings wrong: NaN before, a wrong area now ( cell 227 of the ring of 300:
  5.4e-5 against 6.0e-4 ), depending on how the tree orders its halves; the card's cells agree with the PLAIN storage to 7e-13.
  `test_CardCells` now checks the rings against the plain storage ( exact, tree-blind ). Not fixed ( generic path ).
* `AaBsp` on a card: the per-level kernel's top levels ( one work item over the whole cloud ) took seconds; built on the host now.
* `jax.jit` of a solve needs the tree, the diagram and the density's values CONCRETE at trace time ( they are host-built or
  host-read ): `driver.concrete_eval()` evaluates them when their inputs are concrete ( a traced target or traced positions
  still cannot be solved under a trace: the tree needs the positions ).
* nvcc's link did not bring OpenMP for `sources` compiled with `-fopenmp` ( `undefined symbol: GOMP_critical_end` ): fixed in loom.

## What remains

* The linear solver ( 80-97 % of the card's solve ): smoothed aggregation on the card ( the CPU's `Multigrid.h`: truncated
  `P = ( I - w D^-1 A ) P0`, Galerkin triple product -- ~2x fewer iterations ), or fewer / fused levels in the cycle.
* Densities on the card: an `Image` ( the cell clipped by the pixels, per-pixel constants: measures, facets with the density
  integrated along them, moments ), gaussians ( quadrature ), and the width continuation that needs them ( `Convolved` ).
  The solver itself does not depend on the density ( it reads masses and facet coefficients ): only `Cell2D.cuh::finish_cell`
  and the step's polynomials ( exact for a constant only: a bisection of the mass, as `Bounds.h` does, otherwise ) change.
* 3D ( warp per cell, `doc/05-profils.md` ), other domains than a box, the neighbour memory.
* The tree on the card end to end ( the top levels are on the host: 0.85 s at 1e6, most of an eager call ).

## Risks

* `step = limits` on the card is not the CPU's `limits`: same options, other counts ( said above ); `trials` is the CPU's.
* The float kernel can stagnate on degenerate clouds ( lines equal ): `auto` is `mixed` on the card, `fp32` alone is not safe there.
* A capacity overflow ( spill vertices, COO of the facets ) stops the solve, and loom runs the whole call again ( eager ) or
  raises ( traced ); the first guesses ( 1024 vertices, `3 n + 512` facets ) are far from what the campaign's clouds need.
* The handler reads back after every diagram and every FCG iteration ( stream synchronizations inside the call ): fine under
  `jit`, but the call holds the stream for the whole solve.
* Memory per solve ~1 GB at 1e6 from XLA's pool ( two cards in mixed, two slots of cells, the CSR, the levels ): ~1e7 seeds on
  the 11 GB card.
* Under `jax.jit` the tree is built at TRACE time: a function retraced per call ( new positions ) pays it each time.

# GPU step 5: linear solver ( 2026-10-04 )

The card's linear solve was 80-97 % of the card's Newton solve ( step 4 ). It is now **1.8-2.1x faster on every case**, with the same
Newton counts on the converging cases: t_lin 0.647 -> **0.312 s** at 1e6, 0.100 -> **0.056 s** at 1e5, lines Voronoi 0.431 ->
**0.220 s**, lines equal 0.700 -> **0.372 s** ( jit, errand; table below ).

Code: `include/sdot/gpu/Linear2D.cuh` ( header = the design ). In short:

* **Smoothed aggregation on the first level only**: `P = ( I - 0.7 D^-1 A ) P0` truncated at 0.3 of each row's largest entry
  and renormalized ( the CPU `Multigrid.h` recipe ), `P^t` by a transposition ( atomic placement, then each row sorted: values
  copied, not summed ), `A_c = P^t A P` by **one warp per coarse row and a hash table in shared memory that sums in FIXED
  POINT** ( 64 bits as two native 32-bit atomics: integers add in any order to the same sum, so the coarse matrix is the same at
  every run ). Deeper levels: the plain aggregation ( merge of sorted rows, or the same warp product past 12 entries per fine
  row ) with the **K-cycle moved to levels 2-3** ( the first plain levels ).
* **Float levels by default** ( `Tuning.mg_precision`: `None` = float now ): the outer flexible CG stays in double. Same
  iteration counts as double levels.
* **No double division left in the cycle**: the Chebyshev coefficients and the K-cycle's scalars are computed once by the thread
  that finishes the reduction they depend on ( a finalizer of `reduce_last` ), not by every thread of every elementwise kernel
  ( Turing's double is 1/32 of its float: `smooth_fresh` at 1e6 took 104 us for 24 MB, `kcycle_combine` 54 us for 6 MB ).
* **4 lanes per row** in the matrix-vector kernels of the cycle; reductions on a fixed grid of 272 blocks.
* Fusions: the plain prolongation inside the post-smoothing ( `e[ j >> 2 ]` read at the neighbours ), the float copies of `r`
  and `z` inside the CG's update and dot kernels, the CG scalars in the reductions' finalizers; ONE graph per FCG iteration.
* Options: `LinOptions` ( `smoothed`, `omega`, `truncate`, `nu`, `nu0`, `cheb`, `kcycle`, `kfrom`, `lanes`, ... );
  `Tuning( mg_smoothed = ... )` ( `None`: 1, `0`: the plain aggregation everywhere ), `bench_newton --mg-smoothed`;
  `SDOT_CARD_LIN_DUMP=prefix` / `bench_newton --lin-dump=prefix` writes every system a Newton solve hands to the card.
* Tests ( `test_SdotPlanNd` ): `the_card_linear_solver_solves` ( new ffi hook `SdotPlanNd._card_linear_solves` ->
  `Newton2D.cuh::linear_solves`: the card's solver on a given CSR; against a dense solve at n = 1500, against its own residual at
  n = 4e4 where the hash product runs; 5 variants ( mg float / double levels / plain / smoothed twice, cg ), tol 1e-6 and 1e-10;
  zero mean; two calls equal to the bit; a right-hand side already in the recycled subspace takes <= 2 iterations, the same
  system twice without recycling gives the same solve to the bit ); `the_card_solves_the_transport` has two more variants
  ( double levels, plain aggregation ). `test_SdotPlanNd`, `test_CardCells`: green on lmo-jax.

## Protocol

1. `bench_newton --lin-dump` wrote the systems of the four bench solves ( `uniform 1e6` float limits: 5 systems; `1e5`: 5;
   `lines voronoi`: 12; `lines equal` double: 13 ) to `lmo:/home/leclerc/lindump` ( 0.73 GB ).
2. `bench/linear/linbench.cu` replays them through `CardLinear` exactly as the Newton loop does ( one solver object, recycling,
   hierarchies ): it reproduces the bench's linear iterations to the unit ( 244 / 201 / 848 / 1401 ) and t_lin to 1 %
   ( 0.638 against 0.647 s at 1e6 ). A compile is 10 s instead of minutes: every attempt below was measured this way, then the
   kept ones through errand. `nsys --cuda-graph-trace=node` for the kernels inside the graphs ( kernel time per ( kernel, grid )
   = per level ). The card warms up 0.5 s before timing ( without it the first run of a series read 0.0711 instead of 0.0572 s ).
3. `bench/linear/cpumg.cpp`: the CPU `Multigrid.h` on the same systems, for iteration counts of variants before writing them.
4. Final numbers: `bench_newton` through errand's queue, `--jit=yes --reps=3`, min of 3.

## Where the time went before ( 1e6, nsys, kernel time per FCG iteration: 2.52 ms, 124 launches )

| fine level | level 1 ( 2.5e5 ) | level 2 ( K x2 ) | levels 3-7 ( K x4 ) | reductions' 2nd pass etc |
|---|---|---|---|---|
| 1.11 ms ( `smooth_post` 321 us, `restrict_residual` 254, `SpmvDot` 227, `smooth_fresh` 104 ... ) | 0.64 ms | 0.37 ms | 0.25 ms | 0.1 ms |

The cycle was NOT purely latency bound at 1e6: the fine and first levels were half of it, and double arithmetic ( divisions in
every thread ) inflated the elementwise kernels 2-5x. At 1e5 the small levels' launches ( ~4 us each, 4 visits per iteration
by the K-cycle ) are the floor.

## How many iterations the smoothed aggregation saves ( CPU `Multigrid.h`, V-cycle, LDLT bottom, iterations per solve )

| packets / degree | uniform 1e6 | uniform 1e5 | lines Voronoi | lines equal |
|---|---|---|---|---|
| card ( step 4: plain, K-cycle, degree 1 ) | 48.8 | 40.2 | 70.7 | 107.8 |
| SA 8 / nu 3 ( the CPU default ) | 37.6 | 27.4 | 49.8 | 67.9 |
| SA 8 / nu 1 | 58.2 | 46.0 | 98.4 | 129.6 |
| SA 4 / nu 1 | 34.8 | 30.8 | 62.3 | 97.6 |
| SA 4 / nu 2 | 25.4 ( 22.4 with a hierarchy per solve ) | 22.0 | 46.2 | 65.9 |
| SA 4 / nu 3 | 19.8 | 18.0 | 35.3 | 50.4 |
| SA 4 / nu 2, smoothed on the first 1 / 2 / 3 levels only, V-cycle | 58.0 / 43.0 / 30.0 | | | |
| SA 2 / nu 1 | 360 | | | |

( SA 4 at 1e6: operator complexity 2.5, 17.6 / 38 / 78 / 151 / 237 entries per row from level 1 down; truncation 0.1 / 0.2 / 0.3:
23.6 / 25.4 / 28.8 iterations; omega 0.5 / 0.7 / 0.9: 46 / 25.4 / 32.6; cheb 5 / 10 / 20: no change. )

## Experiment log ( replayed systems; t_lin = the sum over the systems of a solve, min of 2-3; it = iterations per solve )

| # | change | u1e6 | u1e5 | lines V. | lines eq. | kept? |
|---|---|---|---|---|---|---|
| 0 | step 4 as committed ( replay ) | 0.638 s, 48.8 it, 2.61 ms/it | 0.0991, 40.2, 0.49 | 0.4255, 70.7, 0.50 | 0.694, 107.8, 0.49 | |
| 1 | Chebyshev / K-cycle / CG scalars by finalizers ( no division per thread ), prolongation fused into the post-smoothing | 0.598, 2.45 ms/it | 0.0862 | | | yes |
| 2 | float levels ( measured slower at step 4: the divisions and the double accumulators ate it ) | 0.413, 1.69 | 0.0725 | | | **yes** |
| 3 | reductions' first pass on 1024 blocks ( vs 240 ): `SpmvDot` 227 -> 281 us; sweep 136 / 240 / 340 / 512 / 1024 / 2048 blocks | 0.400 / 0.395 / 0.394 / 0.412 / 0.413 / 0.414 | 0.0705 / 0.0701 / 0.0705 / 0.0704 / 0.0725 / 0.0728 | | | 272 ( between the best two ) |
| 4 | matrix kernels capped to 272 / 544 / 1088 blocks ( grid stride ) | 0.391 / 0.392 / 0.393 ( = ) | = | | | no |
| 5 | the whole FCG loop as ONE graph with a while node ( `cudaGraphSetConditional` from the last finalizer, one read back per solve ) | 0.400 | 0.0767 | | | **no** ( slower than a launch + read back per iteration: 0.0703 ) |
| 6 | one graph per iteration instead of two | 0.392 | 0.0703 | | | yes ( = , simpler ) |
| 7 | the V-cycled small levels and the dense bottom in ONE single-block kernel ( barriers instead of launches; the old campaign's negative result ) | tail <= 4096: 0.443 | tail 500: 0.0701, 2000: 0.0805 | 0.3416 ( vs 0.2983 ) | | **no** |
| 8 | SA on every level ( the CPU recipe, packets 4, nu 2, V-cycle ), first card build ( thread per row, list heads in local memory ) | 1.75 s: 22.4 it ( = CPU ), 3.0 ms/it, BUILD 285 ms per hierarchy | 0.234: 20.2 it, build 38 ms | 0.82: 34.3 it | 0.95: 50.2 it | no |
| 9 | the same, solve part only ( build excluded ): SA V(1,1) | 0.28 s ( 31.4 it ) vs 0.39 | 0.042 vs 0.083 | 0.180 vs 0.291 | 0.294 vs 0.481 | ( what a cheap build would give ) |
| 10 | SA on level 0 only, K-cycle on levels 1-2 ( dense level 1 visited twice ) | 0.72: 23.4 it, 3.8 ms/it | | | | no |
| 11 | SA on level 0 only, K-cycle on levels 2-3 ( `kfrom` ) | 0.430: 36.6 it, build 23 ms | 0.0615: 29.6 it | 0.258: 52.0 | 0.411: 79.8 | **the scheme kept** |
| 11b | the same with SA on 2 / 3 levels ( K on the 2 levels after ) | | 0.0719 / 0.0852 | 0.288 / 0.370 | | no |
| 11c | hierarchy reused for 2 / 4 solves ( SA level 0 ) | 0.390 / 0.394 | 0.0593 / 0.0637 | 0.266 / 0.319 ( 56.8 / 70.8 it ) | 0.445 / 0.541 | no ( the lines lose ) |
| 12 | Galerkin by a warp per coarse row, shared hash, fixed-point sums ( instead of k-way merges in local memory ) | build 77 ms ( all SA ) | | | | yes |
| 12b | a dense shared accumulator, a block per coarse row, when the coarse level has <= 4096 rows | 56 ms | 11 ms | | | yes ( deep SA levels ) |
| 12c | ownership instead of the hash ( contributions broadcast by shuffles to the lane owning `col mod 32`, double sums in registers ) | build 83 ms, then 36 ms ( sums in shared ) | | | | no |
| 12d | sort instead of the hash ( contributions in shared, bitonic by ( column, number ), runs summed in order ) | 37 ms | | | | no ( low occupancy ) |
| 12e | the hash, its ( i, j ) pairs flattened over the lanes ( was: lanes over the neighbours of one i, 20 % busy ) | 17 ms | 2.1 ms | 0.248 | 0.399 | yes |
| 12f | `P` and the products in the level's precision ( double was compute bound: ncu 84 % SM ), the plain level 1 -> 2 by the warp product ( a thread per coarse row read 4 rows of 17 uncoalesced: 2.3 ms ) | 11.8 ms; t_lin 0.368 | 0.0571 | 0.244 | 0.394 | yes |
| 12g | `P^t` by transposition + row sort ( instead of a merge of the members' neighbours per coarse row: 3.9 ms ) | 9.1 ms; 0.3506 | 0.0565 | 0.2421 | 0.3927 | yes |
| 13 | truncate 0.3 / cheb 20 ( sweep: truncate 0.1 / 0.2 / 0.25 / 0.3, omega 0.7 / 0.8, cheb 10 / 20, packets 8, K on 3 levels, nu0 = 2 ) | 0.3323: 37.2 it | 0.0568: 30.6 | 0.2326: 51.6 | 0.3840: 80.7 | yes |
| 13b | nu0 = 2 ( degree 2 on the fine level ) | 0.3835: 35.0 it | 0.0603: 28.6 | 0.2268: 44.4 | **0.3344**: 61.7 | no as default ( the lines gain, uniform loses ) |
| 14 | the float copies of `r` / `z` inside the CG's update / dots | 0.3257 | 0.0560 | 0.2292 | 0.3785 | yes |
| 15 | 2 / 4 lanes per row ( 4 lanes measured slower at step 4: the divisions again ) | 0.296 / 0.300 | 0.0548 / 0.0543 | 0.2203 / 0.2147 | 0.3526 ( 4 ) | **4 lanes** |
| 16 | final ( cleaned ): float levels | **0.299 s, 37.2 it, 1.44 ms/it ( 92 launches )**, build 6.3 ms | **0.0535, 30.6 it, 0.31 ms/it ( 80 launches )** | **0.2138, 51.6 it** | **0.3510, 80.7 it** | |
| 16b | final, double levels | 0.4127 | 0.0647 | 0.2595 | 0.4264 | ( same it ) |
| 16c | final, plain aggregation ( `smoothed = 0` ) | 0.3839, 48.2 it | 0.0666, 40.2 | 0.2764, 69.7 | 0.4494, 105.8 | |
| 16d | card CG ( Jacobi ) | 8.58 s, 4323 it | 0.358, 1268 | 2.66, 3884 | 3.79, 5143 | |

Every final variant: the true relative residuals of all systems < 1e-6, the solutions identical to the bit from one run to the next
( checksum ).

## Final table ( `bench_newton`, errand queue, jit wall, min of 3; seconds )

| case | kernel | before: it / diag, lin it, t_lin, **total** | after ( SA 1 + K, float levels ): it / diag, lin it, t_lin, **total** | after, plain aggregation ( `--mg-smoothed=0` ): t_lin, total |
|---|---|---|---|---|
| uniform 1e6 | float | 5 / 6, 244, 0.647, **0.769** | 5 / 6, 186, **0.312**, **0.437** | 0.397, 0.523 ( 241 lin it ) |
| uniform 1e6 | double | 5 / 6, 244, 0.649, **1.161** | 5 / 6, 186, **0.312**, **0.839** | 0.398, 0.931 |
| uniform 1e5 | float | 5 / 6, 201, 0.100, **0.118** | 5 / 6, 153, **0.056**, **0.074** | 0.069, 0.087 |
| lines voronoi | float | 12 / 13, 848, 0.431, **0.510** | 12 / 13, 619, **0.220**, **0.300** | 0.284, 0.364 |
| lines equal | double | 13 / 47 ( 34 bt ), 1401, 0.700, **2.043**, stagn. 3.05e-6 | **14 / 52 ( 38 bt )**, 1084, **0.372**, **1.888**, stagn. 2.35e-6 | 13 / 47, 0.462, **1.832**, stagn. 2.35e-6 |

* ms per FCG iteration ( t_lin / lin it, builds included ): 1e6 2.65 -> 1.68, 1e5 0.50 -> 0.37, lines 0.51 -> 0.36 / 0.50 -> 0.34.
  Kernel time per iteration ( nsys ): 1e6 2.52 -> 1.44 ms, launches per iteration 124 -> 92 ( 80 at 1e5 ).
* The Newton counts and final residuals are unchanged on the three converging cases. Lines equal STAGNATES at the floor of the
  double weights ( step 4 ); any change of the directions at the 1e-6 of the linear tolerance moves its backtracking sequence: the
  plain variant ends at 13 / 47 like before ( 2.35e-6, the CPU's floor, instead of 3.05e-6 ), the smoothed one at 14 / 52 ( 2.35e-6
  ). Per linear solve the smoothed aggregation is 1.24x faster there too, but the total is 3 % SLOWER than the plain variant
  ( 1.888 against 1.832 s ) because of the 5 extra diagrams. Not a property of the solver: of a solve that stagnates.
* The linear solve is now 71 % of the 1e6 float solve ( was 84 % ), 76 % at 1e5, 73 % on lines Voronoi.

## The smoothed aggregation against the plain one, and the old campaign's negative result

The old GPU campaign ( `gpu_des_familles/doc/06-ce-qui-reste.md`, "La prolongation lissée avec `cusparseSpGEMM`: écrite, mesurée,
et elle PERD", `src/gpu/Lisse2D.cuh`, `AMG_LISSE=1` ) built `P = P0 - w D^-1 A P0`, UNTRUNCATED, on packets of 4, every level, with
three `cusparseSpGEMM`, a column sort and a transpose per level. At 2e5 ( K on 2 levels, nu 2 ): plain 65 it, hierarchy 4 ms,
solve 206 ms; smoothed V-cycle 64 it, hierarchy **289 ms**, solve 234 ms; smoothed + K 35 it, 289 + 442 ms ( 3.65 ms per
iteration against 3.17: denser coarse levels ). At 1e6 the smoothed V-cycle did not converge ( 20000 it ) and smoothed + K took
278 it / 6.9 s against 75 / 0.85 s. Its conclusion: the K-cycle on the plain aggregation reaches the smoothed V-cycle's
convergence for a 70x cheaper setup ( AGMG's argument ).

What differs here, and why it wins this time: ( 1 ) the CPU's TRUNCATION and renormalization ( operator complexity 2.5 instead of
the untruncated blow-up: 278 entries per row at the bottom ), ( 2 ) the smoothed transfer on the FIRST level only, the plain one
and the K-cycle below ( smoothing every level on the card costs 2.9-3.8 ms per iteration: the dense coarse rows ), ( 3 ) a
hand-written Galerkin product, one warp per coarse row with fixed-point hashing: **6-9 ms per hierarchy at 1e6, ~1.4 ms at 1e5**
( 285 ms for the first version here, 289 ms with SpGEMM at 2e5 in the old campaign ). Measured both ways on the same cases
( tables above ): the smoothed first level wins on t_lin everywhere ( 1.21-1.29x ) and on the total of every converging case.

## Findings on the way

* `RED_GRID`-like reductions: more resident threads in a fused matrix-vector reduction made it SLOWER ( 227 -> 281 us at 1e6 ).
* Two step-4 rejections were the double divisions in disguise: float levels and 4 lanes per row both win once the per-thread
  divisions are gone.
* A CUDA graph's while node ( device-driven loop, one read back per solve ) is slower on this driver / Turing than a graph launch
  and a read back per iteration ( 0.0767 vs 0.0703 s at 1e5 ): the per-iteration synchronization is not where the time goes.
* A name `d` for an output of an ffi call breaks loom's generated aggregates ( `template<int d>` ): the test hook uses `solution`.

## What remains

* The fine level is ~55 % of an iteration at 1e6: the outer CG's double SpMV ( 221 us, ~430 GB/s ) and the double update / dots
  are at bandwidth; what is left is fusing the direction into the SpMV ( a second buffer of `p`, two graphs ), int32 row pointers
  in the float copy ( 4 of ~52 MB per pass ), maybe 5 %.
* Degree 2 on the fine level ( `nu0 = 2` ) is 13 % faster on lines equal and 15 % slower on uniform 1e6: a per-input choice that
  Python cannot make from `n` alone; not exposed in `Tuning` yet ( `LinOptions::nu0` ).
* The Galerkin count pass ( 1.5 ms of the ~6-9 at 1e6 ) could go with a fixed-width first write; the dense rows of deeper smoothed
  levels make `smoothed >= 2` cost more than it saves on the card.
* 3D ( 15-27 entries per row: the warp product and 4 lanes should carry over; packets of 8 ).

## Risks

* The Galerkin hash table holds at most 512 columns per coarse row ( 64 to 512, retried larger when full ): past it the build
  fails and the Newton reports a linear failure ( not seen: the first smoothed level has ~18 per row on these clouds; deeper
  smoothed levels use the dense block path up to 4096 coarse rows ).
* Fixed point: the scale is `2^61` over a bound of the row's sum of magnitudes; entries smaller than ~1e-18 of the row's largest
  are lost ( harmless for a preconditioner; the fine operator of the outer CG is the exact double laplacian ).
* Float levels: a preconditioner in float on a system that needs 1e-6; same counts as double on the four cases and the tests.
* The stagnating case ( lines equal ) moves its counts with any change of the solver ( above ).
* `lmo:/home/leclerc/lindump` ( 0.73 GB of dumped systems ) and `lmo:/home/leclerc/linbench` are left for further work.

# GPU step 6: overflow passes ( 2026-10-04 )

The fourth pass of the 2D card cells ( `gpu/Cell2D.cuh`, one warp per cell, vertices in global memory ) no longer has a
CAPACITY. Before: a loom ShapeVar `nb_spill` ( vertices per cell, 1024 by default ), a cell that did not fit wrote a status
and asked for more through the error buffer, and loom RAN THE WHOLE CALL AGAIN ( eager; raised under jit ) -- for the card's
Newton, the whole solve. Now:

* ONE FIXED BUDGET per call ( `Cell2D.cuh::Overflow` ): `warps` slots of `cap = min( card_max_vertices, n + 4 )` vertices
  ( a cell of `n` seeds in a box has at most `n + 3` ), taken once from the call's pool. `warps` is Python's
  ( `PowerDiagram_Bsp.card_overflow_warps_for`: 64, fewer if the slots would pass `card_overflow_bytes` = 256 MB, at least 4 ):
  58.7 MB for the float / int variant at the default limit of 32768, 117 MB for double / long long, less on small `n`.
* The fourth pass is a PERSISTENT grid of exactly the slots, striding over the list the third pass filled on the card: the
  cells go through in batches of `warps`, a warp taking its next cell into the slot it freed. No estimate, no read back, no
  second run, the same under jit.
* The only loom interaction left is a FAILURE: a cell past the limit writes zeros and records `ErrorKind::failure` with the
  seed's USER index ( also in Newton, whose outputs are in ranks ): `KernelFailure` "more than N vertices ( seed k )", eager
  and traced; Newton stops at once with `status = failure` ( `S_FAILURE`, was `S_CAPACITY` ).
* The forward, the backward ( `measures_vjp`, which had 8 slots of `max_vertices` ) and the card Newton use the same scheme;
  the Newton takes ONE budget for its three cards ( float, double, moments ), sized on the widest.

Removed: `nb_spill` ( C++, `_CardWork`, `_CardSolveWork` ), `card_spill_capacity`, `Card::report_spill`, `Counters::spill_need`,
`SPILL_WARPS`, the per-cell `status` vector ( `Status` enum, `Problem::status`, `_card_status`, `SdotPlanNd.card_status`,
`scatter_status` ): with no capacity left on the cells it only repeated what the error buffer raises ( the facets' COO overflow
is told to loom by its own ShapeVar `nb_nnz` / `nb_facets` ), and dropping it saves a zero fill of `n` ints per diagram.

Tests ( lmo-jax, green: `test_CardCells`, `test_SdotPlanNd`; local jax CPU `test_SdotPlanNd` green ):
`the_card_fourth_pass_works_in_batches` ( 24 rings of 400 through 4 slots, six batches: the generic plain cells, and the same
bits as 64 slots for the measures and the adjoint, both kernels ), `the_card_raises_past_the_vertex_limit` ( eager, jit, jit of
a grad ), `the_card_solve_on_rings_in_batches_and_past_the_limit` ( 6 rings of 600 through 4 slots: converges fp64 / fp32,
jit = eager to the bit; limit 256: `KernelFailure` naming a centre, eager and jit ).

Timings ( errand bench, lmo, before -> after ):

| bench | before | after |
|---|---|---|
| bench_diagram uniform 1e6 float, kernel ns/seed | 9.1 | 9.1 |
| ... vjp float | 12.3 | 12.3 |
| ... double | 56.4 | 57.0 |
| ... vjp double | 57.5 | 58.1 |
| bench_newton uniform 1e6 float limits, jit wall ( t_diag ) | 0.435 s ( 0.070 ) | 0.438 s ( 0.071 ) |
| bench_newton lines_voronoi float limits, jit wall ( t_diag ) | 0.299 s ( 0.060 ) | 0.301 s ( 0.060 ) |

Within noise ( same counts: 5 / 6 and 12 / 13 iterations / diagrams ). On the campaign's clouds no cell reaches the fourth pass:
it is one launch that reads a zero count, as before.

Risks: the budget is taken from XLA's pool at every call ( 59-117 MB at the default limit: a pool refusal is reported by loom as
an allocation failure, not a capacity ); a cloud with thousands of 400+-gons runs them 64 at a time ( slow, still correct ); a
`card_max_vertices` below the third pass's 384 / 256 lets cells up to that size through ( as before ). The traced failure leaves
JAX's ordered-effect token poisoned, so the process prints an ignored `JaxRuntimeError` at exit ( as the traced capacity error did ).

# GPU step 7: tree on the card ( 2026-10-04 )

The BSP tree of a 2D cloud on a CUDA device is now built ENTIRELY ON THE CARD, in ONE ffi call ( `include/sdot/gpu/Bsp2D.cuh`,
`AaBsp._init_on_card` ), nothing read back: every size is a function of `n` and the leaf size. It runs under `jax.jit` on
traced positions, so `SdotPlanNd` builds the tree of a card solve OUTSIDE its `concrete_eval`: under a jit the tree is part of
the jitted program ( built at every call, from traced or constant positions ), no longer a constant computed while tracing.
`PowerDiagram_Bsp` takes the same build on a card ( 2D ). The CPU path is unchanged; `SDOT_CARD_TREE=0` sends every tree back
to the host-driven build ( `bench_newton --card-tree=no`: the "before" column below, same session ).

## How ( `Bsp2D.cuh` )

* The seeds SORTED ONCE PER AXIS by ( coordinate, index ): a stable LSD radix sort of the ordered-integer image of the double
  ( 8 passes of 8 bits, both axes in the same launches; CUB's headers do not match the pinned `nvcc`, see `Laplacian2D.cuh` ).
  The scatter stages each tile in shared memory sorted by digit and writes it out in digit runs ( coalesced ).
* PER LEVEL ( the classic presorted k-d build, O( n ) per level ): the NODES ( a thread each: the box read off the ends of the
  slice in each sorted list, the first longest axis, the median cut or none, the record at its preorder place, the
  children's slices ); the SIDE of each seed ( a bit per seed, `atomicOr`: the bitset stays in L2 -- a byte per seed went to
  DRAM at 1e7, 3x slower per level ); the OTHER list stably partitioned within each slice ( a ballot scan of the side bits,
  then a scatter ), the cut's own list copied.
* The ORDER inside a leaf: along its PARENT's cut axis ( a left leaf ends next to the cut, its sibling starts there ). The
  multigrid aggregates packets of 4 consecutive ranks; on the SAME Voronoi laplacian re-ranked ( CG iterations, 4 right-hand
  sides, tol 1e-6 ): 1e5 uniform host 33 / own axis 37.5 / parent axis 35.5 / random 33; 1e6 40.5 / 41.3 / 42 / 42; lines 42.3
  / 42.8 / 39.3 / 43.3. A full k-d order inside the leaves ( three more order-only levels ) was WORSE in the solves ( 1e6:
  205 linear iterations against 185-189 ): rejected.
* The weights' majorants of a tree built with weights: `Majorant2D.cuh` ( refactored to run from raw views, `refresh_from` ),
  one more call ( `bsp_refresh_majorants_card`, also `AaBsp.refresh_weight_majorants` on a card: it was the per-node kernel,
  676 ms at 1e6 ). A cold start's zero majorants are zeros ( no call ).
* The same tree as the host's ( slices, boxes, cuts, the seeds of each leaf ) when the coordinates along each cut are
  distinct; on TIES at a median the card sends the lowest indices left ( the host's quickselect: an arbitrary half ). Both
  are median cuts; the tests check the rule on tied clouds. Deterministic: integer scans, a stable sort, order-independent
  atomics -- jit == eager to the bit.
* In eager, the call's TENSORS are given to the aggregates ( `Tensor.set( tensor )` adopts the storage ) and not their buffers:
  `driver.array` turns a concrete device array into numpy, a device -> host copy per field ( 30 of the 50 ms of an eager
  tree at 1e6 ). Same for what `PowerDiagram_Bsp._solver_weights_after` takes back after a solve.

## The tree alone ( uniform 2D unless said; `bench_newton` "tree alone", min of 3; kernels = `LOOM_KERNEL_TIMING` )

| n | before: eager wall ( kernels ) | after: eager wall ( kernels ) | after: jitted wall |
|---|---|---|---|
| uniform 1e5 | 115.8 ms ( 1.64 ) | 4.8 ms ( **0.92** ) | 1.5 ms |
| lines voronoi 1e5 | 110.2 ms ( 1.61 ) | 4.3 ms ( 0.92 ) | - |
| uniform 1e6 | 839.9 ms ( 36.0, the top 10 levels in numpy ) | 11.9 ms ( **5.03** ) | ~5.5 ms |
| uniform 1e7 | 9085 ms ( 447 ) | 91 ms ( **42.9** ) | - |

At 1e6 ( kernels ): sort 1.6 ms ( scatter 1.16, count 0.40 ), per level ~0.18 ms ( partition 0.053, scan 0.037, nodes 0.03,
sides 0.032 ), order 0.3 ms; 137 launches. The eager wall is loom's per-call Python ( ~3.5 ms ) plus the upload of numpy
positions. The old campaign's `Bsp2D.cuh` ( a radix sort of the whole cloud per level ): 44 / 474 ms of kernels at 1e6 / 1e7.
Steps on the way at 1e6 ( kernels ): first version 7.8 ms; partition only the other list + scan fused 7.2; staged radix
scatter + ballot scan 5.8; half-size radix tiles ( occupancy ) 5.6; side bitset 5.06 ( 1e7: 160 -> 43 ms ).

## The solves ( `bench_newton --step=limits --jit=both`, errand queue, min of 3; seconds )

Before: the host-driven tree, built before the call ( eager ) or while TRACING ( jit: not in the jitted call ). After: the
card's tree, built in the eager call sequence and INSIDE the jitted program.

| case | kernel | before: eager / **jit** | after: eager / **jit** | it / diag, linear iterations ( before -> after ) |
|---|---|---|---|---|
| uniform 1e6 | float | 1.709 / **0.435** | **0.567** / **0.440** | 5 / 6 both; 186 -> 189 |
| uniform 1e5 | float | 0.292 / **0.074** | **0.156** / **0.077** | 5 / 6 both; 153 -> 161 |
| lines voronoi | float | 0.530 / **0.300** | **0.404** / **0.305** | 12 / 13 both; 619 -> 639 |
| lines equal | double | 2.122 / **1.901** | **1.817** / **1.709** | 14 / 52 -> 15 / 49 ( stagnation at 2.35e-6 both ); 1084 -> 881 |
| uniform 1e7 | float | does not fit | does not fit | ( see below ) |

* Eager: -1.14 s at 1e6 ( 3.0x ), -0.13 s at 1e5 and on the lines. Jit: the tree is now INSIDE the measured call and costs
  what its kernels cost ( +1 ms at 1e5, +5 ms at 1e6 ); the rest of the difference is the linear iterations, which follow the
  order inside the leaves ( +2 % at 1e6, +5 % at 1e5, +3 % on the lines; the stagnating lines equal moves its counts as any
  change does, and gains here ).
* A retrace ( new positions, `jax.jit` over the positions ) no longer pays a host build: the tree is a traced computation.
* Same solves: the same Newton iterations and diagrams on the three converging cases, the final residuals equal to two digits
  ( 4.02e-7 / 2.06e-7 / 4.73e-7 against 4.02e-7 / 2.05e-7 / 4.74e-7 ).

## Tests ( lmo-jax: `test_SdotPlanNd`, `test_CardCells`, `test_PowerDiagram`; local jax CPU: `test_SdotPlanNd`, `test_PowerDiagram` )

* `the_card_tree_is_the_host_tree`: 17 clouds ( uniform, lines, weighted lines, 30000 uniform, all-kernel host path, 6
  tight clusters, n = 1 .. 1000 ), leaves of 10 and 3: the same slices, boxes, children and leaf contents as
  `_build_in_kernel`, and the rule ( tight boxes, the median cut on the first longest axis, the halves on either side ).
* `the_card_tree_follows_the_rule_on_ties`: a grid, the grid repeated three times, a vertical line, 700 equal seeds + 300,
  lines on a lattice, negative coordinates and zeros: the rule, the host's slice sizes where no tie decides them, the same
  bits at every call; the card's cells of a grid through this tree = the generic ones.
* `the_card_tree_majorants_bound_the_weights`: smooth / random / constant weights: every node bounds its seeds, and a tree
  built with weights = a tree built without + `refresh_weight_majorants`, to the bit.
* `the_card_tree_is_built_under_jit_from_traced_positions`: the tree's tensors, its order's inverse and majorants, jitted
  == eager to the bit; a diagram on it ( FP64, FP32 ): measures and their adjoint with respect to the POSITIONS, jit == eager.
* `the_card_solve_builds_its_tree_in_the_jitted_call` ( `test_SdotPlanNd` ): the positions traced, cold and warm starts:
  jit == eager to the bit, and other positions through the same compiled function give the eager plan.
* `the_card_cells_are_the_generic_cells_with_weights`: `tol64` 1e-9 -> 3e-9 on its `w ~ h^2` cloud. Not the tree ( the host's
  tree gives the same 1.11e-9 ): the generic reference itself is 2e-9 from the exact plain storage on the almost empty cells
  of that cloud, and both move with the MAJORANTS ( they change the order of the cuts, hence the rounding ): 0.98e-9 at HEAD,
  where `PowerDiagram` re-ran the per-node majorant kernel on a tree built with the same weights, 1.1e-9 now that it keeps them.

## Findings on the way

* 1e7 on the card does not fit: the solve's pool runs out in XLA's BFC allocator ( 75 % of 11 GB ) with either tree, and the
  allocator RETRIES every 10 s instead of failing -- the call hangs. Not the tree ( ~0.3 GB of tensors, ~0.55 GB of work for
  its call, released after it ). A pool refusal is not reported as one here; to look at with `XLA_PYTHON_CLIENT_MEM_FRACTION` / the
  allocator's `take`.
* `Tensor.set( device array )` copies to the host ( `JaxDriver.array`: anything not a tracer becomes numpy ): every eager
  ffi output given to an aggregate field as `.raw` pays a device -> host copy and the next call an upload. Fixed here for the
  tree and the solver's outputs ( tensors adopted ); other sites remain ( `RealTensor[ pd.num_point ]( x.raw )` ... ).
* A closure-constant `nu` normalized under `jax.jit` differs from eager by 1e-19 ( XLA's fused reduction ): a jit == eager
  comparison must pass the masses as an argument ( as `the_card_solve_runs_under_jit` does ).

## What remains

* Bottom levels in one kernel ( a warp per node once the slices are <= 32-64 seeds: ~0.6 ms of the 5 at 1e6 ).
* The eager per-call overhead of loom ( ~3.5 ms per call: a jaxpr is traced per eager call ).
* 3D on a card still builds on the host ( `builds_on_card` is 2D: the card's cells and `Majorant2D` are 2D ); the build itself
  is generic in `D` ( template ), only the majorants are not.

## Risks

* The order inside a leaf moves the linear iterations by a few percent either way ( table above ); measured on three clouds.
* Ties at a median: a different ( valid ) tree from the CPU's on tied clouds -- so plans on such clouds may differ from the
  CPU's at the rounding of the solve ( never the cells' validity ).
* The build takes ~55 bytes per seed from the call's pool in 2D ( 0.55 GB at 1e7 ); a refusal of the pool leaves the outputs
  unwritten ( reported by loom's allocator, as for the other card calls ).
* `SDOT_CARD_TREE=0` restores the previous behaviour exactly ( host build, evaluated while tracing ).

# GPU step 8: densities and continuation ( 2026-10-05 )

The card's `SdotPlanNd` ( one ffi call, eager and under `jax.jit` ) now takes, besides a constant density, an `Image` on a
regular grid ( a diagonal frame, uniform knots, covering the box ) and isotropic `SumOfGaussians` in 2D, with the CPU's WIDTH
CONTINUATION ( `sdotplan/Continuation.h` semantics: the same options, stages, triggers and targets, so one `Tuning` gives the
same stages on both ). Refused on the card ( `NotImplementedError`, the CPU solves them ): 3D, a domain that is not a box ( an
image on a rotated grid ), irregular knots, anisotropic gaussians.

## How

* `gpu/Density2D.cuh`: the density is integrated on the EDGES of the finished cell ( `Cell2D.cuh::finish_cell`, a template
  parameter of the cells: `DensConst` compiles the code of before, so the constant path is unchanged ), never by cutting it:
  * image: Green on the rows ( `mass = circ F0 dy`, `F0` the prefix sum of the row, exact at the middle of a piece of edge in a
    pixel; an Amanatides-Woo walk; the reference of the first vertex subtracted ), the same walk gives `int rho ds` ( the
    laplacian's `c_ij` ) and, with the prefix sums of `x` and `x^2` and two Gauss points, the moments -- the old campaigns'
    method ( `solvers_des_familles` § 12, `gpu_des_familles` doc/08 );
  * gaussians: the CPU's polar corner ( `SumOfGaussians::wedge_measure` ) summed per gaussian over the polygon then `| . |`
    ( never negative: the CPU's `| sum |` per fan triangle -- a cell far from a narrow bump weighs 1e-22, and a negative noise
    there flipped the start's similarity test against the CPU ), the facet's `erf` ( `facet_mass` ), and the moments as
    CLOSED-FORM boundary fluxes ( `int ( x - c ) phi = -s^2 circ phi n`, `int |x - c|^2 phi = 2 s^2 m - s^2 circ ( x - c ).n phi` ).
    The corner's Gauss-Legendre takes panels of at most half a unit ( the CPU's 4 panels on long edges, ONE on the short edges of
    a fine diagram: -20 % of the cells' time; one panel of up to 4 units was tried and lost 1e-8 of a cell's mass ); an `erf`
    pair saturated on both ends is skipped ( `J = 0` to the bit ).
* `gpu/DensityHost2D.cuh`: the density of a stage, made on the card -- the image blurred by the CPU's filter ( separable gaussian of
  `s / step` pixels, truncated at 4 sigma, renormalized at the edge, axis 0 then 1, the same sums in the same order; the taps and
  the edge norms computed once per axis ), then the row prefix sums; the gaussians widened ( `sqrt( sigma^2 + s^2 )` ).
* `gpu/Newton2D.cuh::solve`: the stage loop of `Solve.h` ( `s0` = half the box's diagonal or `conv_start`, `/ ratio` down to the
  distribution's `min_scale` -- the image's step / 4, the smallest sigma / 4 -- then 0; `always`, or `auto` when the best start leaves
  a cell under `threshold` times the smallest target; per stage the target rescaled to the domain's mass, a Newton from the previous
  weights, `history.s`; the moments on the true density at the end ).
* The step `limits` with a density: the mass along `w + t d` is no polynomial, but the cell with its edges FROZEN is known at every `t`;
  the forward pass bisects the frozen mass of each cell ( looked for at `t = 1` and at the bottom of its area's parabola ), the
  backward pass that of the trial's cells under the floor ( `AlphaForwardDens`, `AlphaBackwardDens`; the CPU bisects exact cells ).
* Python: `PowerDiagram_Bsp._card_solve_density` ( the kind and its data; the geometry read on the host, the values not: traced
  values work ), `SdotPlanNd._build_card` ( three ffi names, one per kind ), `SumOfGaussians( support_box = ... )` ( the domain of
  § 9: the unit square ).

## Accuracy and parity with the CPU ( tests, n = 1500-2000 )

| case | step | card it / diag | CPU it / diag | weights gap ( / max |w| ) | cost gap | generic measures of the card's plan |
|---|---|---|---|---|---|---|
| soft image 24 x 17 | trials | 8 / 12 | 8 / 12 | 1.3e-13 | 3e-14 | < 1e-8 / n |
| image with a hole 32 x 32, continuation always ( 15 stages ) | trials | 78 / 102 | 78 / 102 | 8e-15 | 3e-15 | < 1e-8 / n |
| the same, continuation auto | limits | 77 / 99 | 78 / 106 | 1e-14 | 2e-14 | < 1e-8 / n |
| overlapping gaussians | trials | 8 / 11 | 8 / 11 | 9e-15 | 6e-5 ( CPU quadrature ) | < 1e-8 / n |
| narrow gaussians sigma 0.04, auto ( 15 stages ) | trials | 77 / 105 | 77 / 105 | 8e-15 | 2.6e-4 ( idem ) | < 1e-8 / n |
| the same, always | limits | 77 / 93 | 79 / 102 | 7e-15 | 2.6e-4 | < 1e-8 / n |

* With `trials` the card's Newton takes the CPU's decisions: the same iterations, diagrams, stages and widths ( `history.s` ).
* The gaussian MOMENTS differ from the CPU's by up to 2.6e-4 ( cost ) and 0.11 of a cell ( barycentres ): the CPU's are an adaptive
  quadrature capped at 8 bisections ( `PointwiseDensity` ), wrong on the huge tail cells and on the needles that reach a bump; the
  card's closed forms agree with a brute-force quadrature of the same polygons to its own accuracy ( 2e-7 relative, a 3000^2 grid
  and a 6000^2 one on a tail cell ). The image moments are exact on both sides ( 3e-15 ).
* jit == eager to the bit ( weights, cost, diagram counts ), image and gaussians with the continuation.

## The cases ( `bench_newton`, errand queue; n = 1e5 gaussians, 2e4 / 1e5 image; seconds; card: min of 3, CPU: lmo-numpy 8 pinned threads )

The continuation from `s = 0.5`, ratio sqrt( 2 ) ( `--continuation=always`, the bench's default on a density ); rtol 1e-6.
Card = jitted wall ( eager in brackets ); old = `reference_lmo.DENSITY` ( the old CPU, its own path: § 9 is the same width
continuation, § 12 a box-blur continuation with an adaptive / hand-tuned scale ).

| case | step | card it / diag | card float | card mixed | card double | CPU it / diag | CPU ( new ) | old CPU | card / CPU |
|---|---|---|---|---|---|---|---|---|---|
| gauss4 sigma 0.05 ( 13 stages ) | trials | 78 / 121 | 2.97 ( 3.37 ) | **2.96** ( 3.48 ) | 3.83 ( 4.26 ) | 78 / 121 | 36.3 | 39 ( 72 it, 123 diag ) | 12x |
| gauss4 sigma 0.05 | limits | 76 / 91 | 3.31 ( 3.72 ) | 3.29 ( 3.71 ) | 3.94 ( 4.35 ) | 77 / 107 | 33.3 | - ( 97 diag ) | 10x |
| gauss4 sigma 0.02 ( 16 stages ) | trials | 146 / 292 | 7.36 ( 7.82 ) | 7.37 ( 7.78 ) | 10.69 ( 11.16 ) | 146 / 292 | 69.4 | 124 ( 139 it, 400 diag ) | 9.4x |
| gauss4 sigma 0.02 | limits | 142 / 160 | 7.25 ( 8.00 ) | **7.23** ( 7.71 ) | 8.88 ( 9.62 ) | 133 / 208 | 62.7 | 108 ( 130 it, 204 diag ) | 8.7x |
| image 512^2, n 2e4 ( 22 stages ) | trials | 94 / 116 | **0.83** ( 1.33 ) | 0.83 ( 1.37 ) | 1.19 ( 1.97 ) | 94 / 116 | 10.1 | 3.7 ( 95 diag, adaptive box-blur ) | 12x |
| image 512^2, n 2e4 | limits | 94 / 116 | 0.86 ( 1.37 ) | 0.87 ( 1.38 ) | 1.23 ( 1.85 ) | 95 / 117 | 9.93 | 3.6 ( 90 diag, hand-tuned ) | 11x |
| image_hole 512^2, n 2e4 | trials | 95 / 121 | 0.85 ( 1.38 ) | 0.86 ( 1.41 ) | 1.24 ( 1.83 ) | 95 / 121 | 10.2 | 4.9 ( 124 diag, adaptive ) | 12x |
| image_hole 512^2, n 2e4 | limits | 95 / 117 | 0.90 ( 1.42 ) | 0.89 ( 1.42 ) | 1.24 ( 2.06 ) | 96 / 122 | 10.4 | 3.9 ( 100 diag, hand-tuned ) | 12x |
| image 512^2, n 1e5 | trials | 103 / 134 | - | 1.88 ( 2.43 ) | 2.89 ( 3.44 ) | 103 / 134 | 29.2 | 188 / 90.7 ( floor path, 792 / 796 diag ) | 16x |
| image 512^2, n 1e5 | limits | 102 / 131 | - | 2.00 ( 2.55 ) | 2.99 ( 3.54 ) | 103 / 133 | 29.3 | idem | 15x |
| image_hole 512^2, n 1e5 | trials | 112 / 160 | - | 2.10 ( 2.69 ) | 3.30 ( 3.84 ) | 112 / 160 | 32.8 | - | 16x |
| image_hole 512^2, n 1e5 | limits | 108 / 133 | - | 2.12 ( 2.67 ) | 3.13 ( 3.67 ) | 110 / 144 | 30.6 | - | 14x |

( card / CPU: the CPU's wall over the card's jitted mixed one, same step. ) The old GPU campaign has no number at these n: its
image Newton ( doc/08, n = 2e5, the 90:1 image without hole, a CONTRAST continuation in 8 steps ) took 164 s; its cells under a
512^2 image cost +7 % at 1e6 ( 96.6 -> 103.9 ns/seed, `reference_lmo_gpu.GPU_DENSITY` ).

Where the card's time goes ( mixed, jitted ): gaussians sigma 0.02 trials: cells 3.6 s ( 292 diagrams, 12 ms each: the corners in
FP64 on Turing, 1/32 rate ), linear solves 3.5 s ( 10634 multigrid iterations: these laplacians are harder than Lebesgue's, 73 per
solve ), the frozen-mass passes 1.8 s with `limits`. Image 2e4: linear solves 0.62 s of 0.83 ( 94 solves, the per-iteration
host round trips at this n ), the 22 blurred images a few ms each. The double kernel costs 1.5-2x the float one on densities
( its finish runs inside the walk kernel; the float kernel's in its own pass ).

## Findings on the way

* A PRE-EXISTING BUG of the card's multigrid ( `Linear2D.cuh::sa_galerkin_dense` ), exposed by the density laplacians: the COUNT pass
  of the dense Galerkin product marked every coarse column touched, the FILL pass emitted only those whose fixed-point sum was not
  zero -- with coefficients from 1e-30 to 1e3 in a row, tiny contributions round to exactly zero at the row's scale, fewer entries
  are written than counted, and the coarse CSR keeps UNINITIALIZED column indices: a NaN residual ( "linear solver failure" at
  n = 1e4 ) or an illegal address ( the gaussian bench at 1e5, after another solve had left garbage in the pool ). Fixed ( the fill
  marks the columns it touches; a zero entry stays a zero ); found with a new debugging aid, `SDOT_CARD_POISON=1` ( every `take`
  of the card's pool filled with 0xff: NaN doubles ) and `compute-sanitizer --tool initcheck`, now clean on the solve ( two benign
  over-reads silenced: the float copy of the fine level stops at `nnz`, the counts' scan reads a zeroed slot ).
* The CPU's moments of a gaussian density ( adaptive quadrature, depth 8 ) are off by up to 2.6e-4 on the cost; its masses are
  exact. Not fixed ( CPU side ): the closed forms of `Density2D.cuh::gauss_lines` / `gauss_cell` would port as they are.
* The CPU's image path is slow at 2e4 ( 10 s, 45 ms per diagram: the cells CUT by the pixels; plus the blur of 22 stages,
  single-threaded ): 2.7x the old bench's 3.7 s, which integrated on the boundary. Not in scope ( CPU ).
* `limits` on a density: fewer diagrams than `trials` on the gaussians ( 160 against 292 ) but each iteration pays the frozen-mass
  passes ( 1.8 s ): about even in time at 1e5; on the images the forward pass never limits and the two are the same.

## Tests ( lmo-jax: `test_SdotPlanNd`, `test_CardCells`; lmo-numpy: `test_SdotPlanNd`; local jax / numpy / torch: `test_SdotPlanNd` )

* `the_card_solves_an_image_as_the_cpu`, `the_card_solves_gaussians_as_the_cpu`: the card against the CPU in a subprocess
  ( `LOOM_DEVICE=cpu` ), the table above; the cells of the card's weights measured by the GENERIC path are the targets.
* `the_card_density_solves_run_under_jit`: image and gaussians, continuation auto: jit == eager to the bit.
* `the_card_refuses_what_it_does_not_solve`: now an image on a rotated grid ( not a box ).

## What remains

* 3D on the card ( cells, tree majorants, the facet circulation becomes a surface one ), domains other than a box, rotated or
  irregular images, anisotropic gaussians.
* The LIFTING of the old campaign ( § 13-16, § 15.17: -28 % of diagrams at 1e5 on the image ) and its adaptive continuation scale
  ( § 12.6 ): neither is in the CPU solver either.
* Speed: the gaussian corners in FP64 ( a far-cell skip would need the CPU's noise semantics, see above ); the double kernel's finish
  in a pass of its own as the float one; the multigrid's iterations on density laplacians ( 73 per solve ); the frozen-mass passes
  of `limits` on all cells ( a pre-filter on the area polynomial ).
* A host fallback of the direction when the card's solver fails ( none seen since the Galerkin fix ).

## Risks

* `limits` on a density is the card's own step ( frozen cells bisected ), not the CPU's ( exact cells ): other counts, the same plans
  ( 1e-14 ). A frozen cell past an edge's vanishing is extrapolated; the trial diagram is the check.
* The gaussian moments differ from the CPU's ( closed forms against a quadrature ): a comparison of costs between the two at 1e-4.
* Parity of counts relies on noise-level agreement of masses in density deserts ( `| sum |` per gaussian ); a case where all the
  gaussians are far from a cell gives masses at the 1e-17 noise on both sides, not the same noise.
* The image's tables are 5 doubles per pixel in the call's pool ( 10 MB at 512^2, 2.7 GB at 16384^2 ); past the pool, the outputs
  are left unwritten as for any card call.

# GPU step 9: the card's memory, a clear error instead of a hang ( 2026-10-05 )

A uniform 2D solve of 1e7 seeds on lmo ( 2080 Ti, XLA's pool 8.51 GB = 75 % of 11 GB ) never ended.

## Why it hung

* The solve takes its buffers during the call from XLA's pool ( `take`, loom's `Scratch` over `ffi::ScratchAllocator` ). When
  the pool cannot give, XLA's BFC allocator does NOT refuse at once: it waits ~10 s for frees ( `AllocatorRetry` ), then refuses.
* Our prepares went on taking after a refusal ( `CardLinear::prepare` takes all its levels before checking ): at 1e7 the first
  refusal came after 7.17 GB, inside the linear solver's prepare, and then every following take waited its 10 s -- 117 refusals
  logged in 20 minutes, not finished. No retry loop of ours or of loom: XLA's wait times the number of takes.
* `SDOT_CARD_TAKES=1` ( new, `Cell2D.cuh::take` ): every take printed with its size, ok / REFUSED and how long it took.

## What changed

* loom `Scratch` ( `fe283f9` ): a refusal is FINAL for the call ( later requests refused without asking the pool: one wait per
  call at most ), the bytes taken are counted, and the handler's `RESOURCE_EXHAUSTED` message says what was refused, what the call
  had taken, and the body's `why` ( the card solve says what it is, for how many seeds, and what to do ).
* A model of the solve's memory ( `sdot/CardMemory.py::card_solve_bytes`, take by take, the multigrid's coarse matrices per
  fine unknown as measured ), checked before the tree and the solve against `memory_stats()` ( `bytes_limit - bytes_in_use` ):
  a `MemoryError` at once ( eager and while tracing ) with the need, the largest parts, what is left, how many seeds would fit
  and what to do ( fewer seeds, `fp32`, `XLA_PYTHON_CLIENT_MEM_FRACTION`, free arrays, the CPU ). Before refusing, Python gives
  up the multigrid's recycled solutions ( 16 B / seed each ) when that makes it fit. `SDOT_CARD_MEMORY_CHECK=0` skips it.
  `stats[ "scratch_bytes" ]`: what the solve really took ( the model is within 1 % at 1e6 and 1e7 ).
* Less memory per seed ( `Newton2D.cuh`, `Cell2D.cuh`, `Linear2D.cuh` ), nothing changed in the arithmetic:
  * the moments' card borrows the card of its kernel ( nodes, seeds, lists, finish store ) and writes in the trial's measures
    and the direction ( -58 B ); the double kernel's card borrows the float card's nodes and lists ( -18 B );
  * one COO for both diagram slots, made again in the rare case it holds a refused start's facets ( -48 B ); between the
    assembly and the next diagram it is the linear solver's: r, z, p, q and the fine float input / output / diagonal are
    carved from it ( -44 B );
  * the linear solver's scan work is the assembly's ( -16 B ); the target is gathered again per stage instead of kept twice ( -8 B );
  * the float kernel's finish store holds at most `DEFER_CAP = 2^20` cells ( 155 MB ): past it the first pass runs by chunks and
    the second leaves its cells by position ( -133 B at 1e7; unchanged below 2^20; `SDOT_CARD_DEFER_CAP` for tests ).

| bytes per seed ( mixed, uniform ) | before | after |
|---|---|---|
| solve's scratch, 1e6 | 1094 | 897 |
| solve's scratch, 1e7 | ~990 ( never completed ) | 663 |

After, at 1e7 ( model, B / seed ): linear solver 203, diagram slots 152 ( edges 2 x 64 ), laplacian 104, cells 82, facets 48,
vectors 40, majorants 27, overflow slots 10; plus the tree's and the outputs' XLA buffers ~100.

## 1e7 on lmo ( mixed, limits, rtol 1e-6 )

* eager: converged, 7 iterations, 13 diagrams, solve 8.4 s; 6.63 GB taken, peak of the pool 8.45 of 8.51 GB.
* jitted: runs ( 8.4 s ), and again on a second call -- at the edge: anything else the program keeps on the card makes it refuse.
* before the fix it hung; a solve that does not fit now raises: `MemoryError` in 0.02-0.07 s ( the model ), or, check skipped,
  `RESOURCE_EXHAUSTED` after one wait of the pool ( 10.8 s ).

## Timings ( `bench_newton --case=uniform --step=limits --jit=yes`, n22 RTX A6000, min of 3; jit wall, seconds )

| case | kernel | before | after | it / diag |
|---|---|---|---|---|
| uniform 1e5 | float | 0.066 | 0.066 | 5 / 6 both |
| uniform 1e5 | mixed | 0.072 | 0.073 | 5 / 6 both |
| uniform 1e6 | float | 0.336 | 0.333 | 5 / 6 both |
| uniform 1e6 | mixed | 0.382 | 0.383 | 5 / 6 both |

( lmo, before only, same protocol: 0.075 / 0.081 / 0.436 / 0.495. ) The same iterations, diagrams and stage times: below
2^20 seeds the kernels are the same; the memory check is one `memory_stats()` per solve.

## Tests

* lmo-jax `test_SdotPlanNd`, `test_CardCells`: all good. Local jax CPU `test_SdotPlanNd`: all good.
* New: `the_card_memory_model_follows_the_solve` ( 2e5 seeds, mixed / fp32 / fp64 double levels / cg: model within 8 % of
  `scratch_bytes` ), `the_card_finish_store_by_chunks_gives_the_same_plan` ( `SDOT_CARD_DEFER_CAP=1000`: the same plan to the bit ),
  `the_card_refuses_a_solve_that_does_not_fit_at_once` ( `XLA_PYTHON_CLIENT_MEM_FRACTION=0.05`, 1e6 seeds: `MemoryError` in
  < 0.1 s eager and jit; without the check `RESOURCE_EXHAUSTED` in 10.8 s with the message ).

## Risks

* The model's coarse matrices are measured on the uniform cloud; a cloud whose Galerkin products are denser takes more than the
  model says -- then the call refuses after one wait of the pool, with its message.
* Under `jax.jit` the program's own buffers are not seen at trace time: the check passes and the call may still refuse.
* 1e7 fits the 2080 Ti with ~60 MB to spare: the edges of the two diagram slots ( 128 B / seed ) are the next thing to shrink.
