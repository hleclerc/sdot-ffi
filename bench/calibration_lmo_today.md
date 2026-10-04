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
