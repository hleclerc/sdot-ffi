# Benchmarks against the old C++ campaign

These entries reproduce the **cases** of `nsdot/solvers_des_familles` (README, ~8000 lines, French) on
the python package `sdot`, so that a number measured here can be set next to the old one.

| file | what |
|---|---|
| `cases.py` | the six clouds: loader of the reference text files, numpy generators, `case( name, n )` |
| `reference_lmo.py` | the old headline numbers as data, each with its README line |
| `bench_diagram.py` | errand `bench`: the cost of **one diagram** ( ns/seed ) |
| `bench_newton.py` | errand `bench`: a **whole Newton solve** ( iterations, diagrams, time split ) |
| `benchlib.py` | threads, environment report, table printing (declares no work) |

`cgal/` and `cuda/` are older, unrelated comparison programs.

## The cases

| name | dim | n | cloud |
|---|---|---|---|
| `uniform2d` / `uniform3d` | 2 / 3 | 1e6 for the diagram, 1e5 for Newton | uniform in `[0.001, 0.999]^d`, weights 0 (or `--wscale`: uniform in `[-w h^2, w h^2]`) |
| `lines_voronoi` | 2 | 99 944 (the file) | 5 lines, sigma 0.005, deduplicated; weights 0 |
| `lines_equal` | 2 | 100 000 | the same cloud, **not** deduplicated; weights = equal areas (the witness of the old campaign) |
| `planes_voronoi` | 3 | 100 000 | 4 planes, sigma 0.02; weights 0 |
| `planes_equal` | 3 | 100 000 | the same cloud, weights = equal volumes (made by the old bench itself) |

Domain: the unit square / cube. The four hard clouds are the text files of
`2d_des_familles/cases` (`#` header lines, then `n`, then `x y [z] w` per line). The directory is
`$SDOT_CASES_DIR`, by default `~/nsdot/2d_des_familles/cases` (the same path on `lmo`). `python cases.py`
lists what is found.

If a Voronoi file is missing it is **regenerated** (port of `gen_cases.py` / `gen_cases_3d.py`, same seed):
the 3D cloud is the file bit for bit, the 2D one is the file *before* its deduplication (100 000 seeds
instead of 99 944; the old Newton numbers for it, 78 diagrams, are for the deduplicated cloud). The `equal`
files cannot be regenerated (their weights are a transport solution): `case()` then raises an error that
says where it looked. With another `n`, a fresh cloud is drawn (for `equal`: weights 0, with a warning): a
smoke run, not a case of the campaign. numpy's generator is not the old `mt19937_64`: uniform clouds have
the same distribution, not the same points.

## Running

errand finds the files by their name (`bench_diagram`, `bench_newton`) anywhere in the tree; flags are the
`Param` names with dashes (`--max-iter`, `--linear-solver`). `errand --help bench_newton` lists them. A
comma is a matrix: one run, one directory per combination. Run from `/Users/hugo.leclerc/Projects`.

Locally (CPU, the laptop; **not** for timings):

```bash
cd ~/Projects
micromamba --root-prefix ~/.mamba run -n prj-jax errand -k bench --env jax --no-queue "bench_diagram,bench_newton" --n=3000 --threads=2 --reps=1   # smoke
```

On `lmo` (an rsync of the whole `~/Projects` tree to `/home/leclerc/Projects-rsync`, environments
`lmo-numpy`, `lmo-torch`, `lmo-jax` of `errand-envs.py`). **The solver and the diagram timed here run on the
CPU**: use `lmo-numpy` (CPU by construction) -- `lmo-jax` is the GPU environment, and `SdotPlanNd` refuses a GPU driver
(`LOOM_DEVICE=cpu` would be needed in it).

```bash
errand -k bench --env lmo-numpy "bench_diagram" --case=uniform --dim=2 --n=1000000 --threads=8 --pin=yes --kernel=double --reps=3
```

One command line per case of the campaign (`--threads=8 --pin=yes --kernel=double` is the old protocol):

```bash
# ONE DIAGRAM  (old numbers: § 3, § 11, § 19.10)
errand -k bench --env lmo-numpy bench_diagram --case=uniform --dim=2 --n=1000000 --threads=8   # 2D uniform 1e6
errand -k bench --env lmo-numpy bench_diagram --case=lines_voronoi --threads=8                 # 2D lines, Voronoi
errand -k bench --env lmo-numpy bench_diagram --case=lines_equal   --threads=8                 # 2D lines, equal areas
errand -k bench --env lmo-numpy bench_diagram --case=uniform --dim=3 --n=1000000 --threads=8   # 3D uniform 1e6
errand -k bench --env lmo-numpy bench_diagram --case=uniform --dim=3 --n=100000  --threads=8   # 3D uniform 1e5 ( § 11 )
errand -k bench --env lmo-numpy bench_diagram --case=planes_voronoi --threads=8                # 3D planes, Voronoi
errand -k bench --env lmo-numpy bench_diagram --case=planes_equal   --threads=8                # 3D planes, equal volumes
errand -k bench --env lmo-numpy bench_diagram --case=uniform --dim=3 --n=100000 --memory=0 --threads=8   # without neighbour memory

# A WHOLE NEWTON SOLVE  (old numbers: § 3, § 24; n = 1e5 )
errand -k bench --env lmo-numpy bench_newton --case=uniform --dim=2 --threads=8 --step=trials,limits
errand -k bench --env lmo-numpy bench_newton --case=lines_voronoi   --threads=8 --step=trials,limits
errand -k bench --env lmo-numpy bench_newton --case=lines_equal     --threads=8 --step=trials,limits
errand -k bench --env lmo-numpy bench_newton --case=uniform --dim=3 --threads=8                # 3D has no `limits`
errand -k bench --env lmo-numpy bench_newton --case=planes_voronoi  --threads=8
errand -k bench --env lmo-numpy bench_newton --case=planes_equal    --threads=8
errand -k bench --env lmo-numpy bench_newton --case=lines_voronoi --linear-solver=cholesky,amg --threads=8   # § 3 linear solvers
```

Each run prints one table row: case, n, threads, kernel, time, ns/seed (diagram) or iterations / diagrams /
`t_diag` / `t_asm` / `t_lin` / `t_lim` / `t_maj` / total (Newton), then the **old value and the ratio
new/old** (> 1: we are slower), the sources of the old value, and the active environment (`LOOM_NB_THREADS`,
`SDOT_NB_THREADS`, `SDOT_PIN_THREADS`, `SDOT_CPU_VARIANT`, `SDOT_CXXFLAGS`, `LOOM_CXXFLAGS`, driver and device...).
Numbers also go to `result.yaml`; `summary.yaml` compares a matrix.

### Threads

loom's pool reads `SDOT_NB_THREADS` (C++) and `LOOM_NB_THREADS` (python: the per-thread scratch is sized on it)
at the first kernel call; `SDOT_PIN_THREADS=1` pins worker `w` to core `w` (Linux; the calling thread, worker 0,
is not pinned). The entries set all three from `--threads` / `--pin` before anything runs, one process per
run. `--pin=env` leaves the variable to the shell. Compile flags: `SDOT_CPU_VARIANT`, `SDOT_NO_MARCH_NATIVE`,
`SDOT_CXXFLAGS` (default `-O3 -march=native`). The Newton linear solver (Eigen / AMGCL) does not use a BLAS here;
if you ever link one, pin its threads (`OPENBLAS_NUM_THREADS`; old README l.6850: 16 BLAS threads cost 1.5x).

## Newton: how the options map onto the old ones

The problem is the old one: seeds = the cloud, target = Lebesgue on the unit domain (an `Image` of ones),
equal masses `1/n` (`equal areas`), start from weights 0 (Voronoi), stop when `max|a-nu|/nu <= 1e-6`
(`--rtol=1e-6 --tol=0`; the library's default `tol = 1e-8` is absolute and much looser), no continuation.
The whole solve is timed from a cold start, tree included (the old `TOTAL` includes the tree and the majorants).

| old (`newton` binary) | here |
|---|---|
| KMT damping, `--pas essais --residu lin` | `--step=trials` (the default of the library in 3D) |
| `--pas essai-limites --facteur 0.9` | `--step=limits` (2D only; the library's `auto` in 2D) |
| `--residu log` + switch to `lin` (the old default after § 24.5) | `--residual=log` (the default), `--residual-switch=2`; `--residual=lin` is KMT, `--residual=power --residual-power=p` the exponent residual |
| `--solver chol \| amg \| cg` | `--linear-solver=cholesky \| amg \| cg \| mg` (see `Linear.cpp` for `auto`; `mg` = `Multigrid.h`, options `--mg-pack --mg-recycle --mg-rebuild --mg-stop`) |
| `--newton-tol`, `--newton-max` | `--rtol`, `--max-iter` |
| `--kernel double \| float` | `--kernel=double \| float` |
| `--threads`, `--no-pin`, `--leaf` | `--threads`, `--pin`, (`bench_diagram` only: `--leaf-size`) |

Reference rows (`reference_lmo.py`, `newton_ref( case, variant )`): `kmt` = KMT + `lin`, `kmt_log`, `best` = limits +
`log` + switch, `model`. By default `--step=trials` is compared with `kmt` and `--step=limits` with `best`
(so `limits` is compared with a number that also has the log residual: it should be *slower* than `best`
until the log residual exists here); `--ref=kmt_log` picks another. `--start=file` starts from the file's
weights (the `equal` files are the solution: it checks the witness, it is not a timing).

## Protocol (for numbers that are compared with the old ones)

* The machine **alone** (errand's `bench` is `exclusive`; do not run anything else, in particular no GPU job).
* **8 threads pinned**, kernel **double**, **minimum of 3** repetitions (the old campaign used 3 for the
  suite, 10 for § 11); warm-up apart (the first call compiles; a diagram is warmed up twice, the second round
  fills the neighbour memory).
* The diagram is timed as `pd.weights = w; pd.measures` on a diagram and tree built once (`tests/test_PowerDiagram.py`,
  `pd accelerated`); the old `diagramme` times the sweep of the same fixed weights.
* Ratios are only printed where the old campaign has a row **at the same n** (1 %); the closest old row is shown
  with its stage (§ 3 first numbers vs. after § 11 memory vs. after § 19.11): several old values exist for one
  case, the most recent is the one used.

## Gaps against the old campaign (what `sdot-ffi` does not have yet, so cannot be compared one for one)

* ~~`log` residual + switch to `lin`~~: ported (`NewtonOptions::residual`), counts match the old binary (see `calibration_lmo_today.md`).
  The old KMT restarts each iteration from t = 1: `--restart-factor=1e9` reproduces its counts.
* **The model step** (span, `K = 2`) and the **exponent residual** `p = 0.25`: options of the old bench only.
* **In-house multigrid `mg`** (3D: 8.46 s against 21.3 s for supernodal Cholesky, old README l.6857-6864): the new AMG is AMGCL.
* **Supernodal CHOLMOD** (3D factor x38 on the factorization): the new Cholesky is Eigen's simplicial LDLT.
* **`fp32 -> fp64` switch** (`--kernel mixte`, § 19.10): `--kernel` is `double` or `float` only. (Measured useless on the
  Xeon, `s ~ 1`; it was meant for GPUs.)
* **Coincident-seed agglomeration** (§ 23.6-23.12): `Iterative( aggregate = ... )` is accepted but not wired (step 7 of
  `notes/2026-10-02-sdotplan.md`): `lines_equal` stays at the old `STAGNATION 2.35e-06` floor, and the
  deduplicated `lines_voronoi` is what makes the old 78-diagram number reproducible.
* **Lifting** (`relèvement` of the old campaign, § 8.5, § 13-16: raising weights of empty cells; multiscale) and the limits step in **3D** (`step = 'limits'`
  raises there).
* **Same random draws**: numpy is not `mt19937_64` (uniform clouds are statistically, not pointwise, the old ones).
* **3D `equal` witness**: the old file was made by the old Newton itself, so it is a timing case, not an independent witness
  (the old campaign says so); it has no Newton row in the old README tables, only its diagram numbers.
