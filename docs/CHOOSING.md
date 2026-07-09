# Choosing a solver

The library ships a portfolio rather than one algorithm, because the best choice
depends on the instance. This guide maps instance features to solvers; if you'd
rather not choose, use **`auto`** (the meta-solver that applies these same rules).

For the model and symbols see [MODEL.md](MODEL.md); to run a solver see
[USAGE.md](USAGE.md).

---

## Quick decision table (minimize makespan)

| If your instance is… | Use | Why |
|---|---|---|
| **small** (`N` ≲ 6) | `exact` (B&B) or `exact-milp` | a proven optimum is affordable |
| small, want an independent check | `exact-dual` | same optimum via the OptV-bisection paradigm |
| **memory-limited** (`Σ Bᵢ < V`, so one round can't carry the load) | `best-rate` | fast, builds a good multi-installment sequence |
| **very large / memory-bound, LP cost matters** | `online` (`--chunk=gss --psr=super`) | LP-free online simulation; O(V/ρ·log m), scales when per-sequence LPs get expensive |
| **ample memory, no startups** (`Sᵢ = 0`) | `single-round` | closed-form **optimum** for this case, instant |
| **ample memory, with startups**, larger `N` | `ga` | general-purpose quality at scale |
| any of the above, don't want to decide | `auto` | picks one of the above from the features |
| want an **ML-predicted T\*** alongside the schedule | `ml-makespan` | GBM predicts log(T\*) in < 1 µs; GA builds the schedule; reports both |

## Different objectives (not makespan)

| Question | Use |
|---|---|
| "How much load fits within a deadline `T`?" (OptV) | `optv` (exact) |
| OptV on infinite-bandwidth (`Cᵢ = 0`), want a guarantee | `fptas-optv` (`≥ (1−ε)·V_OPT`) |
| "Minimum time to process `V`?" on `Cᵢ = 0`, want a guarantee | `fptas-optt` (`≤ (1+ε)·T_OPT`) |

Other problem classes have their own solvers (`--class=mlsd|mapreduce|multilayer`).

---

## Rationale (from the benchmark, `dls-bench`)

On random heterogeneous instances (`A,C,S ~ U[0,1]`), measuring distance from the
optimum (the benchmark's `gap→best`):

- **`exact` / `exact-milp`** give the optimum but cost grows with `N` (and the
  B&B installment cap matters under tight memory) — so reserve them for small `N`.
  `exact-milp` is single-installment only.
- **`single-round`** is the classical closed form: **optimal when `Sᵢ = 0`**, and
  ~1% from optimum otherwise, at essentially zero time — an excellent default
  when memory is ample (`Σ Bᵢ ≥ V`).
- **`best-rate`** shines exactly where `single-round` can't: memory-limited
  instances that need multiple installments. With *unbounded* memory it
  degenerates (one chunk), so don't use it there.
- **`ga`** is the all-rounder — slightly better quality than `single-round`
  (~0.7% from optimum) but seconds, not microseconds; worth it for large
  heterogeneous instances with startups.

## The `auto` meta-solver

`AutoSolver` (registered as `auto`) inspects `N`, `Σ Bᵢ` vs `V`, and whether all
`Sᵢ = 0`, then delegates per the table above. After solving it reports which
solver it used (the `dls` CLI prints `chose : …`). If the pick can't find a
feasible schedule it falls back to `best-rate`, which handles arbitrary
multi-installment cases.

```bash
dls solve --solver=auto instance.txt      # prints e.g. "chose : single-round"
```
```cpp
#include "heuristics/auto/auto_solver.hpp"
AutoSolver a;
DLSSolution s = a.solve(inst, cfg);
// a.chosenSolver() == "exact" | "best-rate" | "single-round" | "ga"
```

`auto` is a *heuristic dispatcher* — it does not guarantee the global optimum
(except when it happens to pick `exact`). For a guaranteed optimum on a small
instance, ask for `exact` directly.

## The `auto-ml` meta-solver

`AutoMlSolver` (registered as `auto-ml`) extracts a **17-feature vector** from the
instance (`N`, memory ratio, heterogeneity in `A/C/S`, startup fraction, mean
rates, speedup ratios, load per processor) and feeds it into a gradient-boosted
tree (GBM, 150 stages × 5 classes, trained on labelled DLT benchmarks) to predict
which portfolio solver will achieve the lowest makespan. Like `auto`, it then runs
that solver and reports the chosen name. The model is embedded as zero-dependency
C++ arithmetic (see `heuristics/auto/solver_selector.hpp`); inference is
sub-microsecond.

The learned selector is ~74% accurate, and a wrong pick can be far worse than the
best solver (not just marginally), so two rule-based guards backstop it:

- **small N** (`N ≤ 6`): exact B&B is affordable and provably optimal, so the
  selector can only match or lose — `auto-ml` bypasses it and solves exactly.
- **best-rate on ample memory** (`Σ Bᵢ ≥ V`): `best-rate` degenerates to a single
  installment there and is never the true best, so such a pick is treated as a
  misprediction and replaced by the rule-based ample-memory choice (`single-round`
  when all `Sᵢ = 0`, else `ga`). On genuinely memory-limited instances
  (`Σ Bᵢ < V`) the guard stays out of the way and `best-rate` is used as picked.

These mirror `auto`'s hand-coded rules, so `auto-ml` is never obviously worse than
`auto` on the cases the rules cover.

The response also includes an instance **difficulty** label (`easy` / `medium` /
`hard`), predicted by a separate GBM classifier that estimates how far the best
heuristic falls from exact — useful for deciding whether to escalate to `exact`.

```bash
dls solve --solver=auto-ml instance.txt   # prints "chose : ..." + difficulty
```

Retrain both models after generating new benchmark data:
```bash
bash tools/retrain.sh --n 50000   # generate → train → rebuild → restart
```

### Training-data provenance and oracles

The generators (`tools/generate_training_data.py`, `tools/generate_mlsd_training_data.py`,
both on the shared `tools/genlib.py` harness) label every row with an **exact
oracle where one is affordable, a heuristic otherwise**, and record which on the
row itself. Each CSV carries provenance columns: `seed` (reproduce the row),
`regime`, `oracle` (the solver that set the label), `label_is_exact` (1 if that
oracle is provably optimal here), and `solver_wall_ms`. A trainer can then filter
to `label_is_exact == 1` to learn only from proven optima.

The oracle is chosen by a **deterministic size gate** (never wall-clock), so
labels are reproducible regardless of host or `--workers` count. For single-load,
`exact` B&B labels instances with `N ≤ 6`. For MLSD, `mlsd-exact` covers the
tiny `n·m ≤ 4` cases, and the exact MILP `mlsd-milp` can be enabled for larger
instances with `--milp-max-n` / `--milp-max-nm` (off by default: each MILP costs
seconds, so it is practical only on capable hardware). Generation parallelises
across `--workers` processes.

**Non-star classes (`tools/generate_topology_training_data.py`).** These are the
opposite case: their solvers are LP-exact (`chain`, `tree`) or closed-form
(`mapreduce`, `multilayer`) and run in well under a millisecond, so datasets of
millions are cheap even on one core (~5–10 k instances/s). Each class trains a
standalone makespan surrogate via `tools/train_topology_predictor.py`, exported
as a zero-dependency C++ header (`heuristics/ml/<class>_predictor.hpp`,
`predict_log_makespan_<class>(const float*)`).

Label quality per class (`label_is_exact`): `chain`/`tree` are LP-optimal and
`mapreduce` is the proven optimum (Berlińska Prop. 4.1) → exact; `multilayer`'s
closed form is a feasible-schedule **upper bound** on the overlapped optimum, so
its rows are tagged non-exact (the surrogate learns the solver's output, which is
what a `multilayer` run returns). These are the classes where an ML surrogate is
most useful — no closed form exists for an arbitrary tree/chain optimum.

**Energy (`tools/generate_energy_training_data.py`).** Minimizing energy directly
(rather than makespan) was previously only reachable by constructing
`MilpParams{minimizeCost=true}` in C++ — no solver name or CLI flag exposed it.
It is now reachable through the existing `exact-milp` solver via a
`minimizeCost=1` option (`bindings/dls_c.cpp`, `core/solver_registry.hpp`): when
the instance uses the energy model (any of `powerIdle`/`powerStartup`/
`powerNetwork`/`energyPieces` set), this minimizes total energy instead of Cmax,
proven optimal via HiGHS (validated against the thesis's hand-computed example
and the MinCost-vs-MinMakespan divergence test in `lib/tests/tests.cpp`).

Unlike the other oracles, energy-mode MILP cost grows steeply with `N`
(calibrated: N=7 ≈3s, N=8 ≈5s, N=9 already exceeds 20s), so
`tools/generate_energy_training_data.py` gates generation to `N ≤ 8` by default
(`--exact-max-n`) — every row is still exact, just capped in scale accordingly
(tens of thousands, not millions, at this instance size). `tools/train_energy_predictor.py`
exports `heuristics/ml/energy_predictor.hpp` (`predict_log_energy(const float*)`),
following the same standalone-header pattern as the topology predictors.

## The `ml-makespan` solver

`MlSolver` (registered as `ml-makespan`) is a two-stage solver whose unique
output is the **ML-predicted optimal makespan T\*** alongside the actual schedule:

- **Stage 1** — a GBM regressor (trained on `log(T*)`) predicts the optimal
  makespan from the 17-feature vector in < 1 µs. The prediction is reported
  separately as `predictedMakespan`.
- **Stage 2** — the best applicable heuristic constructs a valid schedule:
  - `Sᵢ = 0, Cᵢ = 0`, no memory cap → `BestRateSolver` (LP-optimal here)
  - `Sᵢ > 0, Cᵢ = 0`, no memory cap → `FptasOptTSolver` (ε = 0.05 guarantee)
  - General (comm costs or memory caps) → `GASolver` (searches activation orders,
    LP-optimises each split; correctly distributes load across all processors)

This lets you compare `predictedMakespan` (what the ML model thinks is optimal)
against the actual schedule makespan and against a lower bound — a direct
empirical test of ML prediction quality at inference time, with no extra solver
invocation.

```bash
dls solve --solver=ml-makespan instance.txt
# JSON response includes "predictedMakespan": <T_ml> alongside "solution.makespan"
```

The model is in `heuristics/ml/makespan_predictor.hpp` (placeholder until
`tools/train_makespan_predictor.py` runs). Retrain via `bash tools/retrain.sh`.
