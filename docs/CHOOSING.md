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

## The `ml-energy` solver

`EnergyMlSolver` (registered as `ml-energy`) is the energy-model counterpart
of `ml-makespan`, and exists for the same reason: the only way to know the
true minimum energy is `exact-milp` (`minimizeCost=1`), and that MILP's cost
grows steeply with N (calibrated: N=7 ≈3s, N=8 ≈5s, N=9 already >20s — see
`tools/generate_energy_training_data.py`), with **no fast energy-aware
heuristic oracle to fall back on**. That is the genuine "no closed form
exists" gap this ML model fills.

- **Stage 1** — a GBM regressor (trained on `log(minimum energy)`, 15-feature
  vector, test MAPE 16.8%) predicts the lowest achievable energy in < 1 µs,
  reported as `predictedEnergy`.
- **Stage 2** — `GASolver` is steered toward that budget via its existing
  bi-criteria mode (`costLimit = predictedEnergy × 1.30`, the 30% headroom
  covering the prediction's error margin); if that tight budget turns out
  infeasible, it falls back to an unconstrained GA run rather than fail.

`predictedEnergy` and the actual schedule's `solution.energy` are not expected
to match exactly — Stage 2 still ultimately minimizes makespan (no fast
energy-minimizing heuristic exists to hand it instead), so its real energy is
a bi-criteria compromise, not a re-derivation of the Stage 1 estimate.

```bash
dls solve --solver=ml-energy instance.txt
# JSON response includes "predictedEnergy": <E_ml> alongside "solution.energy"
```

The model is in `heuristics/ml/energy_predictor.hpp`, trained on 20,000
exact-MILP-labelled rows. Feature extraction lives in
`heuristics/energy_features.hpp`, matching
`tools/generate_energy_training_data.py`'s `compute_energy_features()` exactly.

### Why chain, tree, MapReduce, and multilayer have no `ml-*` solver

Trained predictors exist for all four (`chain_predictor.hpp`,
`tree_predictor.hpp`, `mapreduce_predictor.hpp`, `multilayer_predictor.hpp`,
2M exact rows each for chain/tree/mapreduce) — but none are registered as
solvers, deliberately. Each class's own training generator
(`tools/generate_topology_training_data.py`) records its own oracle:

| Class | Oracle | `label_is_exact` |
|---|---|---|
| chain | `chain-lp` (allocation LP, `LinearChainSolver`) | `True` |
| tree | `tree-lp` (allocation LP, `TreeSolver`) | `True` |
| mapreduce | `mapreduce-closedform` (`MapReduceSolver`, thesis Prop. 4.1) | `True` |
| multilayer | `multilayer-ub` (`MultilayerSolver`, a feasible-schedule upper bound) | `False` |

Chain, tree, and MapReduce are solved **exactly** by their own closed-form/LP
solver, in well under a millisecond — an ML approximation of an answer that's
already free and provably optimal is pure downside (occasional error, no speed
or capability gain). Multilayer's label is honestly recorded as *not* the true
optimum, but it is the closed-form solver's own upper-bound output — so an ML
model trained on it would just imitate that same heuristic, again with no
value over calling `MultilayerSolver` directly. Unlike MLSD and energy, none
of these four have a genuine "no closed form / oracle too slow" gap for an ML
model to fill. The trained headers remain in the repo as evidence the
generation pipeline scales to millions of rows per class; wiring any of them
in as a selectable solver would mislead a user into an inferior answer for no
benefit.

## Bisection-width-limited MapReduce (exact LP)

`--class=mapreduce-bwidth` (HiGHS builds only) solves the *same* instance as
plain `--class=mapreduce` (§ above) — same text format, same mapper rates,
same reducer count — but drops the plain closed form's assumption that the
network can serve every reducer's read at full rate simultaneously. Instead
it caps concurrent mapper-to-reducer read channels at a `bisection <l>`
value (Berlińska & Drozdowski, technical report RA-09/09, "Dominance
Properties for Divisible MapReduce Computations", §5.2, the "third method").
Mapper load partitioning and the resulting read schedule are optimized
*jointly*, via one LP, rather than assuming a fixed formula.

Mappers still activate in order of non-decreasing rate Aᵢ (the same
dominance result the plain closed form uses). All reducers then read mapper
outputs in that same fixed order (a permutation-flowshop-style
simplification), starting as soon as both the source mapper has finished
and one of the `l` channels is free. `MapReduceBwidthSolver`
(`mapreduce/mapreduce_bwidth_solver.hpp`) builds this directly as an LP
(HiGHS) — the *un*-reduced form the report itself also gives (Eqs. 61-64),
not the further periodicity-compressed one (Eqs. 67-70): O(mr) rows instead
of O(ml), a deliberate trade since mr stays small at realistic cluster
sizes and this form is far simpler to get right.

A mapper assigned zero load still occupies its position's startup slot —
this is a property of the paper's own formulation (fixed activation order,
no "drop the slowest mapper" step like the plain closed form has), not a
bug.

```bash
dls solve --class=mapreduce-bwidth instance.txt
```
```
V 1000
startup 1
readrate 0.01
gamma0 0.1
reducers 4
reducer_startup 0.5
reducer_rate 0.2
bisection 2
mapper 0.5
mapper 0.4
mapper 0.6
```

Validated two ways against ground truth rather than only unit-tested in
isolation: with `reducers=1` the LP collapses to exactly the same linear
system the plain closed form solves (bit-for-bit matching makespan *and*
mapper loads, checked across `bisection` 1-4); with a single mapper, the
schedule length reduces to the textbook `S + A·V + γCV + t_red` formula
when `bisection=1` (fully sequential reads — the only channel-count where
that formula applies). Also checked for monotonicity: widening `bisection`
never increases the makespan, on a many-reducer instance.

Reachable from the CLI, the C/C++ library, the HTTP API
(`/api/mapreduce-bwidth`), and the Solver Studio MapReduce tab (as the
"Bisection-width limited" Variant).

## Reducer read scheduling (branch-and-price)

`--class=reducer-read` (HiGHS builds only) is a distinct problem class from
the rest of the portfolio: heterogeneous multi-channel MapReduce reducer read
scheduling. The plain MapReduce model (§ above) handles "many reducers" two
ways — aggregate them into one r-times-faster resource, or (with a channel /
bisection-width capacity limit) fix the read order by a closed-form
round-robin formula — neither actually *optimizes* who reads what and when.
This class fills that gap: reducers can have different read rates, and the
read schedule under the shared capacity limit is solved to proven optimality
rather than fixed by formula.

Solved by `ReducerReadBpSolver` (`exact/branch_and_price/`), architecturally
distinct from both `exact/enumerative/` (branches with a closed-form bound)
and `exact/milp/` (hands the whole problem to HiGHS as one MILP):

- **Column** = one reducer's complete read plan (order + exact start ticks
  for its whole required mapper set), found in isolation.
- **Master LP** (HiGHS) picks one column per reducer, minimizing the chosen
  objective, subject to the shared channel-capacity limit at every tick.
- **Pricing**: a per-reducer shortest-path DP over (mappers-read-so-far,
  tick) states, given the master's dual prices — returns every distinct-
  finish-tick candidate, not just the single best (a single best-reduced-
  cost column is only guaranteed to find *some* improving column, not every
  column an eventual integer solution needs; this was a real stall this
  solver hit during validation — see the design note in
  `tools/bp_reducer_scheduling.py`, the validated Python prototype this is a
  C++ port of).
- **Branching**: "does reducer *j* start reading mapper *i* at tick *t*" —
  the natural DP-enforceable decision, not whole-column identity (which
  pricing has no way to search around once excluded).

Two switches live in the instance itself, not as CLI flags:
- `objective makespan|balance` — minimize the latest finish (Cmax) or the
  sum of finish times (ΣCⱼ, the classic scheduling-theory alternative to
  Cmax; a reducer left waiting while others hog the channel counts against
  this objective even when Cmax is unchanged).
- per-reducer affinity — every reducer reads every mapper by default;
  listing explicit mapper indices after a `reducer` line restricts that
  reducer to only the outputs it actually needs.

```bash
dls solve --class=reducer-read instance.txt
```
```
capacity 1
objective makespan
mapper 2 2
mapper 3 1
reducer 2 0 1
reducer 2 1
```

Validated against an independent exhaustive brute-force solver on 130+
random tiny instances (Python prototype), then cross-validated again (40/40
matched) between the Python prototype and this C++ port via the actual CLI —
not just unit-tested in isolation. Reachable from the CLI, the C/C++
library, the HTTP API (`/api/reducer-read`), and the Solver Studio MapReduce
tab (as the "Reducer read" Variant).

## Reducer partitioning-skew mitigation

`--class=mapreduce-skew-static` and `--class=mapreduce-skew-dynamic`
(dependency-free — no HiGHS needed) address a gap the rest of the MapReduce
family doesn't: an unbalanced key partition, where some reducers get more
intermediate data than others (Berlińska & Drozdowski, MISTA 2013,
"Mitigating Partitioning Skew in MapReduce Computations"). Every other
MapReduce-family solver assumes each reducer gets an equal γV/r share by
construction; these two model the case where that assumption breaks, and
mitigate it two different ways.

Both share a homogeneous system (all mappers rate A, all reducers
a^sort/a^red, a one-port network with bisection-width limit l) and take the
actual (skewed) key-partition sizes as direct input — a `partition <size>`
line per part — rather than deriving them from V and γ, since real skew data
comes from an external key-frequency source, not a formula.

- **Static** (`skew_static_solver.hpp`, "fine partitioning"): splits the key
  space into k·r parts instead of r; mappers report the kr part sizes to the
  master, which solves the resulting NP-hard bin-packing assignment with an
  LPT-style heuristic (sort parts descending, greedily assign each to the
  currently least-loaded reducer — a min-heap, matching the paper's own
  kr·log(kr) + kr·log(r) master-cost formula exactly), then sends the
  assignment back before any reducer starts reading. Needs `k` and exactly
  k·r `partition` lines.
- **Dynamic** (`skew_dynamic_solver.hpp`, a simplified SkewTune): leaves the
  plain r-way partition alone. Once every reducer has finished sorting and
  at least one has also finished reducing, the master halts the remaining
  busy reducers, learns their remaining (already-sorted, linear) load, and
  rebalances it onto the idle (already-finished) reducers before resuming —
  one rebalancing event, not SkewTune's continual one. Needs k=1 (exactly r
  `partition` lines) — it moves already-computed remainders rather than
  repartitioning.

The paper describes the dynamic method's rebalancing decision only
narratively ("the master checks the scenario where the most loaded reducer
sends part of its data to the least loaded one... continues to more
processors as long as profitable"), with a closed form only for the 2-node
case. This solver derives the general n-sender/n-receiver case via the same
equal-finish-time DLT principle: for n_s busy senders (only their remaining
sum S matters) and n_r idle receivers (rate C+a^red, vs. a^red for load kept
locally), the equalized finish time is `τ = S / (n_s/a^red + n_r/(C+a^red))`
— set n_s=n_r=1 and this is exactly the paper's 2-node formula. Using every
available idle reducer as a receiver is always weakly better, so the only
real choice is how many of the busiest senders to include; this solver
evaluates all n_s=0..r1 exhaustively rather than the paper's own greedy
incremental search, guaranteeing the true best choice under this model
rather than a possibly-premature stop.

```bash
dls solve --class=mapreduce-skew-static instance.txt
dls solve --class=mapreduce-skew-dynamic instance.txt
```
```
V 1000000
mappers 20
mapper_rate 0.000001
readrate 0.00000001
gamma0 0.3
epsilon 8
bisection 4
reducers 8
sort_rate 0.0000001
reduce_rate 0.000001
master_rate 0.0000001
k 1
partition 120000
partition 30000
partition 30000
partition 30000
partition 30000
partition 30000
partition 30000
partition 0
```

Validated against hand-derivable ground truth rather than only unit-tested
in isolation: a perfectly-balanced partition makes the dynamic solver's
result match the paper's own balanced closed form (Eq. 1) exactly, with zero
busy reducers at the trigger point (nothing to rebalance); the static
solver's k=1 case matches the unmitigated baseline plus exactly the LPT
decision cost (its report/assign messages are numerically identical to the
baseline's own location round-trip at k=1, so they cancel); a hand-worked
2-reducer rebalancing example matches the derived formula bit-for-bit; and
the static solver is checked to never beat the perfectly-balanced lower
bound across a range of k (the paper's own Fig. 7 shows more bins is *not*
always better — larger k trades finer balance against more overhead, not a
monotonic improvement — so this solver doesn't assert that trade-off is
monotonic, only that it never beats the physically-impossible lower bound).
Reachable from the CLI, the C/C++ library, the HTTP API
(`/api/skew-static`, `/api/skew-dynamic`), and the Solver Studio "Skew
mitigation" tab.

## Multi-source map-phase scheduling

`--class=multisource` (HiGHS build only) is a genuinely different topology
from the rest of the library — bipartite, not a star or a relay tree (Gu,
Liao, Yang & Li, "Scheduling Divisible Loads from Multiple Input Sources in
MapReduce", IEEE CSE 2013). Every other class here has one root distributing
to m workers; this one has m storage nodes holding the input data and n ≥ m
mapper nodes (the first m of which are the *same* physical nodes as the
storage nodes, so they can process their own data for free) pulling from
potentially several storage nodes at once. Only the map phase is modeled —
the paper explicitly defers reduce-phase optimization to future work, since
"the performance of reduce phase largely depends on map phase".

Node j's finish time is `T_j = L_j·t_j + Σ_i R_ij·(t_j + w_ij)` (`L_j` only
exists for `j < m`, its own local share; `R_ij` is data pulled from storage
node `i` into mapper `j`). The optimum has all n nodes finish
simultaneously — the standard DLT argument that an idle node could always
take on more load to shorten the schedule — solved by `MultiSourceSolver`
(`mapreduce/multisource_solver.hpp`) as an LP: introduce a shared `T`, give
every node an equality row `T_j - T = 0`, plus one of two conservation forms:

- **Free placement** ("Problem A", no `storage` lines in the instance): a
  single row, `Σ L_i + Σ R_ij = S` — the LP is also free to decide how the
  data should have been placed among storage nodes in the first place; `S_i`
  is *derived* from the solution afterward.
- **Fixed supply** ("Problem B", `storage <S_i>` given for each of the m
  nodes): m rows, `L_i + Σ_j R_ij = S_i` — `S_i` is given input; only the
  read schedule (who reads what, from where) is optimized.

This is a genuine LP, not a closed form: L and R have no upper bound and the
row count (n+1 or n+m) is far below the m·(n-1)+m variable count, so
non-negativity can bind — the same reason the bisection-width MapReduce
solver needs HiGHS rather than a recurrence.

```bash
dls solve --class=multisource instance.txt
```
```
S 1000
m 3
n 5
rate 1.0
rate 2.0
rate 1.5
rate 3.0
rate 0.8
transfer 0 1 0.1
transfer 0 2 0.2
transfer 0 3 0.15
transfer 0 4 0.3
transfer 1 0 0.1
transfer 1 2 0.25
transfer 1 3 0.1
transfer 1 4 0.2
transfer 2 0 0.2
transfer 2 1 0.3
transfer 2 3 0.05
transfer 2 4 0.1
```

Validated against hand-derivable ground truth: with `m = n` and homogeneous
rates, the LP exactly reproduces the textbook `T = S/m·t` with zero
transfers (transferring only adds cost when every node computes at the same
rate); with free transfer (`w=0`) and heterogeneous rates, it matches the
classic two-processor DLT equal-finish split by hand; solving free-placement
("Problem A") and then feeding its *own derived* `S_i` back in as fixed
supply ("Problem B") reproduces the identical makespan and load split
bit-for-bit — a strong self-consistency check unique to having both
variants implemented together; and adding an extra mapper node (same total
load) never increases the makespan, matching the paper's own Fig. 4/5
finding that the optimum tends to use every available node.

Reachable from the CLI, the C/C++ library, the HTTP API
(`/api/multisource`), and the Solver Studio "Multi-source" tab.
