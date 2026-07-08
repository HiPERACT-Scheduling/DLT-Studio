# User guide

How to use the DLS library: define a problem, pick a solver, read the result.
For the model and symbols see [MODEL.md](MODEL.md); for building see
[BUILD.md](BUILD.md). All types are in `namespace dls`; includes are relative to
the repository root (the build puts it on the include path).

---

## 1. What the library solves

Four problem families:

- **Single-load DLS** (`core/`, `heuristics/`, `exact/`): one divisible load `V`
  distributed over `N` processors in a single-port star, minimizing the makespan
  `Cmax`. Solvers: a genetic heuristic, a fast best-rate constructive heuristic,
  an ML-assisted solver (`ml-makespan`, two-stage GBM + GA), an exact
  branch-and-bound, and an exact MILP.
- **MLSD** (`mlsd/`): `n` divisible *tasks* over `m` processors. Solvers: an exact
  enumerator and a genetic heuristic.
- **MapReduce** (`mapreduce/`): a map phase feeding a reduce phase (precedence).
  Solver: an exact closed-form scheduler (§9).
- **Linear daisy chain** (`topology/`): a non-star topology — load enters at one
  end and is forwarded down the chain. Solver: an exact allocation LP (§9b).
- **Multi-level tree** (`topology/`): generalizes the star and the chain — load
  flows from the root down a tree. Solver: an exact allocation LP (§9c).
- **General graph / mesh** (`topology/`): an arbitrary connected network. Solver:
  picks the best spanning arborescence and solves its tree LP (§9d).

Every solver shares the same shape: build an *instance*, construct a *solver*
with its parameters, call `solve(...)`, read a *solution*.

---

## 2. Quick start (single-load)

```cpp
#include "heuristics/ga/ga_solver.hpp"
#include <cstdio>
using namespace dls;

int main() {
    // 3 processors {S, C, A, B}: comm-startup, comm-rate, compute-rate, memory.
    DLSInstance inst({
        {0.0, 0.20, 0.30, 1e9},
        {0.1, 0.15, 0.25, 1e9},
        {0.2, 0.25, 0.40, 1e9},
    }, /*V=*/100000.0);

    GAParams p;                       // algorithm parameters (sensible defaults)
    p.installments = 5;               // chunks the load is split into
    GASolver solver(p);

    SolverConfig cfg;                 // runtime config
    cfg.seed = 42;                    // set for reproducibility (omit for random)

    DLSSolution s = solver.solve(inst, cfg);
    if (s.feasible()) {
        std::printf("makespan = %.3f\n", s.makespan);
        for (const LoadFragment& f : s.fragments)
            std::printf("  P%d  load=%.2f\n", f.processorId, f.loadSize);
    }
}
```

Build it (dependency-free): `cmake -B build && cmake --build build`. To compile
a standalone file: `g++ -std=c++20 -I. yourfile.cpp -o app`.

---

## 3. Defining a problem (`DLSInstance`)

A `Processor` carries these fields (all default to neutral values; set only what
you need):

| Field | Symbol | Meaning |
|---|---|---|
| `commStartup` | S | fixed latency per transfer |
| `commRate` | C | comm time per unit load |
| `computeRate` | A | compute time per unit load |
| `memoryLimit` | B | max load per installment (use a large value for "unbounded") |
| `computeStartup` | p | computation startup (default 0) |
| `releaseTime` | r | processor unavailable before this time (default 0) |
| `deadline` | d | completion deadline (≤0 = none, default) |
| `fixedCost` | f | fixed cost of using the processor (default 0) |
| `linearCost` | l | cost per unit of load (default 0) |
| `computePieces` | — | optional convex pieces `{intercept, slope}` for a piecewise processing time (empty = classic `p + A·α`) |

```cpp
Processor q;                 // set fields individually when you use the extras
q.commStartup = 0.1; q.commRate = 0.2; q.computeRate = 0.3;
q.memoryLimit = 5000;        // this processor can hold at most 5000 per installment
q.releaseTime = 10;          // not available before t=10
q.deadline    = 200;         // must finish by t=200
DLSInstance inst({q, /* more processors */}, /*V=*/100000.0);
inst.setResultFraction(0.0); // β: results returned to the master (0 = none)
```

`inst.validate(&error)` checks well-formedness; solvers call it for you.

---

## 4. The three single-load solvers

All implement the same contract: `solve(const DLSInstance&, const SolverConfig&)
-> DLSSolution`, with `name()` and `category()` (`Exact`/`Heuristic`). Not sure
which to pick? Use the **`auto`** meta-solver (it dispatches by instance features
— see [CHOOSING.md](CHOOSING.md)), e.g. `dls solve --solver=auto instance.txt`.

### Genetic algorithm (heuristic, scales to large instances)
```cpp
#include "heuristics/ga/ga_solver.hpp"
GAParams p;
p.populationSize     = 12;
p.maxGenerations     = 200;
p.noImprovementLimit = 60;
p.installments       = 6;
p.selection  = Selection::Tournament;  // or Roulette (default); tournamentSize=...
p.crossover  = Crossover::TwoPoint;    // or SinglePoint
p.mutation   = Mutation::Swap;         // or PerGene
DLSSolution s = GASolver(p).solve(inst, cfg);
```

### Best rate (heuristic, fast constructive — recommended default)
A greedy that builds the activation sequence by always sending the next chunk to
the processor with the best current processing rate, then optimizes the chunk
sizes with the shared LP. Near-optimal at very low cost (Berlińska §3.5.4), and
deterministic. A good first solve and a strong warm bound for the exact solvers.
```cpp
#include "heuristics/best_rate/best_rate_solver.hpp"
BestRateParams b;
b.chunkDivisor = 1;          // the x in BRx: chunk size = Bᵢ/x (1 = no overlap, recommended)
DLSSolution s = BestRateSolver(b).solve(inst, cfg);
```
Compare against the lower bound to gauge quality:
```cpp
#include "core/bounds.hpp"
double lb = divisibleLoadLowerBound(inst);          // LB ≤ Cmax* (Berlińska eq. 3.23)
double gap = (s.makespan - lb) / lb;                // relative distance from the bound
```

### Online PSR + GSS/SSC (heuristic, LP-free, memory-aware)
The online simulation heuristics of Marszałkowski (2020, §6.2) — the only solver
that uses **no LP**. It simulates the single-port master dispatching chunks to
heterogeneous, memory-limited workers and reads the makespan straight off the
schedule, so it is very fast (O(V/ρ · log m) chunks) and well suited to large,
memory-bound instances. Two knobs: a **Processor Sorting Rule** (which ready
worker to feed next) and a **chunk rule** — `SSC` (chunk = the worker's RAM ρ,
never out of core) or `GSS` (shrinking chunks `min{Vˡ, max{minChunk, min{Vˡ/m, ρ}}}`
that even out finish times). `Psr::All` ("super") runs every rule and keeps the
best schedule.
```cpp
#include "heuristics/online/online_solver.hpp"
OnlineParams o;
o.chunk = ChunkRule::GSS;     // or ChunkRule::SSC
o.psr   = Psr::All;           // super: best of compute/comm/startup/memory/energy rules
DLSSolution s = OnlineSolver(o).solve(inst, cfg);   // Feasible; s.makespan, s.energy filled
```
Its fixed-chunk makespan upper-bounds the LP optimum of the same activation order
(re-evaluate `s.sequence` with the LP to tighten). CLI: `--solver=online
--chunk=ssc|gss --psr=compute|comm|startup|memory|energy|super`.

### Single round (closed form, exact for no-startup case)
The classical divisible-load algorithm: one chunk per processor, all finishing
together, activated in non-decreasing `Cᵢ` order. O(m log m), no LP. Exact when
`Sᵢ = 0` (matches the B&B), a fast high-quality heuristic otherwise.
```cpp
#include "heuristics/single_round/single_round_solver.hpp"
DLSSolution s = SingleRoundSolver().solve(inst, cfg);
```

### ML-assisted solver — `ml-makespan` (GBM prediction + GA schedule)

A two-stage solver that exposes the **ML-predicted optimal makespan** alongside
the actual schedule, enabling direct comparison of ML accuracy against heuristics
and exact solvers at runtime.

- **Stage 1** — a GBM regressor (trained on `log(T*)`) predicts the optimal
  makespan from 16 instance features in < 1 µs. The result is exposed via
  `MlSolver::predictedMakespan()` and in the CLI/JSON response as
  `"predictedMakespan"`.
- **Stage 2** — the best applicable heuristic builds the schedule:
  - No `Sᵢ`, no `Cᵢ`, no memory cap → `BestRateSolver` (LP-optimal)
  - `Sᵢ > 0`, `Cᵢ = 0`, no memory cap → `FptasOptTSolver` (ε = 0.05)
  - General (comm costs or memory caps present) → `GASolver`

```cpp
#include "heuristics/ml/ml_solver.hpp"
MlSolverParams p; p.maxInstallments = 5;
MlSolver solver(p);
DLSSolution s = solver.solve(inst, cfg);
double predicted = solver.predictedMakespan();   // ML's estimate of T*
double actual    = s.makespan;                   // schedule makespan (GA/FPTAS/BestRate)
```

The model ships as a closed-form placeholder and is overwritten by
`tools/train_makespan_predictor.py` after the training pipeline runs:
```bash
bash tools/retrain.sh --n 50000
```

### Branch-and-bound (exact, small/moderate instances)
```cpp
#include "exact/enumerative/exact_solver.hpp"
ExactParams e;
e.maxInstallments = 5;       // search schedules of length <= 5
e.allowRepeats    = true;    // true: multi-installment; false: each processor once
e.nodeBudget      = 0;       // 0 = exhaustive (proven Optimal); else best-so-far
e.warmStart       = true;    // seed the incumbent from heuristics (prunes harder, same optimum);
                             // set false for the original B&B (CLI: --warm=0)
DLSSolution s = ExactSolver(e).solve(inst, cfg);   // status Optimal when proven
```

### MILP (exact, HiGHS only)
Requires the HiGHS build (`-DDLS_WITH_HIGHS=ON`, see BUILD.md). `MilpSolver` is
single-installment; `MultiMilpSolver` (`milp-multi`) allows a processor to repeat
and is an independent exact cross-check of the B&B (exact when memory is ample).
```cpp
#include "exact/milp/milp_solver.hpp"
#include "exact/milp/multi_milp_solver.hpp"
DLSSolution s = MilpSolver(MilpParams{/*maxInstallments=*/5}).solve(inst, cfg);
DLSSolution m = MultiMilpSolver(MultiMilpParams{/*maxInstallments=*/4}).solve(inst, cfg);
```

For a **homogeneous** system (identical processors) the optimal single-round
makespan and the diminishing-returns processor count `k*` are closed-form:
```cpp
#include "core/homogeneous.hpp"
HomogeneousOptimum o = homogeneousOptimalProcessors(/*S=*/0.2,/*C=*/0.15,/*A=*/0.25,/*V=*/5000, /*m=*/16);
// o.kStar = how many of the 16 machines are worth using; o.makespan = its T.
```

---

## 5. Reading a `DLSSolution`

```cpp
s.status      // SolveStatus: Optimal | Feasible | Infeasible | Unbounded | Failure
s.feasible()  // true if Optimal or Feasible
s.makespan    // Cmax
s.cost        // G (schedule cost), if a cost model is set
s.sequence    // processor activation order (0-based ids)
s.fragments   // per-installment {processorId, loadSize, commStart, ...}
s.usedSeed    // the RNG seed actually used (for reproducing an unseeded run)
s.iterations  // GA generations / B&B nodes explored
s.conservesLoad(inst.totalLoad());  // sanity check: Σ loadSize ≈ V
```

---

## 6. Extra conditions and objectives

These work across the GA, B&B, and MILP (via the same `DLSInstance` / params).

**Availability & startup** — set `computeStartup`, `releaseTime`, `deadline` on
the processors (section 3). They only add constraints when non-default.

**Memory limits** — set `memoryLimit` (B). A processor never receives more than
`B` in a single installment.

**Memory hierarchy (piecewise processing)** — set `computePieces` to model a
processor whose computation slows once its load spills out of fast core RAM:
```cpp
Processor q; q.commRate = 0.0; q.memoryLimit = 1e9;
q.computePieces = {{0.0, 1.0}, {-10.0, 3.0}};  // core: q ;  disk: 3q-10 ; swap at q=5
```
The processing time becomes `max` over the pieces (`maxₖ aₖ + bₖ·load`), a convex
penalty for large loads. Slopes must be `≥ 0`; the classic single-piece `p + A·α`
is the default. Works on all solvers — the GA, B&B, both LP backends, and the
MILP. The hard cap `memoryLimit` can be set alongside it.

**Cost & bi-criteria** — set `fixedCost`/`linearCost` to get a cost `G`, then:
```cpp
ExactParams e; e.maxInstallments = 5;
e.costLimit = 24.5;                  // minimize Cmax subject to  G <= 24.5
DLSSolution s = ExactSolver(e).solve(inst, cfg);   // s.cost <= 24.5
```
The reverse direction (minimize `G` subject to `Cmax <= C̄`) is available on the
MILP (`MilpParams.minimizeCost = true; m.makespanLimit = C̄;`) and on the
evaluator (section 8).

**Energy (time–energy trade-off)** — model energy as a second criterion
(Marszałkowski 2020). Give each processor its non-computing **state powers** and
a **piecewise running energy** `ε(α)`, and optionally the master's power:
```cpp
Processor p; p.commStartup = 1; p.commRate = 0.1; p.computeRate = 0.2; p.memoryLimit = 1e18;
p.powerIdle = 3; p.powerStartup = 5; p.powerNetwork = 10;   // P^I / P^S / P^N
p.energyPieces = {{0.0, 0.4}, {-7500.0*21.6, 22.0}};        // in-core k₁ ; out-of-core k₂ (meet at ρ)
DLSInstance inst({p}, 100.0);
inst.setOriginatorPowerNetwork(8); inst.setOriginatorPowerIdle(4);   // master M₀ (optional)
```
Setting any of these turns on the **energy model** (`inst.usesEnergyModel()`);
otherwise behaviour is byte-identical to before. Every solved `DLSSolution` then
reports `sol.energy` (total four-state energy `E = E₀ + Σ(E^S+E^N+E^R+E^I)`), and
`scheduleEnergy(inst, sol)` recomputes it in closed form. The running energy is a
convex piecewise penalty just like `computePieces` — the proportional model `l·α`
(empty `energyPieces`) badly underestimates out-of-core work, which is the whole
point of `ε(α)`. With the energy model on, the **`MinCost` objective minimizes
energy** (not the linear cost), and combining it with a makespan limit traces the
time–energy Pareto front:
```cpp
EvaluatorConfig c; c.objective = EvalObjective::MinCost;  // = minimize energy here
c.makespanLimit = 41.5;                                    // min E subject to Cmax <= 41.5
DLSSolution s = SimplexScheduleEvaluator().evaluate(inst, {0}, c);   // s.energy is the optimum
```
The **MILP** (`exact-milp`) optimizes energy over the *whole* problem (it picks the
sequence too): with an energy model, `MilpParams.minimizeCost` minimizes total `E`
(subject to `makespanLimit = C̄`), and `costLimit = Ē` becomes an energy cap
(minimize `Cmax` s.t. `E ≤ Ē`) — the exact time–energy Pareto solver. Single-
installment with affine processing time (piecewise *energy* is fine; piecewise
*time* + energy objective is not yet supported and returns `Failure`).
File format (section 11): `power P^I P^S P^N`, `energy l,k l,k …`, `originator P^N P^I`.

**Result return** — `inst.setResultFraction(β)` adds a result-collection phase
(`Cmax` grows accordingly).

**OptV — maximize load within a deadline** (the dual of minimizing the makespan):
given a deadline `T`, find how much load can be processed. Exact over the same
sequence space as the makespan B&B; the instance's `totalLoad` is ignored.
```cpp
#include "exact/optv/optv_solver.hpp"
OptVParams v; v.deadline = 41.5;     // T (required, finite)
v.maxInstallments = 3; v.allowRepeats = false;
DLSSolution s = OptVSolver(v).solve(inst, cfg);
double maxLoad = s.totalAssignedLoad();   // most load with s.makespan <= 41.5
```
For a single fixed sequence, score it directly with the evaluator (section 8)
using `EvalObjective::MaxLoad` and `makespanLimit = T`.

**OptV with a guarantee (FPTAS, DLS{Cᵢ=0})** — for the infinite-bandwidth case
(`commRate = 0`, only startups `Sᵢ` and compute `Aᵢ`), an approximation scheme
returns a load within a chosen factor of optimal: `V_FPTAS ≥ (1−ε)·V_OPT`.
```cpp
#include "heuristics/fptas/fptas_optv_solver.hpp"
FptasOptVParams f; f.deadline = 20.0; f.epsilon = 0.05;   // 0 < ε < 1
DLSSolution s = FptasOptVSolver(f).solve(inst, cfg);      // inst must have Cᵢ=0
double v = s.totalAssignedLoad();                          // ≥ 0.95·V_OPT
```
The dual `FptasOptTSolver` minimizes the *time* to process the instance's load
`V` with `T_FPTAS ≤ (1+ε)·T_OPT` (also DLS{Cᵢ=0}):
```cpp
#include "heuristics/fptas/fptas_optt_solver.hpp"
DLSSolution t = FptasOptTSolver({/*epsilon=*/0.05}).solve(inst, cfg);
double minTime = t.makespan;                              // ≤ 1.05·T_OPT
```
**Auto-ε** — both FPTAS solvers can derive ε automatically from instance features
(balancing approximation quality against the scheduler's own heuristic error):
```cpp
FptasOptVParams f; f.deadline = 20.0; f.autoEpsilon = true;   // ignores f.epsilon
FptasOptTParams t; t.autoEpsilon = true;
// the computed ε is reported in the CLI output and in FptasOptTSolver::computedEpsilon()
```

---

## 7. Choosing the LP backend (CSimplex vs HiGHS)

The GA and B&B solve an LP per candidate. Default is the built-in CSimplex
(dependency-free). With the HiGHS build you can switch:
```cpp
GAParams p;  p.evaluatorBackend  = "highs";   // or "simplex" (default)
ExactParams e; e.evaluatorBackend = "highs";
```
Selecting `"highs"` in a dependency-free build returns `Failure` (it isn't
compiled in).

---

## 8. Advanced: evaluating a fixed sequence

To score one activation sequence directly (the LP the solvers call internally):
```cpp
#include "core/evaluator_factory.hpp"
auto ev = makeScheduleEvaluator("simplex");        // or "highs"
EvaluatorConfig c;
c.costLimit     = 24.5;                             // optional
c.objective     = EvalObjective::MinCost;           // minimize G instead of Cmax
c.makespanLimit = 41.5;                             //   subject to Cmax <= 41.5
DLSSolution s = ev->evaluate(inst, /*sequence=*/{0,1,2}, c);  // 0-based proc ids
```
The objective can also be `EvalObjective::MaxLoad` — maximize the total load with
`Cmax <= makespanLimit` (the OptV building block; `totalLoad` is ignored and
`makespanLimit` is required).

---

## 9. MapReduce (map + reduce, precedence)

A third problem class (`mapreduce/`): a divisible input `V` is split across `m`
mappers; each emits `γ₀·αᵢ` intermediate bytes that `r` reducers read (rate `C`)
and reduce (time `s_red + a_red·x·log₂x`). The optimal schedule is closed-form
(Berlińska §4.4): activate mappers in increasing rate `Aᵢ`, FIFO reads.
```cpp
#include "mapreduce/mapreduce_solver.hpp"
using namespace dls;

MapReduceInstance inst({1.0, 2.0, 3.0}, /*V=*/100.0);   // 3 mapper rates Aᵢ
inst.setStartup(1.0);          // S: per-processor code-load startup
inst.setReadRate(1.0);         // C: reducer read rate
inst.setResultFraction(0.5);   // γ₀: intermediate output / input ratio
inst.setNumReducers(4);        // r
inst.setReducerStartup(0.1);   // s_red
inst.setReducerRate(1e-6);     // a_red (× x·log₂x)

MapReduceSolution s = MapReduceSolver().solve(inst);
// s.makespan = T; s.mapperLoads[k] = αᵢ for s.mapperOrder[k]; s.reducerTime = t_red.
// Over-provisioned instances drop the slowest mappers automatically.
```

**Multilayer pipelines** (`mapreduce/multilayer_solver.hpp`, ch. 5) generalize to
`R` reducer layers with a sorting cost `τ_p(x) = a·x·log₂x`. The homogeneous
closed form sums sequential phases:
```cpp
#include "mapreduce/multilayer_solver.hpp"
MultilayerInstance ml;
ml.totalLoad = 1e6; ml.numMappers = 100; ml.mapperRate = 1e-7;
ml.readRate = 1e-8; ml.mapperFraction = 0.5; ml.bisectionWidth = 50;
ml.layers = { {/*reducers=*/50, /*s_red=*/0.1, /*a_red=*/1e-9, /*γ=*/0.5},
              {/*reducers=*/10, 0.1, 1e-9, 1.0} };
MultilayerSolution s = MultilayerSolver().solve(ml);   // s.makespan, per-layer breakdown
```
`xLogXConvexPieces(a, maxX)` returns the thesis's convex linearization of
`a·x·log₂x` as `ComputePiece`s — usable as a processor's `computePieces` to give
any DLS solver an `x·log x` cost.

## 9b. Linear daisy chain (non-star topology)

`topology/linear_chain.hpp` solves divisible load on a chain `P0 — P1 — …`: the
load enters at `P0`, each processor keeps a share and forwards the rest. The
optimum is a small LP (CSimplex, dependency-free).
```cpp
#include "topology/linear_chain.hpp"
using namespace dls;
// {computeRate Aᵢ, linkRate Cᵢ}; Cᵢ is the link feeding Pᵢ (C₀ unused).
LinearChainInstance chain({ {1.0, 0.0}, {1.0, 1.0}, {1.0, 1.0} }, /*V=*/30.0);
ChainSolution s = LinearChainSolver().solve(chain);
// s.makespan; s.loads[i] = αᵢ kept by each processor.
```

## 9c. Multi-level tree (generalizes star + chain)

`topology/tree.hpp` solves divisible load on a tree: load enters at the root,
each node keeps a share and forwards each child's subtree-load down. A star is a
1-level tree; a chain is a path. Exact LP (CSimplex).
```cpp
#include "topology/tree.hpp"
using namespace dls;
// {computeRate Aᵢ, linkRate Cᵢ, parent}; node 0 = root (parent -1), parents < index.
TreeInstance tree({ {0.3,0.0,-1}, {0.25,0.2,0}, {0.4,0.15,0}, {0.5,0.3,1} }, /*V=*/1000.0);
TreeSolution s = TreeSolver().solve(tree);   // s.makespan; s.loads[i] = αᵢ
```

## 9d. General graph / mesh

`topology/graph.hpp` solves divisible load on an arbitrary connected graph: the
optimal distribution uses a spanning arborescence rooted at the source, so it
picks the best one (enumerated for small graphs) and solves its tree LP.
```cpp
#include "topology/graph.hpp"
using namespace dls;
GraphInstance g(/*computeRates=*/{1.0, 1.0, 1.0},
                /*edges {u,v,rate}=*/{ {0,1,0.1}, {1,2,0.1}, {0,2,10.0} }, /*V=*/30.0);
GraphSolution s = GraphSolver().solve(g);
// s.makespan; s.loads[i]; s.parent[i] = the chosen arborescence parent of node i.
```
Node 0 is the source. Enumeration is capped (`GraphSolver(maxArborescences)`), so
it targets small graphs / validation.

## 10. MLSD (multiple loads)

```cpp
#include "mlsd/mlsd_solver.hpp"
#include "mlsd/mlsd_ga_solver.hpp"
using namespace dls;

Processor p; p.commStartup=1; p.commRate=1; p.computeRate=1; p.memoryLimit=1e18;
// 2 tasks {size, β}, 3 processors:
MlsdInstance inst({ {32.0, 0.0}, {2.0, 0.0} }, {p, p, p});

MlsdSolution exact = MlsdSolver().solve(inst);               // exact (small instances)
MlsdSolution heur  = MlsdGaSolver({MlsdGaSolver::Params{}}).solve(inst);  // GA heuristic

std::printf("exact Cmax = %.3f\n", exact.makespan);
// exact.taskOrder is the chosen task order; exact.loads[l][k] the per-task loads.
```
The MLSD evaluator also takes a backend: `MlsdScheduleEvaluator("highs")`. Set a
task's `resultFraction` for result return.

**ML-assisted MLSD solver** (`ml-mlsd`) — two-stage: GBM predicts `log(Cmax*)`,
then `MlsdGaSolver` builds the schedule. Exposes `predictedMakespan()` for
ML-accuracy benchmarking:
```cpp
#include "heuristics/ml/ml_mlsd_solver.hpp"
MlMlsdSolver ml;
MlsdSolution s = ml.solve(inst);
double predicted = ml.predictedMakespan();   // GBM estimate of Cmax*
double actual    = s.makespan;               // GA schedule makespan
```

**Instance feature vector** (`MlsdInstanceFeatures`) — a 12-feature fixed-size
vector (nTasks, nProcs, load statistics, memory ratio, processor heterogeneity)
usable for benchmarking or custom ML models:
```cpp
#include "mlsd/mlsd_instance_features.hpp"
MlsdInstanceFeatures f = computeMlsdFeatures(inst);
// f.nTasks, f.nProcs, f.meanV, f.cvV, f.maxMinV, f.totalLoadPerProc,
// f.memoryRatio, f.meanA, f.heteroA, f.speedupA, f.hasStartups, f.hasCommCost
```

CLI: `--solver=mlsd-exact|mlsd-ga|ml-mlsd` with `--class=mlsd`.

---

## 11. Command-line tools

### `dls` — uniform portfolio runner
Selects any registered single-load solver by name and runs it through the shared
contract (see `core/solver_registry.hpp`).
```bash
./build/bin/dls list                       # list solvers + problem classes
./build/bin/dls solve --solver=exact --installments=3 instance.txt
./build/bin/dls solve --class=mlsd mlsd.txt           # other problem classes
./build/bin/dls solve --class=mapreduce mr.txt
./build/bin/dls solve --class=multilayer ml.txt
./build/bin/dls show instance.txt          # re-emit the parsed instance (canonical form)
```
`--class` selects the problem class (default `dls`, the single-load portfolio
above) — including the non-star topologies (§9b–§9d). Each class has its own file
format (see `cli/class_io.hpp`):
- `--class=mlsd` — `task <size> [β]` and `proc <S> <C> <A> <B>` lines; pick the
  solver with `--solver=mlsd-exact` (default), `mlsd-ga`, or `mlsd-milp`
  (independent exact MILP, HiGHS build, β=0).
- `--class=mapreduce` — `V`, `startup`, `readrate`, `gamma0`, `reducers`,
  `reducer_startup`, `reducer_rate`, and `mapper <A>` rows.
- `--class=multilayer` — `V`, `mappers`, `mapper_rate`, `startup`, `readrate`,
  `gamma0`, `bisection`, and `layer <count> <s_red> <a_red> <γ>` rows.
- `--class=chain` — `V` and `node <A> <C>` rows (chain order; P0's `C` unused).
- `--class=tree` — `V` and `node <A> <C> <parent>` rows (node 0 = root, parent -1).
- `--class=graph` — `V`, `node <A>` rows (node 0 = source) and `edge <u> <v> <rate>`.
The instance file (see `core/instance_io.hpp`) exposes the **full** model. A
`V <value>` line sets the load, an optional `beta <value>` line the result-return
fraction, and each processor is a row `S C A B [p r d f l]` — the four required
scalars plus optional compute-startup `p`, release `r`, deadline `d`, fixed cost
`f`, and linear cost `l`. A `pieces a,b a,b …` line attaches convex compute
pieces (memory hierarchy) to the previous processor. For the **energy model**,
`power P^I P^S P^N` attaches the previous processor's state powers, `energy l,k
l,k …` its convex running-energy pieces, and a top-level `originator P^N P^I` sets
the master's power. `#` starts a comment, and a plain `S C A B` row (the legacy
format) still works:
```
V 100000
beta 0.25
originator 8 4
# S    C     A     B      p   r   d    f    l
0.0  0.20  0.30  1e9    5   0   0
0.1  0.15  0.25  1e9
power 3 5 10
energy 0,0.4 -162000,22
0.0  0.0   0.0   1e9    0   0   0    1.0  0.5
pieces 0,0.2 -30,0.7
```
Options (each used by the solvers / fields it applies to): `--load --beta
--installments --repeats --backend=simplex|highs --deadline --epsilon --nodes
--cost --seed --chunk=ssc|gss --psr=compute|comm|startup|memory|energy|super`. `exact-milp` is listed only in the HiGHS build. `solve` prints
status, makespan, load, cost, **energy** (when an energy model is present), the
lower bound, the schedule, and the wall time.

### `dls-bench` — solver comparison
Generates random instances (`A,C,S ~ U[0,1]`) and compares the makespan solvers
on quality (gap from the lower bound) vs. run time — the thesis's quality/time
Pareto view.
```bash
./build/bin/dls-bench --procs=4 --instances=20 --seed=7 [--memory=F] [--installments=L] [--solvers=a,b,c]
```
`--memory=F` sets `Bᵢ = F·V` (small `F` ⇒ memory-limited, multi-installment);
`--homogeneous` generates canonical identical-processor instances. These are
tagged **regime 1** (closed-form-tractable): the closed-form `single-round` *is*
the proven optimum, so the report shows a proven reference and `gap->opt` **without
running the B&B** — the cheap "validate on tractable cases first" baseline.
Heterogeneous instances are **regime 2** (proven only by an exact solver).
The header reports the regime split and how many instances have a **proven-optimal** reference, and
the table shows `gap->opt%` (rigorous distance from the proven optimum, blank
when not proven), `gap->best%` (vs the virtual best) and `gap->LB%` (the tight
lower-bound certificate), alongside time and makespan.

**Isoefficiency / isoenergy maps** (`dls-bench map`, Marszałkowski 2020) — sweep
two system parameters over a grid and emit the performance metric at each point
as CSV; plotting its contours gives constant-time (isoefficiency) and constant-
energy (isoenergy) maps.
```bash
# makespan over processor count × startup latency
./build/bin/dls-bench map --x=procs --y=startup --xrange=1:16:16 --yrange=0:2:9 \
                          --solver=single-round --C=0.1 --A=0.2 --load=100 > iso.csv
# energy over compute rate × running-energy slope k (single-round)
./build/bin/dls-bench map --x=compute --y=k --xrange=0.1:1:10 --yrange=1:20:20 \
                          --metric=energy --solver=single-round --load=100 > isoenergy.csv
```
Axes: `procs|load|comm|compute|startup|pnet|pidle|k`. `--metric=makespan|energy`,
`--solver=NAME`, and base parameters `--C --A --S --k --pidle --pstartup --pnet
--orig-pnet --orig-pidle --memory`. Each grid point is a homogeneous instance
solved by the chosen solver; cells are `inf` where infeasible. Feed the CSV to any
contour plotter (gnuplot `set contour`, matplotlib `contour`).

**Certify an external (e.g. learned) partitioner** against the proven optimum:
```bash
./build/bin/dls-bench labels --procs=4 --instances=100 --seed=7 > labels.csv  # ground truth
./build/bin/dls-bench score --labels labels.csv --policy policy.csv           # report the gaps
```
`labels.csv` is `instance,opt_makespan,proven,lower_bound`; `policy.csv` is the
external model's makespan per instance (one per line, in the same order). `score`
prints the policy's average/max gap to the **proven optimum** and a **certificate**
vs the lower bound: since `LB ≤ optimum`, `(M−LB)/LB` is a valid *upper bound* on
the true optimality gap, so the policy is "certified ≤ Z% above optimal" (avg and
worst-case) — a guarantee that holds **even where exact solving is intractable**.
The benchmark table's `gap->LB%` and its per-solver worst-case footer carry the
same certificate semantics. In code: `generateLabeledInstances(cfg)` /
`scorePolicy(labels, makespans)` (`maxGapToLB` = the worst-case certified bound).

### `ga` — legacy GA driver
The GA also ships as a CLI that reads the legacy three-file input:
```bash
./build/bin/ga [--evaluator=simplex|highs] prog_par.txt proc_par.txt ga_par.txt
```
(see `heuristics/ga/` for the sample input files). It prints the best makespan
and the wall-clock time.

---

## 12. Reproducibility

- Set `SolverConfig.seed` for a deterministic GA run; the same seed reproduces
  the same result. Leave it unset for a random run — the seed used is reported in
  `DLSSolution.usedSeed`, so you can replay it.
- The exact solvers (B&B, MILP) are deterministic; their optimum does not depend
  on the seed.
