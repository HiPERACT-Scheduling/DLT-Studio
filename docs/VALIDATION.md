# Validation

Model validation is the top priority of this library: every solver/model is
backed by tests that confirm **accuracy**, not just that it runs. All tests live
in `tests/tests.cpp` and run via `ctest` (see [BUILD.md](BUILD.md)). Symbols are
as defined in [MODEL.md](MODEL.md).

Coverage: dependency-free build **3189 assertions / 162 test cases**; HiGHS build
**3305 / 187** (adds the HiGHS LP backend + MILP cross-checks).

## 1. Analytic ground truth (strongest check)
Exact results are asserted against hand-derived values:

| Scenario | Expected Cmax | Result |
|---|---|---|
| 1 processor, S=0, one installment | `(C+A)·V` | `230` ✓ |
| 2 identical processors, S=0, single-installment | `(C+A)²·V / (C+2A)` | `180` ✓ |

These pin the model to closed-form correctness, independent of any solver.

## 2. Cross-validation (independent methods must agree)
- **Exact MILP == Exact B&B (single-installment)** — across sizes and a binding-B
  case (`120297`, `2312.26`, `836.596`, …).
- **Multi-installment MILP == Exact B&B (multi-installment)** — `milp-multi`, an
  independent MILP of the carried-load (no same-processor comm/compute overlap)
  model, equals `ExactSolver(allowRepeats=true)` on ample-memory instances, and
  is ≤ the single-installment optimum when pipelining helps.
- **Brute force == Exact B&B** on small instances (validates the B&B pruning).
- **Dual bisection == Exact B&B** — `exact-dual` reaches the makespan optimum by
  a different paradigm (bisect the deadline, solve the OptV decision problem each
  step) and agrees with the primal B&B (and the single-processor closed form).
- **Exact B&B (multi) ≤ GA** — the exact optimum bounds the heuristic.
- **HiGHS LP backend == CSimplex LP** — the two evaluator backends agree.

## 2b. Homogeneous closed form (optimal processor count)
- **Analytic anchors**: `homogeneousSingleRound` reproduces `230` (one proc),
  `180` (two identical), and `5` (the `C=0` parallel-compute branch).
- **Diminishing returns**: makespan is non-increasing in the processor count up
  to `k*`, beyond which the marginal load goes negative.
- **Cross-method**: `homogeneousOptimalProcessors` equals `SingleRoundSolver` and
  the exact B&B on identical-processor instances, with the same `k*`.

## 3. Worst-case heterogeneity (thesis)
The thesis proves naive strategies are arbitrarily bad in heterogeneous systems;
these tests assert the exact solver achieves the optimal value, validating that
each combinatorial aspect matters:
- **Sizing**: optimal balanced split (`≈909.09`) beats equal partitioning (`5000`).
- **Selection**: a uselessly high-startup processor is dropped (`Cmax=20`, not `≥100`).
- **Ordering**: the better activation order is chosen (`≈1.913`, not `11`).

## 4. Boundary conditions
- **Memory B — sub-threshold**: when `Σ Bᵢ < V`, single-installment is
  `Infeasible` while multi-installment is `Feasible` (installments add capacity
  over time). Verified.
- **Memory B — at threshold**: `Σ Bᵢ == V` is feasible and tight.
- **Memory B — binding**: every installment's load satisfies `loadSize ≤ Bᵢ`.
- **Startup S**: with large `S` the B&B prunes heavily (`3` nodes vs `120` at
  `S=0`), confirming the single-port startup bound engages.
- **Warm-started B&B**: the incumbent is seeded from fast heuristics
  (single-round + best-rate, re-scored under the solver's own cost/availability
  limits and the `allowRepeats` policy), so the bound prunes from the first node
  — an N=5 heterogeneous instance drops from 368 to ~153 nodes — while the proven
  optimum is unchanged (seeding only lowers the incumbent). It is a switch
  (`ExactParams.warmStart`, default on; `--warm=0` for the original B&B): both
  modes return the identical optimum, and warm-start never explores more nodes.
- **Availability (p, r, d)**: hand-derived single-processor checks —
  compute startup `p=5 ⇒ Cmax=15`; release `r=7` delays finish to `17`; a
  deadline `d=4` shifts load off the constrained processor (`Cmax 20/3 ⇒ 12`).
  Verified for both the CSimplex and HiGHS LP backends.
- **Cost / bi-criteria (thesis m=4 example)**: the worked 4-processor instance
  with all fields {S,C,A,B,p,r,d,f,l}, V=20, reproduces the thesis reference
  makespans as the cost limit tightens: G-bar large => Cmax=26.333; G-bar=24.25
  => 41.5; G-bar=24.1334 => 60.656; below threshold => Infeasible. Bi-criteria
  ExactSolver/MILP minimise Cmax subject to G<=G-bar (cross-validated, MILP==B&B).
  The reverse direction (min G s.t. Cmax<=C-bar) reproduces the same Pareto
  points: Cmax<=41.5 => G=24.25; Cmax<=60.656 => G=24.1334 (evaluator + MILP).

## 4b. Result return (β, model A)
- **Single-processor closed form**: S=0, C=1, A=1, V=10, β=0.5 => send 10 +
  compute 10 + return 5 = `Cmax=25` (vs `20` without return).
- **Monotonicity + conservation**: larger β never lowers the makespan, and load
  is still conserved. Verified for the CSimplex and HiGHS LP backends.

## 4c. Piecewise-linear convex processing time (memory hierarchy)
- **Swap-point closed form**: one processor, `S=C=0`, core piece `q` and disk
  piece `3q−10` (crossing at `q*=5`). Below the swap the core binds
  (`V=4 ⇒ Cmax=4`), above it the disk binds (`V=10 ⇒ Cmax=20`), and at the swap
  both agree (`V=5 ⇒ Cmax=5`). Verified on CSimplex and HiGHS (which agree).
- **Reduction to the affine model**: a processor stated as two *identical* pieces
  reproduces the classic `p + A·α` makespan — checked through the `(α+y)` coupling
  of a multi-installment sequence, so the carried-load handling is exercised.
- **Dominated piece is inert**: adding a piece that lies strictly below the core
  line leaves the makespan unchanged.
- **Convexity guard**: `validate()` rejects a piece with negative slope.
- **MILP agreement**: the MILP reproduces the swap-point (`Cmax=20`) and, on a
  two-processor memory-hierarchy instance, its optimum equals the
  single-installment B&B optimum (two independent exact methods, both modeling
  the pieces).

## 4d. Best Rate heuristic + lower bound (Berlińska)
- **Lower bound analytic anchor**: on a pure-compute instance (`S=C=0`, two
  `A=1` processors, `V=10`) the optimum is the fluid bound `V/(Σ1/Aᵢ)=5`, and
  `divisibleLoadLowerBound` (eq. 3.23) returns exactly `5`.
- **Bound validity**: `LB ≤ Cmax*` checked against the exact optimum across
  instances (the bound is never violated).
- **Best Rate bracketing**: the heuristic is `Feasible`, conserves load, and its
  makespan lies in `[LB, exact-optimum-can't-be-beaten]` — i.e. `≥ LB` and
  `≥ the multi-installment exact optimum`.
- **Determinism + memory**: no RNG, so repeated runs are identical; with small
  buffers every fragment satisfies `loadSize ≤ Bᵢ`.
- **Selection**: a uselessly high-startup processor is dropped (`Cmax=20`).
- **Backend agreement**: the `simplex` and `highs` backends give the same result.

## 4d1. Online PSR + GSS/SSC heuristics (Marszałkowski 2020 §6.2)
- **Schedule consistency**: the simulated schedule is a valid single-port plan —
  transfers serialize (`commStart ≥ prev commFinish`), compute follows receive,
  `makespan = max computeFinish`, and load is conserved to `V`.
- **Memory invariant**: with `SSC` every chunk satisfies `loadSize ≤ ρ` (the
  heuristic never goes out of core).
- **Bound chain**: re-optimizing the heuristic's own activation order with the LP
  can only help, giving `LB ≤ LP-reoptimum(sequence) ≤ heuristic makespan` — a
  certificate that the heuristic sits above the true optimum without enumerating
  the unbounded number of installments GSS may emit.
- **super ≤ any single rule**: `Psr::All` (best of compute/comm/startup/memory/
  energy) is never worse than any individual PSR.
- **Analytic anchors**: one worker with ample RAM → a single chunk, `Cmax = S +
  C·V + A·V`; with `RAM = 40, V = 100` SSC serializes into chunks `40,40,20`
  finishing at `13,26,33` ⇒ `Cmax = 33` (hand-computed).

## 4e. OptV — maximize load within a deadline (dual)
- **Single-processor closed form**: `S=0, C=A=1, T=20` ⇒ max load `T/(C+A)=10`.
- **Round-trip duality (exact)**: minimize the makespan for `V` (OptT via the
  exact solver), then OptV with deadline `= that makespan` over the *same*
  sequence space recovers exactly `V` — load monotonicity makes this an equality.
- **Monotonicity**: a larger deadline never processes less load.
- **Guard**: `MaxLoad` without a finite `makespanLimit` is rejected (`Failure`).
- **Backend agreement**: `simplex` and `highs` give the same max load.

## 4f. FPTAS for OptV on DLS{Cᵢ=0} (provable guarantee)
- **Single-processor**: `S=2, A=1, T=12` ⇒ load `(T−S)/A = 10`.
- **Guarantee**: against the brute-force subset optimum (closed form eq. 2.1),
  `V_FPTAS ∈ [(1−ε)·V_OPT, V_OPT]` for `ε ∈ {0.5, 0.1, 0.01}`.
- **Convergence**: a tighter `ε` is never worse and reaches `V_OPT` (within 1%).
- **Cross-method**: on a `Cᵢ=0` instance the exact `OptVSolver` (LP) equals the
  closed-form brute force, and the FPTAS lies within `(1−ε)` of it.
- **V-cap correctness**: when `T` is large the DP produces a per-processor load that
  can exceed `totalLoad V`; the solver scales all loads down so `Σαᵢ = V` exactly.
  Verified by a regression test (`T >> V` ⇒ `totalAssignedLoad() ≤ V`) and a
  monotone stress sweep (`assigned load` is non-decreasing in `T` and never
  exceeds `V`).
- **Guards**: a non-`Cᵢ=0` instance and a missing deadline are rejected.

## 4g. FPTAS for OptT on DLS{Cᵢ=0} (provable guarantee)
- **Single-processor**: `S=2, A=1, V=10` ⇒ min time `S + V·A = 12`.
- **Guarantee**: against the brute-force optimum (equal-completion closed form),
  `T_FPTAS ∈ [T_OPT, (1+ε)·T_OPT]` for `ε ∈ {0.2, 0.05}`, with load conserved to `V`.
- **Load conservation**: the binary search may converge to a deadline where OptV
  selects a processor whose cumulative startup exceeds Tprime; such processors are
  trimmed iteratively before the equal-completion formula is applied, ensuring all
  loads are positive and `Σαᵢ = V` exactly.
- **Round-trip**: the returned time suffices to process `V` — exact `OptVSolver`
  at that deadline yields `≥ V`.
- **Convergence**: a tighter `ε` is never worse.
- **ExactSolver regression (ideal-bound fix)**: on a `Cᵢ=0` instance the exact
  B&B now returns the true optimum (`6.0`), not a pruned suboptimum (`6.286`).
  The ideal-processor fill must use zero startup, else it injects forced startup
  time into the bound and can prune the optimum. Cross-checked against brute force.

## 4h. Single-round closed form (classic DLT / §2.4)
- **Analytic anchors**: one processor `(C+A)·V = 230`; two identical
  `(C+A)²V/(C+2A) = 180` — reproduced by the closed form (no LP).
- **Exact for S=0**: the makespan equals the exact B&B optimum on `S=0`
  instances (non-decreasing `Cᵢ` order with all processors is optimal there).
- **LP consistency**: evaluating the solver's own ordered sequence with the LP
  gives the same makespan (its equal-completion split *is* the LP optimum).
- **Ordering**: the smaller-`Cᵢ` processor is activated first, beating the
  reverse order; load is conserved.

## 4i. Energy model (Marszałkowski 2020)
- **Analytic ground truth**: a hand-computed single-installment schedule (one
  worker, `V=100`, `S=1,C=0.1,A=0.2`, `P^S=5,P^N=10,k=2,P^I=3`, master
  `P^N₀=8,P^I₀=4`) gives `Cmax=31` and `E=473` (startup 5 + network 100 +
  running 200 + master 168) — reproduced exactly by the LP + `scheduleEnergy()`.
- **Piecewise vs proportional (§3.3, Tab 3.4)**: `ε(α)` with in-core `{0,k₁}` and
  out-of-core `{l₂,k₂}` meeting at `ρ` returns `k₁ρ` at the swap point and the
  steep value out of core, where the proportional model `k₁·α` underestimates by
  ×5+ — confirming why `energyPieces` is needed.
- **LP optimizes the closed form**: on a fixed sequence the `MinCost` energy
  equals the independent `scheduleEnergy()` value, and is `≤` the energy of the
  makespan-optimal split (the two criteria genuinely diverge).
- **Time–energy Pareto monotonicity**: forcing the schedule down to the minimum
  makespan (`makespanLimit = Tmin`) never lowers the minimum energy below the
  unconstrained optimum — the trade-off front is monotone.
- **Cross-backend + I/O**: CSimplex and HiGHS agree on the energy optimum, and
  power rates / energy pieces / originator power round-trip through the text I/O.
- **Backward compatibility**: instances with no energy data report `energy 0` and
  are byte-identical to the pre-energy makespan/linear-cost path (all 145 prior
  tests unchanged).
- **Energy MILP (#3) == brute force**: the MILP that chooses the sequence *and*
  split to minimize energy matches the minimum over all single-installment
  sequences scored by the evaluator's MinCost (energy) — in both the forward
  (min `E` s.t. `Cmax ≤ C̄`) and reverse (min `Cmax` s.t. `E ≤ Ē`) directions; the
  reported energy equals the closed form, and optimizing energy with piecewise
  *time* is rejected (the idle term needs run time pinned).

## 5. Problem-size scaling
Instances over `N ∈ {2,3,4,5}` and `L ∈ {1..5}`: each feasible case conserves
load to `V`, returns the expected status, and satisfies the cross-checks above.
Exact solvers are kept to small `N·L` for test speed; the GA covers larger sizes.

## 6. MLSD (multiple loads)
- **Thesis Example 1**: `m=3, n=2`, identical processors, `V=(32,2)`. The LP
  evaluator reproduces the reference makespans exactly: full structure
  `Cmax=40`; with P3 dropped from T2 `Cmax=39.333`.
- **Exact solver**: per-task load conservation (`Σₖ loads = Vⱼ`); the fixed-order
  optimum matches `39.333`; the global optimum (over all task orders) is no
  worse than any fixed order.
- **GA**: reproducible from a seed; reaches the exact optimum on small instances.
- **Result return (β)**: single-task/proc matches the closed form (`Cmax=25`);
  larger β never lowers the makespan.
- **HiGHS backend**: the "highs" evaluator agrees with CSimplex on Example 1.
  ASan-clean.
- **MLSD MILP cross-check** (`mlsd-milp`): an independent monolithic MILP (per
  task order, deciding the per-task processor slot assignment + loads) equals the
  brute-force enumerator on Example 1 (`39.333`) and heterogeneous instances,
  with per-task conservation — the second problem class now has two independent
  exact methods.

## 6b. MapReduce (Berlińska ch. 4)
- **Single-mapper closed form**: `A=2, V=10, S=1, C=0.5, γ₀=1` ⇒ `α₁=V` and
  `T = S + V(A+γ₀C) = 26`.
- **Reducer cost** `τ(x)=a_red·x·log₂x`: `V=8, a_red=1, s_red=2, r=1, γ₀=1` ⇒
  `t_red = 2 + 8·log₂8 = 26`, `T = V·A + t_red = 34`.
- **Dual-form identity**: the solved partition satisfies
  `mS + αₘ(Aₘ+g) ≡ S + α₁A₁ + gV` (eq. 4.49, `g=γ₀C/r`) — a built-in check that
  the `O(m)` partition is correct — and `Σαᵢ = V`, with faster mappers receiving
  at least as much load.
- **Over-provisioning**: with many processors for a small load, some mappers are
  dropped (`αₘ<0`), the rest carry non-negative loads summing to `V`.
- **Reducer scaling**: with `a_red=0`, more reducers never lengthen the schedule.

## 6c. Multilayer applications (Berlińska ch. 5)
- **x·log₂x convex pieces** (the thesis's τ linearization, §5.2.1): exact at the
  breakpoints `2^y`, slopes non-decreasing (convex), and an over-estimate
  between — so they are a valid drop-in for the piecewise-convex feature.
- **Single layer / single reducer closed form**: `V=8, m=1, A=1`, one reducer
  `rate=1` ⇒ `T_map=8`, `compute = 1·8·log₂8 = 24`, `T = 32`.
- **γ-chain**: layer inputs follow `L_p = V·γ₀·Π γ_q`; more reducers per layer
  shrink the per-reducer `x·log₂x` compute.
- **Monotonicity**: extra layers never shorten the schedule; a wider bisection
  width speeds up the inter-layer reads.

## 6c1. Auto meta-solver (feature dispatch)
- **Dispatch**: a small instance picks `exact` (and matches it), `Sᵢ=0`+ample
  memory picks `single-round`, `ΣBᵢ<V` picks `best-rate`, and ample-memory +
  startups picks `ga` — each verified via `chosenSolver()`, all feasible.
- **Registry**: `auto` is listed first and reachable through `makeSolver`.

## 6d. Solver registry
- **Listing**: `availableSolvers()` is non-empty and contains every portfolio
  name; an unknown name yields `nullptr`.
- **Construction**: every listed solver builds, reports its own `name()` and a
  valid category.
- **Equivalence**: `makeSolver("exact")` matches a directly-built `ExactSolver`.
- **End-to-end**: the default-configurable solvers run feasibly and conserve
  load through the registry.

## 6e0. Instance I/O (rich text format)
- **Legacy compatibility**: plain `S C A B` rows still parse.
- **Full model**: `beta`, the optional trailing `p r d f l`, and `pieces a,b …`
  all read into the right fields.
- **Round-trip**: `writeInstance` then `readInstance` reproduces every field
  (including convex pieces and β) — 17-digit output guarantees fidelity.
- **Errors**: too-few columns, `pieces` before a processor, a comma-less piece,
  and an empty instance are each reported.

## 6e1. Per-class instance I/O (CLI `--class`)
- **MLSD**: `task`/`proc` text parses and, fed to the solver, reproduces the
  Example 1 optimum `39.333` — tying the reader to a validated result.
- **MapReduce / Multilayer**: every directive parses into the right field, and
  the parsed instance solves successfully.
- **Topologies (chain / tree / graph)**: the `node`/`edge` formats parse and the
  parsed instances solve to the same results as directly-built ones (chain anchor
  `20`, tree match, graph routing `parent[2]==1`).
- **Errors**: missing tasks/processors/mappers/layers/nodes and unknown
  directives are each reported.

## 6e. Benchmark harness
- **Determinism**: a fixed seed reproduces the generated instances and the
  aggregated per-solver results exactly (no hidden global RNG).
- **Optimality on average**: on random instances (`A,C,S ~ U[0,1]`, unbounded
  memory) the exact B&B's average makespan is ≤ every heuristic's — per-instance
  optimality preserved under averaging — and every gap from the lower bound is
  `≥ 0`.
- **Availability**: an unregistered solver name is reported with `total = 0`.
- **Virtual-best reference**: when an exact solver is in the run it is the
  per-instance best, so its `gap→best` is exactly `0`, and no solver's gap is
  negative.

  (Empirically the harness reproduces the thesis picture with now-meaningful
  numbers — `gap→best` is the distance from the optimum: `single-round` ≈1.3% at
  ~0 ms, `ga` ≈0.7%, `best-rate` degenerates under unbounded memory. The tight
  lower bound also cuts the LB-floor gap from ~150% to ~36%.)
- **Status-aware reference**: with an exact solver in the run, every instance has
  a *proven-optimal* reference (`provenOptimal == instances`) and the exact
  solver's `gap→opt` is exactly `0`; with only heuristics, `provenOptimal == 0`
  and no solver is scored against the (non-existent) proven optimum. So the
  rigorous "X% above optimal" claim is reported only when it is actually proven.
- **Labels + policy scoring**: `generateLabeledInstances` labels each instance
  with `(M*, proven, LB)` via an exact solve; `scorePolicy` scores an external
  policy's makespans — an optimal policy gives `avgGapToOpt = 0`, a 10%-above
  policy gives `0.10`, and the LB-gap (a certificate valid even where exact
  solving is intractable) is never below the optimum-gap.
- **Label CSV round-trip + score path**: `writeLabels`→`readLabels` recovers
  `(M*, proven, LB)`; `readPolicyMakespans` takes the last CSV field and skips a
  header; the end-to-end disk path scores an optimal policy at `0%` and an
  8%-above policy at `8%`.
- **LB certificate**: `gap→LB` (per-solver `maxGapToLB`, and the policy's) is a
  valid upper bound on the true optimality gap — verified `maxGapToLB ≥ avgRelGap`
  and `maxGapToLB ≥ gap→opt` (it dominates the true gap), so "certified ≤ X%
  above optimal" holds even where the optimum is unknown.
- **Canonical (homogeneous) instances** (`--homogeneous`, regime 1): the
  generated processors are identical, and the closed-form `single-round` recovers
  the proven optimum (`gap→opt ≈ 0`, matching the exact B&B) at ~0 ms — the
  "validate on tractable cases first" step.
- **Regime tagging**: homogeneous + ample memory is tagged **regime 1**, where the
  closed form *is* the proven optimum — so the report shows a proven reference and
  `gap→opt` **without running the B&B** (verified: heuristics-only run still has
  `provenOptimal == instances`, and that closed-form reference equals the exact
  optimum). Heterogeneous instances are **regime 2** (proven only via an exact
  solver; nothing proven without one).

## 6e2. Isoefficiency / isoenergy maps (Marszałkowski 2020 §4 & §6.3)
- **Grid shape & ticks**: `computePerformanceMap` returns a `ySteps × xSteps`
  grid; axis ticks are evenly spaced and hit both endpoints.
- **Isoenergy analytic**: with `single-round` and zero state powers, every cell of
  an energy map equals `k·V` exactly (running energy `Σ k·αᵢ = k·V`), independent
  of the compute-rate axis — a closed-form anchor for the whole grid.
- **Isoefficiency monotonicity**: makespan is non-increasing along the
  processor-count axis and non-decreasing along the startup axis (more workers
  help, more startup hurts).
- **Single point**: a 1×1 map of one ample-memory processor reproduces the closed
  form `S + C·V + A·V`.
- **CSV writer**: header + one row per `y` tick, `inf` for infeasible cells.

## 6f. Tight lower bound
- **Validity**: `divisibleLoadLowerBoundTight = max(eq.3.23, fluid, port-relaxed)`
  never exceeds the exact optimum, and dominates each component.
- **Port-relaxed analytic**: one processor `S=2, A=1, V=10` (no port contention
  to relax) ⇒ the bound is exactly the optimum `S + V·A = 12`.

## 6g. Linear daisy chain (non-star topology)
- **Analytic anchors**: one processor ⇒ `A₀·V`; two identical with link rate
  `C₁=1` ⇒ `α₀=20, α₁=10, Cmax=20`; a free link (`C₁=0`) ⇒ perfect parallelism
  `V/2`.
- **Optimality principle**: at the LP optimum every participating processor
  finishes at the makespan, and load is conserved.
- **Slow link**: a very slow feeding link starves the downstream processor (it
  keeps only a sliver), and the makespan sits just below `A₀·V`.

## 6h. Multi-level tree (generalizes star + chain)
- **Unification cross-checks**: a *path* tree (each node one child) equals
  `LinearChainSolver`, and a 1-level tree with a non-computing root equals the
  single-port star LP (`SimplexScheduleEvaluator`) — the tree subsumes both.
- **Analytic**: one node ⇒ `A₀·V`.
- **Optimality principle**: on a 2-level tree every participating node finishes
  at the makespan, and load is conserved.

## 6i. General graph / mesh (spanning-arborescence reduction)
- **Reduces to the tree**: when the graph's edges already form a tree, the only
  spanning arborescence is that tree, so `GraphSolver == TreeSolver`.
- **Routing**: with a slow direct link but a fast two-hop path, the optimal
  arborescence is the chain (`gs.parent[2] == 1`), beating the direct-link star.
- **Optimality of the choice**: the best arborescence is never worse than any
  specific one; load is conserved.

## 7. Three-tier cross-solver validation framework
Every makespan-minimising solver is swept through three tiers on each build.

**Tier 1 — structural invariants** (studio-default, single-processor, 5-processor
instances): every feasible result must satisfy `Σαᵢ = V`, `loadSize ≥ 0`,
`commStart ≥ 0`, causal ordering (`commFinish ≥ commStart`, `computeStart ≥
commFinish`, `computeFinish ≥ computeStart`), `computeFinish ≤ makespan`, serialised
communication (no overlapping `[commStart, commFinish]` intervals), and
`makespan ≥ divisibleLoadLowerBoundTight`. Solvers that leave timing fields as 0
(LP-based: `exact`, `best-rate`, `exact-dual`) skip the causal-ordering check
automatically.

**Tier 2 — analytic anchors** (closed-form verification): single-processor `S=0`
instance gives `T=(C+A)·V`; homogeneous 2-processor `C=S=0` gives `T=V·A/2`;
FPTAS OptV with `T >> V` saturates at `V`; FPTAS OptV with `T < min(Sᵢ)` returns an
empty schedule; FPTAS OptT single-processor gives `T=S+V·A`; MapReduce with `C=0`
splits load equally.

**Tier 3 — stress sweeps**: FPTAS OptV `assigned load` is non-decreasing in `T` and
never exceeds `V` (100-point sweep); makespan is non-increasing as identical
processors are added (`C=S=0`, 10-processor sweep).

**Regression guards**: FPTAS OptV never exceeds `V` when `T >> V`; makespan gap
is never negative for any standard makespan solver.

## 8. Always checked
- **Load conservation**: `Σ αᵢ == V` for every feasible solution.
- **Reproducibility**: fixed seed ⇒ identical GA result; exact solvers are
  RNG-independent.
- **Memory safety**: the dependency-free suite is clean under AddressSanitizer +
  UBSan.

## Rule
After implementing or modifying any model/solver, add validation tests confirming
accuracy (analytic anchor where derivable, cross-validation, and the relevant
boundaries), update docs/, and re-run both builds before considering the work done.
