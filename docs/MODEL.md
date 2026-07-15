# DLS model & notation

This document defines the Divisible Load Scheduling (DLS) model used throughout
the library and the meaning of every symbol that appears in the code, the
linear program, and the solvers.

## Model in one paragraph

A single **master** distributes a divisible load of total size **V** to **N**
worker processors over a star network. Communication is **single-port**: the
master sends to one worker at a time, sequentially, in a chosen **sequence** of
installments. A processor may receive several installments (**multi-installment**)
unless restricted to one (**single-installment**). Costs are **affine**: sending
load `α` to processor *i* costs `Sᵢ + Cᵢ·α` time, computing it costs `Aᵢ·α`, and a
single installment can carry at most `Bᵢ`. The objective is to minimize the
**makespan Cmax** — the time the last processor finishes computing (or, when
result return is enabled, the time the last result is collected).

A processor *i* receiving load `α`:
**communication** `= Sᵢ + Cᵢ·α`, **computation** `= Aᵢ·α`, capped by `Bᵢ` per installment.

---

## Per-processor cost coefficients (`Processor` struct)

Each worker *i* has the following coefficients. The first four (`S, C, A, B`) are
the columns of `proc_par.txt`; the rest are set programmatically and default to
neutral values, so existing instances are unaffected:

| Symbol | Code field    | Meaning                                                              | Units            |
|--------|---------------|---------------------------------------------------------------------|------------------|
| **Sᵢ** | `commStartup`    | fixed latency to *begin* a transfer to *i* (independent of load)    | time             |
| **Cᵢ** | `commRate`       | communication time **per unit of load** sent to *i*                 | time / load-unit |
| **Aᵢ** | `computeRate`    | computation time **per unit of load** on *i*                        | time / load-unit |
| **Bᵢ** | `memoryLimit`    | max load *i* can hold in one installment (buffer); `≤0` = unbounded | load-units       |
| **pᵢ** | `computeStartup` | computation startup time, per installment (default 0)               | time             |
| **rᵢ** | `releaseTime`    | *i* is unavailable for computing before this time (default 0)       | time             |
| **dᵢ** | `deadline`       | completion deadline for *i*; `≤0` means no deadline (default)        | time             |
| **fᵢ** | `fixedCost`      | fixed cost of using processor *i* (default 0)                        | cost             |
| **lᵢ** | `linearCost`     | cost per unit of load on *i* (default 0)                             | cost / load-unit |
| —      | `computePieces`  | optional convex piecewise-linear processing time (see below); empty (default) ⇒ the single piece `{pᵢ, Aᵢ}` | — |

Smaller `Aᵢ` / `Cᵢ` mean a faster processor. The availability fields `pᵢ, rᵢ, dᵢ`
are thesis-faithful for the single-installment case and extend the makespan
constraint: a completion is `≥ rᵢ + pᵢ + Aᵢαᵢ` and `≤ dᵢ`. They add LP rows only
when set, so default instances are unaffected.

**Cost / bi-criteria.** The schedule cost is `G = Σ(fᵢ + αᵢ lᵢ)` over the used
processors (reported in `DLSSolution.cost`). Solvers support the bi-criteria
direction **minimize Cmax subject to `G ≤ Ḡ`** via a cost limit (`costLimit` /
`EvaluatorConfig.costLimit`, default ∞) and the reverse direction
**minimize `G` subject to `Cmax ≤ C̄max`** (`EvalObjective::MinCost` /
`makespanLimit`; the MILP also via `minimizeCost`).

**Piecewise-linear convex processing time (memory hierarchy).** By default the
computation time of load `q` on processor *i* is the affine `pᵢ + Aᵢ·q`. Setting
`computePieces` to a list of `{intercept aₖ, slope bₖ}` pieces makes it the
**convex** function `Tᵢ(q) = maxₖ (aₖ + bₖ·q)`. This models a memory hierarchy: a
fast core-RAM piece (small slope) and a steeper virtual/disk piece that takes
over once the load exceeds the core size — the two lines cross at the **swap
point** `q* = (a_core − a_disk)/(b_disk − b_core)`. It also approximates any
convex compute curve. Each slope must be `≥ 0` (time is non-decreasing in load);
intercepts may be negative (the disk line dips below the core line for small
loads). The single implicit piece `{pᵢ, Aᵢ}` is the classic affine special case,
so default instances are unchanged. The hard buffer cap `Bᵢ` (a memory *limit*)
and the soft penalty of `computePieces` (a memory *cost*) coexist. Supported by
**all** solvers — the GA, B&B, both LP backends, and the MILP (which adds a
per-slot `proc_k` variable bounded by `proc_k ≥ aₖ·x[i][k] + bₖ·w[i][k]`; the
assignment binary and load auxiliary both vanish for an unassigned processor, so
the bound binds only for the one in the slot — no big-M needed).

**Result return (β).** With `resultFraction = β > 0`, each processor returns
results of size `β·αᵢ` over the same link (`Sᵢ + Cᵢ·β·αᵢ`). Model A is used: a
**non-overlapping, FIFO collection** phase after all distribution — the master
collects results in installment order over the single port, and the makespan is
the last return completion. Default `β = 0` (no return). Supported by the
evaluator-based solvers (GA, B&B, both LP backends); the MILP does not model
return yet. The fully interleaved single-port model is a future refinement.

## Global instance symbols

| Symbol | Code                | Meaning                                          |
|--------|---------------------|--------------------------------------------------|
| **N**  | `numProcessors()`   | number of worker processors                      |
| **V**  | `totalLoad`         | total divisible load to distribute               |
| **β**  | `resultFraction`    | result-return fraction: results of size β·αᵢ go back to the master (0 = none, default) |
| —      | `CommunicationModel::SinglePort` | master sends to one worker at a time |

## Sequence / installment symbols

- **Installment** — one chunk sent to one processor in one communication.
- **Sequence** — the ordered list of processor indices, one per installment; the
  order in which the single-port master sends.
- **L** — sequence length / number of installments. It is `chromosomeMaxSize` /
  `installments` (GA) and `maxInstallments` (exact solvers, where it is a *cap* —
  schedules of length ≤ L are searched).
- **i** or **k** — installment / slot index, `0 … L-1`.

## Objective

| Symbol   | Code       | Meaning                                                                 |
|----------|------------|-------------------------------------------------------------------------|
| **Cmax** | `makespan` | the **makespan** — time the last processor finishes computing (minimized) |

Three evaluator objectives (`EvalObjective`): **MinMakespan** (default — minimize
`Cmax`, optionally s.t. `G ≤ Ḡ`), **MinCost** (minimize `G` s.t. `Cmax ≤ C̄`), and
**MaxLoad** (the **OptV** dual — maximize the processed load `Σαᵢ` s.t. `Cmax ≤ T`;
the conservation `Σαᵢ = V` is dropped and `V` is ignored). `OptVSolver`
(`exact/optv/`) searches sequences with MaxLoad to answer "most load within `T`",
and is consistent with the makespan direction: `OptV(T = OptT-makespan(V)) = V`.

### Energy criterion (Marszałkowski 2020) — `core/energy_model.hpp`

When an instance carries energy data (`inst.usesEnergyModel()`), a second
criterion **E** (total energy) is modelled alongside time. Each worker passes
through four states with constant power, and the master M₀ draws `P^N` while
distributing and `P^I` afterwards:

| Symbol | Code | Meaning |
|--------|------|---------|
| `P^I` | `powerIdle` | idle power (fills each used machine's slack up to `Cmax`) |
| `P^S` | `powerStartup` | startup power, over the wake/latency `S` |
| `P^N` | `powerNetwork` | networking power, over the transfer `C·α` |
| `ε(α)` | `energyPieces` | **running energy** — convex piecewise `maxₖ(lₖ+kₖ·α)` |
| `P^N₀,P^I₀` | `originatorPower*` | master M₀ networking / idle power |

Total `E = E₀ + Σᵢ(E^S + E^N + E^R + E^I)` with `E^S=P^S·S`, `E^N=P^N·C·α`,
`E^R=ε(α+y)`, `E^I=P^I·(Cmax − busyᵢ)`, and `E₀=P^N₀·dist + P^I₀·(Cmax − dist)`
where `dist=Σ(S+C·α)`. The running energy mirrors `computePieces` exactly: the
empty default is the single proportional piece `{0, linearCost}` = `l·α`, which
the thesis (§3.3) shows badly underestimates out-of-core work — hence the steeper
second piece `{l₂<0, k₂}` meeting the in-core piece at the RAM size `ρ`.
`scheduleEnergy()` computes `E` in closed form; for a fixed sequence `E` is linear
in the LP variables, so **`MinCost` minimizes `E`** (not the linear cost `G`) when
the energy model is on, and pairing it with `makespanLimit` traces the time–energy
Pareto front (min `E` s.t. `Cmax ≤ T̄`). Reduces to the thesis single-installment
formulas when every processor appears once. The **MILP** (`exact/milp/`) carries
the same energy form over its assignment binaries — "machine used" `x_i` becomes a
real binary, so the idle base `P^I·Cmax·x_i` is a McCormick product `cx_i` — and
optimizes energy over sequence *and* split (single-installment, affine time).

For the infinite-bandwidth special case **DLS{Cᵢ=0}** there is also an FPTAS
(`heuristics/fptas/`, Berlińska §2.2, Algorithm 2.2): activate in non-decreasing
`Sᵢ·Aᵢ` order (Prop 2.1), then choose the optimum subset by minimizing the
half-product `f(x) = −Σpᵢxᵢ + Σ_{i<j}qᵢrⱼxᵢxⱼ` (`pᵢ=(T−Sᵢ)/Aᵢ, qᵢ=Sᵢ, rⱼ=1/Aⱼ`)
via the Badics–Boros DP with geometric bucketing of `Σqᵢxᵢ`. It guarantees
`V_FPTAS ≥ (1−ε)·V_OPT` — the library's first solver with a provable
approximation ratio. The dual `FptasOptTSolver` (§2.3, Algorithm 2.4) minimizes
the *time* for a load `V` with `T_FPTAS ≤ (1+ε)·T_OPT`, by binary-searching the
deadline with the OptV FPTAS as a dual-feasibility gauge.

---

## LP variables (fixed sequence) — `core/dls_lp_model.hpp`

For a **fixed** sequence of `L` installments, the optimal load split is a linear
program with `3L + 1` variables (`+L` per active optional block). The `procᵢ` and
`rcᵢ` blocks are appended only when their feature is active, so the default model
keeps the classic `3L + 1` layout. Variable layout (0-based index):

| Symbol | Code            | Index   | Meaning                                                                                   |
|--------|-----------------|---------|-------------------------------------------------------------------------------------------|
| **αᵢ** | `alphaVars[i]`  | `i`     | load assigned to installment *i* (continuous ≥ 0)                                         |
| **tᵢ** | `timeVars[i]`   | `L+i`   | communication timing of installment *i* — when its transfer can start (cumulative comm of prior installments); `t₀ = 0` |
| **yᵢ** | —               | `2L+i`  | "carried load" term — couples an installment with the **previous installment on the same processor**; `0` for a processor's first appearance |
| **procᵢ** | —            | `3L+i` (pieces only) | processing time of installment *i*; bounded by `procᵢ ≥ aₖ + bₖ·(αᵢ+yᵢ)` per piece, so it settles at the binding (max) piece. Present only when some sequenced processor has a multi-piece processing time |
| **rcᵢ** | —             | appended after the above (β>0)| return-completion time of installment *i* (only present when β>0)                  |
| **Cmax** | `cmaxVar`     | last    | the makespan variable (the objective)                                                    |

With a single (implicit) piece the model uses the inline compute term
`pᵢ + Aᵢ·(αᵢ+yᵢ)` and the `procᵢ` block is omitted; with multiple pieces the
makespan/availability/return rows use `procᵢ` in its place.

Every constraint row has the form `coeffs · x ≤ rhs`. The CSimplex backend's
constraint count is `5L + distinct + 3`, where **distinct** is the number of
distinct processors in the sequence.

## MILP-only symbols (single-installment) — `exact/milp/milp_solver.cpp`

The monolithic MILP decides the sequence as well, adding:

| Symbol     | Meaning                                                                                  |
|------------|------------------------------------------------------------------------------------------|
| **K**      | number of slots = `min(maxInstallments, N)`                                              |
| **x[i][k]**| binary: `1` if processor *i* occupies slot *k* (the assignment decision)                 |
| **αₖ**     | load in slot *k*                                                                          |
| **w[i][k]**| McCormick auxiliary linearizing the product `x[i][k]·αₖ` (keeps binary × load linear)    |
| **tₖ**     | communication start time of slot *k*                                                      |
| **U**      | `maxᵢ Bᵢ` — upper bound on any single load; the big-M / variable bound for the McCormick terms |
| **proc_k** | processing time of slot *k* (only when some processor is piecewise); `proc_k ≥ aₖ·x[i][k] + bₖ·w[i][k]` per piece, replacing the inline `pᵢ x + Aᵢ w` compute term in the makespan/availability rows |

## Bound & search symbols (branch-and-bound) — `exact/enumerative/exact_solver.hpp`

| Symbol / name                          | Meaning                                                                              |
|----------------------------------------|--------------------------------------------------------------------------------------|
| **ideal-processor bound** | the primary B&B bound: complete a partial sequence with `L−l` *distinct* ideal processors (`A_id=min A`, `C_id=min C`, `S_id=min S`, B=∞), solve the LP → a valid lower bound on every real completion. Prunes effectively even when `S=0`. |
| **fluid bound** `= V / Σ(1/Aᵢ)`        | secondary lower bound: no communication, full parallel computation (`0` if any `Aᵢ = 0`) |
| **startup bound** `= Σ Sₚ` (over the sequence) | cheap per-child pre-filter from sequential single-port communication         |
| `prefixStartup`                        | running `Σ S` of the current partial sequence in the B&B                             |
| `incumbent`                            | best (lowest) Cmax found so far during the search                                    |
| **distinct**                           | number of distinct processors in a sequence (used in the constraint-count formula)   |

> Provenance: this model and the GA / B&B algorithms come from the DLS thesis
> (chapter SLMD — Single Load Multiple Distributions). Thesis notation uses **m**
> for the processor count (here **N**) and **z** for the installment count (here
> **L**). Availability conditions p_i, r_i, d_i from the thesis are now
> implemented (above). The affine cost model (f_i, l_i, criterion G) and the
> bi-criteria direction min Cmax s.t. G<=G-bar are now implemented. The MLSD
> multiple-loads problem class is fully implemented (see below). Still to come:
> the reverse bi-criteria direction (min G s.t. Cmax<=C-bar) and result return
> beta_j for the MILP.

---

## Mental model

Distribute total load **V** across **N** processors; the single-port master serves
installments sequentially in the chosen **sequence** of length ≤ **L**; each
installment of size `α` on processor *i* costs `Sᵢ + Cᵢ·α` to send and `Aᵢ·α` to
compute, with at most `Bᵢ` per installment; minimize the finish time **Cmax**.

---

## MLSD — Multiple Loads, Single Distribution (`mlsd/`)

A separate problem class: `n` divisible **tasks** of sizes `Vⱼ` on `m` processors
over the single-port star, in a **permutation schedule** (one task order, the
same on all processors; each task uses a processor set `Pⱼ` in some activation
order). Objective: minimize `Cmax`. NP-hard in general.

Same two-layer decomposition as the single-load model:

- **`MlsdInstance`** — `tasks` (`MlsdTask{size Vⱼ, resultFraction βⱼ}`) + `processors`
  (reuses `Processor`; "related" case = processor params independent of task).
- **`MlsdStructure`** — `taskOrder` + per-task `procSeq` (the combinatorial choice).
- **`MlsdScheduleEvaluator`** — fixed structure → LP (thesis eqs lp-eq1/lp-eq2) →
  `Cmax` and per-`(task, processor)` loads. LP variables: `α[l][k]` (load of the
  l-th task to its k-th processor) + `Cmax`. Constraints: per-task conservation
  `Σₖ α[l][k] = Vⱼ`, and a completion-before-`Cmax` row per `(l,k)`.
- **`MlsdSolver`** — exact: enumerate task orders × per-task ordered processor
  subsets, evaluate each, keep the minimum (for small instances; NP-hard).

Solvers/backends:
- `MlsdSolver` — exact (enumerate structures).
- `MlsdGaSolver` — genetic heuristic (genome = MlsdStructure; reproducible from a seed).
- `MlMlsdSolver` — two-stage ML solver: GBM predicts `log(Cmax*)` from
  `MlsdInstanceFeatures` (12-feature vector), then `MlsdGaSolver` constructs the
  schedule. Exposes `predictedMakespan()` for ML-accuracy benchmarking.
- `MlsdScheduleEvaluator(backend)` — "simplex" (default) or "highs" (a generic
  `solveLpViaHighs` LP helper, DLS_WITH_HIGHS).

Result return (β_j > 0): model A (all computation finishes at Tcomp, then results
are collected over the single port: Cmax = Tcomp + Σ collection times). Default
β = 0. Validated against thesis Example 1 (Cmax 40 / 39.333) and the single-load
β closed form; HiGHS == CSimplex.

---

## MapReduce (`mapreduce/`)

A third problem class (Berlińska thesis ch. 4) and the first with **precedence**:
a map phase feeds a reduce phase. A divisible input `V` is split into mapper
chunks `α₁..α_m`; mapper *i* computes at macroscopic rate `Aᵢ` after a one-off
per-processor code startup `S` (sequential ⇒ the `mS` term). Each mapper emits
`γ₀·αᵢ` intermediate bytes; `r` reducers each read `γ₀V/r` bytes (a `γ₀αᵢ/r`
chunk from every mapper) at read rate `C`, then run in `t_red = s_red + a_red·(z·log₂z)`,
`z = γ₀V/r` (the `τ(x)=a_red·x·log₂x` sorting cost). Objective: minimize `T`.

| Symbol | Code | Meaning |
|---|---|---|
| **Aᵢ** | `mapperRates()[i]` | mapper *i* macroscopic rate (time/byte) |
| **V** | `totalLoad` | input load |
| **S** | `startup` | per-processor code-load startup (sequential) |
| **C** | `readRate` | reducer read rate (time/byte) |
| **γ₀** | `resultFraction` | intermediate output / input ratio |
| **r** | `numReducers` | number of reducers |
| **s_red, a_red** | `reducerStartup`, `reducerRate` | reducer startup and `x·log₂x` compute rate |

**Optimal schedule (closed form, `MapReduceSolver`).** By Proposition 4.1 the
optimum activates mappers in non-decreasing `Aᵢ` with sequential FIFO reads, so
computing-plus-reading on `Pᵢ` coincides with startup-plus-computing on `Pᵢ₊₁`:
`(Aᵢ + γ₀C/r)·αᵢ = S + Aᵢ₊₁·αᵢ₊₁`, `Σαᵢ = V`. This `O(m)` system is solved by the
affine reduction `αᵢ = lᵢ + kᵢ·αₘ` (eqs. 4.46–4.48); if `αₘ < 0` the system uses
too many processors, so the slowest mapper is dropped and it is re-solved. The
length is `T = mS + αₘ(Aₘ + γ₀C/r) + t_red`, equivalently `S + α₁A₁ + (γ₀C/r)V + t_red`
(the two forms agree exactly — an internal consistency check). The many-reducer
case here is the "first method" (§4.4.2): single-reducer with `γ₀C` → `γ₀C/r`.

**Bisection-width-limited (exact LP, `MapReduceBwidthSolver`, HiGHS builds
only).** The closed form above assumes the network serves every reducer's
read at full rate `C` at once; `--class=mapreduce-bwidth` instead caps
concurrent read channels at `l` (`bisectionWidth`) — RA-09/09's "third
method" (§5.2, Eqs. 59-64). Mappers still activate in non-decreasing `Aᵢ`;
all `r` reducers then read in that same order, starting once both the
source mapper has finished and a channel is free. `itv(i,j)` (Eq. 59) gives
the interval in which reducer `j` reads mapper `i`'s output; the LP jointly
solves `αᵢ` and the interval boundaries `tᵢ`, minimizing the last transfer's
finish time. Validated at `r=1` (must exactly match the closed form above,
any `l`) and at `m=1, l=1` (must exactly match `S + A·V + γ₀CV + t_red`).

## Multilayer applications (`mapreduce/multilayer_solver.hpp`)

Chapter 5 generalizes MapReduce to a mapper layer feeding `R` reducer layers in a
pipeline, each with a sorting cost `τ_p(x) = a_red_p · x · log₂x`. `MultilayerSolver`
models the **homogeneous** case (identical mappers of rate `A`, identical reducers
per layer ⇒ the optimal split is equal, `δ = 1/r_p`) as sequential, non-overlapping
phases — a feasible schedule / upper bound on the thesis's overlapped optimum:
`T = T_map + Σ_p (read_p + compute_p)`, with `T_map = mS + A·V/m`, input
`L_p = V·γ₀·Π_{q<p} γ_q`, `read_p = C·L_p/min(l, r_p)` (`l` = bisection width), and
`compute_p = s_red_p + τ_p(L_p/r_p)`. The non-linear `τ_p` is linearized (§5.2.1)
by `xLogXConvexPieces`: secant chords of `x·log₂x` at breakpoints `2^y` — convex,
exact at the breakpoints, over-estimating between — reusable as a processor's
`computePieces` (above) so any DLS solver can carry an `x·log x` cost.

## Reducer partitioning skew (`mapreduce/mapreduce_skew_instance.hpp`)

MISTA 2013 (Berlińska & Drozdowski): a **homogeneous** system (mapper rate
`A`, reducer rates `a^sort`/`a^red` shared by all) where the r-way key
partition is *unbalanced* — some reducers get more of the intermediate data
than others, unlike every other MapReduce-family model here, which assumes
an equal `γV/r` share. The instance carries raw partition sizes at `k·r`
granularity directly (`partitionSizes`, k=1 is the plain r-way case).

Shared per-reducer cost, given its assigned size `p`: `read = max(C, C·min(m,r)/l)/min(m,r) · p`,
`sort = a^sort·p·log₂p`, `reduce = a^red·p` — the same per-unit-volume rate
as the balanced baseline (`T* = AV/m + 2Cmrε + max{C,C·min{m,r}/l}γV/min{m,r} + a^sort(γV/r)log₂(γV/r) + a^red·γV/r`),
generalized to an unequal `p` per reducer; setting all `p = γV/r` reproduces
that baseline exactly.

- **StaticSkewSolver**: splits keys into `kr` parts, LPT-assigns them to `r`
  reducers (sort descending, greedily give each to the least-loaded reducer
  — a min-heap), paying `2·C·kr·mε` (report + assign) plus
  `a^master·(krm + kr·log₂(kr) + kr·log₂r)` (the LPT decision itself) before
  any reducer reads.
- **DynamicSkewSolver** (k=1 only): mapping/reading/sorting proceed as the
  plain baseline; once every reducer finishes sorting AND at least one
  finishes reducing, the master halts the `r1` still-busy reducers (3 ×
  `C·r1·ε` control rounds: halt, report remaining load, notify-to-resume —
  the paper gives the first two explicitly, this solver extends the same
  cost to the third), then rebalances. The equalized finish time for
  folding `n_s` busy senders (only their remaining-data sum `S` matters)
  into `n_r` idle receivers is `τ = S / (n_s/a^red + n_r/(C+a^red))` —
  derived here from the same equal-finish-time DLT principle the paper
  invokes only narratively; `n_s=n_r=1` reproduces its 2-node case. Every
  idle reducer is always used as a receiver (weakly optimal), so the only
  real search is over `n_s = 0..r1`, evaluated exhaustively.

Validated: balanced `p_j` ⇒ dynamic result exactly matches the baseline
formula above, with zero reducers busy at the trigger (nothing to
rebalance); static at `k=1` exactly matches baseline + the LPT decision cost
alone (its own report/assign messages are numerically identical to the
baseline's location round-trip at `k=1`, canceling exactly); a hand-derived
2-reducer rebalancing example matches the closed form bit-for-bit; static
never beats the perfectly-balanced lower bound, across a range of `k`
(not asserted monotonic in `k` — the paper's own results show it isn't).

## Multi-source map-phase scheduling (`mapreduce/multisource_solver.hpp`)

Gu, Liao, Yang & Li (IEEE CSE 2013): a **bipartite** topology, not a star or
relay tree — `m` storage nodes hold the input `S`, and `n ≥ m` mapper nodes
(nodes `0..m-1` are the *same* physical nodes as the storage nodes) pull
from potentially several storage nodes concurrently. Map phase only (the
paper defers reduce-phase optimization). Node `j`'s finish time:
`T_j = L_j·t_j + Σ_i R_ij·(t_j + w_ij)` (`L_j` — local share, only for
`j < m`; `R_ij` — data pulled from storage node `i` into mapper `j`; `t_j` —
mapper `j`'s compute rate; `w_ij` — transfer rate). Optimum: every node
finishes simultaneously (classic DLT argument), solved as an LP: a shared
`T`, one equality row `T_j - T = 0` per node (coefficients read directly off
the `T_j` formula — a cleaner equivalent of the paper's own pairwise chained
equations), plus either one total-conservation row `ΣL + ΣR = S` ("Problem
A", `S_i` derived afterward) or `m` per-node rows `L_i + Σ_j R_ij = S_i`
("Problem B", `S_i` fixed input).

Validated: `m = n` + homogeneous rates ⇒ exactly `T = S/m·t`, zero
transfers; free transfer (`w=0`) + heterogeneous rates ⇒ matches the
classic two-processor DLT equal-finish split by hand; Problem A's own
derived `S_i`, fed back in as Problem B's fixed supply, reproduces the
identical makespan and load split bit-for-bit; an extra mapper node (same
total `S`) never increases the makespan.

## Linear daisy chain (`topology/linear_chain.hpp`)

A non-star topology: a chain `P0 — P1 — … — P_{N-1}` where the whole load `V`
enters at `P0`, and each processor keeps a share `αᵢ` and forwards the rest over
the link into the next processor. With front-ends (compute overlaps forwarding),
processor *i* finishes at `fᵢ = Σ_{k≤i} C_k·R_k + Aᵢ·αᵢ` with `R_k = Σ_{j≥k} αⱼ`
(the cumulative link-transfer of the load that reaches `Pᵢ`, plus its own
compute). Minimizing `Cmax = maxᵢ fᵢ` s.t. `Σαᵢ = V` is a small **LP** (variables
`α₀..α_{N-1}, Cmax`), solved exactly with CSimplex — the "express the topology as
an allocation LP" regime. At the optimum all participating processors finish
together (the optimality principle). Per processor: compute rate `Aᵢ`, link rate
`Cᵢ` (the hop into `Pᵢ`); link startups (`S`) are out of scope (they'd need a
used/unused indicator → MILP).

## Multi-level tree (`topology/tree.hpp`)

Generalizes both the star (a 1-level tree) and the chain (a path). Load `V` enters
at the root; each node keeps `αᵢ` and forwards each child's subtree-load down,
sending to children sequentially over one port. With front-ends, node *i* finishes
at `fᵢ = rᵢ + Aᵢ·αᵢ`, where the receive time `rᵢ = r_{parent} + Σ_{siblings c≤i}
C_c·L_c`, `L_c = Σ_{j∈subtree(c)} αⱼ`, and `r_root = 0`. Minimizing `Cmax` s.t.
`Σαᵢ = V` is an LP (variables `αᵢ, rᵢ, Cmax`) solved with CSimplex; at the optimum
every participating node finishes together. Each node has compute rate `Aᵢ` and
parent-link rate `Cᵢ`; sibling send order is the node-index order (optimizing it
is a search — follow-up). A root with leaf children reduces exactly to the
single-port star LP; a path reduces to the linear chain.

## General graph / mesh (`topology/graph.hpp`)

The most general topology. In the single-port store-and-forward model an optimal
divisible-load distribution over a graph uses a *spanning arborescence* rooted at
the source (extra paths only add contention), so the problem reduces to choosing
the best rooted spanning tree and solving its allocation LP (above). For small
graphs `GraphSolver` enumerates the arborescences and keeps the smallest-makespan
one; it reduces to `TreeSolver` when the graph already is a tree, and routes load
through a cheaper multi-hop path when a direct link is slow. Undirected edges with
rate `Cₑ`, node compute rate `Aᵢ`, no link startup; enumeration is capped, so it
targets small graphs / validation rather than large meshes (a heuristic
arborescence — e.g. shortest-path — would scale further).
