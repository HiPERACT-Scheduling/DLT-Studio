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
