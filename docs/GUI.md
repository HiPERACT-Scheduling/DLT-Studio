# Front-end / GUI architecture

How to wire the DLS C++ library (the **backend**) to a friendly **front-end** GUI.
This is the design record; Phase 0 (the data contract) is implemented, the rest is
a roadmap.

## Guiding principle: the data contract is the seam

The boundary that survives every front-end choice is a **stable, serializable
contract** between the library and whatever drives it. Pick the contract first;
the front-end technology then becomes a swappable detail (notebook, web app, and
the CLI can all speak it at once).

## Backend assets vs. gaps

| Capability | Backend today | GUI need |
|---|---|---|
| Solve contract | `DLSSolver` + `makeSolver(name)` + `availableSolvers()` | ideal API surface |
| Problem / result | `DLSInstance`, `DLSSolution` (makespan, cost, energy, fragments) | serialization |
| **Gantt timing** | only the online solver fills all four fragment times | **`expandSchedule()`** ✅ |
| Serialization | text I/O (human-only) | **JSON** ✅ (emitters) |
| Energy / Pareto | `MinCost`+`makespanLimit`, `scheduleEnergy` | a sweep helper (later) |
| Iso-maps | `computePerformanceMap` → CSV/grid | reuse |
| Portfolio compare | `runBenchmark` → `BenchReport` | reuse |
| 7 problem classes | DLS, MLSD, MapReduce, multilayer, chain/tree/graph | class-tagged contract |
| Long solves | `SolverConfig.timeLimitSeconds`, node budgets | async/cancel at the boundary |
| Validation | `DLSInstance::validate()` | reuse for form validation |

## Phase 0 — the contract (implemented)

- **`core/schedule_expand.hpp` — `expandSchedule(instance, sol)`**: fills every
  fragment's `commFinish / computeStart / computeFinish` from the loads +
  coefficients, so a Gantt can be drawn. Uses the makespan model's own finish
  term, so the reconstructed makespan equals the solver's (single-installment,
  β = 0); honours `releaseTime`. (Result-return β collection phase is not drawn.)
- **`core/json_io.hpp` — `writeInstanceJson`, `writeSolutionJson`**: dependency-
  free JSON emitters mirroring `DLSInstance` / `DLSSolution` one-to-one (class tag,
  energy + power sub-objects, full Gantt timing). Write side only — a typed
  binding builds instances directly; display / HTTP / CLI need structured output.
- **CLI `dls solve --json`**: emits `{instance, lowerBound, solution}` with full
  per-fragment timing — the front-end's immediate data source.

Contract shapes (illustrative):
```jsonc
{ "instance": { "class":"dls", "totalLoad":100, "beta":0,
                "originator":{"powerNetwork":8,"powerIdle":4},
                "processors":[ {"S":1,"C":0.1,"A":0.2,"B":1e18,
                                "power":{"idle":3,"startup":5,"network":10},
                                "energyPieces":[[0,2]] } ] },
  "lowerBound": 31,
  "solution": { "status":"Optimal","makespan":31,"cost":0,"energy":473,
                "sequence":[0],
                "fragments":[{"proc":0,"load":100,
                              "commStart":0,"commFinish":11,
                              "computeStart":11,"computeFinish":31}] } }
```

## Integration boundary — options

| Option | How | Best when |
|---|---|---|
| CLI + JSON | shell out to `dls --json` | quickest bridge, scripting |
| **pybind11 module** | `import dls` in Python | **research/notebook workflow (recommended MVP)** |
| Local HTTP server | cpp-httplib serving JSON | polished/shareable web app |
| WASM (Emscripten) | compile lib → browser | zero-install demo (simplex-only) |
| Native (Qt/ImGui) | link lib directly | offline desktop tool |

**Recommended path:** JSON contract (done) → **pybind11 → Streamlit** MVP, keeping
**HTTP + React** and **WASM** as later paths on the same contract. Rationale: the
dependency-free simplex core makes WASM viable later; pybind11 unlocks Python
plotting (Plotly Gantt, heatmaps) and doubles as a scripting layer; HiGHS stays
optional in the module.

## GUI views ↔ backend

1. Instance editor → `DLSInstance` + `validate()`; class selector.
2. Run panel → `makeSolver` + `SolverConfig` (form grouping mirrors the
   problem / runtime / algorithm parameter split).
3. **Schedule Gantt** → `fragments` timeline (needs `expandSchedule`).
4. KPI cards → makespan, energy, cost, lower bound, gap `(Cmax−LB)/LB`, wall time.
5. Time–energy Pareto explorer → makespan-limit sweep over `MinCost` energy.
6. Iso-maps → heatmap/contour from `computePerformanceMap`.
7. Portfolio compare → `runBenchmark` quality/time scatter.

## Cross-cutting concerns

Long solves (async + `timeLimitSeconds` + honest "Feasible, not proven");
reproducibility (`seed`, echoed via `usedSeed`); validation/error mapping from
`SolveStatus`; class tagging for the 7 problem classes (MVP = single-load DLS);
build matrix (with/without HiGHS — the GUI reads `availableSolvers()`); if served,
bind localhost and cap instance/grid sizes.

## Current front-end: FastAPI + SvelteKit (2026-06-16)

The portal is now a **SvelteKit single-page app** talking to a **FastAPI** JSON
API (`frontend/api/main.py`) that also serves the built SPA — one `uvicorn`
process. FastAPI calls the same `dls` ctypes package over the C-ABI; the backend
is untouched. Tooling: Node 20 + SvelteKit (Svelte 4, adapter-static, Tailwind,
bootstrap-icons, plotly.js). Build the UI with `cd frontend/ui && npm run build`
(→ `ui/build`), then `uvicorn api.main:app`. See `frontend/README.md`. The earlier
Streamlit MVP (below) has been retired; the JSON/C-ABI contract it relied on is
unchanged, which is exactly why the swap was cheap.

## Phase 1 — MVP GUI (historical: Streamlit)

Chosen path: a Python module + Streamlit. The intended binding was **pybind11**,
but a minimal sandbox (no `pip`/pybind11/network) made a compiled extension
impractical, so Phase 1 ships the functional equivalent with **zero Python
dependencies for the core**: a **C-ABI shared library driven via `ctypes`**.

Repo layout enforces the one-way `frontend → binding → library` rule:
```
lib/        the self-sustained C++ library + CLIs + tests (LP engines in engines/)
bindings/   C-ABI adapter libdls_c.so (opt-in: -DDLS_WITH_BINDINGS=ON, default)
frontend/   the Streamlit GUI (Python; depends only on the binding's .so)
deploy/     ops: backup_dls.sh, dls-studio.service
```

```
Streamlit app (frontend/app.py)              # editor, solve, KPIs, Gantt
        │  dicts in, dicts out
   dls package (frontend/dls/__init__.py)     # stdlib only: ctypes + json
        │  instance text in, JSON contract out  (C ABI)
   libdls_c.so (bindings/dls_c.cpp)           # dls_solve / dls_solvers / dls_free
```

- `bindings/dls_c.cpp` → `dls_c` target (links the backend's `dls_util`/`dls_highs`).
- `frontend/dls`: `available_solvers()`, `solve(instance_dict_or_text, solver,
  **opts)`, `instance_to_text()` — returns the JSON contract as dicts (full Gantt
  timing via `expandSchedule`).
- `frontend/app.py`: instance editor, solver picker, KPI cards (makespan / LB / gap
  / energy / time), Plotly Gantt; `frontend/requirements.txt`, `frontend/README.md`.

Swapping in a real **pybind11** module later changes only the loader inside
`frontend/dls` — the `available_solvers` / `solve` / `instance_to_text` surface and
the whole Streamlit app stay put.

## Roadmap

- **Phase 0 — contract** ✅ JSON emitters + `expandSchedule` + `dls --json`.
- **Phase 1 — MVP GUI** ✅ C-ABI/ctypes `dls` package + Streamlit (editor, solve,
  Gantt, KPIs). *(pybind11 swap-in pending a pip-capable environment.)*
- **Phase 2 — analysis views** ✅ four-tab app: Solve & Gantt, **time–energy
  Pareto** (new `lib/core/pareto.hpp` `timeEnergyFront` + `dls_pareto`), **iso-map**
  contour (`dls_map` → `computePerformanceMap`), **portfolio compare** (`dls_bench`
  → `runBenchmark`). Python: `dls.pareto / iso_map / benchmark`.
- **Phase 3 — breadth:** the other 6 problem classes; async/cancel; save/load.
- **Phase 4 (optional):** HTTP + React product and/or WASM demo on the same contract.
