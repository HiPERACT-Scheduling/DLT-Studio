# DLT Studio — Divisible Load Theory Library

A C++20 solver portfolio for Divisible Load Theory (DLT) scheduling, with a
C-ABI binding for language-agnostic integration. The library is fully
self-sustained; the dependency flows strictly one way:

```
examples/   language examples (C++, C, Python)
bindings/   C-ABI adapter (libdls_c.so) — any language with a C FFI can use this
lib/        the C++ library: solvers, CLIs, tests, LP engines
docs/       reference documentation
```

---

## What is Divisible Load Theory?

Divisible Load Theory (DLT) models the optimal distribution of a large,
arbitrarily splittable workload across heterogeneous networked processors. A
single **master** node sends chunks of a total load **V** to **N** workers over a
star network. Communication is **single-port**: the master sends to one worker at
a time, in a chosen sequence. Each worker *i* has:

- a communication cost `Sᵢ + Cᵢ·α` to receive load `α` (startup + rate),
- a computation cost `Aᵢ·α` to process it, and
- an optional buffer cap `Bᵢ` per installment.

The objective is to **minimize the makespan Cmax** — the time the last worker
finishes. The optimal load split for a fixed sequence is a linear program; the
optimal sequence is the combinatorial search that the solvers in this library
address.

DLT arises in scientific computing, data-intensive applications, and distributed
processing pipelines wherever the work can be partitioned continuously and sent in
one or more installments.

---

## Features

- **Solver portfolio** — exact, heuristic, and FPTAS solvers for the standard
  single-load star model (see table below); auto-selector picks the best method
  for any given instance.
- **Multi-installment scheduling** — one processor may receive several chunks,
  with buffer limits enforced per installment.
- **Multiple objectives** — minimize makespan, maximize load within a deadline
  (OptV), minimize energy, bi-criteria Pareto front.
- **Energy model** — four-state power model (idle / startup / networking /
  running); convex piecewise running energy for memory-hierarchy effects.
- **Multiple problem classes** — single-load star (core), Multiple Loads Single
  Destination (MLSD), MapReduce with precedence, multilayer pipelines.
- **Topology solvers** — linear chain, multi-level tree, general graph; each
  solved as an allocation LP over the optimal spanning arborescence.
- **Isoefficiency and Pareto maps** — 2-D sweep over instance parameters,
  producing makespan / energy surfaces for visualization.
- **Benchmarking** — built-in `dls-bench` CLI compares solvers on random or
  structured instances and reports gap-to-best statistics.
- **Two LP backends** — built-in CSimplex (no external dependency) and optional
  HiGHS (adds MILP solvers and a faster LP for large instances).
- **C-ABI binding** — `libdls_c.so` exposes the full solver portfolio to any
  language with a C FFI (Python, C, Julia, Rust, …).

---

## Solver portfolio

| Solver | Category | When to use |
|---|---|---|
| `auto` | meta | default — inspects the instance and dispatches to the best method below |
| `exact` | Exact | small instances (N ≤ 6); branch-and-bound, proven optimum |
| `exact-dual` | Exact | small instances; dual bisection via OptV — independent optimum check |
| `exact-milp` | Exact (HiGHS) | single-installment, any N; MILP over assignment + loads |
| `milp-multi` | Exact (HiGHS) | multi-installment MILP; strongest exact method |
| `single-round` | Exact (closed form) | all `Sᵢ = 0`, ample memory — closed-form optimum, instant |
| `optv` | Exact | maximize load within a deadline T (the dual of minimizing makespan) |
| `fptas-optv` | FPTAS | `Cᵢ = 0`; `≥ (1−ε)·V_OPT`, polynomial in `1/ε` |
| `fptas-optt` | FPTAS | `Cᵢ = 0`; `≤ (1+ε)·T_OPT`, polynomial in `1/ε` |
| `ga` | Heuristic | larger N with startups; genetic search, best general-purpose quality |
| `best-rate` | Heuristic | memory-limited instances (`Σ Bᵢ < V`); greedy multi-installment |
| `online` | Heuristic | very large instances; LP-free online simulation (PSR + GSS/SSC) |

Separate solvers exist for the **MLSD** (`mlsd-exact`, `mlsd-ga`),
**MapReduce** (`mapreduce`), and **multilayer** (`multilayer`) problem classes.

See [docs/CHOOSING.md](docs/CHOOSING.md) for guidance on picking a solver.

---

## Quick start

```bash
# dependency-free build (simplex LP engine)
cmake -B build && cmake --build build

# with HiGHS LP/MILP solvers (exact-milp, milp-multi)
cmake -B build-highs -DDLS_WITH_HIGHS=ON && cmake --build build-highs

# run tests
./build-highs/bin/dls_tests          # 187 tests, 0 failures

# run the minimal C++ example
./build/bin/hello_dls
```

Build options: `-DDLS_WITH_HIGHS=ON` enables the HiGHS backend;
`-DDLS_WITH_BINDINGS=OFF` skips the C-ABI binding. All binaries land in
`<build>/bin/`.

---

## Usage examples

### C++ (direct library)

```cpp
#include "core/dls_instance.hpp"
#include "heuristics/auto/auto_solver.hpp"
#include "core/schedule_expand.hpp"
using namespace dls;

DLSInstance inst(
    {
        Processor{.commStartup=0.0, .commRate=0.1, .computeRate=0.50, .memoryLimit=1e18},
        Processor{.commStartup=0.1, .commRate=0.2, .computeRate=0.30, .memoryLimit=1e18},
        Processor{.commStartup=0.2, .commRate=0.3, .computeRate=0.20, .memoryLimit=1e18},
    },
    1000.0   // total load V
);

DLSSolution sol = AutoSolver{}.solve(inst, SolverConfig{});
expandSchedule(inst, sol);   // fills commStart / computeStart / computeFinish

// sol.makespan, sol.fragments[i].{proc, load, computeStart, computeFinish}
```

### Python (via C binding)

```python
import ctypes, json, glob

lib = ctypes.CDLL("build/bin/libdls_c.so")
lib.dls_solve.restype  = ctypes.c_void_p
lib.dls_solve.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
lib.dls_free.argtypes  = [ctypes.c_void_p]

instance = b"""
V 1000.0
0.0 0.1 0.5 1e+18 0.0 0.0 0.0 0.0 0.0
0.1 0.2 0.3 1e+18 0.0 0.0 0.0 0.0 0.0
0.2 0.3 0.2 1e+18 0.0 0.0 0.0 0.0 0.0
"""

ptr = lib.dls_solve(instance, b"auto", b"")
result = json.loads(ctypes.cast(ptr, ctypes.c_char_p).value)
lib.dls_free(ptr)

sol = result["solution"]
print(f"Makespan: {sol['makespan']:.4f}")
```

### Command line

```bash
# solve a text-format instance file
./build/bin/dls solve --solver=auto instance.txt

# compare solvers on random instances
./build/bin/dls-bench --n=6 --runs=50
```

See [examples/](examples/) for full runnable code and
[docs/USAGE.md](docs/USAGE.md) for the complete CLI reference.

---

## Requirements

| Requirement | Version |
|---|---|
| C++ compiler (g++ or clang++) | C++20 support required |
| CMake | 3.16 or later |
| HiGHS (optional) | pulled via vcpkg, pinned in `vcpkg.json` |
| Python (optional, for the Python example) | 3.7 or later; no pip installs |
| OS | Linux (primary); macOS and Windows (community-tested) |

The dependency-free build (`cmake -B build`) needs only CMake and a C++20
compiler. HiGHS is downloaded and built automatically by vcpkg when
`-DDLS_WITH_HIGHS=ON` is passed — no manual installation required.

---

## License

MIT License — see [LICENSE](LICENSE) for the full text.

Copyright (c) 2023-2026 Marcin Lawenda, Poznan Supercomputing and Networking Center

---

## Documentation

| File | Contents |
|---|---|
| [docs/MODEL.md](docs/MODEL.md) | problem model, notation, and field definitions |
| [docs/BUILD.md](docs/BUILD.md) | detailed build instructions and CMake options |
| [docs/USAGE.md](docs/USAGE.md) | CLI and C++ API usage guide |
| [docs/CHOOSING.md](docs/CHOOSING.md) | when to use which solver |
| [docs/EXAMPLES.md](docs/EXAMPLES.md) | annotated examples across languages |
| [docs/VALIDATION.md](docs/VALIDATION.md) | test coverage and validation strategy |
