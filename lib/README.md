# lib/ — DLS C++ library

The self-sustained C++20 header-only scheduling library. It builds and runs
on its own with no dependency on `bindings/` or any GUI.

## Directory layout

```
core/           DLSInstance, DLSSolution, LP model, schedule evaluators, JSON I/O
heuristics/     GA, Best Rate, Online PSR+GSS/SSC, Single Round, Auto (meta-solver)
                fptas/  FPTAS OptV and OptT for infinite-bandwidth instances
exact/          Branch-and-bound (enumerative/), OptV, Dual Bisection, MILP
                (MILP solvers require the HiGHS build)
engines/        LP backends: simplex (dep-free) and HiGHS (optional)
mlsd/           Multiple Loads Single Destination problem class
mapreduce/      MapReduce scheduling (map → reduce precedence)
topology/       Network topology parsers (chain, tree, graph)
bench/          Benchmarking and isoefficiency map utilities
cli/            dls and dls-bench command-line interfaces
tests/          Single-translation-unit doctest suite (187 tests)
util/           RNG, timing
vcpkg/          vcpkg submodule (HiGHS dependency, used only with -DDLS_WITH_HIGHS=ON)
vcpkg.json      vcpkg manifest (declares the HiGHS version)
```

## Build

```bash
# dependency-free (simplex LP engine, 162 tests)
cmake -B build-lib lib && cmake --build build-lib

# full build including HiGHS (187 tests, adds exact-milp and milp-multi)
cmake -B build-highs -DDLS_WITH_HIGHS=ON && cmake --build build-highs

./build-highs/bin/dls_tests
```

See [docs/BUILD.md](../docs/BUILD.md) for all options.
