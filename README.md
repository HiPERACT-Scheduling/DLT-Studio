# DLS — Divisible Load Scheduling

A C++ library and solver portfolio for divisible-load scheduling, with an
optional GUI. The repo is split so the **library is self-sustained** and the
dependency only ever flows one way: **frontend → binding → library**.

```
lib/        the C++ library (the backend): solvers (core/ heuristics/ exact/,
            LP engines in engines/), CLIs (dls, dls-bench, ga), tests, examples,
            vcpkg. Builds and runs ALONE.
bindings/   optional C-ABI adapter (libdls_c.so) so any language can drive the
            library. Depends only inward on lib/; the library never on it.
frontend/   the Streamlit GUI + the ctypes `dls` Python package. Depends only on
            the binding's .so.
deploy/     ops: backup_dls.sh, dls-studio.service.
docs/       USAGE, MODEL, VALIDATION, CHOOSING, GUI, BUILD.
references/ source theses mined for the solver portfolio.
```

## Build

```bash
# library + binding, dependency-free (CSimplex)
cmake -B build && cmake --build build && ./build/bin/dls_tests

# + HiGHS LP/MILP solvers (exact-milp, milp-multi) via in-repo vcpkg
cmake -B build-highs -DDLS_WITH_HIGHS=ON && cmake --build build-highs

# the LIBRARY ALONE (no binding, the self-sustained litmus test)
cmake -B build-lib lib && cmake --build build-lib
```
Options: `-DDLS_WITH_HIGHS=ON` (HiGHS backend), `-DDLS_WITH_BINDINGS=ON|OFF`
(build the C-ABI binding; default ON). Binaries land in `<build>/bin/`.

## Use

```bash
./build/bin/dls solve --solver=best-rate instance.txt      # CLI
./build/bin/dls solve --solver=auto --json instance.txt    # JSON for a front-end
./build/bin/dls-bench --procs=4 --instances=20             # portfolio comparison
./build/bin/dls-bench map --x=procs --y=startup --xrange=1:16:16 --yrange=0:2:9  # iso-map
```
GUI: see [frontend/README.md](frontend/README.md). Model & notation:
[docs/MODEL.md](docs/MODEL.md). Front-end architecture: [docs/GUI.md](docs/GUI.md).
