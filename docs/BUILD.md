# Building DLS

A C++17 library for Divisible Load Scheduling. Two build configurations:

- **dependency-free** (default) — uses the built-in CSimplex LP backend; needs
  only CMake + a C++17 compiler.
- **with HiGHS** (opt-in) — adds the HiGHS LP/MILP backend via vcpkg.

Requirements: CMake >= 3.16 and a C++17 compiler (g++/clang++).

---

## 1. Dependency-free build (default)

```bash
cmake -B build -S .
cmake --build build
```

Run the tests:

```bash
ctest --test-dir build --output-on-failure
# or directly:
./build/bin/dls_tests
```

Run the GA solver CLI (the sample input files live in `heuristics/ga/`):

```bash
cd heuristics/ga
../../build/bin/ga prog_par.txt proc_par.txt ga_par.txt
```

Select the LP backend explicitly (default is `simplex`):

```bash
../../build/bin/ga --evaluator=simplex prog_par.txt proc_par.txt ga_par.txt
```

`--evaluator=highs` is only available in the HiGHS build (below); otherwise it
prints a clear "unavailable" error.

---

## 2. Build with the optional HiGHS backend

HiGHS is pulled in through **vcpkg**, kept entirely inside the project
(`./vcpkg`). The HiGHS version is pinned by `vcpkg.json`'s `builtin-baseline`.

### One-time: bootstrap vcpkg into the repo

```bash
# build prerequisites for vcpkg + building ports
sudo apt-get install -y curl zip unzip tar ninja-build pkg-config

git clone https://github.com/microsoft/vcpkg.git vcpkg
./vcpkg/bootstrap-vcpkg.sh -disableMetrics
```

This creates `./vcpkg` (git-ignored). Both the toolchain and vcpkg's binary
cache (`./vcpkg/.binarycache`) stay inside the project — nothing is written to
`$HOME`.

### Configure, build, test

```bash
cmake -B build-highs -S . -DDLS_WITH_HIGHS=ON
cmake --build build-highs
ctest --test-dir build-highs --output-on-failure
```

`-DDLS_WITH_HIGHS=ON` is all you need: CMake auto-detects `./vcpkg` and sets the
toolchain + binary-cache location for you (no `-DCMAKE_TOOLCHAIN_FILE` required).
The first configure builds HiGHS from source (a few minutes); later configures
restore it from `./vcpkg/.binarycache` in seconds.

Then both backends are selectable:

```bash
cd heuristics/ga
../../build-highs/bin/ga --evaluator=highs   prog_par.txt proc_par.txt ga_par.txt
../../build-highs/bin/ga --evaluator=simplex prog_par.txt proc_par.txt ga_par.txt
```

---

## 3. Layout notes

- Executables go to `<build-dir>/bin` (e.g. `build/bin/ga`, `build/bin/dls`,
  `build/bin/dls-bench`), so the two configurations never clobber each other.
  `dls` is the uniform portfolio CLI (`dls list` / `dls solve --solver=NAME ...`);
  `dls-bench` compares solvers on quality vs. run time.
- `build*/`, `vcpkg/`, and `vcpkg_installed/` are git-ignored (tooling/output,
  not source).
- The dependency-free build never touches vcpkg; selecting `--evaluator=highs`
  there fails gracefully.

---

## 4. Continuous integration

`.github/workflows/ci.yml` runs two GitHub Actions jobs on every push/PR:

- **dependency-free** — `cmake`/`ctest` with no external deps (guards the
  always-builds guarantee).
- **with HiGHS** — configures `-DDLS_WITH_HIGHS=ON` using the runner's
  preinstalled vcpkg (`$VCPKG_INSTALLATION_ROOT`), then builds and tests.
