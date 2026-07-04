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

## Documentation

| File | Contents |
|---|---|
| [docs/MODEL.md](docs/MODEL.md) | problem model, notation, and field definitions |
| [docs/BUILD.md](docs/BUILD.md) | detailed build instructions and CMake options |
| [docs/USAGE.md](docs/USAGE.md) | CLI and C++ API usage guide |
| [docs/CHOOSING.md](docs/CHOOSING.md) | when to use which solver |
| [docs/EXAMPLES.md](docs/EXAMPLES.md) | annotated examples across languages |
| [docs/VALIDATION.md](docs/VALIDATION.md) | test coverage and validation strategy |
