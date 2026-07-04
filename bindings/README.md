# bindings/ — C-ABI adapter (libdls_c.so)

A thin C-ABI wrapper over the DLS library so any language with a C FFI
(Python, Julia, R, Rust, …) can drive the full solver portfolio without
linking against C++ or knowing anything about the library's internals.

## Public API

Declared in [`dls_c.h`](dls_c.h):

| Function | Purpose |
|---|---|
| `dls_solve(instance, solver, opts)` | run a solver, return JSON schedule |
| `dls_pareto(instance, solver, opts)` | time–energy Pareto front |
| `dls_map(opts)` | isoefficiency / isoenergy heatmap grid |
| `dls_bench(opts)` | solver portfolio benchmark |
| `dls_topology(klass, text, opts)` | parse and schedule a network topology |
| `dls_solvers()` | list available solver IDs in this build |
| `dls_free(ptr)` | release any string returned by the above |

All functions return a heap-allocated JSON string that **must** be freed with
`dls_free()`.

## Build

```bash
# from the repo root — binding is ON by default
cmake -B build && cmake --build build --target dls_c
# → build/bin/libdls_c.so

# with HiGHS (adds exact-milp and milp-multi solvers)
cmake -B build-highs -DDLS_WITH_HIGHS=ON && cmake --build build-highs --target dls_c
# → build-highs/bin/libdls_c.so
```

## Instance text format

Instances are passed as plain text strings (same format as the `dls` CLI):

```
V <totalLoad>
<S> <C> <A> <B> <p> <r> <d> <f> <l>    ← one line per processor
[power <P_idle> <P_startup> <P_network>]
[energy <intercept,slope> ...]
```

`B = 1e18` signals an unbounded memory buffer. See
[docs/MODEL.md](../docs/MODEL.md) for field definitions.

## Usage examples

See [`examples/`](../examples/) for complete working code in C++, C, and Python.
