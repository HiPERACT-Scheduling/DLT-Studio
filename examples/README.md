# examples/

Working code showing how to use the DLS library from different languages.
All examples solve the same 3-processor instance so results are comparable.

| Folder | Language | Entry point | Requires |
|---|---|---|---|
| `cpp/` | C++20 | `hello_dls.cpp` | CMake build (`lib/`) |
| `c/` | C99 | `hello_dls.c` | `libdls_c.so` from `bindings/` |
| `python/` | Python 3 | `hello_dls.py` | `libdls_c.so`, stdlib only |

`cpp/portfolio_compare.cpp` runs the full solver portfolio on one instance and
prints a comparison table (requires the HiGHS build).

## Build and run

### C++

```bash
# from the repo root
cmake -B build && cmake --build build --target hello_dls
./build/bin/hello_dls
```

### C

```bash
cmake -B build && cmake --build build --target dls_c
gcc examples/c/hello_dls.c -ldl -o hello_dls_c
DLS_LIB=build/bin/libdls_c.so ./hello_dls_c
```

### Python

```bash
cmake -B build && cmake --build build --target dls_c
DLS_LIB=build/bin/libdls_c.so python examples/python/hello_dls.py
```

## Expected output (all three produce the same result)

```
Makespan : 214.6186
Sequence : P0 P1 P2 P1 P2
Fragments:
  P0  load=357.70  compute [35.7698, 214.6186]
  P1  load=179.23  compute [71.7162, 125.4858]
  P2  load=178.57  compute [125.4858, 161.1989]
  P1  load=178.07  compute [161.1989, 214.6186]
  P2  load=106.44  compute [193.3307, 214.6186]
```
