# Examples

All examples use the same 3-processor instance so results are directly
comparable across languages. Symbols follow [MODEL.md](MODEL.md).

| Processor | S    | C    | A    | B         |
|-----------|------|------|------|-----------|
| P0        | 0.0  | 0.1  | 0.50 | unbounded |
| P1        | 0.1  | 0.2  | 0.30 | unbounded |
| P2        | 0.2  | 0.3  | 0.20 | unbounded |

Total load V = 1000.  Solver: Auto (picks the best method for the instance).

---

## C++ — direct library usage

**File:** `lib/examples/hello_dls.cpp`

Includes the library headers directly; no binding layer involved.

```bash
cmake -B build && cmake --build build --target hello_dls
./build/bin/hello_dls
```

```
Makespan : 214.6186
Sequence : P0 P1 P2 P1 P2
Fragments:
  P0  load=357.70  compute [35.7698, 214.6186]
  P1  load=179.23  compute [71.7162, 125.4858]
  P2  load=178.57  compute [125.4858, 161.1989]
```

---

## C — via libdls_c.so

**File:** `examples/c/hello_dls.c`

Uses `dlopen` to load the C-ABI binding at runtime. No C++ toolchain needed
at link time; any C99 compiler works.

```bash
# build the binding first
cmake -B build && cmake --build build --target dls_c

# compile the example
gcc examples/c/hello_dls.c -ldl -o hello_dls_c

# run
DLS_LIB=build/bin/libdls_c.so ./hello_dls_c
```

Result is a raw JSON string — pipe through `python3 -m json.tool` for
pretty-printing.

---

## Python — via ctypes

**File:** `examples/python/hello_dls.py`

Loads `libdls_c.so` via `ctypes`. No pip installs; standard library only.

```bash
DLS_LIB=build/bin/libdls_c.so python examples/python/hello_dls.py
```

```
Status   : Feasible
Makespan : 214.6186
Sequence : P0 P1 P2 P1 P2
Fragments:
  P0  load=357.70  compute [35.7698, 214.6186]
  P1  load=179.23  compute [71.7162, 125.4858]
  P2  load=178.57  compute [125.4858, 161.1989]
```

---

## Instance text format

The C and Python examples pass the instance as a plain text string — the same
format the CLI and C-ABI binding use:

```
V <totalLoad>
<S> <C> <A> <B> <p> <r> <d> <f> <l>   ← one line per processor
[power <P_idle> <P_startup> <P_network>]
[energy <intercept,slope> ...]
```

All fields after B are optional and default to 0.  `B ≤ 0` means unbounded
memory (use `1e+18` for an explicit large value that parses cleanly).
See [MODEL.md](MODEL.md) for full field definitions.

---

## portfolio_compare — solver portfolio benchmark

**File:** `lib/examples/portfolio_compare.cpp`

Runs GA, Exact B&B, and Exact MILP on a canonical 5-processor instance and
prints a comparison table. Requires the HiGHS build:

```bash
cmake -B build-highs -DDLS_WITH_HIGHS=ON && cmake --build build-highs
./build-highs/bin/portfolio_compare
```

```
Solver       Variant          Status        Makespan   Iter/Nodes   Time(ms)
GA           multi (=L)       Feasible     118252.24           66       27.0
Exact B&B    multi (<=L)      Optimal      117507.42         3905     1572.1
Exact B&B    single (<=L)     Optimal      120296.70          325       75.2
Exact MILP   single (<=L)     Optimal      120296.70            0      563.2
```

Key observations:

- GA is within 0.6% of the proven optimum (118252 vs 117507).
- Exact B&B (multi) and Exact MILP (single) are independent methods that
  agree exactly on the single-installment sub-problem, cross-validating each other.
- Multi-installment (`multi`) always gives a makespan ≤ single-installment (`single`).
