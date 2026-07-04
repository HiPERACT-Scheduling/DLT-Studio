# Examples

Runnable examples live in `examples/`. They use the symbols defined in
[MODEL.md](MODEL.md) (S, C, A, B, V, N, L, Cmax).

## portfolio_compare

Runs the solver portfolio on one instance and prints a comparison table. It
requires the HiGHS build (it uses the HiGHS-based MILP solver), so build with
the optional backend enabled (see [BUILD.md](BUILD.md)):

```bash
cmake -B build-highs -S . -DDLS_WITH_HIGHS=ON
cmake --build build-highs
./build-highs/bin/portfolio_compare
```

Sample output (instance `canonical-5`, N=5, V=1e6, L=5):

```
Solver       Variant          Status        Makespan   Iter/Nodes   Time(ms)
GA           multi (=L)       Feasible     118252.24           66       27.0
Exact B&B    multi (<=L)      Optimal      117507.42         3905     1572.1
Exact B&B    single (<=L)     Optimal      120296.70          325       75.2
Exact MILP   single (<=L)     Optimal      120296.70            0      563.2
```

### How to read it

- **GA** — heuristic, multi-installment, uses exactly L installments. Fast, no
  optimality guarantee. `Iter/Nodes` = generations run.
- **Exact B&B (multi)** — branch-and-bound over sequences of length ≤ L,
  multi-installment. The true optimum the GA approximates: here
  `117507 ≤ 118252`, so the GA is ~0.6% above optimal. `Iter/Nodes` = nodes
  explored (3905 = full enumeration, because this instance has S=0 so the
  startup bound cannot prune — see MODEL.md / the B&B notes).
- **Exact B&B (single)** and **Exact MILP (single)** — two independent exact
  methods for the single-installment problem (each processor used once). They
  agree exactly (`120296.70 == 120296.70`), cross-validating each other.
- **single ≥ multi**: the single-installment optimum (120296) is worse than the
  multi-installment optimum (117507), as expected — more installments give more
  freedom.

`Time(ms)` is wall-clock for that solve. The MILP reports `0` under Iter/Nodes
(it does not expose a node count).
