#!/usr/bin/env python3
"""
tools/generate_topology_training_data.py

Generate exact-labelled training data for non-star topology divisible load:
a linear **chain** or a multi-level **tree**. Both have LP-exact solvers
(`dls_topology`) that run in well under a millisecond, so — unlike the MILP
oracles — these classes yield proven-optimal labels at massive scale even on a
single core. Each row's label is the optimal makespan; every row is exact.

Instance text formats (cli/class_io.hpp), both assume link startup S = 0:
  chain: "V <load>" then "node <A> <C>" in chain order (node 0 = originator,
         its C is unused).
  tree:  "V <load>" then "node <A> <C> <parent>" in node-index order
         (node 0 = root, parent -1).

Every row carries provenance columns (genlib.PROVENANCE_COLS); oracle is the
exact topology solver and label_is_exact is always 1.

Output: tools/{chain,tree}_training_data.csv

Usage:
    python3 tools/generate_topology_training_data.py --class chain [--n 20000] [--workers N]
    python3 tools/generate_topology_training_data.py --class tree  [--n 20000] [--workers N]
"""

import argparse
import os
import math
import random

import genlib   # shared harness: lib loading, parallel driver, provenance, CSV

# Regimes vary size and heterogeneity; weights sum to 1.
REGIMES = [0, 1, 2, 3]
WEIGHTS = [0.30, 0.30, 0.20, 0.20]

def _cv(vals):
    m = sum(vals) / len(vals)
    if m < 1e-12:
        return 0.0
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return math.sqrt(max(var, 0.0)) / m

# ── instance sampling ──────────────────────────────────────────────────────

def sample_nodes(rng, regime):
    """Return (N, V, A[], C[]) shared by chain and tree sampling."""
    if regime == 0:                                  # small, mild heterogeneity
        N = rng.randint(2, 8);  V = rng.uniform(100, 3000)
        A = [rng.uniform(0.2, 1.5) for _ in range(N)]
        C = [rng.uniform(0.02, 0.6) for _ in range(N)]
    elif regime == 1:                                # medium
        N = rng.randint(4, 14); V = rng.uniform(200, 6000)
        A = [rng.uniform(0.1, 2.0) for _ in range(N)]
        C = [rng.uniform(0.01, 1.0) for _ in range(N)]
    elif regime == 2:                                # large
        N = rng.randint(10, 24); V = rng.uniform(500, 12000)
        A = [rng.uniform(0.1, 2.0) for _ in range(N)]
        C = [rng.uniform(0.01, 0.8) for _ in range(N)]
    else:                                            # strong heterogeneity
        N = rng.randint(3, 16); V = rng.uniform(100, 8000)
        A = [rng.uniform(0.05, 3.0) for _ in range(N)]
        C = [rng.uniform(0.005, 1.2) for _ in range(N)]
    return N, V, A, C

def chain_text(V, A, C):
    lines = [f"V {V:.6f}"]
    for a, c in zip(A, C):
        lines.append(f"node {a:.6f} {c:.6f}")
    return "\n".join(lines)

def tree_parents(rng, N):
    """A random rooted tree over nodes 0..N-1: node i>0 attaches to a uniform
    parent in [0, i-1] (node 0 is the root). Output: parent list (parent[0]=-1)."""
    parent = [-1]
    for i in range(1, N):
        parent.append(rng.randint(0, i - 1))
    return parent

def tree_text(V, A, C, parent):
    lines = [f"V {V:.6f}"]
    for a, c, p in zip(A, C, parent):
        lines.append(f"node {a:.6f} {c:.6f} {p}")
    return "\n".join(lines)

# ── feature extraction ─────────────────────────────────────────────────────

CHAIN_FEATURES = ["N", "loadPerNode", "meanA", "heteroA", "speedupA",
                  "meanC", "heteroC", "maxMinC"]
TREE_FEATURES  = CHAIN_FEATURES + ["depth", "meanChildren", "leafFraction"]
MAPREDUCE_FEATURES  = ["nMappers", "loadPerMapper", "meanA", "heteroA", "speedupA",
                       "startup", "readRate", "gamma0",
                       "nReducers", "reducerStartup", "reducerRate"]
MULTILAYER_FEATURES = ["nMappers", "V", "mapperRate", "startup", "readRate",
                       "gamma0", "bisection", "nLayers",
                       "meanLayerCount", "meanLayerRate", "meanLayerGamma"]

def base_features(N, V, A, C):
    # node 0 is the originator/root: its link rate C[0] is unused by the model,
    # so link-rate stats are taken over the child links C[1:].
    Clinks = C[1:] if N > 1 else C
    return {
        "N": float(N),
        "loadPerNode": V / N,
        "meanA": sum(A) / N,
        "heteroA": _cv(A),
        "speedupA": (max(A) / min(A)) if min(A) > 1e-12 else 1.0,
        "meanC": sum(Clinks) / len(Clinks),
        "heteroC": _cv(Clinks),
        "maxMinC": (max(Clinks) / min(Clinks)) if min(Clinks) > 1e-12 else 1.0,
    }

def tree_shape_features(parent):
    """depth (root..deepest leaf), mean children per internal node, leaf fraction."""
    N = len(parent)
    depth = [0] * N
    children = [0] * N
    for i in range(1, N):
        depth[i] = depth[parent[i]] + 1
        children[parent[i]] += 1
    internal = [c for c in children if c > 0]
    leaves = sum(1 for c in children if c == 0)
    return {
        "depth": float(max(depth)),
        "meanChildren": (sum(internal) / len(internal)) if internal else 0.0,
        "leafFraction": leaves / N,
    }

# ── MapReduce and multilayer builders ──────────────────────────────────────

def build_chain(rng, regime):
    N, V, A, C = sample_nodes(rng, regime)
    return chain_text(V, A, C), base_features(N, V, A, C)

def build_tree(rng, regime):
    N, V, A, C = sample_nodes(rng, regime)
    parent = tree_parents(rng, N)
    return tree_text(V, A, C, parent), {**base_features(N, V, A, C),
                                        **tree_shape_features(parent)}

def build_mapreduce(rng, regime):
    m  = rng.randint(2, 6 + 5 * regime)          # more mappers in larger regimes
    V  = rng.uniform(200, 3000 * (regime + 1))
    S  = rng.uniform(0, 1.0)
    C  = rng.uniform(0.01, 0.5)
    g0 = rng.uniform(0.05, 1.0)
    r  = rng.randint(1, 8)
    sr = rng.uniform(0, 1.0)
    ar = rng.uniform(0.05, 1.5)
    A  = [rng.uniform(0.1, 2.0) for _ in range(m)]
    lines = [f"V {V:.6f}", f"startup {S:.6f}", f"readrate {C:.6f}",
             f"gamma0 {g0:.6f}", f"reducers {r}", f"reducer_startup {sr:.6f}",
             f"reducer_rate {ar:.6f}"] + [f"mapper {a:.6f}" for a in A]
    feats = {
        "nMappers": float(m), "loadPerMapper": V / m,
        "meanA": sum(A) / m, "heteroA": _cv(A),
        "speedupA": (max(A) / min(A)) if min(A) > 1e-12 else 1.0,
        "startup": S, "readRate": C, "gamma0": g0,
        "nReducers": float(r), "reducerStartup": sr, "reducerRate": ar,
    }
    return "\n".join(lines), feats

def build_multilayer(rng, regime):
    m  = rng.randint(2, 8 + 4 * regime)
    V  = rng.uniform(200, 4000 * (regime + 1))
    A  = rng.uniform(0.1, 2.0)
    S  = rng.uniform(0, 1.0)
    C  = rng.uniform(0.01, 0.5)
    g0 = rng.uniform(0.1, 1.0)
    l  = rng.randint(2, 8)
    R  = rng.randint(1, 4)
    layers = [(rng.randint(1, 8), rng.uniform(0, 0.5),
               rng.uniform(0.05, 1.0), rng.uniform(0.1, 1.0)) for _ in range(R)]
    lines = [f"V {V:.6f}", f"mappers {m}", f"mapper_rate {A:.6f}", f"startup {S:.6f}",
             f"readrate {C:.6f}", f"gamma0 {g0:.6f}", f"bisection {l}"] + \
            [f"layer {c} {s:.6f} {a:.6f} {g:.6f}" for (c, s, a, g) in layers]
    feats = {
        "nMappers": float(m), "V": V, "mapperRate": A, "startup": S,
        "readRate": C, "gamma0": g0, "bisection": float(l), "nLayers": float(R),
        "meanLayerCount": sum(x[0] for x in layers) / R,
        "meanLayerRate":  sum(x[2] for x in layers) / R,
        "meanLayerGamma": sum(x[3] for x in layers) / R,
    }
    return "\n".join(lines), feats

# Per-class metadata: builder, feature list, (oracle name, label_is_exact).
# MapReduce is the proven optimum (Prop. 4.1); multilayer is a feasible-schedule
# UPPER BOUND on the overlapped optimum, so its label is exact-of-the-solver but
# not the true optimum — recorded honestly as label_is_exact=0.
CLASS_META = {
    "chain":      (build_chain,      CHAIN_FEATURES,      "chain-lp",             True),
    "tree":       (build_tree,       TREE_FEATURES,       "tree-lp",              True),
    "mapreduce":  (build_mapreduce,  MAPREDUCE_FEATURES,  "mapreduce-closedform", True),
    "multilayer": (build_multilayer, MULTILAYER_FEATURES, "multilayer-ub",        False),
}

# ── per-instance task (runs in a worker process) ───────────────────────────

_SEED  = 42
_CLASS = "chain"

def task(index):
    """Goal: sample one non-star instance, solve it exactly, return its row.
    Output: dict {features..., makespan, provenance..., "_i"} or None on failure."""
    import time as _t
    builder, _feats, oracle, is_exact = CLASS_META[_CLASS]
    seed = genlib.derive_seed(_SEED, index)
    rng  = random.Random(seed)
    regime = rng.choices(REGIMES, weights=WEIGHTS)[0]
    text, feats = builder(rng, regime)

    t0 = _t.time()
    try:
        res = genlib.topology_solve(_CLASS, text)
        elapsed = _t.time() - t0
        if res.get("status") not in ("Optimal", "Feasible"):
            return None
        mk = float(res["makespan"])
        if mk <= 0:
            return None
    except Exception:
        return None

    prov = genlib.provenance(seed, regime, oracle=oracle,
                             label_is_exact=is_exact, wall_ms=elapsed * 1000.0)
    return {**feats, "makespan": mk, **prov, "_i": index}

# ── main ────────────────────────────────────────────────────────────────────

def main():
    global _SEED, _CLASS
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="klass",
                    choices=["chain", "tree", "mapreduce", "multilayer"], required=True)
    ap.add_argument("--n",       type=int, default=20000)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--workers", type=int, default=(os.cpu_count() or 1))
    ap.add_argument("--out",     default=None)
    args = ap.parse_args()
    _SEED  = args.seed
    _CLASS = args.klass
    out = args.out or os.path.join(os.path.dirname(__file__), f"{args.klass}_training_data.csv")

    feat_names = CLASS_META[args.klass][1]
    print(f"Generating {args.n} {args.klass} instances using {genlib.find_lib()}")
    results = genlib.run(task, args.n, workers=args.workers,
                         progress_every=5000, label=f"{args.klass} instances")

    cols = feat_names + ["makespan"] + genlib.PROVENANCE_COLS
    genlib.write_csv(out, results, cols)
    exact_n = sum(r["label_is_exact"] for r in results)
    print(f"{len(results)} rows → {out}  (exact: {exact_n}/{len(results)}, "
          f"{100*exact_n/max(len(results),1):.0f}%)")

if __name__ == "__main__":
    main()
