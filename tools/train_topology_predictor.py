#!/usr/bin/env python3
"""
tools/train_topology_predictor.py

Train a GBM regression model on log(makespan) from a topology training dataset
(chain or tree, produced by generate_topology_training_data.py) and export it as
a self-contained, zero-dependency C++ header.

The dataset's labels are exact (LP-optimal), so this is a surrogate for the
optimal makespan of a whole problem class for which no closed form exists.

Feature columns are auto-detected from the CSV header (everything except the
label and the provenance columns), so the same trainer serves chain and tree.

The exported header exposes a standalone `predict(const float* f)` taking the
features in the documented order — it needs no feature struct, so a future
ml-chain / ml-tree solver can adopt it by filling a float array.

Usage:
    python3 tools/train_topology_predictor.py --class chain
    python3 tools/train_topology_predictor.py --class tree [--depth 6] [--trees 150]
"""

import argparse
import csv
import math
import os

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.tree import _tree

# Non-feature columns: the label plus provenance (see genlib.PROVENANCE_COLS).
LABEL       = "makespan"
NON_FEATURE = {LABEL, "seed", "regime", "oracle", "label_is_exact", "solver_wall_ms"}


def load_data(path):
    """Goal: read features (auto-detected) and log(makespan) label.
    Output: (X, y, feature_names)."""
    with open(path) as f:
        reader = csv.DictReader(f)
        feature_names = [c for c in reader.fieldnames if c not in NON_FEATURE]
        X, y = [], []
        for row in reader:
            mk = float(row[LABEL])
            if mk <= 0:
                continue
            X.append([float(row[n]) for n in feature_names])
            y.append(math.log(mk))
    return np.array(X), np.array(y), feature_names


def gbm_regressor_to_cpp(model):
    """Goal: emit the ensemble as a score-accumulating C++ function body."""
    lines = [f"    double score = {model.init_.constant_[0][0]:.10f};",
             f"    const double lr = {model.learning_rate:.10f};"]
    for stage_i, stage in enumerate(model.estimators_):
        t = stage[0].tree_
        lines.append(f"    // stage {stage_i}")

        def recurse(node, depth):
            pad = "    " * (depth + 1)
            if t.feature[node] != _tree.TREE_UNDEFINED:
                fi, thr = t.feature[node], t.threshold[node]
                lines.append(f"{pad}if (f[{fi}] <= {thr:.10f}f) {{")
                recurse(t.children_left[node],  depth + 1)
                lines.append(f"{pad}}} else {{")
                recurse(t.children_right[node], depth + 1)
                lines.append(f"{pad}}}")
            else:
                lines.append(f"{pad}score += lr * {t.value[node][0][0]:.10f};")

        recurse(0, 1)
    lines.append("    return score;")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="klass",
                    choices=["chain", "tree", "mapreduce", "multilayer"], required=True)
    ap.add_argument("--data",  default=None)
    ap.add_argument("--out",   default=None)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--seed",  type=int, default=0)
    args = ap.parse_args()

    here = os.path.dirname(__file__)
    data = args.data or os.path.join(here, f"{args.klass}_training_data.csv")
    out  = args.out  or os.path.join(here, "..", "lib", "heuristics", "ml",
                                     f"{args.klass}_predictor.hpp")

    print(f"Loading {data} ...")
    X, y, feature_names = load_data(data)
    print(f"  {len(X)} {args.klass} samples, {len(feature_names)} features: {feature_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=args.seed)

    print(f"Training GBM regressor (depth={args.depth}, trees={args.trees}) ...")
    model = GradientBoostingRegressor(
        n_estimators=args.trees, max_depth=args.depth, learning_rate=0.1,
        subsample=0.8, random_state=args.seed)
    model.fit(X_train, y_train)

    mape = mean_absolute_percentage_error(np.exp(y_test), np.exp(model.predict(X_test)))
    cv   = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"  Test MAPE on makespan: {mape*100:.2f}%")
    print(f"  5-fold CV R²: {cv.mean():.3f} ± {cv.std():.3f}")
    print("\nFeature importances:")
    for name, imp in sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:16s} {imp:.4f}  {'█' * int(imp * 40)}")

    guard = f"DLS_HEURISTICS_ML_{args.klass.upper()}_PREDICTOR_HPP"
    fn    = f"predict_log_makespan_{args.klass}"
    body  = gbm_regressor_to_cpp(model)
    print(f"\nExporting C++ header to {out} ...")
    header = f"""\
//---------------------------------------------------------------------------
// heuristics/ml/{args.klass}_predictor.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// ML {args.klass}-topology makespan predictor — auto-generated by
// tools/train_topology_predictor.py. DO NOT EDIT — regenerate via the script.
//
// GBM regressor ({args.trees} stages, max_depth {args.depth}) trained on {len(X)}
// LP-exact {args.klass} benchmarks. Predicts log(makespan*); caller takes exp().
// Test MAPE: {mape*100:.1f}%. Zero runtime dependency (plain C++ arithmetic).
//
// Standalone: pass the {len(feature_names)} features as a float array in this order —
//   {', '.join(feature_names)}
//---------------------------------------------------------------------------

#ifndef {guard}
#define {guard}

namespace dls {{

static constexpr int k{args.klass.capitalize()}NumFeatures = {len(feature_names)};

// Goal:   predict log(optimal makespan) for a {args.klass} instance.
// Input:  f - the {len(feature_names)} features in the documented order.
// Output: log(makespan*); caller takes exp() to recover the makespan.
static inline double {fn}(const float* f) {{
"""
    with open(out, "w") as fh:
        fh.write(header)
        for line in body:
            fh.write(line + "\n")
        fh.write(f"}}\n\n}}  // namespace dls\n\n#endif  // {guard}\n")
    print(f"  Written: {out}  ({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    main()
