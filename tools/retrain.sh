#!/usr/bin/env bash
# tools/retrain.sh
#
# Full ML pipeline: generate data → train → rebuild → restart portal.
# Run from the repo root:
#
#   bash tools/retrain.sh [--n 5000] [--depth 6] [--trees 150] [--workers N]
#
# Data generation parallelises across --workers processes (default: all cores).
# On a single-core host this has no wall-clock effect; the generators stay
# deterministic regardless of worker count (seed is pinned per instance).
#
# To re-train only (CSV already exists):
#   bash tools/retrain.sh --skip-generate

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

N=5000; DEPTH=6; TREES=150; SKIP_GEN=0; WORKERS="$(nproc 2>/dev/null || echo 1)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --n)            N="$2";       shift 2;;
    --depth)        DEPTH="$2";   shift 2;;
    --trees)        TREES="$2";   shift 2;;
    --workers)      WORKERS="$2"; shift 2;;
    --skip-generate) SKIP_GEN=1;  shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "=== DLT Studio ML retrain ==="

if [[ $SKIP_GEN -eq 0 ]]; then
  echo "--- Step 1: generate training data ($N instances, $WORKERS worker(s)) ---"
  python3 tools/generate_training_data.py --n "$N" --workers "$WORKERS"
else
  echo "--- Step 1: skipped (--skip-generate) ---"
fi

echo "--- Step 2a: train solver selector GBM and export C++ header ---"
python3 tools/train_solver_selector.py --depth "$DEPTH" --trees "$TREES"

echo "--- Step 2b: train difficulty predictor GBM and export C++ header ---"
python3 tools/train_difficulty_predictor.py --depth "$DEPTH" --trees "$TREES"

echo "--- Step 2c: train makespan predictor GBM and export C++ header ---"
python3 tools/train_makespan_predictor.py --depth "$DEPTH" --trees "$TREES"

echo "--- Step 2d: generate MLSD training data and train MLSD Cmax predictor ---"
# Keep the exact mlsd-milp oracle on for small instances (Phase 2): raises the
# exact-labelled fraction. Set MLSD_MILP_N=0 to disable (falls back to mlsd-ga).
MLSD_MILP_N=${MLSD_MILP_N:-3}
python3 tools/generate_mlsd_training_data.py --workers "$WORKERS" \
        --milp-max-n "$MLSD_MILP_N" --milp-max-nm 9
python3 tools/train_mlsd_predictor.py --depth "$DEPTH" --trees "$TREES"

echo "--- Step 2e: non-star classes (chain, tree, mapreduce, multilayer) — fast exact labels ---"
# These oracles are sub-millisecond (LP-exact / closed-form), so scale up the
# sample count for free. multilayer's label is a feasible upper bound, not the
# proven optimum (recorded as label_is_exact=0 in its rows).
TOPO_N=${TOPO_N:-100000}
for cls in chain tree mapreduce multilayer; do
  python3 tools/generate_topology_training_data.py --class "$cls" --n "$TOPO_N" --workers "$WORKERS"
  python3 tools/train_topology_predictor.py --class "$cls" --depth "$DEPTH" --trees "$TREES"
done

echo "--- Step 3: rebuild libdls_c.so (HiGHS build) ---"
cmake --build build-highs --target dls_c

echo "--- Step 4: restart portal API ---"
if sudo systemctl is-active dls-studio &>/dev/null; then
  sudo systemctl restart dls-studio
  echo "dls-studio restarted"
else
  echo "(dls-studio not running, skip restart)"
fi

echo "=== Done. auto-ml, difficulty, makespan, MLSD and non-star (chain/tree/mapreduce/multilayer) predictors trained. ==="
