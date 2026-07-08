#!/usr/bin/env bash
# tools/retrain.sh
#
# Full ML pipeline: generate data → train → rebuild → restart portal.
# Run from the repo root:
#
#   bash tools/retrain.sh [--n 5000] [--depth 6] [--trees 150]
#
# To re-train only (CSV already exists):
#   bash tools/retrain.sh --skip-generate

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

N=5000; DEPTH=6; TREES=150; SKIP_GEN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --n)            N="$2";     shift 2;;
    --depth)        DEPTH="$2"; shift 2;;
    --trees)        TREES="$2"; shift 2;;
    --skip-generate) SKIP_GEN=1; shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "=== DLT Studio ML retrain ==="

if [[ $SKIP_GEN -eq 0 ]]; then
  echo "--- Step 1: generate training data ($N instances) ---"
  python3 tools/generate_training_data.py --n "$N"
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
python3 tools/generate_mlsd_training_data.py
python3 tools/train_mlsd_predictor.py --depth "$DEPTH" --trees "$TREES"

echo "--- Step 3: rebuild libdls_c.so (HiGHS build) ---"
cmake --build build-highs --target dls_c

echo "--- Step 4: restart portal API ---"
if sudo systemctl is-active dls-studio &>/dev/null; then
  sudo systemctl restart dls-studio
  echo "dls-studio restarted"
else
  echo "(dls-studio not running, skip restart)"
fi

echo "=== Done. auto-ml, difficulty, makespan and MLSD predictors are using trained models. ==="
