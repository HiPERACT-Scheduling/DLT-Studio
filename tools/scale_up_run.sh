#!/usr/bin/env bash
# tools/scale_up_run.sh
#
# One-off scale-up pass on the 8-core box: regenerate every dataset at a much
# larger N (chain/tree/mapreduce/multilayer to genuine millions — their oracles
# are sub-millisecond and exact; single-load/MLSD to a large-but-realistic size,
# since their oracles involve real heuristic-solver work per instance and don't
# scale as cheaply), retrain every predictor, rebuild, and restart the portal.
#
# Energy is intentionally NOT regenerated here: its exact oracle is MILP-bound
# (calibrated: N=8 already ~5s/instance) and the existing 20k-row run (see
# tools/generate_energy_training_data.py) is already at a realistic ceiling for
# that oracle — "millions" is not achievable there regardless of core count.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
W=8

echo "=== Scale-up run (8-core) ==="

echo "--- single-load: 100000 instances ---"
python3 tools/generate_training_data.py --n 100000 --workers "$W"
python3 tools/train_solver_selector.py     --depth 6 --trees 150
python3 tools/train_difficulty_predictor.py --depth 6 --trees 150
python3 tools/train_makespan_predictor.py  --depth 6 --trees 150

echo "--- MLSD: 150000 instances (mlsd-milp gate n<=3, n*m<=9) ---"
python3 tools/generate_mlsd_training_data.py --n 150000 --workers "$W" \
        --milp-max-n 3 --milp-max-nm 9
python3 tools/train_mlsd_predictor.py --depth 6 --trees 150

echo "--- non-star classes: 2,000,000 instances each ---"
for cls in chain tree mapreduce multilayer; do
  python3 tools/generate_topology_training_data.py --class "$cls" --n 2000000 --workers "$W"
  python3 tools/train_topology_predictor.py --class "$cls" --depth 6 --trees 150
done

echo "--- rebuild libdls_c.so (HiGHS build) ---"
cmake --build build-highs --target dls_c

echo "--- restart portal API ---"
if sudo systemctl is-active dls-studio &>/dev/null; then
  sudo systemctl restart dls-studio
  echo "dls-studio restarted"
else
  echo "(dls-studio not running, skip restart)"
fi

echo "=== Done. All predictors retrained at scale-up sizes. ==="
