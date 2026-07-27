#!/usr/bin/env bash
# Chain stages 3-5 for one task on a chosen set of devices.
#   ./experiments/run_task.sh cnn 0            # sweep, landscape, sharpness on GPU 0
#   ./experiments/run_task.sh gpt 0,1 --seeds 2 --budgets 6
set -euo pipefail
TASK=$1; DEVS=$2; shift 2
export STAM_DEVICES=$DEVS
export PYTHONPATH=.
FIRST=${DEVS%%,*}

echo "=== [$TASK] budget sweep on devices $DEVS ==="
python3 -u -W ignore experiments/03_budget_sweep.py --task "$TASK" --split train "$@"

echo "=== [$TASK] certified landscape + animation on cuda:$FIRST ==="
python3 -u -W ignore experiments/04_landscape.py --task "$TASK" --device "cuda:$FIRST"

echo "=== [$TASK] sharpness study on devices $DEVS ==="
python3 -u -W ignore experiments/05_sharpness.py --task "$TASK"

echo "=== [$TASK] complete ==="
