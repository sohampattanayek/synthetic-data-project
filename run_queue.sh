#!/usr/bin/env bash
# Runs the severity study: matched controls + lr spikes at epoch 90 of size
# x3, x10, x30, x100, 5 seeds each, all with 10 TSTR repeats per check.
# Skips any run whose summary.json already exists, so it can be restarted.
# Usage: bash run_queue.sh            (logs one line per finished run)
set -u
cd "$(dirname "$0")"
COMMON="--epochs 150 --check-every 3 --max-rows 10000 --n-eval 2000 --tstr-repeats 10"

jobs=()
for s in 0 1 2 3 4; do jobs+=("ctrl_s$s|--seed $s"); done
for f in 10 3 30 100; do
  for s in 0 1 2 3 4; do jobs+=("spike90x${f}_s$s|--seed $s --spike-epoch 90 --spike-factor $f"); done
done

for j in "${jobs[@]}"; do
  name="${j%%|*}"; args="${j#*|}"
  if [ -f "results/$name/summary.json" ]; then continue; fi
  t0=$(date +%s)
  if python run_experiment.py $COMMON $args --name "$name" > "results/$name.log" 2>&1; then
    echo "DONE $name in $(( $(date +%s) - t0 ))s"
    rm -f "results/$name.log"
  else
    echo "FAILED $name (see results/$name.log)"
  fi
done
echo "QUEUE FINISHED"
