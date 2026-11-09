#!/usr/bin/env bash
# Runs the severity study: matched controls + lr spikes at epoch 90 of size
# x3, x10, x30, x100, 5 seeds each, all with 10 TSTR repeats per check.
# Skips any run whose summary.json already exists, so it can be restarted.
# Usage: bash run_queue.sh            (logs one line per finished run)
set -u
cd "$(dirname "$0")"
mkdir -p "results/${PREFIX:-}"
COMMON="--epochs 150 --check-every 3 --max-rows 10000 --n-eval 2000 --tstr-repeats 10"
# Optional: PREFIX=mmd/ EXTRA="--mmd --per-column" bash run_queue.sh
# reruns the same 25 runs into results/mmd/ with extra logging.
PREFIX="${PREFIX:-}"
COMMON="$COMMON ${EXTRA:-}"

# Optional: SEEDS="5 6 7 8 9" FACTORS="10 100" to run other seeds / spike sizes.
SEEDS="${SEEDS:-0 1 2 3 4}"
FACTORS="${FACTORS:-10 3 30 100}"

# CONTROLS=0 skips the unspiked controls (e.g. for the roll-back runs:
#   CONTROLS=0 PREFIX=rollback/ EXTRA="--mmd --per-column --rollback-series mmd2" bash run_queue.sh)
jobs=()
if [ "${CONTROLS:-1}" = 1 ]; then for s in $SEEDS; do jobs+=("ctrl_s$s|--seed $s"); done; fi
for f in $FACTORS; do
  for s in $SEEDS; do jobs+=("spike90x${f}_s$s|--seed $s --spike-epoch 90 --spike-factor $f"); done
done

for j in "${jobs[@]}"; do
  name="${PREFIX}${j%%|*}"; args="${j#*|}"
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
