#!/usr/bin/env bash
# Rebuilds every analysis table and figure from the logged runs in results/.
# It does not retrain anything: the training runs themselves are made with
# run_queue.sh (see the comments there) and take many hours.
#
# Usage:
#   bash reproduce.sh              # writes into results/ (overwrites the tables there)
#   bash reproduce.sh /tmp/check   # writes into another folder, to compare with results/
#   bash reproduce.sh /tmp/check --compare   # also compares every CSV with the one in results/
set -eu
cd "$(dirname "$0")"
OUT="${1:-results}"
mkdir -p "$OUT"
step() { echo; echo "== $*"; }

step "pilot summary (5 seeds, default settings)"
python analyze_pilot.py results/pilot_s{0,1,2,3,4} --out "$OUT/pilot_summary" > /dev/null

step "change points on the pilot and x100 runs"
python changepoints.py --pilot results/pilot_s{0,1,2,3,4} --spiked results/spike90_s{0,1,2,3,4} --out "$OUT/changepoints" > /dev/null

step "severity study (controls + spikes x3, x10, x30, x100)"
python severity.py --out "$OUT/severity" > /dev/null

step "dependence between checks"
python dependence.py --runs results/pilot_s{0,1,2,3,4} --out "$OUT/dependence" > /dev/null
python dependence.py --runs results/ctrl_s{0,1,2,3,4} --series tstr_val tstr_rep_mean ks_mean score --out "$OUT/dependence_ctrl" > /dev/null

step "stopping rules (10 seeds; results/stopping is the earlier 5-seed table from an older version of the script)"
python stopping_rules.py --runs results/mmd/ctrl_s* results/mmd/spike90x*_s* --out "$OUT/stopping10" > /dev/null

step "MMD and per-column analysis"
python mmd_analysis.py --out "$OUT/mmd_analysis" > /dev/null

step "change in trend (three penalty sizes)"
mkdir -p "$OUT/trend_changes"
for p in 1 2 4; do
  python trend_changes.py --pen-mult $p --out "$OUT/trend_changes/pen$p" > "$OUT/trend_changes/trend_changes_pen$p.txt"
done

step "past-only detectors (all 10 seeds, seeds 0-4, held-out seeds 5-9)"
python causal_detection.py --runs-dir results/mmd --out "$OUT/causal/all" > /dev/null
python causal_detection.py --runs-dir results/mmd --seeds 0 1 2 3 4 --out "$OUT/causal/dev" > /dev/null
python causal_detection.py --runs-dir results/mmd --seeds 5 6 7 8 9 --out "$OUT/causal/heldout" > /dev/null

step "shape of the generator loss"
python loss_shape.py --runs-dir results/mmd --out "$OUT/loss_shape" > /dev/null

step "roll back and continue"
python rollback_analysis.py --out "$OUT/rollback_analysis" > /dev/null

step "settings that are bad from the start"
python realistic_failure.py --out "$OUT/realistic" > /dev/null

step "TVAE"
python tvae_analysis.py --out "$OUT/tvae_analysis" > /dev/null
python stopping_rules.py --runs results/tvae/ctrl_s* results/tvae/spike90x*_s* --out "$OUT/tvae_stopping" > /dev/null

step "other datasets (credit-card default, Covertype)"
for d in credit covertype; do
  if [ -f "results/$d/spike90x30_s4/summary.json" ]; then
    python tvae_analysis.py --runs "results/$d" --out "$OUT/datasets_analysis/$d" > /dev/null
    python stopping_rules.py --runs results/$d/ctrl_s* results/$d/spike90x*_s* --out "$OUT/datasets_analysis/$d/stopping" > /dev/null
  else
    echo "results/$d is not complete, skipped"
  fi
done

step "Mantis map separation"
if [ "$OUT" = results ]; then
  python map_separation.py results/mantis/map_audit.csv.gz audit results/mantis/map_drift.csv.gz drift > /dev/null
else
  echo "skipped (map_separation.py always writes to results/mantis/)"
fi

if [ "${2:-}" = "--compare" ] && [ "$OUT" != results ]; then
  step "comparing CSV files with results/"
  python - "$OUT" <<'EOF'
import sys, glob, os
import numpy as np, pandas as pd
out = sys.argv[1]; same = diff = missing = 0
for p in sorted(glob.glob(f"{out}/**/*.csv", recursive=True)):
    q = os.path.join("results", os.path.relpath(p, out))
    if not os.path.exists(q):
        missing += 1; print("not in results/:", q); continue
    a, b = pd.read_csv(p), pd.read_csv(q)
    ok = a.shape == b.shape and list(a.columns) == list(b.columns)
    if ok:
        for c in a.columns:
            if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
                ok &= bool(np.allclose(a[c].astype(float), b[c].astype(float), rtol=1e-9, atol=1e-12, equal_nan=True))
            else:
                ok &= bool((a[c].astype(str) == b[c].astype(str)).all())
    if ok: same += 1
    else: diff += 1; print("DIFFERENT:", q)
print(f"{same} files identical, {diff} different, {missing} not in results/")
EOF
fi
echo; echo "done"
