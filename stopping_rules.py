"""Compare stopping / checkpoint-selection rules by simulating them on logged runs.

Every run logs, at each check, the fidelity metrics and losses the monitor
can see, plus test-split TSTR (mean of 10 fresh samples, `tstr_test_rep_mean`)
that the monitor never sees. A rule only looks at the monitor-visible columns
up to the current check, decides when to stop and which checkpoint to keep,
and is then scored by the logged test TSTR of the checkpoint it kept. So all
rules are compared on exactly the same training trajectories.

Rules
  fixed_budget            train to the end, keep the last checkpoint
  best_score              never stop, keep the best aggregate-score checkpoint
  best_ks                 never stop, keep the best mean-KS checkpoint
  best_val_tstr           never stop, keep the best validation-TSTR checkpoint
                          (mean of 10; uses held-out utility, so it is the
                          strong but expensive baseline)
  patience_ks             stop after 10 checks without a new best mean KS,
                          keep the best (close to Ferracci et al. 2024, who
                          weight KS by feature importance; here unweighted)
  patience_loss           same with the generator loss
  cusum_score_best        online CUSUM on the aggregate score (reference epochs
                          45-87, k=0.5, h=5 as in changepoints.py); on alarm stop
                          and keep the best-score checkpoint so far
  cusum_score_onset       same alarm, but roll back to the estimated change
                          onset: the last check before the CUSUM sum started
                          its final climb (g was 0)
  cusum_ks_onset          same with mean KS
  oracle                  keep the checkpoint with the best test TSTR (upper bound)

Usage:
    python stopping_rules.py --runs results/ctrl_s* results/spike90x*_s* --out results/stopping
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REF = (45, 87)
START = 48
K_SD, H_SD = 0.5, 5.0
PATIENCE = 10
TARGET = "tstr_test_rep_mean"


def cusum_with_onset(x, epochs, direction):
    """Returns (alarm index, onset index) or (None, None)."""
    m = (epochs >= REF[0]) & (epochs <= REF[1])
    mu, sd = x[m].mean(), x[m].std(ddof=1)
    if not sd > 0:
        return None, None
    g, last_zero = 0.0, None
    for i, (e, v) in enumerate(zip(epochs, x)):
        if e < START:
            continue
        g = max(0.0, g + direction * (v - mu) / sd - K_SD)
        if g == 0.0:
            last_zero = i
        if g > H_SD:
            onset = last_zero if last_zero is not None else i - 1
            return i, onset
    return None, None


def patience_stop(x):
    best, since = np.inf, 0
    for i, v in enumerate(x):
        if v < best:
            best, since = v, 0
        else:
            since += 1
        if i >= 5 and since >= PATIENCE:
            return i
    return None


def argbest(x, upto, lower_better=True):
    seg = x[: upto + 1]
    return int(np.nanargmin(seg) if lower_better else np.nanargmax(seg))


def simulate(m: pd.DataFrame) -> dict:
    ep = m["epoch"].to_numpy()
    last = len(m) - 1
    score, ks, loss = (m[c].to_numpy(float) for c in ["score", "ks_mean", "g_loss"])
    vt = m["tstr_rep_mean"].to_numpy(float)
    picks = {}  # rule -> (stop index, kept index)
    picks["fixed_budget"] = (last, last)
    picks["best_score"] = (last, argbest(score, last))
    picks["best_ks"] = (last, argbest(ks, last))
    picks["best_val_tstr"] = (last, argbest(vt, last, lower_better=False))
    for name, x in [("patience_ks", ks), ("patience_loss", loss)]:
        s = patience_stop(x)
        s = last if s is None else s
        picks[name] = (s, argbest(x, s))
    for name, x, keep in [("cusum_score_best", score, "best"),
                          ("cusum_score_onset", score, "onset"),
                          ("cusum_ks_onset", ks, "onset")]:
        a, onset = cusum_with_onset(x, ep, +1)
        if a is None:
            picks[name] = (last, last)
        else:
            picks[name] = (a, argbest(score, a) if keep == "best" else onset)
    picks["oracle"] = (last, argbest(m[TARGET].to_numpy(float), last, lower_better=False))
    return {r: {"stop_epoch": int(ep[s]), "kept_epoch": int(ep[k]),
                "test_tstr": float(m[TARGET].iloc[k])} for r, (s, k) in picks.items()}


def condition_of(name: str) -> str:
    return name.rsplit("_s", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="results/stopping")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in args.runs:
        p = Path(r)
        if not (p / "summary.json").exists():
            continue
        m = pd.read_csv(p / "metrics.csv")
        if TARGET not in m:
            print(f"skipping {p.name}: no {TARGET} column (run predates test-split logging)")
            continue
        for rule, res in simulate(m).items():
            rows.append({"run": p.name, "condition": condition_of(p.name), "rule": rule, **res})
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_run.csv", index=False, float_format="%.4f")
    oracle = df[df.rule == "oracle"].set_index("run")["test_tstr"]
    df["regret"] = oracle.reindex(df.run).to_numpy() - df.test_tstr
    fixed = df[df.rule == "fixed_budget"].set_index("run")["test_tstr"]
    df["vs_fixed"] = df.test_tstr - fixed.reindex(df.run).to_numpy()
    summ = df.groupby(["condition", "rule"]).agg(
        n=("run", "size"), test_tstr=("test_tstr", "mean"), regret=("regret", "mean"),
        vs_fixed=("vs_fixed", "mean"),
        beats_fixed=("vs_fixed", lambda v: int((v > 0).sum())),
        mean_stop_epoch=("stop_epoch", "mean")).reset_index()
    summ.to_csv(out / "summary.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    for cond, g in summ.groupby("condition"):
        print(f"\n== {cond} ==")
        print(g.drop(columns="condition").round(4).to_string(index=False))
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
