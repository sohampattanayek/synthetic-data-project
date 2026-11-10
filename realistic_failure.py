"""Failures that are there from the first epoch: learning rate too high, batch too small.

Compares runs in results/highlr (lr 2e-3) and results/smallbatch (batch 100) with the
default runs in results/mmd (lr 2e-4, batch 500), same seeds, no spike in any of them.
Reports end-of-run test TSTR and fidelity, and whether the past-only detectors alarm.

Usage: python realistic_failure.py --out results/realistic
"""
import argparse, os
import numpy as np, pandas as pd
from scipy import stats
from causal_detection import cusum_alarm

DET = [("score", "expanding"), ("mmd2", "rolling"), ("ks_mean", "expanding"), ("g_loss", "fixed")]

def ci95(v):
    v = np.asarray(v, float)
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/realistic")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    col = "tstr_test_rep_mean"
    rows = []
    for name, folder in [("default", "results/mmd"), ("lr_2e-3", "results/highlr"), ("batch_100", "results/smallbatch")]:
        for s in a.seeds:
            m = pd.read_csv(f"{folder}/ctrl_s{s}/metrics.csv")
            r = dict(setting=name, seed=s, tstr_last=m[col].iloc[-1], tstr_last5=m[col].iloc[-5:].mean(),
                     tstr_best=m[col].max(), tstr_best_epoch=int(m.epoch[m[col].idxmax()]),
                     score_last5=m.score.iloc[-5:].mean(), ks_last5=m.ks_mean.iloc[-5:].mean(),
                     jsd_last5=m.jsd_mean.iloc[-5:].mean(), mmd2_last5=m.mmd2.iloc[-5:].mean(),
                     corr_last5=m.corr_dist.iloc[-5:].mean(), g_loss_last5=m.g_loss.iloc[-5:].mean())
            for k, mode in DET:
                al, _ = cusum_alarm(m[k].values, m.epoch.values, 1, 5, mode)
                r[f"alarm_{k}_{mode}"] = al if al is not None else np.nan
            rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(f"{a.out}/per_run.csv", index=False)
    num = [c for c in d.columns if c not in ("setting", "seed") and not c.startswith("alarm")]
    summ = d.groupby("setting", sort=False)[num].mean()
    for c in d.columns:
        if c.startswith("alarm"):
            summ[c + "_n"] = d.groupby("setting", sort=False)[c].apply(lambda v: int(v.notna().sum()))
    base = d[d.setting == "default"].set_index("seed")
    for name in ["lr_2e-3", "batch_100"]:
        x = d[d.setting == name].set_index("seed")
        for k in ["tstr_last", "tstr_last5", "score_last5", "ks_last5", "mmd2_last5"]:
            dv = x[k] - base[k]
            summ.loc[name, f"d_{k}"] = dv.mean(); summ.loc[name, f"d_{k}_ci95"] = ci95(dv)
            summ.loc[name, f"worse_{k}"] = int(((dv < 0) if k.startswith("tstr") else (dv > 0)).sum())
    summ.to_csv(f"{a.out}/summary.csv")
    pd.set_option("display.width", 250)
    print(summ.round(4).T.to_string())
    print(d[["setting", "seed"] + [c for c in d.columns if c.startswith("alarm")]].to_string())

if __name__ == "__main__":
    main()
