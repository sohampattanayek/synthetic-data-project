"""TVAE severity study: does a learning-rate spike at epoch 90 damage a TVAE, and do the
past-only detectors see it?

Reads results/tvae (5 controls + spikes x3, x10, x30, x100, seeds 0-4, same seeds as CTGAN).
A run whose weights became NaN is reported as diverged and left out of the end-of-run means.

Usage: python tvae_analysis.py --out results/tvae_analysis
"""
import argparse, json, os
import numpy as np, pandas as pd
from scipy import stats
from causal_detection import cusum_alarm

DET = [("score", "expanding"), ("mmd2", "rolling"), ("ks_mean", "expanding"), ("g_loss", "fixed")]
COL = "tstr_test_rep_mean"
SPIKE = 90

def ci95(v):
    v = np.asarray(v, float)
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/tvae")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--factors", type=int, nargs="+", default=[3, 10, 30, 100])
    ap.add_argument("--out", default="results/tvae_analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for f in [0] + a.factors:
        for s in a.seeds:
            name = f"ctrl_s{s}" if f == 0 else f"spike90x{f}_s{s}"
            m = pd.read_csv(f"{a.runs}/{name}/metrics.csv")
            summ = json.load(open(f"{a.runs}/{name}/summary.json"))
            div = summ.get("diverged_at")
            r = dict(factor=f, seed=s, diverged_at=div if div else np.nan, last_check=int(m.epoch.iloc[-1]))
            if f:  # same seed must give the same run as its control up to the spike
                c = pd.read_csv(f"{a.runs}/ctrl_s{s}/metrics.csv")
                k = ["score", "ks_mean", "jsd_mean", "corr_dist", "g_loss", "mmd2", COL]
                r["matches_control_to_90"] = bool(np.array_equal(m.loc[m.epoch <= SPIKE, k].values,
                                                                 c.loc[c.epoch <= SPIKE, k].values))
            if not div:
                r.update(tstr_last=m[COL].iloc[-1], tstr_last5=m[COL].iloc[-5:].mean(),
                         score_last5=m.score.iloc[-5:].mean(), mmd2_last5=m.mmd2.iloc[-5:].mean())
            r["tstr_at_90"] = float(m.loc[m.epoch == SPIKE, COL].iloc[0])
            for key, mode in DET:
                al, _ = cusum_alarm(m[key].values, m.epoch.values, 1, 5, mode)
                r[f"alarm_{key}_{mode}"] = al if al is not None else np.nan
            rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(f"{a.out}/per_run.csv", index=False)

    ctrl = d[d.factor == 0].set_index("seed")
    out = []
    for f in [0] + a.factors:
        x = d[d.factor == f].set_index("seed")
        ok = x[x.diverged_at.isna()]
        r = dict(factor=f, n=len(x), diverged=int(x.diverged_at.notna().sum()),
                 tstr_last=ok.tstr_last.mean(), tstr_last5=ok.tstr_last5.mean(),
                 score_last5=ok.score_last5.mean(), mmd2_last5=ok.mmd2_last5.mean())
        if f:
            r["all_match_control_to_90"] = bool(x.matches_control_to_90.all())
            for k in ["tstr_last", "tstr_last5", "score_last5"]:
                dv = (ok[k] - ctrl.loc[ok.index, k])
                r[f"d_{k}"] = dv.mean(); r[f"d_{k}_ci95"] = ci95(dv)
            r["tstr_worse_n"] = int((ok.tstr_last < ctrl.loc[ok.index, "tstr_last"]).sum())
        for key, mode in DET:
            al = x[f"alarm_{key}_{mode}"]
            if f == 0:
                r[f"{key}_{mode}_false"] = int(al.notna().sum())
            else:
                r[f"{key}_{mode}_false"] = int((al <= SPIKE).sum())      # alarm before the spike took effect
                hit = al[al > SPIKE]
                r[f"{key}_{mode}_detect"] = int(len(hit))
                r[f"{key}_{mode}_delay"] = float((hit - SPIKE).mean()) if len(hit) else np.nan
        out.append(r)
    b = pd.DataFrame(out)
    b.to_csv(f"{a.out}/by_factor.csv", index=False)
    pd.set_option("display.width", 250)
    print(b.round(4).T.to_string())
    print(d[["factor", "seed", "diverged_at"] + [c for c in d.columns if c.startswith("alarm")]].to_string())

if __name__ == "__main__":
    main()
