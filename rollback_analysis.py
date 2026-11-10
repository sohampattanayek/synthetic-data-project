"""Roll back and continue: does training recover after the rule restores a checkpoint?

Compares three runs per (seed, spike size), all on the same data split and seed:
  control   results/mmd/ctrl_s{seed}            no spike
  fixed     results/mmd/spike90x{f}_s{seed}     spike, train to the end, no rule
  rollback  results/rollback/spike90x{f}_s{seed} spike, rule restores a checkpoint and training continues

Outcome: test TSTR (10 repeats) at the last check, and the mean of the last 5 checks.
Differences are paired by seed; intervals are 95% t-intervals.

Usage: python rollback_analysis.py --out results/rollback_analysis
"""
import argparse, json, os
import numpy as np, pandas as pd
from scipy import stats

def ci95(v):
    v = np.asarray(v, float)
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")

def load(path):
    m = pd.read_csv(os.path.join(path, "metrics.csv"))
    s = json.load(open(os.path.join(path, "summary.json")))
    return m, s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/mmd")
    ap.add_argument("--rollback", default="results/rollback")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--factors", type=int, nargs="+", default=[3, 10, 30, 100])
    ap.add_argument("--out", default="results/rollback_analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    col = "tstr_test_rep_mean"
    rows = []
    for f in a.factors:
        for s in a.seeds:
            c, _ = load(f"{a.base}/ctrl_s{s}")
            x, _ = load(f"{a.base}/spike90x{f}_s{s}")
            r, rs = load(f"{a.rollback}/spike90x{f}_s{s}")
            rb = rs.get("rollbacks", [])
            rows.append(dict(
                factor=f, seed=s, n_rollbacks=len(rb),
                alarm_epoch=rb[0]["alarm_epoch"] if rb else np.nan,
                restored_epoch=rb[0]["restored_epoch"] if rb else np.nan,
                control_last=c[col].iloc[-1], fixed_last=x[col].iloc[-1], rollback_last=r[col].iloc[-1],
                control_last5=c[col].iloc[-5:].mean(), fixed_last5=x[col].iloc[-5:].mean(),
                rollback_last5=r[col].iloc[-5:].mean(),
                control_score=c["score"].iloc[-1], fixed_score=x["score"].iloc[-1], rollback_score=r["score"].iloc[-1],
            ))
    d = pd.DataFrame(rows)
    d.to_csv(f"{a.out}/per_run.csv", index=False)
    out = []
    for f, g in d.groupby("factor"):
        row = dict(factor=f, n=len(g), runs_with_rollback=int((g.n_rollbacks > 0).sum()),
                   max_rollbacks=int(g.n_rollbacks.max()),
                   mean_alarm_epoch=g.alarm_epoch.mean(), mean_restored_epoch=g.restored_epoch.mean())
        for k in ["last", "last5"]:
            row[f"control_{k}"] = g[f"control_{k}"].mean()
            row[f"fixed_{k}"] = g[f"fixed_{k}"].mean()
            row[f"rollback_{k}"] = g[f"rollback_{k}"].mean()
            dv = g[f"rollback_{k}"] - g[f"fixed_{k}"]
            row[f"rb_minus_fixed_{k}_pts"] = 100 * dv.mean()
            row[f"rb_minus_fixed_{k}_ci95"] = 100 * ci95(dv)
            dc = g[f"rollback_{k}"] - g[f"control_{k}"]
            row[f"rb_minus_control_{k}_pts"] = 100 * dc.mean()
            row[f"rb_minus_control_{k}_ci95"] = 100 * ci95(dc)
            row[f"rb_better_than_fixed_{k}"] = int((dv > 0).sum())
        for k in ["control", "fixed", "rollback"]:
            row[f"{k}_score"] = g[f"{k}_score"].mean()
        out.append(row)
    o = pd.DataFrame(out)
    o.to_csv(f"{a.out}/by_factor.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print(o.round(4).T.to_string())

if __name__ == "__main__":
    main()
