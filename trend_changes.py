"""Change-in-trend analysis with `ruptures` (suggested by Prof. Rebecca Killick).

The CUSUM in changepoints.py looks for a change in the MEAN. A series that
rises steadily, or changes slope, can fool it. Here each series is modelled
as piecewise linear and the number and location of slope changes are
estimated, in two ways:

  clinear  continuous piecewise-linear (the line bends but does not jump)
  l2       piecewise-constant (mean shifts; for comparison with CUSUM)

Number of changes: PELT with a BIC-type penalty, pen = 2 * sigma^2 * log(n)
(each change adds a location and one new parameter). sigma is the noise sd,
estimated robustly from differences so that the changes themselves do not
inflate it: second differences for clinear (they remove any linear trend),
first differences for l2.

1. Offline: fit each full series once.
2. Online ("rerun offline at every check"): at every check from epoch 51 on,
   refit on the data seen so far. Alarm if the fit has a change at or after
   epoch 48 and the series moves in the bad direction after it
   (l2: the mean after the change is worse; clinear: the last segment's slope
   points the bad way). Scored like the CUSUM in severity.py: detection on
   spiked runs (alarm after epoch 90), delay, alarms before the spike, and
   alarms on the unspiked controls.

Usage:
    python trend_changes.py --out results/trend_changes/pen1
    python trend_changes.py --pen-mult 2 --out results/trend_changes/pen2   (sensitivity)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt

FACTORS = [3, 10, 30, 100]
SEEDS = [0, 1, 2, 3, 4]
SPIKE = 90
START = 48          # same as the online CUSUM: changes before this epoch are warm-up
FIRST_CHECK = 51
MIN_SIZE = 2        # points per segment

SERIES = {  # column: (label, direction; +1 means an increase is bad)
    "tstr_rep_mean": ("TSTR, mean of 10", -1),
    "ks_mean": ("Mean KS", +1),
    "g_loss": ("Generator loss", +1),
    "jsd_mean": ("Mean JSD", +1),
    "score": ("Aggregate score", +1),
}


def noise_sd(x: np.ndarray, model: str) -> float:
    d = np.diff(x, n=2 if model == "clinear" else 1)
    scale = np.sqrt(6) if model == "clinear" else np.sqrt(2)
    s = np.median(np.abs(d - np.median(d))) / 0.6745 / scale
    return float(s) if s > 0 else float(np.std(d) / scale + 1e-12)


PEN_MULT = 1.0     # sensitivity check: multiply the BIC-type penalty


def fit(x: np.ndarray, model: str) -> list[int]:
    """Indices where a new segment starts (excluding 0 and len(x))."""
    pen = PEN_MULT * 2 * noise_sd(x, model) ** 2 * np.log(len(x))
    algo = rpt.Pelt(model=model, min_size=MIN_SIZE, jump=1).fit(x.reshape(-1, 1))
    return algo.predict(pen=pen)[:-1]


def last_segment_bad(x, bkps, model, direction) -> bool:
    k = bkps[-1]
    if model == "l2":
        before = x[(bkps[-2] if len(bkps) > 1 else 0):k].mean()
        return direction * (x[k:].mean() - before) > 0
    seg = x[k - 1:]                 # continuous fit: segment starts at the bend
    slope = np.polyfit(np.arange(len(seg)), seg, 1)[0]
    return direction * slope > 0


def online_alarm(x, ep, model, direction):
    """First epoch at which the refit-at-every-check rule alarms, else None."""
    for t in range(len(x)):
        if ep[t] < FIRST_CHECK:
            continue
        xs = x[: t + 1]
        bkps = fit(xs, model)
        if bkps and ep[bkps[-1]] >= START and last_segment_bad(xs, bkps, model, direction):
            return int(ep[t])
    return None


def load(name):
    return pd.read_csv(Path("results") / name / "metrics.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/trend_changes")
    ap.add_argument("--pen-mult", type=float, default=1.0,
                    help="multiply the BIC-type penalty (sensitivity check)")
    args = ap.parse_args()
    global PEN_MULT
    PEN_MULT = args.pen_mult
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    runs = [(f"ctrl_s{s}", "control", s) for s in SEEDS] + \
           [(f"spike90x{f}_s{s}", f"x{f}", s) for f in FACTORS for s in SEEDS]

    off, onl = [], []
    for name, cond, seed in runs:
        m = load(name)
        ep = m["epoch"].to_numpy()
        for col, (label, direction) in SERIES.items():
            x = m[col].to_numpy(float)
            for model in ["clinear", "l2"]:
                b = fit(x, model)
                off.append({"run": name, "condition": cond, "seed": seed, "series": label,
                            "model": model, "n_changes": len(b),
                            "change_epochs": " ".join(str(int(ep[i])) for i in b),
                            "change_near_90": any(abs(int(ep[i]) - SPIKE) <= 3 for i in b)})
                a = online_alarm(x, ep, model, direction)
                onl.append({"run": name, "condition": cond, "seed": seed, "series": label,
                            "model": model, "alarm_epoch": a})
    off = pd.DataFrame(off)
    onl = pd.DataFrame(onl)
    off.to_csv(out / "offline_per_run.csv", index=False)
    onl.to_csv(out / "online_per_run.csv", index=False)

    # offline summary: controls and spikes
    offsum = off.groupby(["series", "model", "condition"]).agg(
        n=("run", "size"), mean_changes=("n_changes", "mean"),
        runs_with_change_near_90=("change_near_90", "sum"),
        change_epochs=("change_epochs", lambda v: " | ".join(v))).reset_index()
    offsum.to_csv(out / "offline_summary.csv", index=False)

    # online summary in the same format as severity.py
    ctrl = onl[onl.condition == "control"].set_index(["series", "model", "seed"]).sort_index()["alarm_epoch"]
    rows = []
    for (series, model, cond), g in onl[onl.condition != "control"].groupby(["series", "model", "condition"]):
        a = g["alarm_epoch"]
        det = a.notna() & (a > SPIKE)
        rows.append({"series": series, "model": model, "factor": cond,
                     "detected": int(det.sum()), "n": len(g),
                     "median_delay": float((a[det] - SPIKE).median()) if det.any() else np.nan,
                     "alarm_before_spike": int((a.notna() & (a <= SPIKE)).sum()),
                     "control_alarms": int(ctrl.loc[(series, model)].notna().sum())})
    onsum = pd.DataFrame(rows)
    onsum["factor_num"] = onsum.factor.str[1:].astype(int)
    onsum = onsum.sort_values(["model", "series", "factor_num"]).drop(columns="factor_num")
    onsum.to_csv(out / "online_summary.csv", index=False)

    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 80)
    print("OFFLINE (controls):")
    print(offsum[offsum.condition == "control"].to_string(index=False))
    print("\nOFFLINE (spiked runs, change within 3 epochs of 90):")
    print(offsum[offsum.condition != "control"][["series", "model", "condition", "mean_changes",
                                                 "runs_with_change_near_90"]].to_string(index=False))
    print("\nONLINE (refit at every check):")
    print(onsum.to_string(index=False))
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
