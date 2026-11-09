"""Online change detection that only uses the past ("causal" detectors).

The online CUSUM in changepoints.py / severity.py takes its reference mean
and sd from epochs 45-87. Monitoring starts at epoch 48, so before epoch 87
that detector uses values it could not have seen yet, and the window ends
just before the planted spike at epoch 90. That is fine for asking "is there
a change at all?", but a monitor running during training cannot do it.

This script reruns the detection study with detectors that, at every check,
use only the checks seen so far:

  fixed      reference mean/sd from a fixed burn-in window (epochs 30-45),
             then Page's CUSUM from epoch 48. Simple, but the reference gets
             stale when the series keeps drifting.
  expanding  self-starting CUSUM: at each check the reference mean/sd come
             from all earlier checks from epoch 30 on (the current check is
             not included).
  rolling    reference from the previous W checks only (default W = 12,
             36 epochs), which follows slow trends.

For comparison it also reports `lookahead`, the original detector
(reference epochs 45-87).

Scored exactly like severity.py: on spiked runs, detected = first alarm after
epoch 90; delay; alarms before the spike; and alarms on the matched controls.
The threshold h is varied (k stays 0.5) to show how sensitive the result is.

Usage:
    python causal_detection.py --runs-dir results/mmd --out results/causal
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SPIKE = 90
BURN = (30, 45)
START = 48
LOOKAHEAD_REF = (45, 87)
K_SD = 0.5
H_GRID = [3.0, 4.0, 5.0, 6.0, 8.0]
ROLL_W = 12

SERIES = {  # column: (label, direction; +1 = an increase is bad)
    "score": ("Aggregate score", +1),
    "ks_mean": ("Mean KS", +1),
    "jsd_mean": ("Mean JSD", +1),
    "mmd2": ("MMD^2", +1),
    "corr_dist": ("Correlation distance", +1),
    "g_loss": ("Generator loss", +1),
    "tstr_rep_mean": ("TSTR, mean of 10", -1),
}


def cusum_alarm(x, ep, direction, h, mode, w=ROLL_W):
    """First alarm epoch (or None) and the estimated onset epoch: the last
    check at which the CUSUM sum was zero before the alarm."""
    g, last_zero = 0.0, None
    for i in range(len(x)):
        if ep[i] < START:
            continue
        if mode == "lookahead":
            m = (ep >= LOOKAHEAD_REF[0]) & (ep <= LOOKAHEAD_REF[1])
        elif mode == "fixed":
            m = (ep >= BURN[0]) & (ep <= BURN[1])
        elif mode == "expanding":
            m = (ep >= BURN[0]) & (np.arange(len(x)) < i)
        elif mode == "rolling":
            m = (np.arange(len(x)) < i) & (np.arange(len(x)) >= i - w)
        else:
            raise ValueError(mode)
        ref = x[m]
        if len(ref) < 4:
            continue
        mu, sd = ref.mean(), ref.std(ddof=1)
        if not sd > 0:
            continue
        g = max(0.0, g + direction * (x[i] - mu) / sd - K_SD)
        if g == 0.0:
            last_zero = i
        if g > h:
            onset = last_zero if last_zero is not None else i - 1
            return int(ep[i]), int(ep[onset])
    return None, None


def run_names(runs_dir: Path):
    out = []
    for p in sorted(runs_dir.glob("ctrl_s*/metrics.csv")):
        seed = int(p.parent.name.split("_s")[1])
        out.append((p.parent.name, "control", 0, seed))
    for p in sorted(runs_dir.glob("spike90x*_s*/metrics.csv")):
        f = int(p.parent.name.split("x")[1].split("_")[0])
        seed = int(p.parent.name.split("_s")[1])
        out.append((p.parent.name, "spike", f, seed))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="results/mmd")
    ap.add_argument("--out", default="results/causal")
    args = ap.parse_args()
    runs_dir, out = Path(args.runs_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, cond, factor, seed in run_names(runs_dir):
        m = pd.read_csv(runs_dir / name / "metrics.csv")
        ep = m["epoch"].to_numpy()
        for col, (label, direction) in SERIES.items():
            if col not in m:
                continue
            x = m[col].to_numpy(float)
            for mode in ["lookahead", "fixed", "expanding", "rolling"]:
                for h in H_GRID:
                    a, onset = cusum_alarm(x, ep, direction, h, mode)
                    rows.append({"run": name, "condition": cond, "factor": factor, "seed": seed,
                                 "series": label, "detector": mode, "h": h,
                                 "alarm_epoch": a, "onset_epoch": onset})
    per = pd.DataFrame(rows)
    per.to_csv(out / "per_run.csv", index=False)

    spk = per[per.condition == "spike"].copy()
    spk["detected"] = spk.alarm_epoch > SPIKE
    spk["pre"] = spk.alarm_epoch <= SPIKE
    spk["delay"] = np.where(spk.detected, spk.alarm_epoch - SPIKE, np.nan)
    by_factor = spk.groupby(["series", "detector", "h", "factor"]).agg(
        detected=("detected", "sum"), n=("detected", "size"),
        median_delay=("delay", "median"), alarm_before_spike=("pre", "sum")).reset_index()
    by_factor.to_csv(out / "by_factor.csv", index=False)

    ctl = per[per.condition == "control"]
    ctl_tab = ctl.groupby(["series", "detector", "h"]).agg(
        control_alarms=("alarm_epoch", lambda v: int(v.notna().sum())),
        n_controls=("alarm_epoch", "size")).reset_index()
    tot = spk.groupby(["series", "detector", "h"]).agg(
        detected=("detected", "sum"), n_spiked=("detected", "size"),
        alarm_before_spike=("pre", "sum"), median_delay=("delay", "median")).reset_index()
    summary = tot.merge(ctl_tab, on=["series", "detector", "h"])
    summary.to_csv(out / "summary.csv", index=False)

    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
    n_ctl = int(ctl_tab.n_controls.iloc[0]); n_spk = int(tot.n_spiked.iloc[0])
    print(f"{n_spk} spiked runs, {n_ctl} controls. k = {K_SD}, monitoring from epoch {START}.")
    print("\nAll spike sizes pooled, h = 5:")
    print(summary[summary.h == 5.0].drop(columns=["h", "n_spiked", "n_controls"]).to_string(index=False))
    print("\nBy spike size, h = 5, causal detectors (detected / median delay):")
    v = by_factor[(by_factor.h == 5.0) & (by_factor.detector != "lookahead")].copy()
    v["cell"] = v.detected.astype(int).astype(str) + "/" + v.n.astype(str) + " (" + v.median_delay.astype(str) + ")"
    print(v.pivot_table(index=["series", "detector"], columns="factor", values="cell", aggfunc="first").to_string())
    print("\nThreshold sensitivity (detected of spiked / before spike / control alarms):")
    s = summary.copy()
    s["cell"] = (s.detected.astype(int).astype(str) + " / " + s.alarm_before_spike.astype(int).astype(str)
                 + " / " + s.control_alarms.astype(str))
    print(s.pivot_table(index=["series", "detector"], columns="h", values="cell", aggfunc="first").to_string())
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
