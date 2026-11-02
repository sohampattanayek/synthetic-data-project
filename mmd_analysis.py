"""MMD (product kernel) and per-column analysis of the severity runs.

Runs are the 25 severity runs rerun with --mmd --per-column into results/mmd/.
The extra logging uses no random numbers, so each rerun should match the
original run in results/ exactly; part 0 checks this.

0. Reproduction: every original column identical to the original run.
1. Saturation: how much of its range C2ST and MMD^2 use over a normal run.
2. Tracking utility (5 controls): Spearman rho of each metric with the
   mean-of-10 validation TSTR, for levels and for check-to-check changes,
   next to the epoch number.
3. Online CUSUM on MMD^2 (and C2ST, KS for reference), same settings and
   format as severity.py: detection, delay, alarms before the spike, alarms
   on the matched controls.
4. Runtime per check.
5. Per-column view of the spike (for Prof. Cho's joint/sparse question):
   effect of the spike on each column = mean over the checks at epochs
   93-105 of (spiked - matched control), in units of that column's
   check-to-check noise in the control before the spike (sd of first
   differences, epochs 45-87). Columns with effect > 3 are counted as hit.

Usage:
    python mmd_analysis.py --out results/mmd_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from changepoints import online_cusum

FACTORS = [3, 10, 30, 100]
SEEDS = [0, 1, 2, 3, 4]
SPIKE = 90
AFTER = (93, 105)
REF = (45, 87)
HIT = 3.0

CUSUM_SERIES = {"mmd2": ("MMD^2 (product kernel)", +1), "c2st": ("C2ST", +1),
                "ks_mean": ("Mean KS", +1), "jsd_mean": ("Mean JSD", +1),
                "score": ("Aggregate score", +1)}


def load(name):
    return pd.read_csv(Path("results/mmd") / name / "metrics.csv")


def names():
    return [f"ctrl_s{s}" for s in SEEDS] + [f"spike90x{f}_s{s}" for f in FACTORS for s in SEEDS]


def part0():
    rows = []
    for n in names():
        new, old = load(n), pd.read_csv(Path("results") / n / "metrics.csv")
        cols = [c for c in old.columns if c != "eval_seconds"]
        rows.append({"run": n, "identical_to_original": bool(new[cols].equals(old[cols]))})
    return pd.DataFrame(rows)


def part1(ctrl):
    rows = []
    for s, m in ctrl.items():
        late = m[m.epoch >= 30]
        for col in ["c2st", "mmd2", "ks_mean"]:
            x = m[col]
            rows.append({"seed": s, "metric": col, "first": x.iloc[0], "min": x.min(),
                         "last": x.iloc[-1],
                         "drop_first_to_min_pct": 100 * (x.iloc[0] - x.min()) / x.iloc[0],
                         "share_checks_ge_0.99": float((x >= 0.99).mean()) if col == "c2st" else np.nan,
                         "min_after_epoch30": late[col].min(), "max_after_epoch30": late[col].max()})
    return pd.DataFrame(rows)


def part2(ctrl):
    rows = []
    for s, m in ctrl.items():
        d = m.diff().dropna()
        for col in ["mmd2", "c2st", "ks_mean", "jsd_mean", "corr_dist", "score", "epoch"]:
            rows.append({"seed": s, "metric": col,
                         "rho_levels": spearmanr(m[col], m["tstr_rep_mean"])[0],
                         "rho_changes": spearmanr(d[col], d["tstr_rep_mean"])[0] if col != "epoch" else np.nan})
    per = pd.DataFrame(rows)
    return per, per.groupby("metric")[["rho_levels", "rho_changes"]].agg(["mean", "min", "max"]).round(3)


def part3(ctrl, spiked):
    rows = []
    for (f, s), m in spiked.items():
        c = ctrl[s]
        for col, (label, direction) in CUSUM_SERIES.items():
            a = online_cusum(m[col].to_numpy(float), m.epoch.to_numpy(), REF, 48, direction)
            ca = online_cusum(c[col].to_numpy(float), c.epoch.to_numpy(), REF, 48, direction)
            rows.append({"series": label, "factor": f, "seed": s,
                         "detected": a is not None and a > SPIKE,
                         "pre": a is not None and a <= SPIKE,
                         "delay": a - SPIKE if (a is not None and a > SPIKE) else np.nan,
                         "control_alarm": ca is not None})
    per = pd.DataFrame(rows)
    tab = per.groupby(["series", "factor"]).agg(
        detected=("detected", "sum"), n=("detected", "size"), median_delay=("delay", "median"),
        alarm_before_spike=("pre", "sum"), control_alarms=("control_alarm", "sum")).reset_index()
    return per, tab


def part4(all_runs):
    m = pd.concat(all_runs.values())
    return {"mmd_seconds_mean": m.mmd_seconds.mean(), "mmd_seconds_max": m.mmd_seconds.max(),
            "eval_seconds_mean": m.eval_seconds.mean()}


def part5(ctrl, spiked):
    cols = [c for c in ctrl[0].columns if c.startswith("col_")]
    rows = []
    for (f, s), m in spiked.items():
        c = ctrl[s]
        w = (m.epoch >= AFTER[0]) & (m.epoch <= AFTER[1])
        ref = (c.epoch >= REF[0]) & (c.epoch <= REF[1])
        for col in cols:
            noise = np.diff(c.loc[ref, col]).std(ddof=1)
            eff = (m.loc[w, col].to_numpy() - c.loc[w, col].to_numpy()).mean()
            rows.append({"factor": f, "seed": s, "column": col.split("__")[1],
                         "metric": "KS" if col.startswith("col_ks") else "TVD",
                         "raw_effect": eff, "effect_in_noise_sd": eff / noise if noise > 0 else np.nan})
    per = pd.DataFrame(rows)
    per["hit"] = per.effect_in_noise_sd > HIT
    by_col = per.groupby(["column", "metric", "factor"]).agg(
        raw_effect=("raw_effect", "mean"), effect_sd=("effect_in_noise_sd", "mean"),
        seeds_hit=("hit", "sum")).reset_index()
    by_run = per.groupby(["factor", "seed"]).agg(cols_hit=("hit", "sum"), n_cols=("hit", "size")).reset_index()
    return per, by_col, by_run


def figure(by_col, out):
    piv = by_col.pivot_table(index="column", columns="factor", values="effect_sd")
    piv = piv.loc[piv[100].sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(5.5, 6))
    im = ax.imshow(piv.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"x{f}" for f in piv.columns])
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.iat[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if piv.iat[i, j] < piv.to_numpy().max() / 2 else "black")
    ax.set_xlabel("learning-rate spike at epoch 90")
    ax.set_title("Spike effect per column, epochs 93-105\n(spiked - control, in control noise sds; mean of 5 seeds)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout(); fig.savefig(out / "per_column_effect.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/mmd_analysis")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ctrl = {s: load(f"ctrl_s{s}") for s in SEEDS}
    spiked = {(f, s): load(f"spike90x{f}_s{s}") for f in FACTORS for s in SEEDS}
    pd.set_option("display.width", 220)

    rep = part0(); rep.to_csv(out / "reproduction_check.csv", index=False)
    print(f"0. reruns identical to originals: {int(rep.identical_to_original.sum())}/{len(rep)}")

    sat = part1(ctrl); sat.to_csv(out / "saturation.csv", index=False, float_format="%.5g")
    print("\n1. saturation (controls):\n", sat.groupby("metric").agg(
        first=("first", "mean"), min=("min", "mean"), last=("last", "mean"),
        drop_pct=("drop_first_to_min_pct", "mean"), share_c2st_ge_099=("share_checks_ge_0.99", "mean"),
        min_after30=("min_after_epoch30", "min"), max_after30=("max_after_epoch30", "max")).to_string())

    per2, tab2 = part2(ctrl)
    per2.to_csv(out / "spearman_per_seed.csv", index=False, float_format="%.4f")
    tab2.to_csv(out / "spearman_summary.csv")
    print("\n2. Spearman rho with mean-of-10 TSTR (5 controls):\n", tab2.to_string())

    per3, tab3 = part3(ctrl, spiked)
    per3.to_csv(out / "cusum_per_run.csv", index=False); tab3.to_csv(out / "cusum_summary.csv", index=False)
    print("\n3. online CUSUM:\n", tab3.to_string(index=False))

    rt = part4({**{("c", s): m for s, m in ctrl.items()}, **spiked})
    pd.Series(rt).to_csv(out / "runtime.csv")
    print("\n4. runtime per check (s):", {k: round(v, 3) for k, v in rt.items()})

    per5, by_col, by_run = part5(ctrl, spiked)
    per5.to_csv(out / "per_column_per_run.csv", index=False, float_format="%.4f")
    by_col.to_csv(out / "per_column_summary.csv", index=False, float_format="%.4f")
    by_run.to_csv(out / "per_column_hits_per_run.csv", index=False)
    print(f"\n5. columns hit (effect > {HIT} noise sds), per spiked run:\n",
          by_run.groupby("factor").cols_hit.agg(["min", "median", "max"]).to_string(),
          f"\n(of {by_run.n_cols.iloc[0]} columns)")
    piv = by_col.pivot_table(index=["column", "metric"], columns="factor", values="effect_sd").round(1)
    print("\nper-column effect (noise sds, mean of 5 seeds):\n", piv.sort_values(100, ascending=False).to_string())
    figure(by_col, out)
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
