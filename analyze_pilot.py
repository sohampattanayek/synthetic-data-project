"""Aggregate pilot runs across seeds and answer the pilot question:
do any fidelity metrics move with downstream utility when the losses do not?

Usage:
    python analyze_pilot.py results/pilot_s0 results/pilot_s1 ... --out results/pilot_summary

For each run, computes the Spearman rank correlation between each logged
quantity and per-check TSTR accuracy (validation split). Because every metric
here is "lower is better" while TSTR is "higher is better", a useful metric
should show a NEGATIVE correlation with TSTR.

Two versions are reported:
  level   - correlation of the raw series. Inflated by any shared time trend:
            two series that both drift over training correlate even if they
            are unrelated, so the epoch number itself is included as a reference.
  change  - correlation of check-to-check differences. This asks whether a
            metric moves in the same direction as utility at the same check,
            which is what a monitor actually needs.

Writes:
    correlations.csv       per-seed and mean/sd correlation for each quantity
    selection.csv          TSTR (test split) of final vs. best-by-score checkpoint
    pilot_over_training.png  mean +/- range across seeds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

QUANTITIES = {
    "epoch": "Epoch number (reference: time trend alone)",
    "g_loss": "Generator loss",
    "d_loss": "Discriminator loss",
    "score": "Aggregate fidelity score",
    "ks_mean": "Mean KS statistic",
    "jsd_mean": "Mean JSD",
    "tvd_mean": "Mean categorical TVD",
    "corr_dist": "Correlation distance",
    "c2st": "C2ST 2|AUC-0.5|",
    "excess_point_mass_max": "Max excess point mass",
}


def load(run_dir: Path):
    m = pd.read_csv(run_dir / "metrics.csv")
    s = json.loads((run_dir / "summary.json").read_text())
    return m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/pilot_summary")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = [(Path(r).name, *load(Path(r))) for r in args.runs]

    # ---- 1. correlation with per-check TSTR -------------------------------
    def rho(a, b):
        a, b = pd.Series(a), pd.Series(b)
        ok = a.notna() & b.notna()
        if a[ok].nunique() < 2 or b[ok].nunique() < 2:
            return np.nan
        return spearmanr(a[ok], b[ok]).statistic

    rows = []
    for name, m, _ in runs:
        for q in QUANTITIES:
            rows.append({"run": name, "quantity": q,
                         "level": rho(m[q], m["tstr_val"]),
                         "change": rho(m[q].diff(), m["tstr_val"].diff())})
    corr = pd.DataFrame(rows)
    corr.to_csv(out / "correlations_per_seed.csv", index=False, float_format="%.3f")
    agg = corr.groupby("quantity")[["level", "change"]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    table = agg.reindex(list(QUANTITIES))
    table.insert(0, "label", [QUANTITIES[q] for q in table.index])
    table.to_csv(out / "correlations.csv", float_format="%.3f")

    # ---- 2. checkpoint selection (test split) -----------------------------
    sel = []
    for name, m, s in runs:
        u = s["downstream_accuracy_on_real_test"]
        best_tstr_epoch = int(m.loc[m["tstr_val"].idxmax(), "epoch"])
        sel.append({"run": name,
                    "train_real": u["train_real"],
                    "final_epoch": u["train_final_synth"],
                    "best_by_score": u["train_best_synth"],
                    "best_epoch_by_score": s["best_epoch"],
                    "best_epoch_by_val_tstr": best_tstr_epoch,
                    "max_val_tstr": float(m["tstr_val"].max()),
                    "final_val_tstr": float(m["tstr_val"].iloc[-1])})
    sel = pd.DataFrame(sel)
    sel.to_csv(out / "selection.csv", index=False, float_format="%.4f")

    # ---- 3. figure: mean and range across seeds ---------------------------
    stacked = pd.concat([m.assign(run=n) for n, m, _ in runs])
    g = stacked.groupby("epoch")
    fig, axes = plt.subplots(4, 1, figsize=(8, 10.5), sharex=True)
    panels = [
        ("tstr_val", "TSTR accuracy\n(validation)", "downstream utility (higher = better)"),
        ("score", "fidelity score\n(lower = better)", "aggregate fidelity score"),
        ("ks_mean", "metric\n(lower = better)", None),
        ("g_loss", "loss", "generator loss"),
    ]
    for ax, (col, ylab, lab) in zip(axes, panels):
        if col == "ks_mean":
            for c, l in [("ks_mean", "mean KS"), ("tvd_mean", "mean categorical TVD"),
                         ("corr_dist", "correlation distance"), ("c2st", "C2ST")]:
                mu = g[c].mean()
                ax.plot(mu.index, mu.values, label=l)
                ax.fill_between(mu.index, g[c].min(), g[c].max(), alpha=0.15)
        else:
            mu = g[col].mean()
            ax.plot(mu.index, mu.values, color="C0", label=f"{lab} (mean of {len(runs)} seeds)")
            ax.fill_between(mu.index, g[col].min(), g[col].max(), alpha=0.2, color="C0",
                            label="min-max across seeds")
            if col == "tstr_val":
                real = sel["train_real"].mean()
                ax.axhline(real, ls="--", c="gray", label=f"trained on real data ({real:.3f}, test split)")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7, loc="best")
    axes[0].set_title(f"Pilot: CTGAN on UCI Adult, {len(runs)} seeds, default settings")
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out / "pilot_over_training.png", dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 160)
    print("Spearman correlation with per-check TSTR (negative = metric falls as utility rises):")
    print(table.round(3).to_string())
    print("\nCheckpoint selection, TSTR on test split:")
    print(sel.round(4).to_string(index=False))
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
