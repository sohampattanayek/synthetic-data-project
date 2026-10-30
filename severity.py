"""Severity study (A) and TSTR noise (B).

A. Learning-rate spikes of different sizes at epoch 90 (x3, x10, x30, x100),
   5 seeds each, against seed-matched controls run on the same machine
   (ctrl_s*). For each spike size and monitored series: how much damage the
   spike does, whether online CUSUM detects it, how fast, and how often it
   false-alarms on the matched control. Also checks that each spiked run is
   identical to its control up to the spike (the matched-control property).

B. Every check in these runs also scored 10 fresh synthetic samples by TSTR.
   From the controls: how big per-check TSTR sampling noise is, compared with
   the check-to-check movement of the single-sample TSTR series, and whether
   averaging over samples lets fidelity metrics track check-to-check changes
   in utility.

Usage:
    python severity.py --out results/severity
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

from changepoints import cusum_stat, noise_sd, online_cusum, CRIT_05

FACTORS = [3, 10, 30, 100]
SEEDS = [0, 1, 2, 3, 4]
SPIKE = 90

SERIES = {  # column: (label, direction where +1 means an increase is bad)
    "score": ("Aggregate fidelity score", +1),
    "ks_mean": ("Mean KS", +1),
    "jsd_mean": ("Mean JSD", +1),
    "corr_dist": ("Correlation distance", +1),
    "excess_point_mass_max": ("Pile-up", +1),
    "g_loss": ("Generator loss", +1),
    "tstr_val": ("TSTR, 1 sample", -1),
    "tstr_rep_mean": ("TSTR, mean of 10", -1),
}


def load(name: str):
    d = Path("results") / name
    m = pd.read_csv(d / "metrics.csv")
    s = json.loads((d / "summary.json").read_text())
    return m, s


def part_a(ctrl: dict, out: Path) -> pd.DataFrame:
    rows, damage, identity = [], [], []
    for f in FACTORS:
        for s in SEEDS:
            name = f"spike90x{f}_s{s}"
            if not (Path("results") / name / "summary.json").exists():
                continue
            m, summ = load(name)
            c, csum = ctrl[s]
            # matched-control check: identical up to and including the spike epoch
            pre = [col for col in SERIES if col in m]
            a = m.loc[m.epoch <= SPIKE, pre].reset_index(drop=True)
            b = c.loc[c.epoch <= SPIKE, pre].reset_index(drop=True)
            identity.append({"run": name, "identical_before_spike": bool(a.equals(b))})

            u, uc = summ["downstream_accuracy_on_real_test"], csum["downstream_accuracy_on_real_test"]
            last = lambda df: df.loc[df.epoch > 135, "tstr_rep_mean"].mean()
            damage.append({"factor": f, "seed": s,
                           "test_tstr_final_spiked": u["train_final_synth"],
                           "test_tstr_final_control": uc["train_final_synth"],
                           "val_tstr_last5_spiked": last(m), "val_tstr_last5_control": last(c)})

            ep = m["epoch"].to_numpy()
            for col, (label, direction) in SERIES.items():
                alarm = online_cusum(m[col].to_numpy(float), ep, (45, 87), 48, direction)
                calarm = online_cusum(c[col].to_numpy(float), c["epoch"].to_numpy(), (45, 87), 48, direction)
                # offline: where does a single change point land on the spiked run?
                x = m[col].to_numpy(float)
                k, stat = cusum_stat(x, noise_sd(x))
                rows.append({
                    "factor": f, "seed": s, "series": label,
                    "detected": alarm is not None and alarm > SPIKE,
                    "false_alarm_pre_spike": alarm is not None and alarm <= SPIKE,
                    "delay": (alarm - SPIKE) if (alarm is not None and alarm > SPIKE) else np.nan,
                    "control_alarm": calarm is not None,
                    "offline_break_epoch": int(ep[k - 1]),
                    "offline_significant": stat >= CRIT_05,
                })
    det = pd.DataFrame(rows)
    dmg = pd.DataFrame(damage)
    ident = pd.DataFrame(identity)
    det.to_csv(out / "detection_per_run.csv", index=False)
    dmg.to_csv(out / "damage_per_run.csv", index=False, float_format="%.4f")
    ident.to_csv(out / "matched_control_check.csv", index=False)

    if len(det):
        tab = det.groupby(["series", "factor"]).agg(
            detected=("detected", "sum"), n=("detected", "size"),
            median_delay=("delay", "median"),
            false_alarm_pre=("false_alarm_pre_spike", "sum"),
            control_alarms=("control_alarm", "sum"),
            offline_at_90=("offline_break_epoch", lambda e: int((e == SPIKE).sum())),
        ).reset_index()
        tab.to_csv(out / "detection_summary.csv", index=False)
    else:
        tab = pd.DataFrame()
    dsum = dmg.groupby("factor").mean(numeric_only=True).drop(columns="seed") if len(dmg) else dmg
    if len(dmg):
        dsum["n"] = dmg.groupby("factor").size()
        dsum.to_csv(out / "damage_summary.csv", float_format="%.4f")
    return det, tab, dsum, ident


def part_b(ctrl: dict, out: Path):
    rows, corr_rows = [], []
    for s, (m, _) in ctrl.items():
        single = m["tstr_val"].to_numpy(float)
        rows.append({
            "seed": s,
            "mean_sd_across_samples": float(m["tstr_rep_sd"].mean()),
            # what pure sampling noise alone would make the check-to-check sd of the single-sample series
            "expected_diff_sd_from_noise": float(m["tstr_rep_sd"].mean() * np.sqrt(2)),
            "observed_diff_sd_single": float(np.diff(single).std(ddof=1)),
            "observed_diff_sd_mean10": float(np.diff(m["tstr_rep_mean"]).std(ddof=1)),
        })
        d = m.diff().dropna()
        for col in ["ks_mean", "jsd_mean", "corr_dist", "score", "g_loss"]:
            corr_rows.append({
                "seed": s, "metric": col,
                "rho_levels_single": spearmanr(m[col], m["tstr_val"])[0],
                "rho_levels_mean10": spearmanr(m[col], m["tstr_rep_mean"])[0],
                "rho_changes_single": spearmanr(d[col], d["tstr_val"])[0],
                "rho_changes_mean10": spearmanr(d[col], d["tstr_rep_mean"])[0],
            })
        corr_rows.append({"seed": s, "metric": "epoch",
                          "rho_levels_single": spearmanr(m["epoch"], m["tstr_val"])[0],
                          "rho_levels_mean10": spearmanr(m["epoch"], m["tstr_rep_mean"])[0],
                          "rho_changes_single": np.nan, "rho_changes_mean10": np.nan})
    noise = pd.DataFrame(rows)
    corr = pd.DataFrame(corr_rows)
    noise.to_csv(out / "tstr_noise_per_seed.csv", index=False, float_format="%.4f")
    corr.to_csv(out / "correlations_per_seed.csv", index=False, float_format="%.4f")
    corr_sum = corr.groupby("metric").agg(["mean", "std"]).drop(columns="seed").round(3)
    corr_sum.to_csv(out / "correlations_summary.csv")
    return noise, corr_sum


def figure(det, dsum, ctrl, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    if len(det):
        rate = det.groupby(["series", "factor"])["detected"].mean().unstack("factor")
        for i, (label, r) in enumerate(rate.iterrows()):
            axes[0].plot(r.index, r.values, marker="o", label=label, color=f"C{i}")
        axes[0].set_xscale("log"); axes[0].set_xticks(FACTORS); axes[0].set_xticklabels([f"x{f}" for f in FACTORS])
        axes[0].set_ylim(-0.05, 1.05); axes[0].set_xlabel("learning-rate spike at epoch 90")
        axes[0].set_ylabel("detection rate (5 seeds)"); axes[0].set_title("A. Online CUSUM detection vs spike size")
        axes[0].legend(fontsize=7)
    if len(dsum):
        axes[1].plot(dsum.index, dsum["test_tstr_final_spiked"], marker="o", label="spiked")
        axes[1].plot(dsum.index, dsum["test_tstr_final_control"], marker="o", ls="--", label="matched control")
        axes[1].set_xscale("log"); axes[1].set_xticks(FACTORS); axes[1].set_xticklabels([f"x{f}" for f in FACTORS])
        axes[1].set_ylabel("final test TSTR accuracy (mean of seeds)"); axes[1].set_title("A. Damage vs spike size")
        axes[1].legend(fontsize=8)
    for i, (s, (m, _)) in enumerate(ctrl.items()):
        axes[2].plot(m.epoch, m.tstr_val, color=f"C{i}", alpha=0.35, lw=0.8)
        axes[2].plot(m.epoch, m.tstr_rep_mean, color=f"C{i}", lw=1.6, label=f"seed {s}")
    axes[2].set_title("B. Controls: TSTR from 1 sample (thin) vs mean of 10 (thick)")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("validation TSTR"); axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "severity.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/severity")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ctrl = {s: load(f"ctrl_s{s}") for s in SEEDS if (Path("results") / f"ctrl_s{s}" / "summary.json").exists()}
    det, tab, dsum, ident = part_a(ctrl, out)
    noise, corr = part_b(ctrl, out)
    figure(det, dsum, ctrl, out)
    pd.set_option("display.width", 220)
    print("matched-control check (spiked == control up to epoch 90):",
          f"{int(ident.identical_before_spike.sum())}/{len(ident)}" if len(ident) else "n/a")
    print("\nA. detection by series and spike size:\n", tab.to_string(index=False) if len(tab) else "no spiked runs yet")
    print("\nA. damage:\n", dsum.round(4).to_string() if len(dsum) else "n/a")
    print("\nB. TSTR noise per control seed:\n", noise.round(4).to_string(index=False))
    print("\nB. Spearman rho with TSTR (5 controls):\n", corr.to_string())
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
