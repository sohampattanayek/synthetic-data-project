"""Change-point analysis of training-metric time series.

Part A (offline, natural breaks): for the default-setting pilot runs, estimate
where each series changes, using the CUSUM statistic for a change in mean
and binary segmentation for multiple change points.

  * TSTR accuracy and mean KS are analysed in levels (change in mean).
  * Generator loss is analysed in first differences: a change in the mean of
    the differences is a change in slope, which is what the loss curve shows
    (falling, then rising).

Part B (online, planted break): for runs where the learning rate is multiplied
at a known epoch, run a one-sided CUSUM detector check by check and record
the detection delay. The same detector is run on the seed-matched runs
without a spike (the pilot) to count false alarms.

Usage:
    python changepoints.py --pilot results/pilot_s{0..4} --spiked results/spike90_s{0..4} --out results/changepoints
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

# 5% critical value of sup|Brownian bridge| (Kolmogorov distribution). Valid
# for independent noise; serial dependence in training metrics would require a
# long-run variance estimate instead, so p-values here are approximate.
CRIT_05 = 1.358


# ---------------------------------------------------------------------------
# Offline CUSUM for a change in mean
# ---------------------------------------------------------------------------

def noise_sd(x: np.ndarray) -> float:
    """Robust noise scale from first differences (insensitive to mean shifts)."""
    d = np.diff(x)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def cusum_stat(x: np.ndarray, sigma: float):
    """Standardized CUSUM process |S_k - (k/n) S_n| / (sigma sqrt(n)), k=1..n-1.
    Returns (argmax k, max value). The change is estimated to occur after
    observation k."""
    n = len(x)
    s = np.cumsum(x - x.mean())
    stat = np.abs(s[:-1]) / (sigma * np.sqrt(n))
    k = int(np.argmax(stat)) + 1
    return k, float(stat.max())


def binary_segmentation(x: np.ndarray, sigma: float, min_seg: int = 5,
                        crit: float = CRIT_05, offset: int = 0, out=None):
    """Recursive CUSUM binary segmentation. Returns list of (index, stat)."""
    out = [] if out is None else out
    if len(x) < 2 * min_seg:
        return out
    k, stat = cusum_stat(x, sigma)
    if stat < crit or k < min_seg or len(x) - k < min_seg:
        return out
    out.append((offset + k, stat))
    binary_segmentation(x[:k], sigma, min_seg, crit, offset, out)
    binary_segmentation(x[k:], sigma, min_seg, crit, offset + k, out)
    return out


def analyse_series(epochs: np.ndarray, x: np.ndarray, differenced: bool):
    if differenced:
        x = np.diff(x)
        epochs = epochs[1:]
    sigma = noise_sd(x)
    k, stat = cusum_stat(x, sigma)
    multi = sorted(binary_segmentation(x, sigma))
    # report the epoch of the last observation BEFORE the change
    return {
        "single_epoch": int(epochs[k - 1]),
        "single_stat": stat,
        "significant": stat >= CRIT_05,
        "mean_before": float(x[:k].mean()),
        "mean_after": float(x[k:].mean()),
        "all_epochs": [int(epochs[i - 1]) for i, _ in multi],
    }


# ---------------------------------------------------------------------------
# Online one-sided CUSUM
# ---------------------------------------------------------------------------

def online_cusum(x: np.ndarray, epochs: np.ndarray, ref: tuple[int, int],
                 start: int, direction: int, k_sd: float = 0.5, h_sd: float = 5.0):
    """Page's CUSUM for a shift in the 'bad' direction (direction=+1 means
    an increase is bad). Reference mean and sd come from the checks with
    epoch in [ref[0], ref[1]]; monitoring starts at epoch `start`.
    Returns the first alarm epoch or None."""
    m = (epochs >= ref[0]) & (epochs <= ref[1])
    mu, sd = x[m].mean(), x[m].std(ddof=1)
    if sd == 0:
        return None
    g = 0.0
    for e, v in zip(epochs, x):
        if e < start:
            continue
        z = direction * (v - mu) / sd
        g = max(0.0, g + z - k_sd)
        if g > h_sd:
            return int(e)
    return None


ONLINE_SERIES = {
    "tstr_val": ("TSTR accuracy", -1),          # a drop is bad
    "score": ("Aggregate fidelity score", +1),  # a rise is bad
    "excess_point_mass_max": ("Max excess point mass", +1),
    "ks_mean": ("Mean KS statistic", +1),
    "g_loss": ("Generator loss", +1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", nargs="+", required=True)
    ap.add_argument("--spiked", nargs="*", default=[])
    ap.add_argument("--out", default="results/changepoints")
    ap.add_argument("--ref", type=int, nargs=2, default=[45, 87],
                    help="epochs used to estimate the pre-change mean/sd for online CUSUM")
    ap.add_argument("--monitor-from", type=int, default=48)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- Part A: natural breaks (offline) ----------------
    offline_rows = []
    pilot = {}
    for r in args.pilot:
        m = pd.read_csv(Path(r) / "metrics.csv")
        pilot[Path(r).name] = m
        ep = m["epoch"].to_numpy()
        for col, label, diff in [("tstr_val", "TSTR accuracy (levels)", False),
                                 ("ks_mean", "Mean KS (levels)", False),
                                 ("g_loss", "Generator loss (slope, via differences)", True)]:
            res = analyse_series(ep, m[col].to_numpy(dtype=float), diff)
            offline_rows.append({"run": Path(r).name, "series": label, **res})
    off = pd.DataFrame(offline_rows)
    off.to_csv(out / "offline_changepoints.csv", index=False, float_format="%.4f")

    # ---------------- Part B: planted break (online) ------------------
    online_rows = []
    spiked = {}
    for r in args.spiked:
        m = pd.read_csv(Path(r) / "metrics.csv")
        cfg = json.loads((Path(r) / "summary.json").read_text())["config"]
        spiked[Path(r).name] = (m, cfg["spike_epoch"])
    for name, (m, spike) in spiked.items():
        seed = name.split("_s")[-1]
        control = pilot.get(f"pilot_s{seed}")
        for col, (label, direction) in ONLINE_SERIES.items():
            alarm = online_cusum(m[col].to_numpy(float), m["epoch"].to_numpy(),
                                 tuple(args.ref), args.monitor_from, direction)
            ctrl = None
            if control is not None:
                ctrl = online_cusum(control[col].to_numpy(float), control["epoch"].to_numpy(),
                                    tuple(args.ref), args.monitor_from, direction)
            online_rows.append({
                "run": name, "series": label, "spike_epoch": spike,
                "alarm_epoch": alarm,
                "false_alarm_before_spike": alarm is not None and alarm <= spike,
                "delay_epochs": (alarm - spike) if (alarm is not None and alarm > spike) else None,
                "control_alarm_epoch": ctrl,
            })
    onl = pd.DataFrame(online_rows)
    if len(onl):
        onl.to_csv(out / "online_detection.csv", index=False)

    # ---------------- Figure ------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(8, 11.5), sharex=True)
    for i, (name, m) in enumerate(pilot.items()):
        c = f"C{i}"
        ep = m["epoch"]
        axes[0].plot(ep, m["tstr_val"], color=c, alpha=0.7, label=name)
        axes[1].plot(ep, m["g_loss"], color=c, alpha=0.7)
        r_t = off[(off.run == name) & (off.series.str.startswith("TSTR"))].iloc[0]
        r_g = off[(off.run == name) & (off.series.str.startswith("Generator"))].iloc[0]
        axes[0].axvline(r_t.single_epoch, color=c, ls=":", alpha=0.8)
        axes[1].axvline(r_g.single_epoch, color=c, ls=":", alpha=0.8)
    axes[0].set_ylabel("TSTR accuracy")
    axes[0].set_title("A. Natural breaks (default settings): dotted = CUSUM estimate per seed")
    axes[0].legend(fontsize=7, ncol=5)
    axes[1].set_ylabel("generator loss")
    for i, (name, (m, spike)) in enumerate(spiked.items()):
        seed = name.split("_s")[-1]
        c = f"C{i}"
        ctrl = pilot.get(f"pilot_s{seed}")
        for ax, col in [(axes[2], "score"), (axes[3], "tstr_val")]:
            ax.plot(m["epoch"], m[col], color=c, alpha=0.8, label=f"spiked s{seed}")
            if ctrl is not None:
                ax.plot(ctrl["epoch"], ctrl[col], color=c, alpha=0.3, ls="--")
        if len(onl):
            a = onl[(onl.run == name) & (onl.series == "Aggregate fidelity score")].alarm_epoch.iloc[0]
            if pd.notna(a):
                v = m.loc[m.epoch == a, "score"].iloc[0]
                axes[2].plot([a], [v], marker="v", color=c, ms=8)
    if spiked:
        spike = next(iter(spiked.values()))[1]
        for ax in axes[2:]:
            ax.axvline(spike, color="k", lw=1.5)
        axes[2].set_title(f"B. Planted break at epoch {spike} (learning rate x100). Solid = spiked, dashed = same seed without spike",
                          fontsize=9)
        axes[2].set_ylabel("fidelity score\n(lower = better)")
        axes[2].plot([], [], "kv", label="online CUSUM alarm")
        axes[2].legend(fontsize=7, ncol=3)
    axes[3].set_ylabel("TSTR accuracy")
    axes[3].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out / "changepoints.png", dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 200)
    print("PART A - offline single change point (epoch of last check before the change):")
    print(off.pivot(index="run", columns="series", values="single_epoch").to_string())
    print("\nsignificant at 5% (independence approx.):")
    print(off.pivot(index="run", columns="series", values="significant").to_string())
    print("\nmeans before/after:")
    print(off.groupby("series")[["mean_before", "mean_after", "single_stat"]].mean().round(4).to_string())
    print("\nmultiple change points (binary segmentation):")
    print(off[["run", "series", "all_epochs"]].to_string(index=False))
    if len(onl):
        print("\nPART B - online CUSUM:")
        print(onl.to_string(index=False))
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
