"""Draw the fidelity-over-training figure from a finished run.

Usage:
    python plot_results.py results/<name>

Reads metrics.csv and summary.json from that folder and writes
fidelity_over_training.png next to them. Needs only pandas and matplotlib,
so figures can be redrawn without retraining.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot(hist: pd.DataFrame, noise_floor: float, best_epoch, stopped_epoch,
         title: str, path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    ax = axes[0]
    ax.plot(hist["epoch"], hist["score"], marker=".", label="aggregate score")
    ax.axhline(noise_floor, ls="--", c="gray", label="real vs. real (noise floor)")
    if best_epoch is not None:
        ax.axvline(best_epoch, ls=":", c="green", label=f"best checkpoint (epoch {best_epoch})")
    if stopped_epoch:
        ax.axvline(stopped_epoch, ls="-", c="red", alpha=0.6,
                   label=f"detector alarm (epoch {stopped_epoch})")
    ax.set_ylabel("score (lower = better)")
    ax.legend(fontsize=8)
    ax.set_title(title)

    ax = axes[1]
    for col, lab in [("ks_mean", "mean KS statistic"), ("c2st", "C2ST  2|AUC-0.5|"),
                     ("tvd_mean", "mean categorical TVD"),
                     ("excess_point_mass_max", "max excess point mass")]:
        ax.plot(hist["epoch"], hist[col], marker=".", label=lab)
    ax.set_ylabel("metric (lower = better)")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(hist["epoch"], hist["g_loss"], label="generator loss")
    ax.plot(hist["epoch"], hist["d_loss"], label="discriminator loss")
    ax.set_ylabel("loss")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_run(run_dir: Path):
    run_dir = Path(run_dir)
    hist = pd.read_csv(run_dir / "metrics.csv")
    s = json.loads((run_dir / "summary.json").read_text())
    cfg = s["config"]
    title = (f"CTGAN on Adult — detector={cfg['detector']}, "
             f"lr={cfg['lr']:g}, seed={cfg['seed']}")
    out = run_dir / "fidelity_over_training.png"
    plot(hist, s["noise_floor"]["score"], s["best_epoch"], s["stopped_early_at"], title, out)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(f"wrote {plot_run(Path(sys.argv[1]))}")
