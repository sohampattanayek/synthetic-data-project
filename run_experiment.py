"""Train CTGAN while tracking fidelity every few epochs.

Example:
    python run_experiment.py --epochs 100 --max-rows 10000
    python run_experiment.py --epochs 100 --detector cusum
    python run_experiment.py --epochs 100 --lr 2e-3 --name high_lr   # try to induce degradation

Outputs go to results/<name>/:
    metrics.csv                  per-check fidelity metrics, losses, alarms
    summary.json                 config, best epoch, final vs. best comparison
    fidelity_over_training.png   the main figure
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data import load_adult
from monitor import MonitoredCTGAN, make_detector
from plot_results import plot_run
from validator import FidelityReport, aggregate


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tstr_accuracy(train: pd.DataFrame, test: pd.DataFrame, target: str,
                  categorical_cols: list[str], seed: int) -> float:
    """Train-on-X, test-on-real: accuracy of a classifier predicting `target`."""
    feats = [c for c in train.columns if c != target]
    cats = [c for c in feats if c in categorical_cols]

    def encode(df, ref):
        X = df[feats].copy()
        for c in cats:
            X[c] = pd.Categorical(X[c], categories=sorted(ref[c].unique())).codes
        return X.to_numpy(dtype=float)

    ref = pd.concat([train, test])
    clf = HistGradientBoostingClassifier(
        categorical_features=np.array([c in cats for c in feats]), random_state=seed)
    clf.fit(encode(train, ref), train[target].astype(str))
    return float(accuracy_score(test[target].astype(str), clf.predict(encode(test, ref))))


def _round(obj, digits=5):
    if isinstance(obj, float):
        return float(f"{obj:.{digits}g}")
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--check-every", type=int, default=2, help="evaluate every k epochs")
    ap.add_argument("--max-rows", type=int, default=10000, help="subsample Adult for speed (0 = all)")
    ap.add_argument("--n-eval", type=int, default=2000, help="synthetic rows sampled per check")
    ap.add_argument("--detector", choices=["none", "patience", "cusum"], default="none")
    ap.add_argument("--lr", type=float, default=2e-4, help="generator and discriminator learning rate")
    ap.add_argument("--batch-size", type=int, default=500, help="must be a multiple of 10 (CTGAN pac)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="results subfolder name")
    args = ap.parse_args()

    set_seed(args.seed)
    name = args.name or f"adult_{args.detector}_lr{args.lr:g}_s{args.seed}"
    out = Path("results") / name
    out.mkdir(parents=True, exist_ok=True)

    df, num_cols, cat_cols = load_adult(args.max_rows or None, args.seed)
    train, holdout = train_test_split(df, test_size=0.3, random_state=args.seed,
                                      stratify=df["income"])
    print(f"Adult: {len(train)} train rows, {len(holdout)} held-out rows")

    # Noise floor: how different do two REAL samples look under the same metrics?
    report = FidelityReport(num_cols, cat_cols, seed=args.seed)
    n_floor = min(args.n_eval, len(train))
    floor_summary = report.summary(holdout, train.sample(n_floor, random_state=args.seed))
    noise_floor = aggregate(floor_summary)
    print(f"real-vs-real noise floor (aggregate score): {noise_floor:.4f}")

    mon = MonitoredCTGAN(
        real_eval=holdout, numeric_cols=num_cols, categorical_cols=cat_cols,
        check_every=args.check_every, n_eval=args.n_eval,
        detector=make_detector(args.detector), seed=args.seed,
        batch_size=args.batch_size, generator_lr=args.lr, discriminator_lr=args.lr,
        verbose=True,
    )
    mon.fit(train, epochs=args.epochs)
    hist = mon.history_frame()
    hist.to_csv(out / "metrics.csv", index=False, float_format="%.5g")

    # Compare the final generator with the best checkpoint.
    n_final = len(holdout)
    final_synth = mon.sample(n_final)
    final_summary = report.summary(holdout, final_synth)
    mon.restore_best()
    best_synth = mon.sample(n_final)
    best_summary = report.summary(holdout, best_synth)

    utility = {
        "train_real": tstr_accuracy(train, holdout, "income", cat_cols, args.seed),
        "train_final_synth": tstr_accuracy(final_synth, holdout, "income", cat_cols, args.seed),
        "train_best_synth": tstr_accuracy(best_synth, holdout, "income", cat_cols, args.seed),
    }

    last_epoch = int(hist["epoch"].iloc[-1])
    summary = {
        "config": vars(args),
        "epochs_trained": last_epoch,
        "stopped_early_at": mon.stopped_epoch,
        "best_epoch": mon.best_epoch,
        "noise_floor": {"score": noise_floor, **floor_summary},
        "final": {"score": aggregate(final_summary), **final_summary},
        "best_checkpoint": {"score": aggregate(best_summary), **best_summary},
        "downstream_accuracy_on_real_holdout": utility,
    }
    (out / "summary.json").write_text(json.dumps(_round(summary), indent=2))
    plot_run(out)

    print(f"\nbest checkpoint: epoch {mon.best_epoch}  (score {summary['best_checkpoint']['score']:.4f})")
    print(f"final epoch {last_epoch}:      score {summary['final']['score']:.4f}")
    print("downstream accuracy (train on X, test on real):",
          {k: round(v, 4) for k, v in utility.items()})
    print(f"results written to {out}/")


if __name__ == "__main__":
    main()
