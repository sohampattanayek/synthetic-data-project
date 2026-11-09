"""Train CTGAN while tracking fidelity every few epochs.

Example:
    python run_experiment.py --epochs 100 --max-rows 10000
    python run_experiment.py --epochs 100 --detector cusum
    python run_experiment.py --epochs 100 --lr 2e-3 --name high_lr   # try to induce degradation

Real data is split 60/20/20 into train / validation / test. Fidelity and
per-check downstream utility (TSTR) are measured on the validation split; the
final comparison uses the untouched test split.

Outputs go to results/<name>/:
    metrics.csv                  per-check fidelity metrics, losses, TSTR, alarms
    summary.json                 config, best epoch, final vs. best comparison
    fidelity_over_training.png   the main figure
"""

from __future__ import annotations

import argparse
import json
import random
import time
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
from validator import FidelityReport, MMDProductKernel, aggregate


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


def tstr_two_splits(train, val, test, target, categorical_cols, seed) -> tuple[float, float]:
    """One classifier fitted on `train`, scored on both real splits."""
    feats = [c for c in train.columns if c != target]
    cats = [c for c in feats if c in categorical_cols]
    ref = pd.concat([train, val, test])

    def encode(df):
        X = df[feats].copy()
        for c in cats:
            X[c] = pd.Categorical(X[c], categories=sorted(ref[c].unique())).codes
        return X.to_numpy(dtype=float)

    clf = HistGradientBoostingClassifier(
        categorical_features=np.array([c in cats for c in feats]), random_state=seed)
    clf.fit(encode(train), train[target].astype(str))
    return (float(accuracy_score(val[target].astype(str), clf.predict(encode(val)))),
            float(accuracy_score(test[target].astype(str), clf.predict(encode(test)))))


def per_check_utility(synth, mon, val, test, cat_cols, seed, repeats, n_eval) -> dict:
    """TSTR on the check's own sample (tstr_val, exactly as in the pilot), plus
    optional repeats on fresh samples from the same generator:
      tstr_rep_mean / tstr_rep_sd   validation TSTR over the repeats (noise estimate)
      tstr_test_rep_mean            test TSTR over the same repeats

    The test-split numbers are LOGGED ONLY and never used by the monitor. They
    let any stopping/selection rule be simulated after the fact: whichever
    check a rule picks, its test utility is already recorded.

    The repeats draw extra random numbers, which would change the rest of the
    training run. All RNG states are saved and restored around them, so a run
    with repeats follows exactly the same trajectory as one without.
    """
    out = {"tstr_val": tstr_accuracy(synth, val, "income", cat_cols, seed)}
    if repeats <= 0:
        return out
    states = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    try:
        reps = np.array([tstr_two_splits(mon.sample(n_eval), val, test, "income", cat_cols, seed)
                         for _ in range(repeats)])
    finally:
        torch.set_rng_state(states[0])
        np.random.set_state(states[1])
        random.setstate(states[2])
    out.update({"tstr_rep_mean": float(reps[:, 0].mean()),
                "tstr_rep_sd": float(reps[:, 0].std(ddof=1)) if repeats > 1 else np.nan,
                "tstr_test_rep_mean": float(reps[:, 1].mean()),
                "tstr_test_rep_sd": float(reps[:, 1].std(ddof=1)) if repeats > 1 else np.nan,
                "tstr_rep_n": repeats})
    return out


def fidelity_extras(synth, val, report, mmd) -> dict:
    """Optional extra fidelity logging (same synthetic sample as the check):
      mmd2, mmd_seconds        product-kernel MMD^2 vs validation (Kim's suggestion)
      col_ks__<c>, col_tvd__<c> per-column KS (numeric) / TVD (categorical)
    Deterministic; uses no global random state."""
    out = {}
    if mmd is not None:
        t0 = time.time()
        out["mmd2"] = mmd.mmd2(val, synth)
        out["mmd_seconds"] = time.time() - t0
    if report is not None:
        for r in report.per_column(val, synth).itertuples():
            if r.type == "numeric":
                out[f"col_ks__{r.column}"] = r.ks
            else:
                out[f"col_tvd__{r.column}"] = r.tvd
    return out


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
    ap.add_argument("--spike-epoch", type=int, default=None,
                    help="planted failure: multiply learning rates after this epoch")
    ap.add_argument("--spike-factor", type=float, default=100.0)
    ap.add_argument("--tstr-repeats", type=int, default=0,
                    help="extra fresh synthetic samples per check, each scored by TSTR, "
                         "to measure how noisy per-check TSTR is (0 = off)")
    ap.add_argument("--mmd", action="store_true",
                    help="log product-kernel MMD^2 vs validation at every check")
    ap.add_argument("--per-column", action="store_true",
                    help="log per-column KS / TVD at every check")
    ap.add_argument("--rollback-series", default=None,
                    help="roll back and continue: run a past-only CUSUM on this logged "
                         "column (e.g. mmd2, score); on an alarm restore the checkpoint "
                         "from the estimated onset and keep training")
    ap.add_argument("--rollback-h", type=float, default=5.0, help="CUSUM threshold (in sds)")
    ap.add_argument("--rollback-max", type=int, default=3, help="most rollbacks per run")
    ap.add_argument("--rollback-keep-lr", action="store_true",
                    help="after a rollback keep the current learning rate instead of "
                         "resetting it to the configured one")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="results subfolder name")
    args = ap.parse_args()

    set_seed(args.seed)
    name = args.name or f"adult_{args.detector}_lr{args.lr:g}_s{args.seed}"
    out = Path("results") / name
    out.mkdir(parents=True, exist_ok=True)

    df, num_cols, cat_cols = load_adult(args.max_rows or None, args.seed)
    train, rest = train_test_split(df, test_size=0.4, random_state=args.seed,
                                   stratify=df["income"])
    val, test = train_test_split(rest, test_size=0.5, random_state=args.seed,
                                 stratify=rest["income"])
    holdout = val  # fidelity is always measured against held-out real data
    print(f"Adult: {len(train)} train / {len(val)} validation / {len(test)} test rows")

    # Noise floor: how different do two REAL samples look under the same metrics?
    report = FidelityReport(num_cols, cat_cols, seed=args.seed)
    n_floor = min(args.n_eval, len(train))
    floor_summary = report.summary(holdout, train.sample(n_floor, random_state=args.seed))
    noise_floor = aggregate(floor_summary)
    print(f"real-vs-real noise floor (aggregate score): {noise_floor:.4f}")

    mmd = MMDProductKernel(train, num_cols, cat_cols, seed=args.seed) if args.mmd else None
    if mmd is not None:
        print(f"MMD bandwidths from training data: h = {mmd.h:.3f}, lambda = {mmd.lam:.3f}")

    mon = MonitoredCTGAN(
        real_eval=holdout, numeric_cols=num_cols, categorical_cols=cat_cols,
        check_every=args.check_every, n_eval=args.n_eval,
        detector=make_detector(args.detector), seed=args.seed,
        batch_size=args.batch_size, generator_lr=args.lr, discriminator_lr=args.lr,
        verbose=True,
        lr_spike=(args.spike_epoch, args.spike_factor) if args.spike_epoch else None,
        rollback=dict(series=args.rollback_series, h=args.rollback_h,
                      max_rollbacks=args.rollback_max,
                      reset_lr=not args.rollback_keep_lr) if args.rollback_series else None,
        extra_metrics=lambda synth: {
            **per_check_utility(synth, mon, val, test, cat_cols, args.seed,
                                args.tstr_repeats, args.n_eval),
            **fidelity_extras(synth, val, report if args.per_column else None, mmd)},
    )
    mon.fit(train, epochs=args.epochs)
    hist = mon.history_frame()
    hist.to_csv(out / "metrics.csv", index=False, float_format="%.5g")

    # Compare the final generator with the best checkpoint.
    n_final = len(train)
    final_synth = mon.sample(n_final)
    final_summary = report.summary(test, final_synth)
    mon.restore_best()
    best_synth = mon.sample(n_final)
    best_summary = report.summary(test, best_synth)

    utility = {
        "train_real": tstr_accuracy(train, test, "income", cat_cols, args.seed),
        "train_final_synth": tstr_accuracy(final_synth, test, "income", cat_cols, args.seed),
        "train_best_synth": tstr_accuracy(best_synth, test, "income", cat_cols, args.seed),
    }

    last_epoch = int(hist["epoch"].iloc[-1])
    summary = {
        "config": vars(args),
        "epochs_trained": last_epoch,
        "stopped_early_at": mon.stopped_epoch,
        "best_epoch": mon.best_epoch,
        "rollbacks": mon.rollbacks,
        "noise_floor": {"score": noise_floor, **floor_summary},
        "final": {"score": aggregate(final_summary), **final_summary},
        "best_checkpoint": {"score": aggregate(best_summary), **best_summary},
        "downstream_accuracy_on_real_test": utility,
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
