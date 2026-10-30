"""Two fixes to the change-point analysis in changepoints.py.

1. Serial dependence. changepoints.py scales the CUSUM statistic by a noise
   sd estimated from first differences, and compares it with the 5% critical
   value of sup|Brownian bridge| (1.358). That critical value is only valid
   for independent noise. With autocorrelated noise the right scale is the
   long-run standard deviation (the square root of the sum of all
   autocovariances). Here it is estimated with a Bartlett kernel
   (Newey-West) on the residuals of the fitted one-change model, so the
   change itself does not inflate it. The test is then repeated with that
   scale.

2. Break vs. trend. A mean-shift CUSUM also "finds" a change in a series that
   just rises smoothly. Four models are fitted by least squares to each
   series and compared with BIC:
       constant mean          (1 parameter + variance)
       mean shift at k        (3: two means + location)
       linear trend           (2)
       broken trend at k      (4: intercept, slope, change in slope, location;
                               continuous at k)
   BIC assumes independent errors, so with autocorrelation it tends to favour
   the more complex models; read it as a comparison, not a test.

Usage:
    python dependence.py --runs results/pilot_s{0..4} --out results/dependence
    python dependence.py --runs results/ctrl_s{0..4} --series tstr_val tstr_rep_mean ks_mean --out results/dependence_ctrl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from changepoints import CRIT_05, cusum_stat, noise_sd


def bartlett_lrv_sd(resid: np.ndarray, bandwidth: int | None = None) -> float:
    """Long-run sd of a mean-zero series, Bartlett kernel (Newey & West, 1987)."""
    e = resid - resid.mean()
    n = len(e)
    L = bandwidth if bandwidth is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    gamma0 = e @ e / n
    lrv = gamma0
    for lag in range(1, L + 1):
        w = 1 - lag / (L + 1)
        lrv += 2 * w * (e[lag:] @ e[:-lag]) / n
    return float(np.sqrt(max(lrv, 1e-12)))


def lag1_autocorr(e: np.ndarray) -> float:
    e = e - e.mean()
    return float((e[1:] @ e[:-1]) / (e @ e))


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def _rss(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(r @ r), r


def fit_models(y: np.ndarray, t: np.ndarray, min_seg: int = 5) -> dict:
    n = len(y)
    one = np.ones(n)
    out = {}
    rss, r = _rss(y, one[:, None]); out["constant"] = (rss, 1, None, r)
    rss, r = _rss(y, np.c_[one, t]); out["trend"] = (rss, 2, None, r)
    best_shift = best_kink = (np.inf, None, None)
    for k in range(min_seg, n - min_seg + 1):
        step = (np.arange(n) >= k).astype(float)
        rs, rr = _rss(y, np.c_[one - step, step])
        if rs < best_shift[0]:
            best_shift = (rs, k, rr)
        hinge = np.maximum(0, t - t[k - 1])
        rs, rr = _rss(y, np.c_[one, t, hinge])
        if rs < best_kink[0]:
            best_kink = (rs, k, rr)
    out["mean_shift"] = (best_shift[0], 3, best_shift[1], best_shift[2])
    out["broken_trend"] = (best_kink[0], 4, best_kink[1], best_kink[2])
    return out


def bic(rss: float, n: int, p: int) -> float:
    return n * np.log(rss / n) + (p + 1) * np.log(n)  # +1 for the error variance


# ---------------------------------------------------------------------------

def analyse(m: pd.DataFrame, col: str) -> dict:
    y = m[col].to_numpy(float)
    ep = m["epoch"].to_numpy()
    t = (ep - ep.min()) / (ep.max() - ep.min())   # rescaled time, for conditioning
    n = len(y)

    # CUSUM with the original (iid) scale and with the long-run scale
    k, stat_raw = cusum_stat(y, noise_sd(y))
    sd_iid = noise_sd(y)
    resid = np.r_[y[:k] - y[:k].mean(), y[k:] - y[k:].mean()]
    sd_lr = bartlett_lrv_sd(resid)
    _, stat_lr = cusum_stat(y, sd_lr)

    models = fit_models(y, t)
    bics = {name: bic(rss, n, p) for name, (rss, p, _, _) in models.items()}
    winner = min(bics, key=bics.get)
    loc = models[winner][2]
    return {
        "series": col,
        "cusum_epoch": int(ep[k - 1]),
        "stat_iid_scale": stat_raw, "significant_iid": stat_raw >= CRIT_05,
        "lag1_autocorr_resid": lag1_autocorr(resid),
        "sd_iid": sd_iid, "sd_longrun": sd_lr,
        "stat_longrun_scale": stat_lr, "significant_longrun": stat_lr >= CRIT_05,
        **{f"bic_{k_}": v for k_, v in bics.items()},
        "best_model": winner,
        "best_model_break_epoch": int(ep[loc - 1]) if loc is not None else None,
        "bic_gap_to_runner_up": float(sorted(bics.values())[1] - bics[winner]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--series", nargs="+", default=["tstr_val", "ks_mean", "score"])
    ap.add_argument("--out", default="results/dependence")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in args.runs:
        m = pd.read_csv(Path(r) / "metrics.csv")
        for col in args.series:
            if col in m:
                rows.append({"run": Path(r).name, **analyse(m, col)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "dependence.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    cols = ["run", "series", "cusum_epoch", "lag1_autocorr_resid", "stat_iid_scale",
            "significant_iid", "stat_longrun_scale", "significant_longrun",
            "best_model", "best_model_break_epoch", "bic_gap_to_runner_up"]
    print(df[cols].round(3).to_string(index=False))
    print("\nsummary by series:")
    g = df.groupby("series").agg(
        n=("run", "size"),
        sig_iid=("significant_iid", "sum"), sig_longrun=("significant_longrun", "sum"),
        lag1_mean=("lag1_autocorr_resid", "mean"),
        sd_ratio=("sd_longrun", "mean"),
    )
    g["sd_ratio"] = (df.sd_longrun / df.sd_iid).groupby(df.series).mean()
    print(g.round(3).to_string())
    print("\nbest model counts:")
    print(df.groupby(["series", "best_model"]).size().unstack(fill_value=0).to_string())
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
