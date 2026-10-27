"""Fidelity metrics comparing a real table to a synthetic table.

All metrics are effect sizes (distances), not p-values. With large samples a
two-sample test rejects even negligible differences, and with small samples it
accepts large ones, so p-values are not used to decide anything here.

Every metric is oriented so that 0 means "indistinguishable" and larger means
"worse".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Univariate (marginal) metrics
# ---------------------------------------------------------------------------

def ks_statistic(real: pd.Series, synth: pd.Series) -> float:
    """Two-sample Kolmogorov-Smirnov statistic D = sup_x |F_real(x) - F_synth(x)|.

    Returns the statistic itself (an effect size in [0, 1]), not the p-value.
    """
    return float(ks_2samp(real.to_numpy(), synth.to_numpy()).statistic)


def js_divergence(real: pd.Series, synth: pd.Series, bins: int = 20) -> float:
    """Jensen-Shannon divergence (base 2, so in [0, 1]) between binned columns.

    Both histograms use the SAME bin edges, computed from the pooled data.
    (Binning each column separately compares histograms over different
    intervals, which makes the result meaningless.)
    """
    r = real.to_numpy(dtype=float)
    s = synth.to_numpy(dtype=float)
    edges = np.histogram_bin_edges(np.concatenate([r, s]), bins=bins)
    p, _ = np.histogram(r, bins=edges)
    q, _ = np.histogram(s, bins=edges)
    return _jsd_from_counts(p, q)


def tv_distance(real: pd.Series, synth: pd.Series) -> float:
    """Total variation distance between category frequencies, in [0, 1]."""
    p = real.value_counts(normalize=True)
    q = synth.value_counts(normalize=True)
    p, q = p.align(q, fill_value=0.0)
    return float(0.5 * np.abs(p - q).sum())


def excess_point_mass(real: pd.Series, synth: pd.Series) -> float:
    """How much more probability the synthetic column puts on its single most
    common exact value than the real column puts on ITS most common value.

    Catches a common tabular-GAN failure: samples piling up on one value
    (often a boundary such as the column min or max). Real columns can have
    legitimate point masses (e.g. capital-gain = 0), so we report the excess
    over the real data rather than the raw share. Clipped at 0.
    """
    real_top = real.value_counts(normalize=True).iloc[0]
    synth_top = synth.value_counts(normalize=True).iloc[0]
    return float(max(0.0, synth_top - real_top))


def _jsd_from_counts(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(float) / max(p.sum(), 1)
    q = q.astype(float) / max(q.sum(), 1)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ---------------------------------------------------------------------------
# Multivariate (joint-structure) metrics
# ---------------------------------------------------------------------------

def correlation_distance(real: pd.DataFrame, synth: pd.DataFrame,
                         numeric_cols: list[str]) -> float:
    """Mean absolute difference between the Pearson correlation matrices of
    the numeric columns (upper triangle only). In [0, 2]."""
    if len(numeric_cols) < 2:
        return 0.0
    cr = real[numeric_cols].corr().to_numpy()
    cs = synth[numeric_cols].corr().fillna(0.0).to_numpy()
    iu = np.triu_indices(len(numeric_cols), k=1)
    return float(np.mean(np.abs(cr[iu] - cs[iu])))


def c2st_score(real: pd.DataFrame, synth: pd.DataFrame,
               categorical_cols: list[str], seed: int = 0) -> float:
    """Classifier two-sample test (Lopez-Paz & Oquab, 2017).

    Train a classifier to tell real rows from synthetic rows and measure its
    held-out ROC AUC. AUC = 0.5 means the classifier cannot tell them apart.
    Returned as 2 * |AUC - 0.5|, so 0 = indistinguishable, 1 = trivially
    separable. Unlike the column-wise metrics, this sees joint structure.
    """
    n = min(len(real), len(synth))
    both = pd.concat([real.sample(n, random_state=seed),
                      synth.sample(n, random_state=seed)], ignore_index=True)
    y = np.r_[np.zeros(n), np.ones(n)]

    X = both.copy()
    cat_mask = []
    for c in X.columns:
        is_cat = c in categorical_cols
        cat_mask.append(is_cat)
        if is_cat:
            X[c] = X[c].astype("category").cat.codes
    X = X.to_numpy(dtype=float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=seed)
    clf = HistGradientBoostingClassifier(
        max_iter=100, categorical_features=np.array(cat_mask), random_state=seed)
    clf.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
    return float(2 * abs(auc - 0.5))


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

class FidelityReport:
    """Computes all metrics for one (real, synthetic) pair."""

    def __init__(self, numeric_cols: list[str], categorical_cols: list[str],
                 bins: int = 20, seed: int = 0):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.bins = bins
        self.seed = seed

    def per_column(self, real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for c in self.numeric_cols:
            rows.append({
                "column": c, "type": "numeric",
                "ks": ks_statistic(real[c], synth[c]),
                "jsd": js_divergence(real[c], synth[c], self.bins),
                "excess_point_mass": excess_point_mass(real[c], synth[c]),
            })
        for c in self.categorical_cols:
            rows.append({
                "column": c, "type": "categorical",
                "tvd": tv_distance(real[c], synth[c]),
            })
        return pd.DataFrame(rows)

    def summary(self, real: pd.DataFrame, synth: pd.DataFrame) -> dict:
        cols = self.per_column(real, synth)
        num = cols[cols["type"] == "numeric"]
        cat = cols[cols["type"] == "categorical"]
        return {
            "ks_mean": float(num["ks"].mean()) if len(num) else np.nan,
            "ks_max": float(num["ks"].max()) if len(num) else np.nan,
            "jsd_mean": float(num["jsd"].mean()) if len(num) else np.nan,
            "tvd_mean": float(cat["tvd"].mean()) if len(cat) else np.nan,
            "excess_point_mass_max": float(num["excess_point_mass"].max()) if len(num) else np.nan,
            "corr_dist": correlation_distance(real, synth, self.numeric_cols),
            "c2st": c2st_score(real, synth, self.categorical_cols, self.seed),
        }


def aggregate(summary: dict) -> float:
    """Single scalar used for checkpoint selection and change detection.

    An unweighted mean of metrics that all live on [0, 1]. This is a
    deliberately simple choice; how to weight or combine metrics is one of
    the open questions of the study.
    """
    keys = ["ks_mean", "tvd_mean", "c2st", "excess_point_mass_max"]
    vals = [summary[k] for k in keys if not np.isnan(summary.get(k, np.nan))]
    return float(np.mean(vals))
