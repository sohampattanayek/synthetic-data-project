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


class MMDProductKernel:
    """Maximum mean discrepancy with a product kernel for mixed data.

    Suggested by Prof. Ilmun Kim (Oct 31) as one option for mixed
    numeric/categorical tables, because C2ST saturates: it answers *whether*
    two distributions differ, not *by how much*. MMD^2 grows with the size
    of the difference.

        k(x, y) = exp(-||x_num - y_num||^2 / (2 h^2)) * exp(-d_H(x_cat, y_cat) / lam)

    x_num are the numeric columns standardized with the TRAINING split's
    mean and sd; d_H is the number of categorical columns that differ
    (Hamming distance). The bandwidths h and lam are fixed once, from real
    training rows only (median heuristic: median pairwise distance), so the
    metric means the same thing at every check.

    `mmd2` returns the unbiased estimate of MMD^2 (it can be slightly
    negative when the two samples come from the same distribution).
    Uses its own random generator, so it never touches the global RNG.
    """

    def __init__(self, real_train: pd.DataFrame, numeric_cols, categorical_cols,
                 categories: dict | None = None, n_bandwidth: int = 1000, seed: int = 0):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.mu = real_train[self.numeric_cols].mean().to_numpy(float)
        self.sd = real_train[self.numeric_cols].std().replace(0, 1).to_numpy(float)
        self.categories = categories or {
            c: sorted(real_train[c].astype(str).unique()) for c in self.categorical_cols}
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(real_train), size=min(n_bandwidth, len(real_train)), replace=False)
        xn, xc = self._encode(real_train.iloc[idx])
        iu = np.triu_indices(len(idx), k=1)
        self.h = float(np.median(np.sqrt(_sq_dists(xn, xn)[iu])))
        self.lam = float(np.median(_hamming(xc, xc)[iu]))
        if not self.h > 0:
            self.h = 1.0
        if not self.lam > 0:
            self.lam = 1.0

    def _encode(self, df: pd.DataFrame):
        xn = (df[self.numeric_cols].to_numpy(float) - self.mu) / self.sd
        xc = np.column_stack([
            pd.Categorical(df[c].astype(str), categories=self.categories[c]).codes
            for c in self.categorical_cols]) if self.categorical_cols else np.zeros((len(df), 0), int)
        return xn, xc

    def kernel(self, a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
        an, ac = self._encode(a)
        bn, bc = self._encode(b)
        return self._kernel(an, ac, bn, bc)

    def _kernel(self, an, ac, bn, bc):
        return np.exp(-_sq_dists(an, bn) / (2 * self.h ** 2) - _hamming(ac, bc) / self.lam)

    def mmd2(self, real: pd.DataFrame, synth: pd.DataFrame) -> float:
        xn, xc = self._encode(real)
        yn, yc = self._encode(synth)
        m, n = len(xn), len(yn)
        kxx = self._kernel(xn, xc, xn, xc)
        kyy = self._kernel(yn, yc, yn, yc)
        kxy = self._kernel(xn, xc, yn, yc)
        return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                     + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
                     - 2 * kxy.mean())


def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    return np.maximum(d, 0.0)


def _hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros((len(a), len(b)), dtype=np.float32)
    for j in range(a.shape[1]):
        out += a[:, j][:, None] != b[:, j][None, :]
    return out


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
