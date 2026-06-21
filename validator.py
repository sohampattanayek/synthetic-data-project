import numpy as np
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

class DataValidator:
    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def distribution_check(self, real_col, synthetic_col):
        stat, p_value = ks_2samp(real_col, synthetic_col)
        return p_value

    def calculate_js_divergence(self, real_col, synthetic_col):
        p, _ = np.histogram(real_col, bins=20, density=True)
        q, _ = np.histogram(synthetic_col, bins=20, density=True)
        p += 1e-10
        q += 1e-10
        return jensenshannon(p, q)

    def evaluate(self, real_df, synthetic_df, columns):
        results = {}
        for col in columns:
            p_val = self.distribution_check(real_df[col], synthetic_df[col])
            js_div = self.calculate_js_divergence(real_df[col], synthetic_df[col])
            needs_retrain = p_val < self.threshold or js_div > 0.2
            results[col] = {"p_value": p_val, "js_divergence": js_div, "retrain": needs_retrain}
        return results