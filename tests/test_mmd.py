"""Sanity checks for the product-kernel MMD (run: python tests/test_mmd.py).

1. Real vs real (two disjoint real samples): MMD^2 close to 0.
2. A shifted copy of real data: MMD^2 > 0, clearly above real vs real.
3. MMD^2 increases with the size of the shift (numeric and categorical).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import load_adult  # noqa: E402
from validator import MMDProductKernel  # noqa: E402


def shifted(df, num_cols, cat_cols, size, seed=0):
    """Shift every numeric column by `size` training sds, and replace a
    fraction `size/2` of each categorical column with its most common value."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in num_cols:
        out[c] = out[c] + size * df[c].std()
    for c in cat_cols:
        mask = rng.random(len(out)) < min(size / 2, 1.0)
        out.loc[mask, c] = df[c].mode().iloc[0]
    return out


def main():
    df, num_cols, cat_cols = load_adult(10000, 0)
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    train, a, b = df.iloc[:6000], df.iloc[6000:8000], df.iloc[8000:10000]
    mmd = MMDProductKernel(train, num_cols, cat_cols, seed=0)
    print(f"bandwidths: h = {mmd.h:.3f}, lambda = {mmd.lam:.3f}")

    rr = mmd.mmd2(a, b)
    print(f"real vs real: {rr:.5f}")
    sizes = [0.1, 0.25, 0.5, 1.0]
    vals = [mmd.mmd2(a, shifted(b, num_cols, cat_cols, s)) for s in sizes]
    for s, v in zip(sizes, vals):
        print(f"shift {s:<4}: {v:.5f}")

    assert abs(rr) < 0.002, rr
    assert vals[0] > 5 * max(abs(rr), 1e-4), (vals[0], rr)
    assert all(x < y for x, y in zip(vals, vals[1:])), vals
    print("all MMD checks passed")


if __name__ == "__main__":
    main()
