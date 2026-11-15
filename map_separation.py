# Usage: python map_separation.py <mantis_export.csv> <name> [<csv> <name> ...]
# Example: python map_separation.py results/mantis/map_audit.csv.gz audit results/mantis/map_drift.csv.gz drift
# Input: CSV from Mantis Export -> CSV (needs columns x, y, source, epoch, impossible, age,
#        hours_per_week, fnlwgt, education_num). Output: results/mantis/separation_<name>.csv
"""Measure how well two groups separate on a Mantis map, using only the map's 2-D positions.
Score = 5-fold cross-validated accuracy of a 10-nearest-neighbour classifier on (x, y),
for a balanced pair of groups (chance = 0.5).

Columns written per epoch:
  ctrl_vs_spike            all rows, one balanced draw and one 5-fold split (seed 0)
  ctrl_vs_spike_clean      rows with no impossible values, same single draw/split
  ctrl_vs_spike_clean_mean, _sd   the clean comparison repeated over 20 seeds
                           (new balanced draw and new folds each time), because the clean
                           groups are small (55-577 rows) and one split is noisy
  real_vs_ctrl, real_vs_spike     the 1,000 real rows against that epoch's synthetic rows
"""
import os, sys, numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

REPEATS = 20

def sep(df, a, b, k=10, seed=0):
    sub = df[df.source.isin([a, b])]
    n = sub.source.value_counts()
    if len(n) < 2 or n.min() < 20: return np.nan, int(n.min()) if len(n) else 0
    # balance the two groups
    sub = pd.concat([sub[sub.source == s].sample(n.min(), random_state=seed) for s in (a, b)])
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    acc = cross_val_score(KNeighborsClassifier(k), sub[["x", "y"]].values, sub.source.values, cv=cv).mean()
    return acc, int(n.min())

def tell_v3(d):  # impossible values still visible in the v3 text
    return (d.age < 17) | (d.age > 90) | (d.hours_per_week < 1) | (d.hours_per_week > 99) | (d.fnlwgt < 0) | (d.education_num < 1) | (d.education_num > 16)

if __name__ == "__main__":
    os.makedirs("results/mantis", exist_ok=True)
    for path, name in zip(sys.argv[1::2], sys.argv[2::2]):
        d = pd.read_csv(path)
        assert len(d) == 17000 and d[["x", "y"]].notna().all().all()
        real = d[d.source == "real"]
        rows = []
        for e in [30, 60, 87, 90, 93, 99, 120, 150]:
            g = pd.concat([d[d.epoch == e], real])
            clean = g[(g.impossible == 0) & ~tell_v3(g)]
            r = {"epoch": e}
            r["ctrl_vs_spike"], r["n"] = sep(g, "ctrl", "spike")
            r["ctrl_vs_spike_clean"], r["n_clean"] = sep(clean, "ctrl", "spike")
            reps = [sep(clean, "ctrl", "spike", seed=s)[0] for s in range(REPEATS)]
            r["ctrl_vs_spike_clean_mean"], r["ctrl_vs_spike_clean_sd"] = np.mean(reps), np.std(reps, ddof=1)
            r["real_vs_ctrl"], _ = sep(g, "real", "ctrl")
            r["real_vs_spike"], _ = sep(g, "real", "spike")
            rows.append(r)
        out = pd.DataFrame(rows).round(3)
        print(f"== {name}\n{out.to_string(index=False)}\n")
        out.to_csv(f"results/mantis/separation_{name}.csv", index=False)
