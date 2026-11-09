"""Shape of each series over a normal run: smooth curve or change point?

Follow-up to Prof. Killick's advice (call, Nov 4): a mean-shift model for
TSTR; for the generator loss a piecewise-linear model only if a specific
break point is wanted, otherwise a quadratic; and a cosine cost as another
option.

Part 1. For every unspiked control run and every series, six models are
fitted by least squares and compared with BIC (lower = better):

  constant, trend (straight line), quadratic (one smooth curve),
  mean_shift (one jump), broken_trend (two lines joined at a bend),
  quadratic has 3 mean parameters, mean_shift 3 (two levels + location),
  broken_trend 4 (line + slope change + location).

If the quadratic wins for the generator loss, the loss has no change point
in a normal run: its "turn" is a smooth minimum, not a break.
BIC assumes independent errors, so this is a comparison, not a formal test.

Part 2. Cosine cost on the per-column fidelity vector. For one series a
cosine cost is not informative (it only sees the sign), so it is applied to
the 15-dimensional vector of per-column KS/TVD values at each check:
`ruptures` kernel change point detection with the cosine kernel looks for a
change in the DIRECTION of that vector, i.e. in which columns are off, not
in how far off they are in total. Offline, PELT, penalty chosen on the
controls (the smallest of the grid that gives no change after epoch 45 in
any control), then applied to the spiked runs.

Usage:
    python loss_shape.py --runs-dir results/mmd --out results/loss_shape
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt

from dependence import _rss, bic, fit_models

SERIES = {"g_loss": "Generator loss", "tstr_rep_mean": "TSTR, mean of 10",
          "ks_mean": "Mean KS", "jsd_mean": "Mean JSD", "mmd2": "MMD^2",
          "score": "Aggregate score"}
SPIKE = 90
START = 45
PEN_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def model_bics(y, ep):
    t = (ep - ep.min()) / (ep.max() - ep.min())
    n = len(y)
    models = fit_models(y, t)
    rss, _ = _rss(y, np.c_[np.ones(n), t, t ** 2])
    models["quadratic"] = (rss, 3, None, None)
    b = {k: bic(v[0], n, v[1]) for k, v in models.items()}
    win = min(b, key=b.get)
    loc = models[win][2]
    return b, win, (int(ep[loc - 1]) if loc is not None else None)


def part1(ctrl):
    rows = []
    for seed, m in ctrl.items():
        ep = m["epoch"].to_numpy()
        for col, label in SERIES.items():
            if col not in m:
                continue
            b, win, loc = model_bics(m[col].to_numpy(float), ep)
            best_break = min(b["mean_shift"], b["broken_trend"])
            rows.append({"seed": seed, "series": label, "best_model": win, "break_epoch": loc,
                         **{f"bic_{k}": v for k, v in b.items()},
                         "quadratic_minus_best_break": b["quadratic"] - best_break})
    per = pd.DataFrame(rows)
    counts = per.groupby(["series", "best_model"]).size().unstack(fill_value=0)
    gap = per.groupby("series")["quadratic_minus_best_break"].agg(["mean", "min", "max"]).round(1)
    return per, counts, gap


def column_matrix(m):
    cols = [c for c in m.columns if c.startswith("col_")]
    return m[cols].to_numpy(float)


def cosine_changes(m, pen):
    X = column_matrix(m)
    algo = rpt.KernelCPD(kernel="cosine", min_size=2).fit(X)
    return [int(m["epoch"].iloc[i - 1]) for i in algo.predict(pen=pen)[:-1]]


def part2(ctrl, spiked):
    rows = []
    for pen in PEN_GRID:
        late_ctrl = sum(any(e >= START for e in cosine_changes(m, pen)) for m in ctrl.values())
        near = sum(any(abs(e - SPIKE) <= 3 for e in cosine_changes(m, pen)) for m in spiked.values())
        rows.append({"penalty": pen, "controls_with_change_after_45": late_ctrl,
                     "n_controls": len(ctrl), "spiked_with_change_near_90": near,
                     "n_spiked": len(spiked)})
    grid = pd.DataFrame(rows)
    ok = grid[grid.controls_with_change_after_45 == 0]
    chosen = float(ok.penalty.min()) if len(ok) else None
    detail = []
    if chosen is not None:
        for (f, s), m in spiked.items():
            ch = cosine_changes(m, chosen)
            detail.append({"factor": f, "seed": s, "changes": " ".join(map(str, ch)),
                           "near_90": any(abs(e - SPIKE) <= 3 for e in ch)})
    return grid, chosen, pd.DataFrame(detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="results/mmd")
    ap.add_argument("--out", default="results/loss_shape")
    args = ap.parse_args()
    d, out = Path(args.runs_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ctrl = {int(p.parent.name.split("_s")[1]): pd.read_csv(p) for p in sorted(d.glob("ctrl_s*/metrics.csv"))}
    spiked = {(int(p.parent.name.split("x")[1].split("_")[0]), int(p.parent.name.split("_s")[1])): pd.read_csv(p)
              for p in sorted(d.glob("spike90x*_s*/metrics.csv"))}
    pd.set_option("display.width", 200)

    per, counts, gap = part1(ctrl)
    per.to_csv(out / "bic_per_run.csv", index=False, float_format="%.3f")
    counts.to_csv(out / "best_model_counts.csv")
    print(f"1. Best model by BIC, {len(ctrl)} control runs (count of runs):\n", counts.to_string())
    print("\n   BIC(quadratic) - BIC(best change-point model); negative = the smooth curve fits better:\n",
          gap.to_string())

    grid, chosen, detail = part2(ctrl, spiked)
    grid.to_csv(out / "cosine_penalty_grid.csv", index=False)
    print("\n2. Cosine-kernel change detection on the per-column vector:\n", grid.to_string(index=False))
    if chosen is None:
        print("   no penalty in the grid is free of late changes on the controls")
    else:
        detail.to_csv(out / "cosine_spiked_detail.csv", index=False)
        print(f"   chosen penalty {chosen}: spiked runs with a change within 3 epochs of 90, by spike size:\n",
              detail.groupby("factor").near_90.agg(["sum", "size"]).to_string())
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
