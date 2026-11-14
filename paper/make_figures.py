"""Figure for paper v4, made only from logged metrics.csv files in the repo.
Usage: python make_figures.py <repo>/results <outdir>"""
import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R, OUT = sys.argv[1], sys.argv[2]
def load(path):
    d = pd.read_csv(path)
    # after a rollback the same epochs are trained again; keep the final trajectory
    return d.drop_duplicates("epoch", keep="last").set_index("epoch").sort_index()
def mean_of(paths, col):
    return pd.concat([load(p)[col] for p in paths], axis=1).mean(axis=1)
S10, S5 = range(10), range(5)
ctrl = [f"{R}/mmd/ctrl_s{s}/metrics.csv" for s in S10]
sp = [f"{R}/mmd/spike90x100_s{s}/metrics.csv" for s in S10]
rb = [f"{R}/rollback/spike90x100_s{s}/metrics.csv" for s in S10]
kl = [f"{R}/rollback_keeplr/spike90x100_s{s}/metrics.csv" for s in S5]
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "legend.fontsize": 6})
fig, ax = plt.subplots(2, 1, figsize=(3.45, 3.6), sharex=True)
ax[0].plot(mean_of(ctrl, "score"), color="0.35", lw=1, label="no spike (10 seeds)")
ax[0].plot(mean_of(sp, "score"), color="C3", lw=1, label=r"lr $\times$100 at epoch 90 (10 seeds)")
ax[0].set_ylabel("aggregate score\n(lower = better)"); ax[0].set_title("(a) Fidelity score")
ax[1].plot(mean_of(ctrl, "tstr_test_rep_mean"), color="0.35", lw=1, label="no spike")
ax[1].plot(mean_of(sp, "tstr_test_rep_mean"), color="C3", lw=1, label=r"$\times$100, no rule")
ax[1].plot(mean_of(rb, "tstr_test_rep_mean"), color="C0", lw=1, label=r"$\times$100, roll back + reset lr")
ax[1].plot(mean_of(kl, "tstr_test_rep_mean"), color="C1", lw=1, ls="--", label=r"$\times$100, roll back, lr kept (5 seeds)")
ax[1].set_ylabel("test TSTR\n(mean of 10 samples)"); ax[1].set_xlabel("epoch"); ax[1].set_title("(b) Downstream utility")
ax[1].set_ylim(0.52, 0.82); ax[1].legend(frameon=False, loc="lower left")
ax[0].legend(frameon=False, loc="upper left")
for a in ax:
    a.axvline(90, color="k", lw=0.6, ls=":"); a.grid(alpha=0.25, lw=0.4)
fig.tight_layout(); fig.savefig(f"{OUT}/spike_rollback.pdf"); fig.savefig(f"{OUT}/spike_rollback.png", dpi=200)
