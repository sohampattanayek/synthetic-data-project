"""Export real and synthetic rows for a Mantis map.

Step 1 (train): rerun one seed's control and spiked run with the usual settings
and save synthetic rows sampled at chosen epochs. The sampled rows were not
kept by the original runs, so they are regenerated here; the script checks that
the rerun's logged fidelity matches the original run exactly.
    python mantis_export.py train --seed 0 --factor 100
Step 2 (probe / full): build the files Mantis gets from the saved samples.
    python mantis_export.py probe
    python mantis_export.py full

Files (results/mantis/):
  samples_{ctrl,spike}_s{seed}.csv   raw sampled rows, column `epoch`
  probe.csv, export.csv              source, epoch, label, impossible, row_text, 15 columns
  probe_v2.csv, export_v2.csv        same, but row_text leaves out capital_gain and capital_loss
  probe_v3.csv, export_v3.csv        same, but row_text also leaves out education_num
  probe_vectors.csv, export_vectors.csv   source, epoch, label, vector (20 numbers)
The vector is: numeric columns standardized with the real training mean/sd,
categoricals one-hot, then PCA to 20 dimensions fitted on the real training rows.
"""
import argparse, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from data import load_adult
from monitor import MonitoredCTGAN
from run_experiment import set_seed

OUT = Path("results/mantis")
EPOCHS = [30, 60, 87, 90, 93, 99, 120, 150]


def splits(seed, max_rows=10000):
    df, num, cat = load_adult(max_rows, seed)
    train, rest = train_test_split(df, test_size=0.4, random_state=seed, stratify=df["income"])
    val, test = train_test_split(rest, test_size=0.5, random_state=seed, stratify=rest["income"])
    return train, val, num, cat


def train_one(seed, factor, n_rows, original):
    set_seed(seed)
    train, val, num, cat = splits(seed)
    saved = []

    def grab(synth):
        ep = mon._hook_calls
        if ep in EPOCHS:
            states = (torch.get_rng_state(), np.random.get_state(), random.getstate())
            try:  # extra sampling must not change the rest of the run
                s = mon.sample(n_rows)
            finally:
                torch.set_rng_state(states[0]); np.random.set_state(states[1]); random.setstate(states[2])
            s.insert(0, "epoch", ep)
            saved.append(s)
        return {}

    mon = MonitoredCTGAN(real_eval=val, numeric_cols=num, categorical_cols=cat, check_every=3,
                         n_eval=2000, seed=seed, batch_size=500, generator_lr=2e-4,
                         discriminator_lr=2e-4, verbose=False,
                         lr_spike=(90, factor) if factor else None, extra_metrics=grab)
    mon.fit(train, epochs=150)
    h = mon.history_frame()
    o = pd.read_csv(Path(original) / "metrics.csv")
    cols = ["score", "ks_mean", "jsd_mean", "corr_dist", "g_loss"]
    same = bool(np.allclose(h[cols].to_numpy(float), o[cols].to_numpy(float), rtol=1e-4, atol=1e-6))
    name = "spike" if factor else "ctrl"
    pd.concat(saved).to_csv(OUT / f"samples_{name}_s{seed}.csv", index=False)
    print(f"{name}: saved {sum(len(s) for s in saved)} rows; matches {original}: {same}")
    return same


def serialize(df, cols):
    return df[cols].astype(str).apply(lambda r: "; ".join(f"{c}: {v}" for c, v in r.items()), axis=1)


def build(seed, epochs, n_per_group, stem):
    train, val, num, cat = splits(seed)
    cols = list(train.columns)
    parts = [val.sample(n_per_group, random_state=seed).assign(source="real", epoch=0)]
    for name in ["ctrl", "spike"]:
        s = pd.read_csv(OUT / f"samples_{name}_s{seed}.csv")
        for ep in epochs:
            parts.append(s[s.epoch == ep].head(n_per_group).assign(source=name))
    d = pd.concat(parts, ignore_index=True)
    d["label"] = d["source"] + "|" + d["epoch"].astype(str)
    # 1 if the row has a value no real Adult row can have (negative capital, age under 17).
    # Lets the map be colored by it, to see whether such rows explain the separation.
    d["impossible"] = ((d.capital_gain < 0) | (d.capital_loss < 0) | (d.age < 17)).astype(int)
    # 1 if education_num is not the number real Adult rows have for that education level
    # (in real data the two columns carry the same information).
    edu_map = pd.concat([train, val]).groupby("education").education_num.agg(lambda v: v.mode().iloc[0])
    d["edu_mismatch"] = (d.education.map(edu_map) != d.education_num).astype(int)
    head = ["source", "epoch", "label", "impossible", "edu_mismatch", "row_text"]
    d["row_text"] = serialize(d, cols)
    d[head + cols].to_csv(OUT / f"{stem}.csv", index=False)
    # v2: the text leaves out capital_gain/capital_loss, which give synthetic rows away
    # (zero in most real rows, never exactly zero in CTGAN rows). The columns themselves stay.
    v2 = d.copy()
    v2["row_text"] = serialize(d, [c for c in cols if c not in ("capital_gain", "capital_loss")])
    v2[head + cols].to_csv(OUT / f"{stem}_v2.csv", index=False)
    # v3: also leaves education_num out of the text (education stays).
    v3 = d.copy()
    v3["row_text"] = serialize(d, [c for c in cols if c not in ("capital_gain", "capital_loss", "education_num")])
    v3[head + cols].to_csv(OUT / f"{stem}_v3.csv", index=False)

    mu, sd = train[num].mean(), train[num].std()
    def features(x):
        z = (x[num].astype(float) - mu) / sd
        oh = pd.concat([pd.get_dummies(pd.Categorical(x[c].astype(str), categories=sorted(train[c].astype(str).unique())),
                                       prefix=c) for c in cat], axis=1)
        return np.hstack([z.to_numpy(), oh.to_numpy(float)])
    pca = PCA(n_components=20, random_state=seed).fit(features(train))
    v = pca.transform(features(d))
    out = d[["source", "epoch", "label", "impossible", "edu_mismatch"]].copy()
    out["vector"] = [json.dumps([round(float(a), 4) for a in row]) for row in v]
    out["coordinate1"], out["coordinate2"] = v[:, 0].round(4), v[:, 1].round(4)  # first 2 PCA components
    out.to_csv(OUT / f"{stem}_vectors.csv", index=False)
    print(d.groupby("label", sort=False).size().to_string())
    print(f"PCA: 20 of {features(train).shape[1]} dimensions, variance kept {pca.explained_variance_ratio_.sum():.3f}")
    print(d.groupby("source", sort=False)[["impossible", "edu_mismatch"]].mean().round(3).to_string())
    print(f"wrote {OUT/stem}.csv, {OUT/stem}_v2.csv, {OUT/stem}_v3.csv ({len(d)} rows each) and {OUT/stem}_vectors.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "probe", "full"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--factor", type=float, default=100.0)
    ap.add_argument("--rows", type=int, default=1000, help="synthetic rows saved per epoch")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "train":
        ok1 = train_one(a.seed, None, a.rows, f"results/mmd/ctrl_s{a.seed}")
        ok2 = train_one(a.seed, a.factor, a.rows, f"results/mmd/spike90x{a.factor:g}_s{a.seed}")
        if not (ok1 and ok2):
            raise SystemExit("rerun does not match the original runs; do not use these samples")
    elif a.step == "probe":
        build(a.seed, [150], 200, "probe")
    else:
        build(a.seed, EPOCHS, 1000, "export")


if __name__ == "__main__":
    main()
