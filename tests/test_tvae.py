"""Smoke test for the TVAE monitor (run: python tests/test_tvae.py).

Short runs on 2,000 rows. Checks that
1. the per-epoch hook runs and logs one row per epoch, with a loss,
2. the monitor finds TVAE's single optimizer and its learning rate (1e-3),
3. a planted spike multiplies that learning rate from the right epoch on,
4. a spiked run is identical to an unspiked one up to the spike epoch,
5. two runs with the same seed log exactly the same metrics.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import load_adult  # noqa: E402
from monitor import MonitoredTVAE  # noqa: E402
from run_experiment import set_seed  # noqa: E402


def run(spike=None, epochs=10):
    set_seed(0)
    df, num, cat = load_adult(2000, 0)
    train, val = train_test_split(df, test_size=0.4, random_state=0, stratify=df["income"])
    mon = MonitoredTVAE(real_eval=val, numeric_cols=num, categorical_cols=cat,
                        check_every=1, n_eval=500, seed=0, batch_size=500,
                        verbose=False, lr_spike=spike)
    mon.fit(train, epochs=epochs)
    return mon


def main():
    cols = ["score", "ks_mean", "jsd_mean", "corr_dist", "g_loss"]
    a = run()
    h = a.history_frame()
    assert list(h["epoch"]) == list(range(1, 11)), "hook did not run once per epoch"
    assert h["g_loss"].notna().all(), "loss not logged"
    assert len(a._optimizers) == 1, "expected exactly one optimizer"
    assert np.allclose(a._base_lrs, 1e-3), a._base_lrs

    b = run()
    assert np.allclose(h[cols], b.history_frame()[cols]), "same seed gave different metrics"

    s = run(spike=(6, 100.0))
    hs = s.history_frame()
    assert np.isclose(s._optimizers[0].param_groups[0]["lr"], 0.1), "spike did not change the learning rate"
    assert np.allclose(h[cols].iloc[:6], hs[cols].iloc[:6]), "spiked run differs before the spike"
    assert not np.allclose(h[cols].iloc[6:], hs[cols].iloc[6:]), "spike had no effect"
    print("loss after the spike (control vs spiked):",
          h["g_loss"].iloc[6:].round(2).tolist(), hs["g_loss"].iloc[6:].round(2).tolist())
    print("all TVAE checks passed")


if __name__ == "__main__":
    main()
