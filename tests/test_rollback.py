"""Smoke test for the roll-back-and-continue rule (run: python tests/test_rollback.py).

A short run on 2,000 rows with a x100 learning-rate spike after epoch 12.
Checks that
1. the rule alarms after the spike and restores an earlier checkpoint,
2. right after the restore, the generator's weights equal the saved ones,
3. the learning rates are back at their configured value,
4. training carries on to the last epoch,
5. a run with the rule switched on but no alarm logs exactly the same metrics
   as a run without the rule (the rule itself changes nothing).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import load_adult  # noqa: E402
from monitor import MonitoredCTGAN  # noqa: E402
from run_experiment import set_seed  # noqa: E402


def run(spike, rollback, epochs=24):
    set_seed(0)
    df, num, cat = load_adult(2000, 0)
    train, val = train_test_split(df, test_size=0.4, random_state=0, stratify=df["income"])
    mon = MonitoredCTGAN(real_eval=val, numeric_cols=num, categorical_cols=cat,
                         check_every=1, n_eval=500, seed=0, batch_size=500,
                         verbose=False, lr_spike=spike, rollback=rollback)
    seen = {}
    if rollback:
        original = mon._restore

        def checked_restore(snap, reset_lr):
            original(snap, reset_lr)
            now = mon.model._generator.state_dict()
            seen["weights_match"] = all(torch.equal(now[k], v) for k, v in snap["gen"].items())
            seen["lrs"] = [g["lr"] for o in mon._optimizers for g in o.param_groups]
        mon._restore = checked_restore
    mon.fit(train, epochs=epochs)
    return mon, seen


def main():
    rule = dict(series="score", h=3.0, burn_from=2, start=8, max_rollbacks=2)
    mon, seen = run(spike=(12, 100.0), rollback=rule)
    hist = mon.history_frame()
    print("rollbacks:", mon.rollbacks)
    assert mon.rollbacks, "no rollback happened"
    first = mon.rollbacks[0]
    assert first["alarm_epoch"] > 12 and first["restored_epoch"] <= first["alarm_epoch"]
    assert seen["weights_match"], "generator weights differ from the checkpoint after restore"
    assert np.allclose(seen["lrs"], 2e-4), seen["lrs"]
    assert int(hist["epoch"].iloc[-1]) == 24, "training did not continue to the end"

    quiet = dict(series="score", h=1e9, burn_from=2, start=8)
    a, _ = run(spike=None, rollback=quiet, epochs=12)
    b, _ = run(spike=None, rollback=None, epochs=12)
    cols = [c for c in b.history_frame().columns if c != "eval_seconds"]
    assert a.history_frame()[cols].equals(b.history_frame()[cols]), "the rule changed a run with no alarm"
    print("all rollback checks passed")


if __name__ == "__main__":
    main()
