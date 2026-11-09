"""In-training fidelity monitoring for CTGAN.

CTGAN's `fit()` has no per-epoch callback. We add one by swapping the progress
iterator (`tqdm`) used inside `ctgan.synthesizers.ctgan` for an iterator that
calls our hook after every epoch. The hook can sample from the generator as it
trains, log fidelity metrics, keep the best checkpoint, and stop training
early by ending the iteration.

Verified against ctgan 0.11.x. If a future ctgan version changes how the
training loop iterates over epochs, `MonitoredCTGAN.fit` raises instead of
silently training without monitoring.
"""

from __future__ import annotations

import copy
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import ctgan.synthesizers.ctgan as _ctgan_module
from ctgan import CTGAN

from validator import FidelityReport, aggregate


# ---------------------------------------------------------------------------
# Change detectors
# ---------------------------------------------------------------------------

class NoDetector:
    """Baseline: never stop early; just keep the best checkpoint."""
    name = "none"

    def update(self, score: float) -> bool:
        return False


class PatienceDetector:
    """Baseline: stop if the best score has not improved for `patience` checks."""
    name = "patience"

    def __init__(self, patience: int = 10, warmup: int = 5):
        self.patience, self.warmup = patience, warmup
        self.best, self.since_best, self.t = np.inf, 0, 0

    def update(self, score: float) -> bool:
        self.t += 1
        if score < self.best:
            self.best, self.since_best = score, 0
        else:
            self.since_best += 1
        return self.t > self.warmup and self.since_best >= self.patience


class CusumDetector:
    """One-sided CUSUM (Page, 1954) on degradation relative to the best score.

        g_t = max(0, g_{t-1} + (s_t - s_best) - drift)
        alarm when g_t > threshold

    `drift` absorbs normal epoch-to-epoch noise; `threshold` sets how much
    accumulated degradation triggers an alarm. Training metrics trend downward
    while the model learns, so measuring against the running best (rather than
    a fixed pre-change mean) is one simple way to handle non-stationarity.
    Choosing drift/threshold in a principled way is an open question.
    """
    name = "cusum"

    def __init__(self, drift: float = 0.01, threshold: float = 0.15, warmup: int = 5):
        self.drift, self.threshold, self.warmup = drift, threshold, warmup
        self.best, self.g, self.t = np.inf, 0.0, 0

    def update(self, score: float) -> bool:
        self.t += 1
        self.best = min(self.best, score)
        self.g = max(0.0, self.g + (score - self.best) - self.drift)
        return self.t > self.warmup and self.g > self.threshold


class RollbackRule:
    """Roll back and continue: a change detector that only uses the past.

    Self-starting one-sided CUSUM on one logged series (higher = worse). At
    each check from epoch `start` on, the reference mean and sd come from all
    earlier checks with epoch >= `burn_from` that are still "valid"; the
    current check is never part of its own reference.

        g_t = max(0, g_{t-1} + (x_t - mean) / sd - k),   alarm when g_t > h

    On an alarm, `update` returns the epoch to restore: the last check at
    which g was 0 (the estimated onset; the check before the alarm if g was
    never 0). The checks after that epoch are dropped from the reference,
    because the training that produced them is being undone, and g restarts
    at 0. At most `max_rollbacks` alarms act; later ones are ignored.

    `reset_lr`: after restoring, also set the learning rates back to their
    configured values (True), or keep whatever they are now (False).
    """

    def __init__(self, series: str = "mmd2", h: float = 5.0, k: float = 0.5,
                 burn_from: int = 30, start: int = 48, max_rollbacks: int = 3,
                 reset_lr: bool = True):
        self.series, self.h, self.k = series, h, k
        self.burn_from, self.start = burn_from, start
        self.max_rollbacks, self.reset_lr = max_rollbacks, reset_lr
        self.epochs: list[int] = []
        self.values: list[float] = []
        self.g, self.last_zero, self.n_rollbacks = 0.0, None, 0
        self.g_before_reset = 0.0

    def update(self, epoch: int, value: float):
        """Returns the epoch to restore, or None."""
        ref = [v for e, v in zip(self.epochs, self.values) if e >= self.burn_from]
        self.epochs.append(epoch)
        self.values.append(value)
        self.g_before_reset = self.g
        if epoch < self.start or len(ref) < 4:
            return None
        mu, sd = float(np.mean(ref)), float(np.std(ref, ddof=1))
        if not sd > 0:
            return None
        self.g = max(0.0, self.g + (value - mu) / sd - self.k)
        self.g_before_reset = self.g
        if self.g == 0.0:
            self.last_zero = epoch
        if self.g > self.h and self.n_rollbacks < self.max_rollbacks:
            onset = self.last_zero if self.last_zero is not None else self.epochs[-2]
            keep = [i for i, e in enumerate(self.epochs) if e <= onset]
            self.epochs = [self.epochs[i] for i in keep]
            self.values = [self.values[i] for i in keep]
            self.g, self.last_zero = 0.0, onset
            self.n_rollbacks += 1
            return onset
        return None


def make_detector(name: str, **kw):
    return {"none": NoDetector, "patience": PatienceDetector,
            "cusum": CusumDetector}[name](**kw)


# ---------------------------------------------------------------------------
# Monitored training
# ---------------------------------------------------------------------------

class MonitoredCTGAN:
    def __init__(self, real_eval: pd.DataFrame, numeric_cols, categorical_cols,
                 check_every: int = 1, n_eval: int = 2000, detector=None,
                 seed: int = 0, verbose: bool = True, extra_metrics=None,
                 lr_spike: tuple[int, float] | None = None,
                 rollback: dict | None = None,
                 **ctgan_kwargs):
        self.real_eval = real_eval
        self.categorical_cols = list(categorical_cols)
        self.report = FidelityReport(numeric_cols, categorical_cols, seed=seed)
        self.check_every = check_every
        self.n_eval = n_eval
        self.detector = detector or NoDetector()
        self.verbose = verbose
        # Optional callable(synthetic_df) -> dict of extra values to log at each
        # check (e.g. downstream utility). Logged only; never used for decisions.
        self.extra_metrics = extra_metrics
        # Optional planted failure: (epoch, factor). After that epoch finishes,
        # both optimizers' learning rates are multiplied by `factor`, giving a
        # break point whose location is known exactly.
        self.lr_spike = lr_spike
        self._optimizers = []
        # Optional roll-back-and-continue rule (see RollbackRule). A dict with
        # keys: series, h, k, burn_from, start, max_rollbacks, reset_lr.
        self.rollback = RollbackRule(**rollback) if rollback else None
        self._snapshots: dict[int, dict] = {}
        self._base_lrs: list[list[float]] | None = None
        self.rollbacks: list[dict] = []
        self.model = CTGAN(**ctgan_kwargs)

        self.history: list[dict] = []
        self.best_score = np.inf
        self.best_epoch = None
        self.best_state = None
        self.stopped_epoch = None
        self._hook_calls = 0

    # -- the per-epoch hook ------------------------------------------------
    def _on_epoch_end(self, epoch: int) -> bool:
        """Called after each epoch. Returns True to stop training."""
        self._hook_calls += 1
        if self.lr_spike and epoch == self.lr_spike[0]:
            if len(self._optimizers) != 2:
                raise RuntimeError("could not find CTGAN's two optimizers for the lr spike")
            for opt in self._optimizers:
                for g in opt.param_groups:
                    g["lr"] *= self.lr_spike[1]
            if self.verbose:
                print(f"--- epoch {epoch}: learning rate multiplied by {self.lr_spike[1]:g} ---", flush=True)
        if epoch % self.check_every != 0:
            return False

        t0 = time.time()
        with torch.no_grad():
            synth = self.model.sample(self.n_eval)
        summary = self.report.summary(self.real_eval, synth)
        score = aggregate(summary)

        losses = self.model.loss_values
        g_loss = float(losses["Generator Loss"].iloc[-1]) if len(losses) else np.nan
        d_loss = float(losses["Discriminator Loss"].iloc[-1]) if len(losses) else np.nan

        if score < self.best_score:
            self.best_score, self.best_epoch = score, epoch
            self.best_state = copy.deepcopy(self.model._generator.state_dict())

        extra = self.extra_metrics(synth) if self.extra_metrics else {}

        stop = self.detector.update(score)
        row = {"epoch": epoch, **summary, "score": score,
               "g_loss": g_loss, "d_loss": d_loss, **extra, "alarm": stop}
        if self.rollback:
            row.update(self._rollback_step(epoch, row))
        row["eval_seconds"] = time.time() - t0
        self.history.append(row)
        if self.verbose:
            print(f"epoch {epoch:4d} | score {score:.4f} | KS {summary['ks_mean']:.3f} "
                  f"| C2ST {summary['c2st']:.3f} | pile-up {summary['excess_point_mass_max']:.3f}"
                  + (f" | TSTR {extra['tstr_val']:.3f}" if 'tstr_val' in extra else "")
                  + ("  <-- ALARM, stopping" if stop else ""), flush=True)
        if stop:
            self.stopped_epoch = epoch
        return stop

    # -- roll back and continue ---------------------------------------------
    def _all_params(self, opt):
        return [p for g in opt.param_groups for p in g["params"]]

    def _snapshot(self) -> dict:
        """Everything needed to resume training from this check: generator
        (weights + batch-norm statistics), every parameter the two optimizers
        own (this is how the discriminator, a local variable inside CTGAN's
        fit(), is reached), and both Adam states."""
        return {"gen": copy.deepcopy(self.model._generator.state_dict()),
                "params": [[p.detach().clone() for p in self._all_params(o)] for o in self._optimizers],
                "opt": [copy.deepcopy(o.state_dict()) for o in self._optimizers]}

    def _restore(self, snap: dict, reset_lr: bool):
        current = [[g["lr"] for g in o.param_groups] for o in self._optimizers]
        self.model._generator.load_state_dict(snap["gen"])
        for i, o in enumerate(self._optimizers):
            with torch.no_grad():
                for p, saved in zip(self._all_params(o), snap["params"][i]):
                    p.copy_(saved)
            o.load_state_dict(copy.deepcopy(snap["opt"][i]))
            for g, lr0, lr_now in zip(o.param_groups, self._base_lrs[i], current[i]):
                g["lr"] = lr0 if reset_lr else lr_now

    def _rollback_step(self, epoch: int, row: dict) -> dict:
        if len(self._optimizers) != 2:
            raise RuntimeError("could not find CTGAN's two optimizers for the rollback rule")
        if self._base_lrs is None:
            raise RuntimeError("base learning rates were not recorded")
        rule = self.rollback
        self._snapshots[epoch] = self._snapshot()
        onset = rule.update(epoch, float(row[rule.series]))
        out = {"cusum_g": rule.g_before_reset, "rollback_to": np.nan}
        if onset is not None:
            self._restore(self._snapshots[onset], rule.reset_lr)
            for e in [e for e in self._snapshots if e > onset]:
                del self._snapshots[e]
            self.rollbacks.append({"alarm_epoch": epoch, "restored_epoch": onset})
            out["rollback_to"] = onset
            if self.verbose:
                print(f"--- epoch {epoch}: ALARM on {rule.series}; restored the epoch-{onset} "
                      f"checkpoint and continuing"
                      + (" at the configured learning rate" if rule.reset_lr else "") + " ---", flush=True)
        return out

    @contextmanager
    def _patched_epoch_iterator(self):
        original = _ctgan_module.tqdm
        hook = self._on_epoch_end

        class _HookedIterator:
            def __init__(self, iterable, *args, **kwargs):
                self._n = len(iterable)

            def __iter__(self):
                for i in range(self._n):
                    yield i
                    if hook(i + 1):
                        return  # ends CTGAN's `for i in epoch_iterator` loop

            def set_description(self, *a, **k):
                pass

        # Capture the optimizers CTGAN creates inside fit() so the planted
        # learning-rate spike can reach them.
        original_optim = _ctgan_module.optim
        registry = self._optimizers
        monitor = self

        class _RecordingAdam(original_optim.Adam):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                registry.append(self)
                monitor._base_lrs = [[g["lr"] for g in o.param_groups] for o in registry]

        class _OptimShim:
            Adam = _RecordingAdam

            def __getattr__(self, name):
                return getattr(original_optim, name)

        _ctgan_module.tqdm = _HookedIterator
        _ctgan_module.optim = _OptimShim()
        try:
            yield
        finally:
            _ctgan_module.tqdm = original
            _ctgan_module.optim = original_optim

    def fit(self, train: pd.DataFrame, epochs: int):
        with self._patched_epoch_iterator():
            self.model.fit(train, discrete_columns=self.categorical_cols, epochs=epochs)
        if self._hook_calls == 0:
            raise RuntimeError(
                "Per-epoch hook never ran: this ctgan version's training loop "
                "differs from the one monitor.py was written for. Pin the "
                "ctgan version in requirements.txt.")
        return self

    def restore_best(self):
        """Load the generator weights from the best-scoring checkpoint."""
        if self.best_state is not None:
            self.model._generator.load_state_dict(self.best_state)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        with torch.no_grad():
            return self.model.sample(n)

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
