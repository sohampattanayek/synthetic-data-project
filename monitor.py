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


def make_detector(name: str, **kw):
    return {"none": NoDetector, "patience": PatienceDetector,
            "cusum": CusumDetector}[name](**kw)


# ---------------------------------------------------------------------------
# Monitored training
# ---------------------------------------------------------------------------

class MonitoredCTGAN:
    def __init__(self, real_eval: pd.DataFrame, numeric_cols, categorical_cols,
                 check_every: int = 1, n_eval: int = 2000, detector=None,
                 seed: int = 0, verbose: bool = True, **ctgan_kwargs):
        self.real_eval = real_eval
        self.categorical_cols = list(categorical_cols)
        self.report = FidelityReport(numeric_cols, categorical_cols, seed=seed)
        self.check_every = check_every
        self.n_eval = n_eval
        self.detector = detector or NoDetector()
        self.verbose = verbose
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

        stop = self.detector.update(score)
        self.history.append({"epoch": epoch, **summary, "score": score,
                             "g_loss": g_loss, "d_loss": d_loss,
                             "alarm": stop, "eval_seconds": time.time() - t0})
        if self.verbose:
            print(f"epoch {epoch:4d} | score {score:.4f} | KS {summary['ks_mean']:.3f} "
                  f"| C2ST {summary['c2st']:.3f} | pile-up {summary['excess_point_mass_max']:.3f}"
                  + ("  <-- ALARM, stopping" if stop else ""), flush=True)
        if stop:
            self.stopped_epoch = epoch
        return stop

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

        _ctgan_module.tqdm = _HookedIterator
        try:
            yield
        finally:
            _ctgan_module.tqdm = original

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
