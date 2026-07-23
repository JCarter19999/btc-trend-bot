"""Recursive (Kalman-filter) linear regression.

Standard Ridge in the equity walk-forward pipeline is refit from scratch each
fold (see ``run_equity_real_data_walkforward.py``): every `step_bars` the
coefficients are thrown away and re-estimated on a fresh trailing window, so
predictions inside a fold are made from a single static coefficient vector,
and coefficients can jump discontinuously at each fold boundary.

This module implements the alternative: a linear-Gaussian state-space model
where the coefficient vector follows a random walk,

    beta_t = beta_{t-1} + w_t,      w_t ~ N(0, Q)
    y_t    = x_t^T beta_t + v_t,    v_t ~ N(0, r)

and is updated one observation at a time via the standard Kalman filter
recursion. This is algebraically the same thing as recursive least squares
with a forgetting mechanism -- it lets the fitted relationship drift smoothly
as new bars arrive instead of jumping at fixed retrain boundaries. `process_var`
(Q = process_var * I) controls how fast coefficients are allowed to drift;
`process_var = 0` degenerates to ordinary recursive least squares (a fixed
"true" coefficient vector, estimated with ever-increasing confidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KalmanRegressor:
    """Online linear regression via a Kalman filter with random-walk coefficients.

    Parameters
    ----------
    n_features: dimensionality of x_t.
    process_var: diagonal of the process-noise covariance Q added to the
        coefficient covariance at every step. Larger values let the fitted
        relationship drift faster; 0.0 gives ordinary recursive least squares.
    obs_var: observation-noise variance r (residual variance of y_t | x_t).
    prior_var: diagonal of the initial coefficient covariance P0. Larger
        values mean a weaker (less regularizing) prior on beta_0 = 0, acting
        like the inverse of a ridge penalty at the start of the series.
    """

    n_features: int
    process_var: float = 1e-5
    obs_var: float = 1.0
    prior_var: float = 1.0

    beta: np.ndarray = field(init=False)
    covariance: np.ndarray = field(init=False)
    n_updates: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.beta = np.zeros(self.n_features)
        self.covariance = np.eye(self.n_features) * self.prior_var

    def predict_one(self, x: np.ndarray) -> float:
        return float(x @ self.beta)

    def update_one(self, x: np.ndarray, y: float) -> None:
        # Predict step: random-walk coefficient drift.
        p_pred = self.covariance + np.eye(self.n_features) * self.process_var
        # Innovation.
        residual = float(y - x @ self.beta)
        innovation_var = float(x @ p_pred @ x) + self.obs_var
        if innovation_var <= 0 or not np.isfinite(innovation_var):
            return
        gain = (p_pred @ x) / innovation_var
        self.beta = self.beta + gain * residual
        self.covariance = p_pred - np.outer(gain, x @ p_pred)
        self.n_updates += 1


def online_predict_and_update(
    frame,
    features: list[str],
    label_col: str,
    signal_time_col: str,
    ready_time_col: str,
    process_var: float = 1e-5,
    obs_var: float = 1.0,
    prior_var: float = 1.0,
    warmup_updates: int = 200,
) -> "np.ndarray":
    """Causal online pass over `frame`, sorted internally by signal time.

    For each row, predicts `label_col` using only information available
    strictly before that row's `signal_time_col` -- i.e. only rows whose
    `ready_time_col` (the time the true label became knowable, e.g. a trade's
    exit_time) is at or before the current row's `signal_time_col` are folded
    into the filter's state beforehand. This mirrors the purge/embargo logic
    the fold-based Ridge walk-forward uses, but applied continuously instead
    of at fold boundaries.

    Returns an array of predictions aligned to `frame`'s original row order.
    NaN where the filter has not yet seen `warmup_updates` observations
    (matching the `len(train) < 200` cold-start guard used elsewhere in the
    walk-forward runner).
    """
    import pandas as pd

    order = frame[signal_time_col].argsort(kind="stable").to_numpy()
    signal_times = frame[signal_time_col].to_numpy()
    ready_times = frame[ready_time_col].to_numpy()
    X = frame[features].to_numpy(dtype=float)
    y = frame[label_col].to_numpy(dtype=float)

    n = len(frame)
    predictions = np.full(n, np.nan)
    model = KalmanRegressor(
        n_features=X.shape[1], process_var=process_var, obs_var=obs_var, prior_var=prior_var
    )

    # Pending observations, sorted by ready_time, waiting to be folded into
    # the filter once their ready_time has passed relative to the row
    # currently being predicted.
    pending_order = np.argsort(ready_times, kind="stable")
    pending_ptr = 0

    for idx in order:
        current_time = signal_times[idx]
        while pending_ptr < n and ready_times[pending_order[pending_ptr]] <= current_time:
            j = pending_order[pending_ptr]
            if np.all(np.isfinite(X[j])) and np.isfinite(y[j]):
                model.update_one(X[j], y[j])
            pending_ptr += 1
        if model.n_updates >= warmup_updates and np.all(np.isfinite(X[idx])):
            predictions[idx] = model.predict_one(X[idx])

    return predictions
