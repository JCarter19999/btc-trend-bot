from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(rows: int = 4000, seed: int = 42, timeframe: str = "4h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2020-01-01", periods=rows, freq=timeframe, tz="UTC")

    regimes = np.zeros(rows)
    segment = 400
    drifts = [0.0005, -0.00025, 0.0, 0.0008, -0.0005]
    vols = [0.012, 0.018, 0.008, 0.016, 0.024]
    returns = np.zeros(rows)
    for start in range(0, rows, segment):
        regime = (start // segment) % len(drifts)
        stop = min(start + segment, rows)
        returns[start:stop] = rng.normal(drifts[regime], vols[regime], stop - start)
        regimes[start:stop] = regime

    close = 10_000 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.0, 0.006, rows))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(mean=8.0, sigma=0.6, size=rows) * (1 + np.abs(returns) * 10)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "synthetic_regime": regimes,
        }
    )
