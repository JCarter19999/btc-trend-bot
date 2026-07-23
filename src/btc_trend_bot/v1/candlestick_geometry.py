from __future__ import annotations

import numpy as np
import pandas as pd

# Named textbook pattern flags were intentionally removed from the model feature
# set. Continuous geometry is more stable, preserves strength, and avoids
# arbitrary hard boundaries that vary between data vendors and markets.
CONTINUOUS_GEOMETRY_COLUMNS = [
    "signed_body_fraction",
    "body_atr",
    "range_atr",
    "upper_wick_body",
    "lower_wick_body",
    "upper_wick_range",
    "lower_wick_range",
    "close_location",
    "open_location",
    "gap_bps",
    "body_change_ratio",
    "range_change_ratio",
    "overlap_fraction",
    "engulfing_strength_signed",
    "doji_score",
    "hammer_score",
    "inverted_hammer_score",
    "marubozu_score",
    "three_bar_body_balance",
    "three_bar_close_progress",
]

# Backward-compatible alias used elsewhere in the repository.
GEOMETRY_COLUMNS = CONTINUOUS_GEOMETRY_COLUMNS


def add_candlestick_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    """Add continuous, scale-normalized candlestick geometry measurements.

    No named pattern flags are emitted. A downstream model can learn smooth
    combinations of wick, body, overlap, and context measurements directly.
    """
    out = frame.copy()
    eps = 1e-12
    o, h, l, c = (out[x].astype(float) for x in ("open", "high", "low", "close"))
    rng = (h - l).clip(lower=eps)
    body_signed = c - o
    body = body_signed.abs()
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    atr = out.get("atr", rng).astype(float).replace(0, np.nan)

    out["signed_body_fraction"] = body_signed / rng
    out["body_atr"] = body / atr
    out["range_atr"] = rng / atr
    out["upper_wick_body"] = upper / (body + eps)
    out["lower_wick_body"] = lower / (body + eps)
    out["upper_wick_range"] = upper / rng
    out["lower_wick_range"] = lower / rng
    out["close_location"] = (c - l) / rng
    out["open_location"] = (o - l) / rng
    out["gap_bps"] = (o / c.shift(1) - 1.0) * 10_000.0
    out["body_change_ratio"] = body / (body.shift(1) + eps)
    out["range_change_ratio"] = rng / (rng.shift(1) + eps)

    previous_high = h.shift(1)
    previous_low = l.shift(1)
    overlap = (pd.concat([h, previous_high], axis=1).min(axis=1)
               - pd.concat([l, previous_low], axis=1).max(axis=1)).clip(lower=0.0)
    union = (pd.concat([h, previous_high], axis=1).max(axis=1)
             - pd.concat([l, previous_low], axis=1).min(axis=1)).clip(lower=eps)
    out["overlap_fraction"] = overlap / union

    previous_body = body.shift(1)
    previous_direction = np.sign((c - o).shift(1))
    current_direction = np.sign(c - o)
    body_engulf = body / (previous_body + eps)
    opposite = current_direction == -previous_direction
    out["engulfing_strength_signed"] = np.where(
        opposite,
        current_direction * body_engulf,
        0.0,
    )

    out["doji_score"] = (1.0 - body / rng).clip(0.0, 1.0)
    out["hammer_score"] = (
        (lower / (body + eps) / 3.0).clip(0.0, 1.0)
        * (1.0 - upper / rng).clip(0.0, 1.0)
        * (1.0 - body / rng).clip(0.0, 1.0)
    )
    out["inverted_hammer_score"] = (
        (upper / (body + eps) / 3.0).clip(0.0, 1.0)
        * (1.0 - lower / rng).clip(0.0, 1.0)
        * (1.0 - body / rng).clip(0.0, 1.0)
    )
    out["marubozu_score"] = (1.0 - (upper + lower) / rng).clip(0.0, 1.0)

    body_sum_3 = body.rolling(3).sum().replace(0, np.nan)
    out["three_bar_body_balance"] = body_signed.rolling(3).sum() / body_sum_3
    out["three_bar_close_progress"] = (c / c.shift(3) - 1.0) / (
        out.get("atr", rng).rolling(3).mean() / c
    ).replace(0, np.nan)
    return out
