"""Market-regime classifier scaffold: TRENDING / DAX_SIGNAL_DAY / CHOPPY_SIDELINED.

First piece of a planned strategy-rotation system (rotate between
independently-validated strategies depending on what regime the market is
in). This module only classifies regimes -- it does not size positions,
allocate capital, or decide what to do when a regime flips mid-trade. Those
are explicitly out of scope until a rotation/dispatch backtest is built on
top of this, once all the strategy arms it would route into (including the
short-strangle chop strategy, backtested separately) have real results.

Two regime signals are wired in, both reusing exact, already-validated
formulas from this project's research rather than re-deriving new ones:

- TRENDING: the live safety layer's own trend gate (see
  EQUITY_REGIME_STRESS_FINAL.md / experiments/run_equity_regime_stress_final.py,
  `trend_ok = return_26 > -0.01 and ema_slope_atr > -0.20`), evaluated
  MARKET-WIDE on the benchmark (SPY) rather than on whichever symbol
  `simple_trend`'s selector would pick that day.

  RESOLVED DESIGN FORK (was open in the first scaffold pass): the initial
  version evaluated this gate on simple_trend's actual daily winner (top
  relative_strength_20 among symbols above their 50-day benchmark trend).
  That inflates the pass rate by construction -- taking the max of 4
  correlated-but-partially-independent names over a lenient gate passes
  93.1% of real trade-days, far above the ~21% whole-calendar
  CHOPPY_SIDELINED rate, leaving too thin a complement (62 of 905 trades)
  for a rotation system to route meaningful capital through. Measured
  directly (see EQUITY_REGIME_CLASSIFIER_SCAFFOLD.md's resolution
  section): the market-wide SPY version (a) still separates simple_trend's
  real per-trade returns in the same direction, slightly BETTER
  (+1.64pp real diff, 100th percentile of a 1,000-seed shuffled null, vs.
  the per-candidate version's +1.20pp / 95th percentile), and (b) yields a
  materially larger, more usable non-trending complement (113 of 905
  trade-days, 12.5%, and 34.5% at the whole-calendar level vs. 21%) for a
  future rotation system to actually route into the chop/strangle arm.
  Kept as `compute_trend_flag_percandidate` below for reference -- it is
  no longer what `classify_regimes` uses for the canonical `trending`
  column, exposed instead as the secondary `trending_percandidate` column.

- DAX_SIGNAL_DAY: the validated DAX pre-US-open top-quartile |move| gate
  (EUROPEAN_LEAD_US_FIRST_HOUR_BACKTEST.txt -- "exactly what's live in the
  shadow deployment" per EDGE_DECOMPOSITION_SECTOR_AND_MARKET_STATE.md),
  AND-conditioned on the three calm-regime state variables that doc found
  the edge is stronger under (prior-day VIX close, overnight gap, DAX
  session range, each <= their established sample medians of 17.22 / 0.44%
  / 1.40%).

  IMPORTANT CAVEAT: that AND-combination (quartile gate + all three calm
  conditions simultaneously) is itself a NEW hypothesis, not something this
  project has backtested with real option prices. The +146.6%/+88.6% real
  ThetaData results in EQUITY_OPTIONS_REAL_DATA_RETEST.md section 3 were
  measured on the raw top-quartile gate (116 days), not this narrower
  calm-conditioned subset. What's validated here is that DAX_SIGNAL_DAY is
  constructed with no lookahead and is a proper subset of those 116 days --
  NOT that this narrower subset's real-money profitability has been
  re-verified. Treat it as "the events where the underlying calls/puts trade
  is available AND additionally described as calmer by three independent
  state variables," not as "a separately proven, more profitable filter."

CHOPPY_SIDELINED is a placeholder: neither TRENDING nor DAX_SIGNAL_DAY. The
short-strangle chop strategy (backtested in a separate, parallel piece of
work) is intended to eventually fill this bucket. No strangle logic lives
in this module.

Schema: TRENDING and DAX_SIGNAL_DAY are independent boolean flags, not a
single mutually-exclusive label -- they describe different things at
different timeframes (TRENDING is a multi-day regime state gating a
10-day-hold rotation strategy across AAPL/MSFT/NVDA/TSLA; DAX_SIGNAL_DAY is
a same-day intraday event trigger for a SPY 0DTE/1DTE options trade), so
they are not mutually exclusive and can co-occur on the same calendar date.
`primary_bucket` is a convenience mutually-exclusive summary column for
reporting, with an explicit 'TRENDING+DAX_SIGNAL_DAY' bucket for the overlap
case rather than silently picking a precedence order.

No lookahead anywhere: every input feature used to classify date t is
computed from bars/quotes that closed at or before t (pct_change, ewm,
rolling are all backward-looking; VIX uses the PRIOR trading day's close;
gap and DAX session range use only pre-13:30-UTC information). Classifying
date t is a separate question from *trading* on date t -- this module does
not introduce same-bar entries; it only labels what regime a given
signal date belongs to, same convention as everywhere else in this project.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

# Trend gate -- exact values from experiments/run_equity_regime_stress_final.py's
# `trend_ok` predicate (SafetyConfig.require_positive_trend), which is itself
# what the live equity_v2_4 deployment's safety layer applies.
TREND_RETURN_26_MIN = -0.01
TREND_EMA_SLOPE_ATR_MIN = -0.20

# DAX signal gate -- exact values from experiments/run_european_lead_us_first_hour_backtest.py
# (quartile gate) and EDGE_DECOMPOSITION_SECTOR_AND_MARKET_STATE.md (calm-regime medians,
# computed on that study's 116 gated trades).
DAX_QUANTILE_GATE = 0.75
VIX_PRIOR_CLOSE_MEDIAN = 17.22
OVERNIGHT_GAP_PCT_MEDIAN = 0.0044
DAX_SESSION_RANGE_PCT_MEDIAN = 0.0140

US_OPEN_UTC = pd.Timestamp("13:30:00").time()


def compute_trend_flag_market_wide(symbol_frames: dict[str, pd.DataFrame], benchmark_symbol: str = "SPY") -> pd.Series:
    """Bool Series indexed by date (tz-naive, midnight-normalized): True where
    the live trend gate (`return_26 > -0.01 and ema_slope_atr > -0.20`)
    passes on the BENCHMARK itself, independent of which stock
    simple_trend's selector would pick that day. This is the canonical
    `trending` definition used by `classify_regimes` -- see the module
    docstring's "RESOLVED DESIGN FORK" note for why this beat the
    per-candidate version empirically."""
    from run_equity_real_data_walkforward import add_features  # noqa: E402

    bench = symbol_frames[benchmark_symbol]
    feats = add_features(bench, bench)
    feats["return_26"] = bench["close"].pct_change(26)
    trending = (feats["return_26"] > TREND_RETURN_26_MIN) & (feats["ema_slope_atr"] > TREND_EMA_SLOPE_ATR_MIN)
    trending.index = pd.DatetimeIndex(trending.index).tz_localize(None).normalize()
    return trending.astype(bool)


def compute_trend_flag_percandidate(symbol_frames: dict[str, pd.DataFrame], benchmark_symbol: str = "SPY") -> pd.Series:
    """Bool Series indexed by date (tz-naive, midnight-normalized): True where
    the live trend gate passes for whichever symbol simple_trend's selector
    would pick that day, mirroring `_select_fold_winners`'s simple_trend
    branch (pool = symbols with market_above_ema50 >= 1, ranked by
    relative_strength_20; falls back to the full pool if none qualify).
    Superseded by `compute_trend_flag_market_wide` as the canonical
    `trending` definition (see module docstring) -- kept for reference and
    exposed as the secondary `trending_percandidate` diagnostic column."""
    from run_equity_real_data_walkforward import add_features  # noqa: E402

    bench = symbol_frames[benchmark_symbol]
    feats: dict[str, pd.DataFrame] = {}
    for sym, frame in symbol_frames.items():
        f = add_features(frame, bench)
        f["return_26"] = frame["close"].pct_change(26)
        feats[sym] = f

    all_dates = sorted(set.union(*(set(f.index) for f in feats.values())))
    trending = pd.Series(False, index=pd.DatetimeIndex(all_dates))

    for d in all_dates:
        rows = [(sym, f.loc[d]) for sym, f in feats.items() if d in f.index]
        if not rows:
            continue
        qualified = [
            (sym, row) for sym, row in rows
            if pd.notna(row.get("market_above_ema50")) and row["market_above_ema50"] >= 1
            and pd.notna(row.get("relative_strength_20"))
        ]
        pool = qualified if qualified else rows
        pool.sort(
            key=lambda sr: sr[1]["relative_strength_20"] if pd.notna(sr[1].get("relative_strength_20")) else -np.inf,
            reverse=True,
        )
        _, winner = pool[0]
        r26, slope = winner.get("return_26"), winner.get("ema_slope_atr")
        if pd.isna(r26) or pd.isna(slope):
            continue
        trending.loc[d] = bool(r26 > TREND_RETURN_26_MIN and slope > TREND_EMA_SLOPE_ATR_MIN)

    trending.index = pd.DatetimeIndex(trending.index).tz_localize(None).normalize()
    return trending


def compute_dax_signal_day_flag() -> tuple[pd.Series, dict]:
    """Bool Series indexed by date (tz-naive, midnight-normalized) + a
    diagnostics dict. Reuses build_dataset() from
    run_european_lead_us_first_hour_backtest.py for the quartile gate and
    download_hourly() from run_european_lead_us_first_hour_study.py for the
    SPY/DAX intraday bars, matching run_market_state_decomposition_study.py's
    construction of the three calm-regime state variables exactly."""
    from run_european_lead_us_first_hour_backtest import build_dataset  # noqa: E402
    from run_european_lead_us_first_hour_study import download_hourly  # noqa: E402
    import yfinance as yf

    df = build_dataset()
    abs_dax = df["eu_pre_open_return"].abs()
    quartile_gate = abs_dax >= abs_dax.quantile(DAX_QUANTILE_GATE)

    vix = yf.download("^VIX", start="2023-08-01", progress=False, auto_adjust=True)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix_prior_close = vix["Close"].shift(1).reindex(df.index, method="ffill")

    spy = download_hourly("SPY")
    spy_daily = spy.groupby(spy.index.date).agg(open=("open", "first"), close=("close", "last"))
    spy_daily.index = pd.to_datetime(spy_daily.index)
    gap = ((spy_daily["open"] - spy_daily["close"].shift(1)).abs() / spy_daily["close"].shift(1)).reindex(df.index)

    dax_hourly = download_hourly("^GDAXI").copy()
    dax_hourly["date"] = dax_hourly.index.date
    dax_hourly["time"] = dax_hourly.index.time
    pre_open = dax_hourly[dax_hourly["time"] < US_OPEN_UTC]
    dax_range = pre_open.groupby("date").apply(
        lambda d: (d["high"].max() - d["low"].min()) / d["open"].iloc[0], include_groups=False
    )
    dax_range.index = pd.to_datetime(dax_range.index)
    dax_range = dax_range.reindex(df.index)

    calm = (
        (vix_prior_close <= VIX_PRIOR_CLOSE_MEDIAN)
        & (gap <= OVERNIGHT_GAP_PCT_MEDIAN)
        & (dax_range <= DAX_SESSION_RANGE_PCT_MEDIAN)
    )
    dax_signal_day = (quartile_gate & calm.fillna(False)).rename("dax_signal_day")
    dax_signal_day.index = pd.DatetimeIndex(dax_signal_day.index).tz_localize(None).normalize()

    diagnostics = {
        "coverage_start": str(df.index.min()),
        "coverage_end": str(df.index.max()),
        "n_days_total": int(len(df)),
        "n_quartile_gate_only": int(quartile_gate.sum()),
        "n_quartile_and_calm": int(dax_signal_day.sum()),
        "vix_median_used": VIX_PRIOR_CLOSE_MEDIAN,
        "gap_median_used": OVERNIGHT_GAP_PCT_MEDIAN,
        "range_median_used": DAX_SESSION_RANGE_PCT_MEDIAN,
    }
    return dax_signal_day, diagnostics


def classify_regimes(symbol_frames: dict[str, pd.DataFrame], benchmark_symbol: str = "SPY") -> tuple[pd.DataFrame, dict]:
    """Returns (DataFrame[date_index, trending, trending_percandidate,
    dax_signal_day, primary_bucket], diagnostics dict). TRENDING has full
    coverage of whatever history symbol_frames spans (e.g. 2018-2026 daily
    data); DAX_SIGNAL_DAY is only computable over the yfinance-hourly-data
    window (~2023-09 onward as of this writing) -- outside that window
    dax_signal_day is False by construction (no data, not "confirmed
    absent"), stated plainly rather than silently treated as a real
    negative.

    `trending` is the market-wide (SPY) definition -- the resolved choice
    for bucketing/rotation (see module docstring). `trending_percandidate`
    is retained purely as a secondary diagnostic column, not used by
    `primary_bucket`."""
    trending = compute_trend_flag_market_wide(symbol_frames, benchmark_symbol)
    trending_percandidate = compute_trend_flag_percandidate(symbol_frames, benchmark_symbol)
    dax_signal_day, dax_diag = compute_dax_signal_day_flag()

    idx = trending.index.union(dax_signal_day.index)
    out = pd.DataFrame(index=idx)
    out["trending"] = trending.reindex(idx).fillna(False).astype(bool)
    out["trending_percandidate"] = trending_percandidate.reindex(idx).fillna(False).astype(bool)
    out["dax_signal_day"] = dax_signal_day.reindex(idx).fillna(False).astype(bool)
    out["dax_coverage"] = out.index.isin(dax_signal_day.index)

    def bucket(row) -> str:
        if row.trending and row.dax_signal_day:
            return "TRENDING+DAX_SIGNAL_DAY"
        if row.trending:
            return "TRENDING"
        if row.dax_signal_day:
            return "DAX_SIGNAL_DAY"
        return "CHOPPY_SIDELINED"

    out["primary_bucket"] = out.apply(bucket, axis=1)
    return out, dax_diag
