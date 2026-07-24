"""Short strangle, gated to sideways/choppy SPY regimes -- real ThetaData
bid/ask, following this project's established real-data rigor pattern.

Motivation (Joey, 2026-07-24): `simple_trend` is a trend-following strategy
that is known to degrade in chop (its whole edge is "don't cut positions
early, ride the drift" -- see EQUITY_EXIT_REGIME_SIMPLE_TREND.md). This
tests the natural complementary idea for choppy/sideways conditions:
SELL premium instead of buying it. Two pieces of existing evidence in this
project motivate testing the short side specifically, not just repeating
the long-premium structures that already failed:

1. `EQUITY_OPTIONS_REAL_DATA_RETEST.md`'s real-priced long volatility
   breakout straddle was a decisive loser (PF 0.36, -83.5% total return).
   That doc never tested the mirror trade -- selling the same structure.
2. That doc's P&L decomposition on real SPY 0DTE/1DTE trades found mean
   entry IV -> exit IV fell on average (31.1%->28.0% / 21.0%->19.9%) -- a
   real, quantified volatility-crush tailwind that benefits premium
   SELLERS, not buyers.
3. `EDGE_DECOMPOSITION_SECTOR_AND_MARKET_STATE.md` found the DAX/SPY
   directional edge is a CALM-regime edge (works better in low-VIX,
   small-gap, small-range conditions) -- independent evidence that calm/
   sideways SPY conditions in this dataset don't produce large realized
   moves, which is exactly the condition a strangle SELLER wants.

Regime filter: three interchangeable modes, selected with --regime-filter,
sharing every other part of the pipeline so any difference in outcome is
attributable to the filter, not a confound (same "same signal/window, only
X differs" discipline as every other real-data retest in this project):

- `trend_slope` (original/default): |ema_slope_atr| below a threshold --
  the inverse of the concept used elsewhere in this project to detect
  "trending enough to trade" (slope near zero = no trend = chop).
  REJECTED: barely-breakeven (+3.4%, PF 1.04), no advantage over a
  random-date control. Diagnosed cause: slope measures DIRECTION, not
  magnitude of movement -- lets through "flattening before/after a move"
  days, not genuinely tight-range ones.
- `realized_vol`: realized_vol_20 below its own sample median in the
  real-data window -- directly measures the MAGNITUDE of recent
  movement, unlike trend_slope which measures direction. Added per the
  first pass's own "what's next" note. Primary threshold is the SAMPLE
  MEDIAN (50th percentile of realized_vol_20 in the 2021-06-01+ window),
  pre-registered before any pricing, mirroring the median-split
  convention EDGE_DECOMPOSITION_SECTOR_AND_MARKET_STATE.md already uses
  for VIX/gap/range -- not a threshold search.
  REJECTED, more decisively than trend_slope: -41.0% total return, PF
  0.42, sits BELOW all random-date control seeds by up to 83pp. Diagnosed
  cause, confirmed via P&L decomposition: mean IV moved 24.9%->32.7%
  entry-to-exit on this filter's selected days -- IV EXPANDED, the
  opposite of the crush a seller needs. Low-realized-vol days are
  disproportionately vol-COMPRESSION points, and vol mean-reverts, so this
  filter adversely selects into vol about to rise.
- `iv_rv_spread`: the classic volatility-risk-premium signal -- real ATM
  implied vol (solved from a real ThetaData quote via
  `implied_greeks.implied_volatility`, same 30-DTE convention as the
  strangle itself) divided by realized_vol_20 (annualized: *sqrt(252), to
  put it on the same scale as IV before ratioing -- realized_vol_20 itself
  is a raw daily-return std, NOT already annualized). Gate is
  `iv_rv_ratio >= threshold` -- the OPPOSITE direction from the previous
  two ("calm" meant transformed-value-below-threshold; this one means
  "IV rich relative to what's actually been realized," a high-ratio
  condition), because the hypothesis is about being PAID a premium that
  exceeds realized movement, not about the past having been quiet by
  either direction or magnitude. Primary threshold is the sample median
  of iv_rv_ratio across the full real-data-window population (pre-
  registered before any strangle pricing, same median-split convention as
  `realized_vol`). Motivation: the first two filters are both backward-
  looking price-action proxies that never looked at what the options
  market itself is pricing; a strangle seller's real edge, if one exists,
  is IV priced above what actually gets realized, not "the past was
  quiet" under either definition.

Structure: sell 5%-OTM call + 5%-OTM put (independent strikes, same
expiration), 30 DTE at entry, held 10 days (same DTE/hold convention as
every other options structure in this project) -- so every trade is closed
well before expiration, avoiding settlement/early-assignment modeling.

Sizing convention: net_return is on the CREDIT collected (+1.0 = both legs
closed worthless, credit fully kept). This is NOT floored at -1.0 the way
a long option's return is -- a strangle that gets run over on either leg
can lose several multiples of the credit collected, and that must show up
here as net_return << -1.0, not be silently capped. P&L dollars scale off
a SizingMode fixed-notional budget matching the project's existing
"$250-of-$2,500" convention (`PREMIUM_BUDGET` in the sleeve ladder /
straddle retest) -- stated explicitly as a scaling convention, NOT a real
margin/collateral model. Real-world short-strangle margin requirements
(Reg-T or portfolio margin) are not modeled here at all; this only tests
whether the structure has positive expectancy, not whether a $2,500
account could actually hold the position.

Methodology note vs. the tail-hedge trap documented in
EQUITY_OPTIONS_REAL_DATA_RETEST.md section 4: candidate dedup (single-
position, no-overlap) is done on CALENDAR DATES first, using pure
timestamp logic, before any option is priced -- not on the priced
candidate pool. This avoids the "a candidate with no real quote gets
silently dropped before the overlap filter runs, mechanically inflating
trades-taken" artifact that document had to root-cause after the fact.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btc_trend_bot.implied_greeks import decompose_option_pnl  # noqa: E402
from btc_trend_bot.portfolio_sim import SizingMode, simulate_single_position  # noqa: E402
from btc_trend_bot.thetadata_pricing import (  # noqa: E402
    real_atm_iv_on_date, real_short_strangle_trade, reset_client,
)
from run_equity_real_data_walkforward import add_features, load_config, load_csv_dir  # noqa: E402

SYMBOL = "SPY"
DTE_AT_ENTRY = 30
HOLD_DAYS = 10
CALL_MONEYNESS = 1.05
PUT_MONEYNESS = 0.95
REAL_DATA_START = pd.Timestamp("2021-06-01", tz="UTC")
CREDIT_BUDGET = 250.0  # scaling convention only, see module docstring -- NOT a margin model
SIZING = SizingMode("fixed_credit_250_equiv", "fixed_notional", CREDIT_BUDGET)
RANDOM_CONTROL_SEEDS = (0, 1, 2)
TRADING_DAYS_PER_YEAR = 252  # annualizes realized_vol_20 (a raw daily-return std) onto IV's scale

# Regime-filter configs. Each maps a feature column + a transform (identity
# or abs) to a set of thresholds, plus a `direction`: "lt" means "calm" is
# transformed-value-BELOW-threshold (trend_slope, realized_vol); "ge" means
# the gate is transformed-value-AT-OR-ABOVE-threshold (iv_rv_spread -- "IV
# rich relative to realized" is a HIGH-ratio condition, the opposite sense
# from the first two filters' "quiet" framing). trend_slope's thresholds
# are absolute |ema_slope_atr| cutoffs (unchanged from the first pass).
# realized_vol's and iv_rv_spread's thresholds are ACTUAL feature values
# corresponding to pre-registered sample percentiles in the 2021-06-01+
# window (25th/37.5th/50th-primary/62.5th/75th) -- median split is the
# primary threshold for both, mirroring the existing median-split
# convention in EDGE_DECOMPOSITION_SECTOR_AND_MARKET_STATE.md. No threshold
# search: primary is always the pre-registered median: sweep values are an
# explicit robustness check reported alongside, never used to pick primary.
FILTER_CONFIGS = {
    "trend_slope": dict(
        feature="ema_slope_atr", transform=np.abs, direction="lt",
        primary_threshold=0.20, sweep_thresholds=(0.10, 0.15, 0.20, 0.30, 0.40),
        primary_quantile=None, sweep_quantiles=None,
    ),
    "realized_vol": dict(
        feature="realized_vol_20", transform=lambda x: x, direction="lt",
        primary_threshold=None, sweep_thresholds=None,
        primary_quantile=0.50, sweep_quantiles=(0.25, 0.375, 0.50, 0.625, 0.75),
    ),
    "iv_rv_spread": dict(
        feature="iv_rv_ratio", transform=lambda x: x, direction="ge",
        primary_threshold=None, sweep_thresholds=None,
        primary_quantile=0.50, sweep_quantiles=(0.25, 0.375, 0.50, 0.625, 0.75),
    ),
}


def _cfg():
    base = load_config(ROOT / "configs/real_data.yaml")
    return dataclasses.replace(base, symbols=(SYMBOL,), stop_atr=100.0, target_atr=100.0,
                                max_hold_bars=HOLD_DAYS, safety_enabled=False, hard_shutdown_drawdown=1.0,
                                minimum_equity=0.0)


def build_feature_frame() -> pd.DataFrame:
    frames = load_csv_dir((SYMBOL,), ROOT / "data/real")
    spy = frames[SYMBOL]
    # SPY is its own benchmark here -- add_features only needs *a* benchmark
    # frame to compute benchmark_return_5/20/relative_strength_20/market_above_ema50,
    # none of which this study uses; ema_slope_atr/atr_pct (the features we
    # actually need) don't depend on the benchmark argument at all.
    f = add_features(spy, spy)
    return f[f.index >= REAL_DATA_START].copy()


def compute_iv_rv_ratio(f: pd.DataFrame, cache_path: Path) -> pd.Series:
    """Real ATM IV / annualized realized_vol_20 for every row in f, no
    lookahead: IV is fetched as of the SIGNAL date itself (the same date
    whose row this value attaches to), matching how ema_slope_atr and
    realized_vol_20 are themselves point-in-time features in this
    project's feature frame. `real_atm_iv_on_date` deliberately does NOT
    take this frame's `close` as a spot reference -- that column is
    dividend/split-adjusted (this project's standard OHLCV convention),
    while real option strikes are unadjusted nominal levels; feeding the
    adjusted close in as "spot" was tried first and confirmed to badly
    mis-locate the ATM strike (see that function's docstring for the
    caught example). IV is derived entirely from real quotes via put-call
    parity instead. One ThetaData option-chain call per date (~1-3s) --
    checkpointed to `cache_path` as a date->iv CSV so a rerun doesn't
    refetch already-computed dates. realized_vol_20 is a raw daily-return
    std (NOT already annualized) -- multiplied by sqrt(252) here so it's on
    the same scale as IV before ratioing; getting this unit conversion
    wrong would silently produce a meaningless ratio."""
    cache: dict[str, float | None] = {}
    if cache_path.exists():
        cached_df = pd.read_csv(cache_path, index_col=0)
        cache = {k: (None if pd.isna(v) else float(v)) for k, v in cached_df["atm_iv"].items()}
        print(f"Loaded {len(cache)} cached ATM IV values from {cache_path}")

    # Only treat a cached date as "done" if it actually resolved. A None in
    # the cache is retried on every run -- confirmed necessary (see
    # reset_client's docstring): a long sequential run degrades over time
    # and produces long streaks of spurious None that succeed immediately
    # on retry with a fresh client, so caching None as terminal would
    # silently lock in that degradation instead of recovering from it.
    dates = [d for d in f.index if cache.get(d.date().isoformat()) is None]
    print(f"Fetching real ATM IV for {len(dates)} dates needing (re)try "
          f"({len(f) - len(dates)} already resolved from cache)...")
    consecutive_failures = 0
    for i, d in enumerate(dates):
        key = d.date().isoformat()
        result = real_atm_iv_on_date(SYMBOL, d.date(), DTE_AT_ENTRY)
        if result is None:
            # one retry with a forced-fresh client before accepting "no data"
            reset_client()
            result = real_atm_iv_on_date(SYMBOL, d.date(), DTE_AT_ENTRY)
        cache[key] = result
        consecutive_failures = 0 if result is not None else consecutive_failures + 1
        if consecutive_failures >= 10:
            print(f"  !! {consecutive_failures} consecutive failures as of {key} -- forcing client reset", flush=True)
            reset_client()
            consecutive_failures = 0
        # proactive periodic reset regardless of failures -- the degradation
        # observed was gradual, not a hard break, so don't wait for a streak
        if (i + 1) % 100 == 0:
            reset_client()
        if (i + 1) % 50 == 0 or (i + 1) == len(dates):
            pd.Series(cache, name="atm_iv").to_csv(cache_path)
            n_ok = sum(v is not None for v in cache.values())
            print(f"  [{i+1}/{len(dates)}] checkpointed ({n_ok}/{len(cache)} total resolved)", flush=True)

    atm_iv = f.index.map(lambda d: cache.get(d.date().isoformat()))
    rv_annualized = f["realized_vol_20"] * np.sqrt(TRADING_DAYS_PER_YEAR)
    ratio = pd.Series(atm_iv, index=f.index, dtype=float) / rv_annualized
    return ratio


def build_all_candidates(f: pd.DataFrame, feature: str) -> pd.DataFrame:
    rows = []
    for i in range(0, len(f) - HOLD_DAYS - 1):
        r = f.iloc[i]
        val = r.get(feature, np.nan)
        if not np.isfinite(val):
            continue
        entry_i, exit_i = i + 1, i + HOLD_DAYS
        rows.append({
            "signal_time": f.index[i], "symbol": SYMBOL,
            "entry_time": f.index[entry_i], "exit_time": f.index[exit_i],
            "entry_spot": float(f.iloc[entry_i].open),
            "gate_value": float(val),
        })
    return pd.DataFrame(rows)


def dedup_single_position(candidates: pd.DataFrame) -> pd.DataFrame:
    """Pure calendar-date single-position dedup for the REAL (chronological)
    trading sequence: sorted by signal_time, skip any candidate whose
    entry_time is before the previous TAKEN candidate's exit_time. No
    pricing involved -- see module docstring on why this is done before
    fetching any real quote. Valid because all windows here share the same
    fixed HOLD_DAYS duration and entries are monotonic in time once sorted,
    so a single "next_available" marker is equivalent to a full pairwise-
    overlap check -- see dedup_no_overlap_arbitrary_order for the general
    version needed when candidates are NOT processed in chronological order."""
    candidates = candidates.sort_values("signal_time").reset_index(drop=True)
    next_available = None
    keep = []
    for row in candidates.itertuples():
        if next_available is not None and row.entry_time < next_available:
            continue
        keep.append(row.Index)
        next_available = row.exit_time
    return candidates.loc[keep].reset_index(drop=True)


def dedup_no_overlap_arbitrary_order(candidates: pd.DataFrame) -> pd.DataFrame:
    """General non-overlap dedup that respects the GIVEN row order (does
    NOT re-sort by signal_time) -- required for the random-date control.

    Bug this replaces: dedup_single_position re-sorts chronologically
    internally, so shuffling the input before calling it had zero effect --
    every random seed produced the identical chronological greedy sequence.
    This version checks each candidate against every previously-accepted
    interval (not just a single "next available" marker, which only works
    when processing order is guaranteed chronological)."""
    accepted: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    keep = []
    for row in candidates.itertuples():
        overlaps = any(not (row.exit_time < a_entry or row.entry_time > a_exit) for a_entry, a_exit in accepted)
        if not overlaps:
            accepted.append((row.entry_time, row.exit_time))
            keep.append(row.Index)
    return candidates.loc[keep].reset_index(drop=True)


def price_strangles(cands: pd.DataFrame, tag: str) -> pd.DataFrame:
    rows = []
    n_priced, n_skipped = 0, 0
    for i, t in enumerate(cands.itertuples()):
        r = real_short_strangle_trade(SYMBOL, t.signal_time.date(), t.entry_time.date(), t.exit_time.date(),
                                       t.entry_spot, DTE_AT_ENTRY, CALL_MONEYNESS, PUT_MONEYNESS)
        if r is None:
            rows.append({"net_return": np.nan})
            n_skipped += 1
        else:
            rows.append(r)
            n_priced += 1
        if (i + 1) % 25 == 0:
            print(f"  [{tag}] {i+1}/{len(cands)} priced ({n_priced} ok, {n_skipped} skipped)", flush=True)
    priced = pd.DataFrame(rows)
    out = cands.copy().reset_index(drop=True)
    for col in priced.columns:
        out[col] = priced[col]
    print(f"[{tag}] done: {n_priced} priced, {n_skipped} skipped ({len(cands)} candidates)", flush=True)
    return out


def apply_stress(trades: pd.DataFrame, tick: float, bps: float) -> pd.Series:
    """Re-derive net_return from the already-fetched raw call/put entry-bid
    and exit-ask prices under an extra fill-quality haircut -- no new API
    calls, matching the "cost stress" checks elsewhere in this project."""
    valid = trades.dropna(subset=["net_return"]).copy()
    call_entry = (valid["call_entry_bid"] - tick).clip(lower=0.0) * (1 - bps)
    put_entry = (valid["put_entry_bid"] - tick).clip(lower=0.0) * (1 - bps)
    entry_credit = call_entry + put_entry
    call_exit = (valid["call_exit_ask"] + tick) * (1 + bps)
    put_exit = (valid["put_exit_ask"] + tick) * (1 + bps)
    exit_debit = call_exit + put_exit
    net_return = 1.0 - exit_debit / entry_credit.replace(0, np.nan)
    return net_return


def summarize(trades: pd.DataFrame, label: str, cfg) -> dict:
    t = trades.dropna(subset=["net_return"]).copy()
    if t.empty:
        return {"label": label, "trades_taken": 0}
    _, summary = simulate_single_position(t, cfg, SIZING)
    summary["label"] = label
    return summary


def worst_trades(trades: pd.DataFrame, n: int = 5) -> list[dict]:
    t = trades.dropna(subset=["net_return"]).sort_values("net_return").head(n)
    return [
        {
            "signal_time": str(r.signal_time.date()), "entry_time": str(r.entry_time.date()),
            "exit_time": str(r.exit_time.date()), "call_strike": r.call_strike, "put_strike": r.put_strike,
            "entry_spot": r.entry_spot, "entry_credit": r.entry_credit, "exit_debit": r.exit_debit,
            "net_return": r.net_return,
        }
        for r in t.itertuples()
    ]


def pnl_decomposition(trades: pd.DataFrame, spot_by_date: pd.Series) -> dict | None:
    t = trades.dropna(subset=["net_return"]).copy()
    rows = []
    for r in t.itertuples():
        entry_spot = float(spot_by_date.get(pd.Timestamp(r.entry_time), np.nan))
        exit_spot = float(spot_by_date.get(pd.Timestamp(r.exit_time), np.nan))
        if not (np.isfinite(entry_spot) and np.isfinite(exit_spot)):
            continue
        entry_dte_years = max((r.expiration - r.entry_time.date()).days, 1) / 365.0
        exit_dte_years = max((r.expiration - r.exit_time.date()).days, 1) / 365.0
        call = decompose_option_pnl(r.call_entry_price, r.call_exit_price, entry_spot, exit_spot,
                                     r.call_strike, entry_dte_years, exit_dte_years, "call", HOLD_DAYS)
        put = decompose_option_pnl(r.put_entry_price, r.put_exit_price, entry_spot, exit_spot,
                                    r.put_strike, entry_dte_years, exit_dte_years, "put", HOLD_DAYS)
        if call is None or put is None:
            continue
        # decompose_option_pnl is written for the LONG holder (entry=ask paid,
        # exit=bid received); we sold, so our P&L is the negative of the
        # combined long-holder attribution for both legs.
        rows.append({
            "underlying_move_pnl": -(call["underlying_move_pnl"] + put["underlying_move_pnl"]),
            "iv_change_pnl": -(call["iv_change_pnl"] + put["iv_change_pnl"]),
            "theta_pnl": -(call["theta_pnl"] + put["theta_pnl"]),
            "residual_gamma_pnl": -(call["residual_gamma_pnl"] + put["residual_gamma_pnl"]),
            "total_pnl": -(call["total_pnl"] + put["total_pnl"]),
            "entry_iv_call": call["entry_iv"], "exit_iv_call": call["exit_iv"],
            "entry_iv_put": put["entry_iv"], "exit_iv_put": put["exit_iv"],
        })
    if not rows:
        return None
    d = pd.DataFrame(rows)
    abs_total = d["total_pnl"].abs().replace(0, np.nan)
    return {
        "n": len(d),
        "mean_total_pnl": float(d["total_pnl"].mean()),
        "underlying_move_share_of_abs_pnl": float((d["underlying_move_pnl"].abs() / abs_total).mean()),
        "iv_change_share_of_abs_pnl": float((d["iv_change_pnl"].abs() / abs_total).mean()),
        "theta_share_of_abs_pnl": float((d["theta_pnl"].abs() / abs_total).mean()),
        "residual_share_of_abs_pnl": float((d["residual_gamma_pnl"].abs() / abs_total).mean()),
        "mean_entry_iv_call": float(d["entry_iv_call"].mean()), "mean_exit_iv_call": float(d["exit_iv_call"].mean()),
        "mean_entry_iv_put": float(d["entry_iv_put"].mean()), "mean_exit_iv_put": float(d["exit_iv_put"].mean()),
        "losers_mean_underlying_move_pnl": float(d.loc[d.total_pnl < 0, "underlying_move_pnl"].mean())
        if (d.total_pnl < 0).any() else None,
        "losers_mean_theta_pnl": float(d.loc[d.total_pnl < 0, "theta_pnl"].mean())
        if (d.total_pnl < 0).any() else None,
    }


def main(regime_filter: str = "trend_slope") -> None:
    fcfg = FILTER_CONFIGS[regime_filter]
    feature = fcfg["feature"]
    transform = fcfg["transform"]
    direction = fcfg["direction"]

    cfg = _cfg()
    f = build_feature_frame()
    print(f"Regime filter: {regime_filter} (feature={feature}, direction={direction})")
    print(f"SPY feature frame: {len(f)} rows, {f.index.min().date()} to {f.index.max().date()}")

    if feature == "iv_rv_ratio":
        out_dir_early = ROOT / f"outputs/short_strangle_chop_backtest_{regime_filter}"
        out_dir_early.mkdir(parents=True, exist_ok=True)
        cache_path = out_dir_early / "atm_iv_cache.csv"
        print(f"\n=== Computing real ATM IV / annualized realized_vol_20 for every candidate date ===")
        f["iv_rv_ratio"] = compute_iv_rv_ratio(f, cache_path)
        n_resolved = f["iv_rv_ratio"].notna().sum()
        print(f"iv_rv_ratio resolved for {n_resolved}/{len(f)} rows "
              f"({n_resolved / len(f) * 100:.1f}%; NaN rows drop out of build_all_candidates below)")

    raw = f[feature]
    print(f"{feature} distribution (real-data window): "
          f"mean={raw.mean():.4f} std={raw.std():.4f} "
          f"p10={raw.quantile(.1):.4f} p50={raw.quantile(.5):.4f} "
          f"p90={raw.quantile(.9):.4f}")

    all_cands = build_all_candidates(f, feature)
    all_cands["gate_value"] = transform(all_cands["gate_value"])
    print(f"{len(all_cands)} total daily candidates in real-data window")

    # Resolve concrete thresholds: trend_slope uses fixed absolute cutoffs
    # (unchanged from the first pass); realized_vol uses PRE-REGISTERED
    # sample-percentile cutoffs (computed from the transformed candidate
    # pool itself, before any pricing -- not tuned on an outcome).
    if fcfg["sweep_thresholds"] is not None:
        sweep = list(fcfg["sweep_thresholds"])
        primary_threshold = fcfg["primary_threshold"]
    else:
        sweep = [float(all_cands["gate_value"].quantile(q)) for q in fcfg["sweep_quantiles"]]
        primary_threshold = float(all_cands["gate_value"].quantile(fcfg["primary_quantile"]))
        print(f"Pre-registered percentile->value mapping: "
              + ", ".join(f"q{q:.3f}->{v:.5f}" for q, v in zip(fcfg["sweep_quantiles"], sweep)))
        print(f"Primary threshold (median, q={fcfg['primary_quantile']}): {primary_threshold:.5f}")

    op = "<" if direction == "lt" else ">="

    def gate(pool: pd.DataFrame, th: float) -> pd.DataFrame:
        return pool[pool.gate_value < th] if direction == "lt" else pool[pool.gate_value >= th]

    print(f"\n=== Threshold sensitivity (calendar-date dedup only, no pricing yet) ===")
    threshold_counts = {}
    for th in sweep:
        gated = gate(all_cands, th)
        taken = dedup_single_position(gated)
        pct_days = len(gated) / len(all_cands) * 100
        threshold_counts[th] = {"candidate_days": int(len(gated)), "pct_of_days": pct_days,
                                 "trades_after_dedup": int(len(taken))}
        print(f"  {feature} {op} {th:.5f}: {len(gated):5d} candidate days ({pct_days:5.1f}%), "
              f"{len(taken):3d} trades after single-position dedup")

    primary_gated = gate(all_cands, primary_threshold)
    primary_taken = dedup_single_position(primary_gated)
    print(f"\nPrimary threshold {feature} {op} {primary_threshold:.5f}: "
          f"{len(primary_taken)} trades to price (real ThetaData quotes)")

    print("\n=== Pricing chop-gated real strangle trades ===")
    priced = price_strangles(primary_taken, "chop-gated")
    priced.to_csv(ROOT / f"outputs_chop_gated_raw_{regime_filter}.csv.tmp", index=False)  # safety checkpoint

    baseline_summary = summarize(priced, "short_strangle_chop_gated_baseline", cfg)
    print("\nBaseline summary:")
    print(json.dumps(baseline_summary, indent=2, default=str))

    n_priced_valid = priced["net_return"].notna().sum() if "net_return" in priced.columns else 0
    if n_priced_valid == 0:
        print("\n!! 0 of the gated candidates priced -- cannot run cost stress / OOS split / random "
              "control / P&L decomposition on an empty trade set. Stopping here rather than crashing "
              "on a missing column further down (this is a real data-availability finding, not a bug "
              "to paper over).")
        out_dir = ROOT / f"outputs/short_strangle_chop_backtest_{regime_filter}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps({
            "regime_filter": regime_filter, "feature": feature, "primary_threshold": primary_threshold,
            "baseline": baseline_summary, "threshold_sensitivity": threshold_counts,
            "verdict": "0 of gated candidates priced -- no real-data trade set to evaluate",
        }, indent=2, default=str))
        return

    print("\n=== Cost stress (re-derived from already-fetched quotes, no new API calls) ===")
    stress_summaries = {}
    for name, tick, bps in [("plus_1_tick", 0.01, 0.0), ("plus_2_tick", 0.02, 0.0),
                             ("plus_5bp", 0.0, 0.05), ("plus_10bp", 0.0, 0.10)]:
        stressed = priced.copy()
        stressed["net_return"] = apply_stress(priced, tick, bps)
        s = summarize(stressed, f"stress_{name}", cfg)
        stress_summaries[name] = s
        print(f"  {name}: total_return={s.get('total_return', float('nan')):+.3f} "
              f"expectancy_bps={s.get('expectancy_bps', float('nan')):+.1f} "
              f"win_rate={s.get('win_rate', float('nan')):.3f}")

    print("\n=== Out-of-sample split (chop-gated trades, first half vs second half by date) ===")
    priced_valid = priced.dropna(subset=["net_return"]).sort_values("signal_time").reset_index(drop=True)
    mid = len(priced_valid) // 2
    first_half, second_half = priced_valid.iloc[:mid], priced_valid.iloc[mid:]
    first_summary = summarize(first_half, "chop_gated_first_half", cfg)
    second_summary = summarize(second_half, "chop_gated_second_half", cfg)
    print(f"  first half  (n={len(first_half)}, {first_half.signal_time.min().date() if len(first_half) else '-'} "
          f"to {first_half.signal_time.max().date() if len(first_half) else '-'}): "
          f"total_return={first_summary.get('total_return', float('nan')):+.3f}")
    print(f"  second half (n={len(second_half)}, {second_half.signal_time.min().date() if len(second_half) else '-'} "
          f"to {second_half.signal_time.max().date() if len(second_half) else '-'}): "
          f"total_return={second_summary.get('total_return', float('nan')):+.3f}")

    print("\n=== Worst 5 trades (raw net_return, unlimited-downside check) ===")
    worst = worst_trades(priced, 5)
    for w in worst:
        print(f"  {w['signal_time']} entry {w['entry_time']} exit {w['exit_time']}: "
              f"call_strike={w['call_strike']} put_strike={w['put_strike']} entry_spot={w['entry_spot']:.2f} "
              f"credit={w['entry_credit']:.2f} debit={w['exit_debit']:.2f} net_return={w['net_return']:+.2f}")

    print("\n=== Random-date control (single-position dedup on UNGATED random draws, real quotes) ===")
    random_summaries = {}
    n_target = len(primary_taken)
    for seed in RANDOM_CONTROL_SEEDS:
        shuffled = all_cands.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        random_taken = dedup_no_overlap_arbitrary_order(shuffled).head(n_target)
        print(f"  seed {seed}: pricing {len(random_taken)} randomly-drawn non-overlapping dates...")
        rp = price_strangles(random_taken, f"random-seed{seed}")
        rs = summarize(rp, f"random_control_seed{seed}", cfg)
        random_summaries[seed] = rs
        print(f"    total_return={rs.get('total_return', float('nan')):+.3f} "
              f"expectancy_bps={rs.get('expectancy_bps', float('nan')):+.1f} "
              f"win_rate={rs.get('win_rate', float('nan')):.3f} trades={rs.get('trades_taken', 0)}")

    print("\n=== P&L decomposition (theta/IV-crush vs. underlying-move risk) ===")
    spot_by_date = f["close"]
    decomp = pnl_decomposition(priced, spot_by_date)
    if decomp:
        print(json.dumps(decomp, indent=2, default=str))
    else:
        print("  Decomposition unavailable (implied-vol solve failed for all trades)")

    out_dir = ROOT / f"outputs/short_strangle_chop_backtest_{regime_filter}"
    out_dir.mkdir(parents=True, exist_ok=True)
    priced.to_csv(out_dir / "chop_gated_trades.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "regime_filter": regime_filter,
        "feature": feature,
        "primary_threshold": primary_threshold,
        "baseline": baseline_summary,
        "threshold_sensitivity": threshold_counts,
        "cost_stress": stress_summaries,
        "out_of_sample": {"first_half": first_summary, "second_half": second_summary},
        "worst_trades": worst,
        "random_control": random_summaries,
        "pnl_decomposition": decomp,
        "design": {
            "symbol": SYMBOL, "dte_at_entry": DTE_AT_ENTRY, "hold_days": HOLD_DAYS,
            "call_moneyness": CALL_MONEYNESS, "put_moneyness": PUT_MONEYNESS,
            "credit_budget": CREDIT_BUDGET,
        },
    }, indent=2, default=str))
    tmp = ROOT / f"outputs_chop_gated_raw_{regime_filter}.csv.tmp"
    if tmp.exists():
        tmp.unlink()
    print(f"\nWrote {out_dir / 'summary.json'} and {out_dir / 'chop_gated_trades.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime-filter", choices=list(FILTER_CONFIGS), default="trend_slope")
    args = parser.parse_args()
    main(args.regime_filter)
