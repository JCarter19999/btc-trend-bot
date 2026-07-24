"""Regime-rotation backtest: a single account whose active strategy depends
on `regime_classifier.classify_regimes()`'s market-wide TRENDING flag, plus
an always-on DAX_SIGNAL_DAY options overlay sleeve. This is the first actual
dispatch backtest built on top of the classifier scaffold
(EQUITY_REGIME_CLASSIFIER_SCAFFOLD.md) and the two arms that survived real-
data validation:

- TRENDING -> `simple_trend` (EQUITY_EXIT_REGIME_SIMPLE_TREND.md): rotate
  into the strongest-momentum AAPL/MSFT/NVDA/TSLA candidate, hold 10 days,
  $2,500 fixed notional, gated to only fire when the market-wide (SPY)
  trend gate is True on the signal date.
- DAX_SIGNAL_DAY -> SPY 0DTE ATM calls/puts on DAX top-quartile pre-open-move
  days (EQUITY_OPTIONS_REAL_DATA_RETEST.md section 3 /
  run_european_signal_options_real_data_retest.py), $250 fixed-premium
  sleeve, reusing the ALREADY-CACHED real ThetaData trades in
  outputs/european_signal_options_real_data_retest/raw_trades_0dte.parquet
  -- no new API calls.
- CHOPPY_SIDELINED -> cash. The short-strangle chop strategy was built and
  real-data-tested separately (EQUITY_SHORT_STRANGLE_CHOP_STUDY.md) and
  REJECTED (barely-above-breakeven baseline, flips negative under modest
  cost stress, sign-flipping OOS split, no demonstrated edge over an
  unconditional random-date control, uncapped tail risk). Routing
  CHOPPY_SIDELINED to it would inject a rejected strategy into a backtest
  meant to demonstrate validated rotation. This is a deliberate, evidence-
  based choice, not a placeholder -- see the module docstring below for the
  capital-sleeve design and CLAUDE.md's rigor conventions this follows.

IMPORTANT DEFINITION CHOICE: the DAX_SIGNAL_DAY trigger used for actual
dispatch here is the FULL validated 116-day top-quartile-|DAX move| gate,
recomputed directly from build_dataset() -- NOT `classify_regimes()`'s
`dax_signal_day` column, which is a narrower 27-day subset additionally
conditioned on three "calm regime" variables shown (in
EDGE_DECOMPOSITION_SECTOR_AND_MARKET_STATE.md) to correlate with the
underlying STOCK signal's Sharpe, but never independently re-tested against
real option P&L. Using that narrower, unvalidated subset as if it were
proven would silently smuggle an untested hypothesis into a backtest whose
whole point is combining only what's actually been validated with real
quotes.

CAPITAL-SLEEVE DESIGN (a real modeling choice, stated explicitly rather than
assumed): TRENDING and DAX_SIGNAL_DAY do not compete for the same capital or
time window. TRENDING's stock position is a multi-day hold sized at the full
$2,500 fixed notional (matching simple_trend's own validated sizing);
DAX_SIGNAL_DAY's options trade is a same-day, ~1-hour round-trip sized at a
separate $250 (10%) fixed-premium sleeve, matching the DAX study's own
validated sizing and this project's existing live-deployment precedent of
running each arm as an independently-ledgered account
(equity_yfinance_paper_simpletrend.sqlite3 vs. equity_call_paper.sqlite3 in
the live repo) rather than one pool of undivided cash. This backtest reports
the SUM of both sleeves' dollar P&L against one $2,500 base as "the combined
account" for comparison purposes -- it does NOT model shared-capital
competition (e.g., the DAX sleeve being blocked because the TRENDING sleeve's
$2,500 is "tied up" in a stock position). That would be a materially
different, harder assumption nobody has validated, and is flagged here as an
open question for a future pass, not silently decided.

NO-LOOKAHEAD AT THE ROTATION LEVEL: the TRENDING flag for signal date d uses
only data available at d's close (return_26, ema_slope_atr are backward-
looking pct_change/ewm), and simple_trend's own selection already enters on
d+1 (next-bar entry) -- so gating d+1's entry decision on information known
at d's close introduces no new lookahead. The DAX_SIGNAL_DAY gate uses only
pre-13:30-UTC (pre-US-open) DAX data for a trade entered at ~9:30 ET the same
day -- unchanged from the already-validated original study.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btc_trend_bot.portfolio_sim import SizingMode, simulate_single_position  # noqa: E402
from btc_trend_bot.regime_classifier import classify_regimes  # noqa: E402
from run_equity_real_data_walkforward import (  # noqa: E402
    build_candidates, load_config, load_csv_dir, walk_forward,
)
from run_european_lead_us_first_hour_backtest import build_dataset  # noqa: E402

TRENDING_NOTIONAL = SizingMode("trending_fixed_2500", "fixed_notional", 2500.0)
DAX_NOTIONAL = SizingMode("dax_fixed_premium_250", "fixed_notional", 250.0)
DAX_DTE_PRIMARY = 0  # 0DTE chosen as primary (higher real return, +146.6% vs 1DTE's +88.6%); 1DTE reported as reference


class Cfg:
    initial_capital = 2500.0
    minimum_equity = 0.0
    hard_shutdown_drawdown = 1.0
    safety_enabled = False
    drawdown_pause = 1.0
    cooldown_trades = 0
    consecutive_loss_limit = 10_000


def build_trending_arm(frames: dict, cfg, regimes: pd.DataFrame, gated: bool) -> tuple[pd.DataFrame, dict]:
    candidates = build_candidates(frames, cfg.benchmark, cfg)
    _, winners = walk_forward(candidates, cfg, selection="simple_trend")
    winners = winners.copy()
    winners["signal_date"] = pd.to_datetime(winners["signal_time"]).dt.tz_localize(None).dt.normalize()
    winners["trending"] = winners["signal_date"].map(regimes["trending"]).fillna(False)

    trades = winners[winners["trending"]] if gated else winners
    trades = trades.rename(columns={"signal_time": "signal_time"})
    return simulate_single_position(trades, Cfg(), TRENDING_NOTIONAL)


def build_dax_arm(dte: int) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    raw = pd.read_parquet(ROOT / "outputs" / "european_signal_options_real_data_retest" / f"raw_trades_{dte}dte.parquet")
    trades = pd.DataFrame({
        "signal_time": raw.index, "symbol": "SPY",
        "entry_time": raw.index, "exit_time": raw.index,
        "net_return": raw["net_return"].to_numpy(),
    }).dropna(subset=["net_return"])
    path, summary = simulate_single_position(trades, Cfg(), DAX_NOTIONAL)
    return path, summary, trades


def combined_equity_curve(trending_path: pd.DataFrame, dax_path: pd.DataFrame) -> pd.DataFrame:
    """Additive combination of two independent capital sleeves against one
    $2,500 base -- see module docstring's capital-sleeve design note."""
    t = trending_path[trending_path.get("trade_taken", False) == True][["signal_time", "trade_pnl"]].copy() if len(trending_path) else pd.DataFrame(columns=["signal_time", "trade_pnl"])
    d = dax_path[dax_path.get("trade_taken", False) == True][["signal_time", "trade_pnl"]].copy() if len(dax_path) else pd.DataFrame(columns=["signal_time", "trade_pnl"])
    t["signal_time"] = pd.to_datetime(t["signal_time"]).dt.tz_localize(None)
    d["signal_time"] = pd.to_datetime(d["signal_time"]).dt.tz_localize(None)
    all_pnl = pd.concat([t.assign(sleeve="trending"), d.assign(sleeve="dax")], ignore_index=True)
    all_pnl = all_pnl.sort_values("signal_time").reset_index(drop=True)
    all_pnl["equity"] = 2500.0 + all_pnl["trade_pnl"].cumsum()
    return all_pnl


def shuffled_regime_control(frames: dict, cfg, regimes: pd.DataFrame, n_seeds: int = 200) -> dict:
    """Null control: does gating simple_trend to TRENDING days beat gating to
    a random same-size subset of days? Same discipline as every other
    finding in this project -- don't trust a gated result without checking
    it against a randomization null."""
    candidates = build_candidates(frames, cfg.benchmark, cfg)
    _, winners = walk_forward(candidates, cfg, selection="simple_trend")
    winners = winners.copy()
    winners["signal_date"] = pd.to_datetime(winners["signal_time"]).dt.tz_localize(None).dt.normalize()
    winners["trending"] = winners["signal_date"].map(regimes["trending"]).fillna(False)
    n_trending = int(winners["trending"].sum())

    real_trades = winners[winners["trending"]]
    _, real_summary = simulate_single_position(real_trades, Cfg(), TRENDING_NOTIONAL)
    real_stat = real_summary["total_return"]

    null_stats = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        mask = np.zeros(len(winners), dtype=bool)
        mask[rng.choice(len(winners), size=n_trending, replace=False)] = True
        sub = winners[mask]
        _, s = simulate_single_position(sub, Cfg(), TRENDING_NOTIONAL)
        null_stats.append(s["total_return"])
    null_stats = np.array(null_stats)
    pct = float((null_stats < real_stat).mean() * 100)
    return {
        "n_trending_trades": n_trending, "real_total_return": real_stat,
        "null_mean": float(null_stats.mean()), "null_std": float(null_stats.std()),
        "null_percentile": pct, "n_seeds": n_seeds,
    }


def main() -> None:
    cfg = load_config(ROOT / "configs" / "real_data.yaml")
    all_symbols = tuple(dict.fromkeys((*cfg.symbols, cfg.benchmark)))
    frames = load_csv_dir(all_symbols, ROOT / "data" / "real")

    print("Classifying regimes (market-wide TRENDING, resolved definition)...", flush=True)
    regimes, _ = classify_regimes(frames, cfg.benchmark)

    print("\n=== TRENDING arm: gated (only fires when market-wide trend gate is True) ===", flush=True)
    gated_path, gated_summary = build_trending_arm(frames, cfg, regimes, gated=True)
    print(json.dumps(gated_summary, indent=2, default=str))

    print("\n=== TRENDING arm benchmark: UNGATED (plain simple_trend, no rotation) ===", flush=True)
    ungated_path, ungated_summary = build_trending_arm(frames, cfg, regimes, gated=False)
    print(json.dumps(ungated_summary, indent=2, default=str))

    print(f"\n=== DAX_SIGNAL_DAY arm: SPY {DAX_DTE_PRIMARY}DTE ATM (full validated 116-day quartile gate) ===", flush=True)
    dax_path, dax_summary, dax_trades = build_dax_arm(DAX_DTE_PRIMARY)
    print(json.dumps(dax_summary, indent=2, default=str))

    print("\n=== DAX_SIGNAL_DAY arm reference: SPY 1DTE ATM ===", flush=True)
    dax_path_1dte, dax_summary_1dte, _ = build_dax_arm(1)
    print(json.dumps(dax_summary_1dte, indent=2, default=str))

    print("\n=== Shuffled-regime-label null control (200 seeds): TRENDING-gated vs. random-same-size-subset ===", flush=True)
    null_check = shuffled_regime_control(frames, cfg, regimes)
    print(json.dumps(null_check, indent=2, default=str))

    print("\n=== Combined rotation account: TRENDING(gated) + DAX_SIGNAL_DAY(0DTE) sleeves ===", flush=True)
    combined = combined_equity_curve(gated_path, dax_path)
    combined_total_return = float(combined["equity"].iloc[-1] / 2500.0 - 1) if len(combined) else float("nan")
    combined_dd = float((1 - combined["equity"] / combined["equity"].cummax()).max()) if len(combined) else float("nan")
    print(f"Combined trades: {len(combined)} (trending={int((combined.sleeve=='trending').sum()) if len(combined) else 0}, "
          f"dax={int((combined.sleeve=='dax').sum()) if len(combined) else 0})")
    print(f"Combined total_return={combined_total_return*100:.1f}%  max_drawdown={combined_dd*100:.1f}%  "
          f"final_equity=${combined['equity'].iloc[-1]:.2f}" if len(combined) else "no combined trades")

    print("\n=== Reference: UNGATED simple_trend alone vs. GATED (rotation) TRENDING arm alone ===", flush=True)
    print(f"Ungated total_return: {ungated_summary['total_return']*100:.1f}%  "
          f"(trades_taken={ungated_summary['trades_taken']})")
    print(f"Gated (rotation) total_return: {gated_summary['total_return']*100:.1f}%  "
          f"(trades_taken={gated_summary['trades_taken']})")

    print(f"\n=== Day-count accounting (whole calendar, {len(regimes)} classified days) ===", flush=True)
    print(regimes["primary_bucket"].value_counts())

    out_dir = ROOT / "outputs" / "regime_rotation_backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "combined_equity_curve.csv", index=False)
    gated_path.to_csv(out_dir / "trending_gated_trades.csv", index=False)
    ungated_path.to_csv(out_dir / "trending_ungated_trades.csv", index=False)
    dax_path.to_csv(out_dir / "dax_0dte_trades.csv", index=False)

    results = {
        "trending_gated": gated_summary,
        "trending_ungated_benchmark": ungated_summary,
        "dax_signal_day_0dte": dax_summary,
        "dax_signal_day_1dte_reference": dax_summary_1dte,
        "shuffled_regime_null_control": null_check,
        "combined_account": {
            "total_return": combined_total_return, "max_drawdown": combined_dd,
            "final_equity": float(combined["equity"].iloc[-1]) if len(combined) else None,
            "n_trades": int(len(combined)),
        },
        "calendar_bucket_counts": regimes["primary_bucket"].value_counts().to_dict(),
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
