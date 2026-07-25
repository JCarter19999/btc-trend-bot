"""Aggregate 3-sleeve paper-trading portfolio test, per Joey's proposed
"finalized paper-trading configuration": tech rotation (with crash/drawdown
overlay) 60%, DAX-informed SPY 0DTE/1DTE options 35%, BTC high-vol bot 5%,
$10,000 total base. Compares the blend against each sleeve run ALONE at the
full $10,000, and against SPY buy-and-hold.

Reuses only real, already-validated trade-level data from prior forks -- no
new backtesting, no synthetic returns. All three sleeves use fixed-notional
sizing (position size = a fixed dollar amount per trade, decided at entry,
never re-based on realized gains, no compounding) -- the same convention
used throughout this project. Rescaling a sleeve to a different base means
scaling its per-trade dollar P&L proportionally (pnl_new = net_return *
new_base), not compounding percentage returns sequentially, which would be
the wrong methodology for this sizing convention (confirmed directly: the
crash-overlay CSV's own equity_after column reconciles to 2500 + cumsum(pnl),
NOT sequential (1+r).cumprod() -- verified before trusting anything else).

Sleeve data sources:
- Tech rotation (crash overlay): outputs/crash_overlay_test/crash_overlay_trades.csv
  (166 trades, 2018-05 to 2026-07, $2,500 base -- from EQUITY_CRASH_OVERLAY_TEST.md)
- DAX 0DTE/1DTE options: outputs/european_signal_options_real_data_retest/
  raw_trades_{0,1}dte.parquet (116 trades each, 2023-09 to 2026-07, $2,500
  base, fixed_premium_250 sizing -- from EQUITY_OPTIONS_REAL_DATA_RETEST.md)
- BTC vol-gated (ema_pullback_15m_4h, >=0.75 high-vol regime): re-derived
  from /home/joey/btc-paper-5m's outputs/intraday_matrix_50000/feature_frame.csv
  + outputs/popular_matrix_50000/trade_episodes.csv, using the ORIGINAL
  5-minute-bar/2016-bar/7-day regime construction (the one that was put
  through full rigor -- OOS split, random-date control, cost stress --
  in BTC_VOLATILITY_GATED_DAYTRADE_PRELIM.md sections 6-9), NOT the
  4-hour-bar-recomputed version built for the live paper bot (that version
  was only verified via a single live dry-run, not the same full-backtest
  rigor). 38 trades, 2026-03 to 2026-07.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BTC_REPO = Path("/home/joey/btc-paper-5m")


def load_tech_rotation_trades() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "outputs/crash_overlay_test/crash_overlay_trades.csv",
                      parse_dates=["entry_date", "exit_date"])
    return df[["entry_date", "exit_date", "net_return"]].rename(columns={"exit_date": "date"}).sort_values("date")


def load_dax_options_trades() -> tuple[pd.DataFrame, pd.DataFrame]:
    d0 = pd.read_parquet(ROOT / "outputs/european_signal_options_real_data_retest/raw_trades_0dte.parquet")
    d1 = pd.read_parquet(ROOT / "outputs/european_signal_options_real_data_retest/raw_trades_1dte.parquet")
    for d in (d0, d1):
        d.index = pd.to_datetime(d.index).tz_localize(None)
    d0 = d0.reset_index().rename(columns={"index": "date"})[["date", "net_return"]].sort_values("date")
    d1 = d1.reset_index().rename(columns={"index": "date"})[["date", "net_return"]].sort_values("date")
    return d0, d1


def load_btc_hv_trades() -> pd.DataFrame:
    feat = pd.read_csv(BTC_REPO / "outputs/intraday_matrix_50000/feature_frame.csv", parse_dates=["timestamp"])
    feat = feat.sort_values("timestamp").reset_index(drop=True)
    feat["atr_norm"] = feat["atr_5m"] / feat["close"]
    window = 2016
    feat["vol_pctile"] = feat["atr_norm"].rolling(window, min_periods=window).apply(
        lambda s: (s < s.iloc[-1]).mean(), raw=False)
    feat["vol_pctile"] = feat["vol_pctile"].shift(1)
    feat["high_vol"] = feat["vol_pctile"] >= 0.75

    trades = pd.read_csv(BTC_REPO / "outputs/popular_matrix_50000/trade_episodes.csv",
                          parse_dates=["entry_timestamp", "exit_timestamp"])
    ema = trades[(trades.strategy_id == "ema_pullback_15m_4h") & (trades.is_open == False)].sort_values("entry_timestamp").copy()
    feat_sorted = feat[["timestamp", "high_vol"]].dropna().sort_values("timestamp")
    merged = pd.merge_asof(ema.sort_values("entry_timestamp"), feat_sorted,
                            left_on="entry_timestamp", right_on="timestamp", direction="backward")
    hv = merged[merged.high_vol == True].copy().sort_values("exit_timestamp")
    hv["date"] = pd.to_datetime(hv["exit_timestamp"]).dt.tz_localize(None)
    return hv[["date", "net_portfolio_return_pct"]].rename(columns={"net_portfolio_return_pct": "net_return"})


def build_equity_curve(trades: pd.DataFrame, base_capital: float, per_trade_fraction: float = 1.0) -> pd.DataFrame:
    """Fixed-notional-per-trade equity curve: pnl = net_return * (base_capital
    * per_trade_fraction), equity = base_capital + cumsum(pnl). No compounding,
    matching this project's established sizing convention throughout.

    per_trade_fraction matters: the tech-rotation and BTC sleeves are
    single-position (one trade at a time, full sleeve notional per trade,
    fraction=1.0). The DAX options sleeve uses fixed_premium_250 on a $2,500
    base = 10% allocation per trade (options already carry embedded leverage,
    so risking full notional per trade the way the stock strategies do would
    be a different, much riskier sizing scheme than what was actually
    validated) -- fraction=0.10, preserved when rescaling to any other base.
    Verified this distinction directly: treating the DAX sleeve at
    fraction=1.0 produced a mathematically impossible >100% max drawdown
    (154%) before this fix -- fixed_premium_250 must be interpreted as a
    PERCENTAGE allocation to carry forward, not a fixed dollar amount that
    happens to equal 10% of the original $2,500."""
    t = trades.sort_values("date").copy()
    t["pnl"] = t["net_return"] * base_capital * per_trade_fraction
    t["equity"] = base_capital + t["pnl"].cumsum()
    t["peak"] = t["equity"].cummax()
    t["drawdown"] = 1 - t["equity"] / t["peak"]
    return t


def summarize(equity_curve: pd.DataFrame, base_capital: float, label: str) -> dict:
    if equity_curve.empty:
        return {"label": label, "trades": 0, "total_return_pct": 0.0, "max_drawdown_pct": 0.0,
                "final_equity": base_capital, "start": None, "end": None}
    final_equity = float(equity_curve["equity"].iloc[-1])
    return {
        "label": label,
        "trades": len(equity_curve),
        "total_return_pct": (final_equity / base_capital - 1) * 100,
        "max_drawdown_pct": float(equity_curve["drawdown"].max()) * 100,
        "final_equity": final_equity,
        "start": str(equity_curve["date"].min().date()),
        "end": str(equity_curve["date"].max().date()),
    }


def spy_buy_hold(start: str, end: str, base_capital: float) -> dict:
    spy = pd.read_csv(ROOT / "data/real/SPY.csv", parse_dates=["date"])
    spy["date"] = pd.to_datetime(spy["date"]).dt.tz_localize(None)
    spy = spy.sort_values("date").reset_index(drop=True)
    window = spy[(spy["date"] >= pd.Timestamp(start)) & (spy["date"] <= pd.Timestamp(end))]
    if window.empty:
        return {"label": "SPY buy-and-hold", "total_return_pct": None, "max_drawdown_pct": None}
    start_price, end_price = window["close"].iloc[0], window["close"].iloc[-1]
    ret = end_price / start_price - 1
    window = window.copy()
    window["peak"] = window["close"].cummax()
    window["dd"] = 1 - window["close"] / window["peak"]
    return {
        "label": "SPY buy-and-hold", "total_return_pct": ret * 100, "max_drawdown_pct": float(window["dd"].max()) * 100,
        "final_equity": base_capital * (1 + ret), "start": str(window["date"].min().date()), "end": str(window["date"].max().date()),
    }


def main() -> None:
    tech_trades = load_tech_rotation_trades()
    dax0_trades, dax1_trades = load_dax_options_trades()
    btc_trades = load_btc_hv_trades()

    print("=" * 100)
    print("PART 1: EACH SLEEVE ALONE AT FULL $10,000")
    print("=" * 100)

    tech_alone = build_equity_curve(tech_trades, 10000.0)
    tech_alone_summary = summarize(tech_alone, 10000.0, "Tech rotation (crash overlay) ALONE @ $10,000")

    # DAX alone @ $10,000: split 50/50 between 0DTE/1DTE, same split logic as
    # the blend. DAX_PER_TRADE_FRACTION=0.10 preserves the original
    # fixed_premium_250-on-$2,500-base (10% allocation) convention -- options
    # already carry leverage, so this must NOT be scaled as if 100% of
    # sleeve capital is risked per trade (see build_equity_curve's docstring).
    DAX_PER_TRADE_FRACTION = 0.10
    dax0_alone = build_equity_curve(dax0_trades, 5000.0, DAX_PER_TRADE_FRACTION)
    dax1_alone = build_equity_curve(dax1_trades, 5000.0, DAX_PER_TRADE_FRACTION)
    dax_alone_final = 5000.0 + dax0_alone["pnl"].sum() + 5000.0 + dax1_alone["pnl"].sum()
    dax_alone_total_return = (dax_alone_final / 10000.0 - 1) * 100
    dax_alone_start = min(dax0_trades["date"].min(), dax1_trades["date"].min())
    dax_alone_end = max(dax0_trades["date"].max(), dax1_trades["date"].max())
    # combined drawdown: build a merged daily equity series for max-DD purposes
    dax_combo = pd.concat([
        dax0_alone[["date", "pnl"]].assign(pnl=dax0_alone["pnl"]),
        dax1_alone[["date", "pnl"]].assign(pnl=dax1_alone["pnl"]),
    ]).groupby("date", as_index=False)["pnl"].sum().sort_values("date")
    dax_combo["equity"] = 10000.0 + dax_combo["pnl"].cumsum()
    dax_combo["peak"] = dax_combo["equity"].cummax()
    dax_combo["drawdown"] = 1 - dax_combo["equity"] / dax_combo["peak"]
    dax_alone_summary = {
        "label": "DAX 0DTE/1DTE options ALONE @ $10,000 (50/50 split)",
        "trades": len(dax0_trades) + len(dax1_trades),
        "total_return_pct": dax_alone_total_return,
        "max_drawdown_pct": float(dax_combo["drawdown"].max()) * 100,
        "final_equity": dax_alone_final,
        "start": str(dax_alone_start.date()), "end": str(dax_alone_end.date()),
    }

    btc_alone = build_equity_curve(btc_trades, 10000.0)
    btc_alone_summary = summarize(btc_alone, 10000.0, "BTC vol-gated bot ALONE @ $10,000")

    for s in (tech_alone_summary, dax_alone_summary, btc_alone_summary):
        print(f"\n{s['label']}")
        print(f"  Window: {s['start']} to {s['end']}  Trades: {s['trades']}")
        print(f"  Total return: {s['total_return_pct']:+.1f}%   Max drawdown: {s['max_drawdown_pct']:.1f}%   Final equity: ${s['final_equity']:,.2f}")

    print("\n" + "=" * 100)
    print("PART 2: BLENDED 60/35/5 PORTFOLIO (phased deployment, real start dates)")
    print("=" * 100)

    tech_blend = build_equity_curve(tech_trades, 6000.0)
    dax0_blend = build_equity_curve(dax0_trades, 1750.0, DAX_PER_TRADE_FRACTION)
    dax1_blend = build_equity_curve(dax1_trades, 1750.0, DAX_PER_TRADE_FRACTION)
    btc_blend = build_equity_curve(btc_trades, 500.0)

    all_pnl = pd.concat([
        tech_blend[["date", "pnl"]], dax0_blend[["date", "pnl"]], dax1_blend[["date", "pnl"]], btc_blend[["date", "pnl"]],
    ]).groupby("date", as_index=False)["pnl"].sum().sort_values("date")
    all_pnl["equity"] = 10000.0 + all_pnl["pnl"].cumsum()
    all_pnl["peak"] = all_pnl["equity"].cummax()
    all_pnl["drawdown"] = 1 - all_pnl["equity"] / all_pnl["peak"]

    blend_full_summary = {
        "label": "BLENDED 60/35/5 portfolio (full phased-deployment window)",
        "trades": len(tech_trades) + len(dax0_trades) + len(dax1_trades) + len(btc_trades),
        "total_return_pct": (float(all_pnl["equity"].iloc[-1]) / 10000.0 - 1) * 100,
        "max_drawdown_pct": float(all_pnl["drawdown"].max()) * 100,
        "final_equity": float(all_pnl["equity"].iloc[-1]),
        "start": str(all_pnl["date"].min().date()), "end": str(all_pnl["date"].max().date()),
    }
    print(f"\n{blend_full_summary['label']}")
    print(f"  Window: {blend_full_summary['start']} to {blend_full_summary['end']}  Total trades across sleeves: {blend_full_summary['trades']}")
    print(f"  Total return: {blend_full_summary['total_return_pct']:+.1f}%   Max drawdown: {blend_full_summary['max_drawdown_pct']:.1f}%   Final equity: ${blend_full_summary['final_equity']:,.2f}")
    print("  NOTE: from 2018-04 to 2023-09, only the tech sleeve ($6,000) is deployed;")
    print("  the other $4,000 sits in cash (0% return) since those strategies had no real")
    print("  data/deployment yet. DAX sleeve activates 2023-09; BTC sleeve activates ~2026-03.")

    print("\n--- Secondary check: short fully-overlapping window (all 3 sleeves have real data) ---")
    overlap_start = btc_trades["date"].min()
    overlap_end = min(tech_trades["date"].max(), dax0_trades["date"].max(), dax1_trades["date"].max(), btc_trades["date"].max())
    overlap_pnl = all_pnl[(all_pnl["date"] >= overlap_start) & (all_pnl["date"] <= overlap_end)].copy()
    if not overlap_pnl.empty:
        start_eq = 10000.0 + all_pnl[all_pnl["date"] < overlap_start]["pnl"].sum()
        overlap_pnl["equity"] = start_eq + overlap_pnl["pnl"].cumsum()
        overlap_pnl["peak"] = overlap_pnl["equity"].cummax()
        overlap_pnl["drawdown"] = 1 - overlap_pnl["equity"] / overlap_pnl["peak"]
        overlap_return = (float(overlap_pnl["equity"].iloc[-1]) / start_eq - 1) * 100
        print(f"  Window: {overlap_start.date()} to {overlap_end.date()} ({(overlap_end-overlap_start).days} days)")
        print(f"  Return over this short window: {overlap_return:+.2f}%   Max DD in-window: {overlap_pnl['drawdown'].max()*100:.2f}%")
        print(f"  ** Small sample -- {len(overlap_pnl)} portfolio-affecting trade-days in ~{(overlap_end-overlap_start).days} days. Directional only. **")

    print("\n" + "=" * 100)
    print("PART 3: SPY BUY-AND-HOLD COMPARISON")
    print("=" * 100)
    spy_full = spy_buy_hold(tech_trades["date"].min(), blend_full_summary["end"], 10000.0)
    print(f"\nSPY buy-and-hold, {spy_full['start']} to {spy_full['end']}:")
    print(f"  Total return: {spy_full['total_return_pct']:+.1f}%   Max drawdown: {spy_full['max_drawdown_pct']:.1f}%   Final equity: ${spy_full['final_equity']:,.2f}")

    spy_tech_window = spy_buy_hold(tech_alone_summary["start"], tech_alone_summary["end"], 10000.0)
    spy_dax_window = spy_buy_hold(dax_alone_summary["start"], dax_alone_summary["end"], 10000.0)
    spy_btc_window = spy_buy_hold(btc_alone_summary["start"], btc_alone_summary["end"], 10000.0)

    print("\n" + "=" * 100)
    print("FINAL SIDE-BY-SIDE TABLE")
    print("=" * 100)
    rows = [
        ("Tech rotation ALONE", tech_alone_summary["start"], tech_alone_summary["end"], tech_alone_summary["total_return_pct"], tech_alone_summary["max_drawdown_pct"]),
        ("  (SPY, same window)", spy_tech_window["start"], spy_tech_window["end"], spy_tech_window["total_return_pct"], spy_tech_window["max_drawdown_pct"]),
        ("DAX options ALONE", dax_alone_summary["start"], dax_alone_summary["end"], dax_alone_summary["total_return_pct"], dax_alone_summary["max_drawdown_pct"]),
        ("  (SPY, same window)", spy_dax_window["start"], spy_dax_window["end"], spy_dax_window["total_return_pct"], spy_dax_window["max_drawdown_pct"]),
        ("BTC vol-gated ALONE", btc_alone_summary["start"], btc_alone_summary["end"], btc_alone_summary["total_return_pct"], btc_alone_summary["max_drawdown_pct"]),
        ("  (SPY, same window)", spy_btc_window["start"], spy_btc_window["end"], spy_btc_window["total_return_pct"], spy_btc_window["max_drawdown_pct"]),
        ("BLENDED 60/35/5", blend_full_summary["start"], blend_full_summary["end"], blend_full_summary["total_return_pct"], blend_full_summary["max_drawdown_pct"]),
        ("  (SPY, full window)", spy_full["start"], spy_full["end"], spy_full["total_return_pct"], spy_full["max_drawdown_pct"]),
    ]
    print(f"{'Config':<24} {'Start':<12} {'End':<12} {'Return':>10} {'Max DD':>8}")
    for label, start, end, ret, dd in rows:
        print(f"{label:<24} {start:<12} {end:<12} {ret:>+9.1f}% {dd:>7.1f}%")

    out_dir = ROOT / "outputs" / "aggregate_portfolio_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_pnl.to_csv(out_dir / "blended_portfolio_equity_curve.csv", index=False)
    tech_alone.to_csv(out_dir / "tech_alone_equity_curve.csv", index=False)
    btc_alone.to_csv(out_dir / "btc_alone_equity_curve.csv", index=False)
    dax_combo.to_csv(out_dir / "dax_alone_equity_curve.csv", index=False)
    print(f"\nWrote equity curves to {out_dir}")


if __name__ == "__main__":
    main()
