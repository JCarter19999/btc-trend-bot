"""Regenerate real, exact monthly return series for the two strategies in
this project that are actually validated as profitable with real data:
`simple_trend` (full 2018-2026 real-data backtest) and the European-lead
SPY 0DTE/1DTE options overlay (real ThetaData, 2023-09 onward).

`simple_trend` full-history reconstruction: `walk_forward()`'s standard
harness truncates the start date via its 756-bar training warmup even
though simple_trend needs no training -- bypassed here by calling
`_select_fold_winners` directly against the FULL candidate set (fold=0,
train=test=candidates), then running the result through
`portfolio_sim.simulate_single_position` (NOT `simulate_capital`, which
allows overlapping positions -- see that module's docstring) with
fixed-notional $2,500 sizing and the exit-regime deployment's config
(`config/simple_trend_exit_regime_strategy.yaml` in equity_v2_4,
stop_atr/target_atr=100). Verified to reproduce
EQUITY_EXIT_REGIME_SIMPLE_TREND.md's documented characterization exactly:
188 trades, 58.0% win rate, PF 2.04, 243.3bps expectancy, +457.4% total
return, 22.1% max drawdown.

European-lead options: sizing is fixed_premium_250 (no compounding), so
monthly return = sum(net_return * 250 for that month's trades) / 2500.
Verified both 0DTE and 1DTE monthly series sum to the documented
+146.6%/+88.6% totals exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from run_equity_real_data_walkforward import build_candidates, load_config, load_csv_dir, _select_fold_winners
from btc_trend_bot.portfolio_sim import SizingMode, simulate_single_position

LIVE_SIMPLE_TREND_CONFIG = Path("/home/joey/equity_v2_4/config/simple_trend_exit_regime_strategy.yaml")


def monthly_from_pnl(dates: pd.Series, pnl: pd.Series, initial_capital: float) -> pd.DataFrame:
    df = pd.DataFrame({"date": pd.to_datetime(dates).dt.tz_localize(None) if pd.to_datetime(dates).dt.tz is not None else pd.to_datetime(dates), "pnl": pnl})
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month").agg(trades=("pnl", "count"), pnl=("pnl", "sum")).reset_index()
    full_range = pd.period_range(df.date.min().to_period("M"), df.date.max().to_period("M"), freq="M")
    monthly = monthly.set_index("month").reindex(full_range, fill_value=0).rename_axis("month").reset_index()
    monthly["trades"] = monthly["trades"].astype(int)
    monthly["return_pct"] = monthly["pnl"] / initial_capital * 100
    monthly["cum_equity"] = initial_capital + monthly["pnl"].cumsum()
    return monthly


def simple_trend_monthly(out_dir: Path) -> pd.DataFrame:
    cfg = load_config(LIVE_SIMPLE_TREND_CONFIG)
    all_symbols = tuple(dict.fromkeys((*cfg.symbols, cfg.benchmark)))
    frames = load_csv_dir(all_symbols, ROOT / "data" / "real")
    candidates = build_candidates(frames, cfg.benchmark, cfg)
    rng = np.random.default_rng(0)
    winners = _select_fold_winners(candidates, candidates, cfg, 0, rng, False, "simple_trend")

    sizing = SizingMode(name="fixed_notional_2500", kind="fixed_notional", value=2500.0)
    path, summary = simulate_single_position(winners, cfg, sizing)
    assert summary["trades_taken"] == 188, f"reconstruction mismatch: {summary['trades_taken']} != 188"
    path.to_csv(out_dir / "simple_trend_trades_full_history.csv", index=False)

    taken = path[path.trade_taken]
    monthly = monthly_from_pnl(taken["signal_time"], taken["trade_pnl"], cfg.initial_capital)
    monthly.to_csv(out_dir / "simple_trend_monthly.csv", index=False)
    return monthly


def european_lead_monthly(out_dir: Path) -> dict[str, pd.DataFrame]:
    results = {}
    for tag, fname in [("0dte", "raw_trades_0dte.parquet"), ("1dte", "raw_trades_1dte.parquet")]:
        df = pd.read_parquet(ROOT / "outputs" / "european_signal_options_real_data_retest" / fname)
        df = df.reset_index().rename(columns={"index": "trade_date"})
        pnl = df["net_return"] * 250.0
        monthly = monthly_from_pnl(df["trade_date"], pnl, 2500.0)
        monthly.to_csv(out_dir / f"european_lead_{tag}_monthly.csv", index=False)
        results[tag] = monthly
    return results


if __name__ == "__main__":
    out_dir = ROOT / "outputs" / "monthly_returns_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    st = simple_trend_monthly(out_dir)
    print(f"simple_trend: {len(st)} months, total return {st.pnl.sum()/2500:.4f} (expect 4.5742)")
    eu = european_lead_monthly(out_dir)
    print(f"european_lead 0DTE total return {eu['0dte'].pnl.sum()/2500:.4f} (expect 1.4660)")
    print(f"european_lead 1DTE total return {eu['1dte'].pnl.sum()/2500:.4f} (expect 0.8864)")
