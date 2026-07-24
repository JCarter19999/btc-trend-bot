"""Re-tests the European lead signal (DAX-top-quartile direction) expressed
via SPY options (0DTE and 1DTE ATM, real quotes) instead of shares --
Joey's priority #1 item from his ThetaData task list. Same signal dates
as the validated backtest (EUROPEAN_LEAD_US_FIRST_HOUR_BACKTEST.txt,
top-quartile-by-|DAX move| filter, static cutoff over the full sample --
matching what was actually out-of-sample tested, not the live shadow
deployment's expanding-percentile variant), same entry (~9:30 ET) and
exit (~10:30 ET) timestamps. Real intraday 1-minute quotes, ask-in/bid-out.

Data window: SPY options confirmed available back to the full 2023-2026
hourly-bar backtest window (unlike the 2021-cutoff volatile-universe
equities -- SPY/QQQ are far more liquid and have deep chain history at
this subscription tier).
"""

from __future__ import annotations

import json
import sys
import time as time_module
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btc_trend_bot.portfolio_sim import SizingMode, simulate_single_position  # noqa: E402
from btc_trend_bot.thetadata_intraday_pricing import real_first_hour_option_trade  # noqa: E402
from run_european_lead_us_first_hour_backtest import build_dataset  # noqa: E402
from run_european_lead_us_first_hour_study import download_hourly  # noqa: E402

PREMIUM_SIZING = SizingMode("fixed_premium_250", "fixed_notional", 250.0)


def price_options(df: pd.DataFrame, spy_open: pd.Series, dte_days: int, instrument: str) -> pd.DataFrame:
    rows = []
    n_priced, n_skipped = 0, 0
    t0 = time_module.time()
    for i, (dt, row) in enumerate(df.iterrows()):
        trade_date = dt.date()
        spot = spy_open.get(dt.normalize())
        if spot is None or not np.isfinite(spot):
            rows.append({"net_return": np.nan})
            n_skipped += 1
            continue
        r = real_first_hour_option_trade(instrument, trade_date, int(row["direction"]), dte_days, float(spot))
        if r is None:
            rows.append({"net_return": np.nan})
            n_skipped += 1
        else:
            rows.append(r)
            n_priced += 1
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(df)} priced ({n_priced} ok, {n_skipped} skipped), {time_module.time()-t0:.0f}s elapsed", flush=True)
    priced = pd.DataFrame(rows, index=df.index)
    print(f"{instrument} {dte_days}DTE: {n_priced} priced, {n_skipped} skipped", flush=True)
    return priced


class Cfg:
    initial_capital = 2500.0
    minimum_equity = 0.0
    hard_shutdown_drawdown = 1.0
    safety_enabled = False
    drawdown_pause = 1.0
    cooldown_trades = 0
    consecutive_loss_limit = 10_000


def summarize(net_returns: pd.Series, label: str) -> dict:
    trades = pd.DataFrame({
        "signal_time": net_returns.index, "symbol": "SPY",
        "entry_time": net_returns.index, "exit_time": net_returns.index,
        "net_return": net_returns.to_numpy(),
    }).dropna(subset=["net_return"])
    if trades.empty:
        return {"label": label, "trades_taken": 0}
    _, summary = simulate_single_position(trades, Cfg(), PREMIUM_SIZING)
    summary["label"] = label
    return summary


def rigor_report(priced: pd.DataFrame, label: str) -> dict:
    """Everything Joey specifically asked to see before trusting this:
    payoff ratio / avg winner / avg loser, longest losing streak, return
    concentration (top-3/top-5 share of total P&L), Sharpe, max drawdown,
    CAGR, log growth. Computed on the $250-premium-per-trade dollar P&L
    (not raw % returns) so "concentration" and "CAGR" are in the same
    units as the account actually experiences."""
    t = priced.dropna(subset=["net_return"]).copy()
    if t.empty:
        return {"label": label, "trades_taken": 0}
    t["pnl_dollars"] = t["net_return"] * 250.0
    wins = t.loc[t.pnl_dollars > 0, "pnl_dollars"]
    losses = t.loc[t.pnl_dollars <= 0, "pnl_dollars"]

    equity = 2500.0 + t["pnl_dollars"].cumsum()
    running_max = equity.cummax()
    drawdown = 1 - equity / running_max
    daily_returns = t["net_return"] * (250.0 / 2500.0)  # portfolio-level return per trade (10% allocation)
    sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else None

    years = len(t) / 116 * (462 / 252) if len(t) else None  # rough: 116 trades spans the ~1.83yr top-quartile-eligible window
    total_return = float(equity.iloc[-1] / 2500.0 - 1)
    cagr = float((equity.iloc[-1] / 2500.0) ** (1 / years) - 1) if years and years > 0 else None

    sorted_pnl = t["pnl_dollars"].sort_values(ascending=False)
    top3_share = float(sorted_pnl.head(3).sum() / t["pnl_dollars"].sum()) if t["pnl_dollars"].sum() != 0 else None
    top5_share = float(sorted_pnl.head(5).sum() / t["pnl_dollars"].sum()) if t["pnl_dollars"].sum() != 0 else None

    is_loss = (t["pnl_dollars"] <= 0).to_numpy()
    longest_losing_streak, current = 0, 0
    for loss in is_loss:
        current = current + 1 if loss else 0
        longest_losing_streak = max(longest_losing_streak, current)

    return {
        "label": label, "trades_taken": int(len(t)),
        "win_rate": float((t.pnl_dollars > 0).mean()),
        "avg_winner_dollars": float(wins.mean()) if len(wins) else None,
        "avg_loser_dollars": float(losses.mean()) if len(losses) else None,
        "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else None,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else None,
        "longest_losing_streak": longest_losing_streak,
        "top3_trades_share_of_total_pnl": top3_share,
        "top5_trades_share_of_total_pnl": top5_share,
        "sharpe_annualized": sharpe,
        "max_drawdown": float(drawdown.max()),
        "total_return": total_return,
        "cagr_approx": cagr,
        "expected_log_growth_per_trade": float(np.log1p(daily_returns).mean()),
        "final_equity": float(equity.iloc[-1]),
    }


def main() -> None:
    print("Building DAX-top-quartile signal dataset (same as the validated backtest)...", flush=True)
    df = build_dataset()
    abs_eu = df["eu_pre_open_return"].abs()
    top_quartile = df[abs_eu >= abs_eu.quantile(0.75)].copy()
    if top_quartile.index.tz is None:
        top_quartile.index = top_quartile.index.tz_localize("UTC")
    print(f"{len(df)} total days, {len(top_quartile)} top-quartile-by-|DAX move| days", flush=True)

    spy_hourly = download_hourly("SPY")
    spy_open = spy_hourly[spy_hourly.index.time == pd.Timestamp("13:30:00").time()]["open"]
    spy_open.index = spy_open.index.normalize()

    out = ROOT / "outputs" / "european_signal_options_real_data_retest"
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for dte in (0, 1):
        print(f"\n=== SPY ATM {dte}DTE options on top-quartile DAX signal ===", flush=True)
        priced = price_options(top_quartile, spy_open, dte, "SPY")
        priced.to_parquet(out / f"raw_trades_{dte}dte.parquet")  # save raw trades -- stress tests reuse this, no new API calls

        key = f"spy_atm_{dte}dte"
        results[key] = summarize(priced["net_return"], key)
        rigor = rigor_report(priced, key)
        results[f"{key}_rigor"] = rigor
        print(f"{key}: trades={rigor.get('trades_taken',0)} win={rigor.get('win_rate',float('nan')):.3f} "
              f"PF={rigor.get('profit_factor') or float('nan'):.2f} payoff_ratio={rigor.get('payoff_ratio') or float('nan'):.2f} "
              f"avg_win=${rigor.get('avg_winner_dollars') or float('nan'):.1f} avg_loss=${rigor.get('avg_loser_dollars') or float('nan'):.1f} "
              f"longest_losing_streak={rigor.get('longest_losing_streak')} "
              f"top3_share={rigor.get('top3_trades_share_of_total_pnl') or float('nan'):.1%} "
              f"top5_share={rigor.get('top5_trades_share_of_total_pnl') or float('nan'):.1%} "
              f"sharpe={rigor.get('sharpe_annualized') or float('nan'):.2f} maxDD={rigor.get('max_drawdown',float('nan')):.1%} "
              f"CAGR={rigor.get('cagr_approx') or float('nan'):.1%} total_return={rigor.get('total_return',float('nan')):.1%}", flush=True)

        # Cost stress: +1/+2 tick (SPY 0DTE/1DTE ticks in $0.01 increments,
        # applied on both entry and exit = 2 crossings x tick size), +5bp/+10bp
        # -- all post-hoc on the SAME real quotes, no new API calls needed.
        stress = {}
        for extra_cost_bps, tag in [(0.0, "no_extra_cost"), (None, "plus_1tick"), (None, "plus_2tick"),
                                     (5.0, "plus_5bp"), (10.0, "plus_10bp")]:
            t2 = priced.dropna(subset=["net_return"]).copy()
            if tag == "plus_1tick":
                t2["net_return"] = t2["exit_bid"].sub(0.01).clip(lower=0) / t2["entry_ask"].add(0.01) - 1
            elif tag == "plus_2tick":
                t2["net_return"] = t2["exit_bid"].sub(0.02).clip(lower=0) / t2["entry_ask"].add(0.02) - 1
            else:
                t2["net_return"] = t2["net_return"] - (extra_cost_bps or 0.0) / 10000.0
            stress[tag] = summarize(t2["net_return"], tag)
        results[f"{key}_cost_stress"] = stress
        print("  cost stress:", {k: f"{v.get('total_return',float('nan'))*100:.1f}%" for k, v in stress.items()}, flush=True)

        # Out-of-sample split: first half vs second half of the 116 signal dates
        mid = len(priced) // 2
        first_half = summarize(priced.iloc[:mid]["net_return"], f"{key}_first_half")
        second_half = summarize(priced.iloc[mid:]["net_return"], f"{key}_second_half")
        results[f"{key}_oos_split"] = {"first_half": first_half, "second_half": second_half}
        print(f"  OOS split: first_half total_return={first_half.get('total_return',float('nan'))*100:.1f}% "
              f"(n={first_half.get('trades_taken')})  second_half total_return={second_half.get('total_return',float('nan'))*100:.1f}% "
              f"(n={second_half.get('trades_taken')})", flush=True)

    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")
    print("\nReference: SPY shares (top-quartile arm, 2bp cost) from the validated backtest:")
    print("  win 61.2%, mean 9.57bps/trade, total_return +12.9%, Sharpe 3.76")


if __name__ == "__main__":
    main()
