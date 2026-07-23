"""Expresses the DAX-direction -> SPY/QQQ first-hour signal via same-day
(0DTE-style) options instead of shares -- Joey's follow-up question after
the equity backtest checked out. SPY and QQQ both have real daily
expirations in practice, so a same-day call/put bought at the open and
sold an hour later is a realistic structure, not a contrived one.

Important new caveat, on top of every other options caveat in this
project: 0DTE option pricing via Black-Scholes-off-realized-vol is even
LESS reliable than the daily-DTE studies elsewhere in this repo. Real
0DTE dynamics (gamma exposure, dealer hedging flows, pinning toward
round strikes near expiry) are not well captured by a Black-Scholes
model at all -- BS assumes continuous smooth price evolution and
constant vol, exactly the assumptions 0DTE options violate hardest.
Treat this as an even-more-generous-than-usual upper bound on what a
real 0DTE overlay could deliver, not a real backtest of one.

Prices with `years_to_expiry` as a FRACTION OF A DAY (not the integer-DTE
convention used elsewhere in this project) -- same-day options expire a
few hours after entry, not weeks/months out. Realized vol is computed from
TRAILING HOURLY bars (not daily closes), annualized via sqrt(252*6.5)
(trading hours/year) instead of sqrt(252) -- the appropriate scaling for
an hourly sampling frequency.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btc_trend_bot.options_pricing import bs_call_price, bs_put_price, MIN_TRADABLE_PREMIUM  # noqa: E402
from run_european_lead_us_first_hour_study import download_hourly, european_pre_us_open_return  # noqa: E402
from run_european_lead_us_first_hour_backtest import random_control  # noqa: E402
from btc_trend_bot.portfolio_sim import SizingMode, simulate_single_position  # noqa: E402

HOURS_TO_CLOSE_AT_ENTRY = 6.5  # 9:30am -> 4:00pm ET
HOURS_HELD = 1.0
TRADING_HOURS_PER_YEAR = 252 * 6.5
US_OPEN_UTC = pd.Timestamp("13:30:00").time()
PREMIUM_FRACTION = 0.10  # same 10%-of-capital convention used throughout this project


def hourly_realized_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    log_ret = np.log(df["close"] / df["close"].shift(1))
    return log_ret.rolling(window).std() * np.sqrt(TRADING_HOURS_PER_YEAR)


def build_zero_dte_trades(symbol: str, dax_signal: pd.Series, spread: float) -> pd.DataFrame:
    df = download_hourly(symbol).copy()
    df["date"] = df.index.date
    df["time"] = df.index.time
    df["vol"] = hourly_realized_vol(df)

    rows = []
    for date, day in df.groupby("date"):
        day = day.sort_index()
        open_bars = day[day["time"] == US_OPEN_UTC]
        if open_bars.empty or date not in dax_signal.index:
            continue
        entry_row = open_bars.iloc[0]
        vol = float(entry_row["vol"])
        if not np.isfinite(vol) or vol <= 0:
            continue
        entry_spot = float(entry_row["open"])
        exit_spot = float(entry_row["close"])
        direction = float(np.sign(dax_signal.loc[date]))
        if direction == 0:
            continue

        dte_entry_years = HOURS_TO_CLOSE_AT_ENTRY / (24 * 365)
        dte_exit_years = (HOURS_TO_CLOSE_AT_ENTRY - HOURS_HELD) / (24 * 365)
        strike = entry_spot  # ATM, the realistic 0DTE liquidity concentration point
        price_fn = bs_call_price if direction > 0 else bs_put_price

        entry_premium = price_fn(entry_spot, strike, dte_entry_years, vol)
        entry_fill = entry_premium * (1 + spread)
        exit_premium = price_fn(exit_spot, strike, max(dte_exit_years, 1e-6), vol)
        exit_fill = exit_premium * (1 - spread)

        if entry_fill < MIN_TRADABLE_PREMIUM:
            continue
        rows.append({
            "signal_time": pd.Timestamp(date), "symbol": symbol,
            "entry_time": pd.Timestamp(date).tz_localize("UTC") + pd.Timedelta(hours=13, minutes=30),
            "exit_time": pd.Timestamp(date).tz_localize("UTC") + pd.Timedelta(hours=14, minutes=30),
            "direction": direction, "entry_premium": entry_fill, "exit_premium": exit_fill,
            "net_return": exit_fill / entry_fill - 1,
            "stock_return": direction * (exit_spot / entry_spot - 1),
        })
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {"label": label, "trades_taken": 0}

    class Cfg:
        initial_capital = 2500.0
        minimum_equity = 0.0
        hard_shutdown_drawdown = 1.0
        safety_enabled = False
        drawdown_pause = 1.0
        cooldown_trades = 0
        consecutive_loss_limit = 10_000

    sizing = SizingMode("fixed_premium", "fixed_notional", PREMIUM_FRACTION * 2500.0)
    _, s = simulate_single_position(trades, Cfg(), sizing)
    s["label"] = label
    return s


def main() -> None:
    dax_signal = european_pre_us_open_return(download_hourly("^GDAXI"))

    print("=== 0DTE-style calls/puts on the DAX-direction signal ===")
    print("(SPY and QQQ both have real daily option expirations -- this isn't a contrived structure,")
    print(" but 0DTE BS-off-realized-vol pricing is the least reliable pricing regime tested in this project)\n")

    all_results = {}
    for symbol in ("SPY", "QQQ"):
        for spread in (0.05, 0.10):
            trades = build_zero_dte_trades(symbol, dax_signal, spread)
            key = f"{symbol}_0dte_spread{spread}"
            s = summarize(trades, key)
            if s.get("trades_taken", 0) == 0:
                print(f"{key:28s} NO TRADES (all below min tradable premium -- see note below)")
                continue
            print(f"{key:28s} trades={s['trades_taken']:4d} win={s.get('win_rate',float('nan')):.3f} "
                  f"mean_bps={s.get('mean_return_per_trade',float('nan'))*10000:8.1f} "
                  f"total_return={s.get('total_return',float('nan'))*100:8.1f}% "
                  f"PF={s.get('profit_factor') or float('nan'):.2f} maxDD={s.get('max_drawdown',float('nan'))*100:.1f}%")
            all_results[key] = s

        n_dropped_check = build_zero_dte_trades(symbol, dax_signal, 0.05)
        print(f"  ({symbol}: {len(n_dropped_check)} of {len(dax_signal.dropna())} signal days produced a tradable 0DTE option)")

    import json
    out = ROOT / "outputs" / "european_signal_options_overlay"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
