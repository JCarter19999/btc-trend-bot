"""Re-tests the recurring SPY tail hedge (EQUITY_OPTIONS_SLEEVE_LADDER_STUDY.md)
with real ThetaData quotes instead of Black-Scholes-off-realized-vol.
Same structure: 45-DTE, ~10%-OTM SPY put, re-entered every ~21 trading
days, sold (not exercised) at the end of each cycle. Joey's ask: "measure
its true carrying cost, see whether crash protection justifies the drag."

Real data window: SPY confirmed liquid back through this project's full
backtest history in the intraday work; scoped to 2021-06-01+ here for
consistency with every other real-data re-test tonight and because
that's the empirically-confirmed floor for this account's data tier.
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

from btc_trend_bot.options_pricing import realized_vol  # noqa: E402
from btc_trend_bot.portfolio_sim import SizingMode, simulate_single_position  # noqa: E402
from btc_trend_bot.thetadata_pricing import real_put_trade  # noqa: E402
from run_equity_options_real_data_retest import REAL_DATA_START  # noqa: E402
from run_equity_real_data_walkforward import load_config, load_csv_dir  # noqa: E402

HEDGE_DTE = 45
HEDGE_HOLD = 21  # trading days
HEDGE_MONEYNESS = 0.90
MONTHLY_BUDGET_FRACTION = 0.02  # same 2% convention as the synthetic version
BUDGET = MONTHLY_BUDGET_FRACTION * 2500.0


class Cfg:
    initial_capital = 2500.0
    minimum_equity = 0.0
    hard_shutdown_drawdown = 1.0
    safety_enabled = False
    drawdown_pause = 1.0
    cooldown_trades = 0
    consecutive_loss_limit = 10_000


def main() -> None:
    cfg_base = load_config(ROOT / "configs/real_data.yaml")
    frames = load_csv_dir((cfg_base.benchmark,), ROOT / "data/real")
    spy = frames[cfg_base.benchmark].copy()
    spy["realized_vol"] = realized_vol(spy["close"].to_numpy(), 20)
    spy.index = pd.to_datetime(spy.index, utc=True)

    start_idx = spy.index.searchsorted(REAL_DATA_START)
    print(f"SPY data: {len(spy)} bars, starting real-data window at index {start_idx} ({spy.index[start_idx].date()})", flush=True)

    rows = []
    i = max(60, start_idx)
    while i < len(spy) - HEDGE_HOLD - 1:
        signal_date = spy.index[i].date()
        entry_i, exit_i = i + 1, min(i + 1 + HEDGE_HOLD - 1, len(spy) - 1)
        entry_date, exit_date = spy.index[entry_i].date(), spy.index[exit_i].date()
        entry_spot = float(spy.iloc[entry_i].open)

        r = real_put_trade(cfg_base.benchmark, signal_date, entry_date, exit_date, entry_spot, HEDGE_DTE, HEDGE_MONEYNESS)
        rows.append({
            "signal_time": spy.index[i], "symbol": cfg_base.benchmark,
            "entry_time": spy.index[entry_i], "exit_time": spy.index[exit_i],
            "net_return": r["net_return"] if r else np.nan,
            "priced": r is not None,
        })
        i += HEDGE_HOLD

    trades = pd.DataFrame(rows)
    n_priced = int(trades["priced"].sum())
    print(f"{len(trades)} hedge cycles, {n_priced} priced with real quotes, {len(trades)-n_priced} skipped (no real data)", flush=True)

    t = trades.dropna(subset=["net_return"])
    sizing = SizingMode("hedge", "fixed_notional", BUDGET)
    _, summary = simulate_single_position(t, Cfg(), sizing)
    summary["label"] = "real_tail_hedge"
    print(json.dumps(summary, indent=2, default=str))

    # Crash-window check -- same three windows as the synthetic study, but
    # only 2022 bear falls inside the real-data-confirmed window (2021-06+);
    # 2018 Q4 and 2020 COVID predate what this subscription tier can verify.
    crash_windows = {"2022_bear": ("2022-01-01", "2022-10-31")}
    for name, (start, end) in crash_windows.items():
        mask = (t.signal_time >= pd.Timestamp(start, tz="UTC")) & (t.signal_time <= pd.Timestamp(end, tz="UTC"))
        window_trades = t[mask]
        spy_mask = (spy.index >= pd.Timestamp(start, tz="UTC")) & (spy.index <= pd.Timestamp(end, tz="UTC"))
        spy_window = spy.loc[spy_mask, "close"]
        spy_dd = float(spy_window.iloc[-1] / spy_window.max() - 1) if len(spy_window) else None
        print(f"\n{name}: SPY return in window={spy_dd}, hedge trades={len(window_trades)}, "
              f"hedge mean return={window_trades.net_return.mean() if len(window_trades) else None}")

    out = ROOT / "outputs" / "tail_hedge_real_data_retest"
    out.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(out / "raw_trades.parquet")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")
    print("\nOriginal synthetic study (full history, for reference, NOT the same window):")
    print("  win 20%, PF 2.96, total_return +147.1%, annual cost 24% of hedge budget")


if __name__ == "__main__":
    main()
