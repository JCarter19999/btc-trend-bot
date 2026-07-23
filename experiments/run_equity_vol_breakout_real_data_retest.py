"""Re-tests the volatility breakout straddle (EQUITY_OPTIONS_SLEEVE_LADDER_STUDY.md's
strongest synthetic finding) with REAL ThetaData quotes instead of
Black-Scholes-off-realized-vol. Joey's stated top priority for the real-data
re-tests: "Your earlier results were the most promising but depended on
Black-Scholes assumptions. Now you can test with actual quoted IV and
market prices."

Same signal exactly (compressed realized vol bottom-quartile + expanding
relative volume, same volatile universe), same 30-DTE/10-day-hold framing
-- only the pricing changes: real call+put ask-in/bid-out on the SAME
real listed strike, not a theoretical ATM price built from a single
realized-vol number.

Scope: signals from 2021-06-01 onward only, same empirically-confirmed
real-data window as the calls re-test.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btc_trend_bot.options_pricing import realized_vol  # noqa: E402
from btc_trend_bot.portfolio_sim import SizingMode, simulate_single_position  # noqa: E402
from btc_trend_bot.thetadata_pricing import real_straddle_trade  # noqa: E402
from run_equity_options_sleeve_ladder import VOLATILE_UNIVERSE, DTE_AT_ENTRY, HOLD_DAYS, _cfg  # noqa: E402
from run_equity_real_data_walkforward import add_features, load_csv_dir  # noqa: E402

REAL_DATA_START = pd.Timestamp("2021-06-01", tz="UTC")
SIZING = SizingMode("fixed_premium_250", "fixed_notional", 250.0)


def build_signal_pool(frames: dict, cfg) -> pd.DataFrame:
    bench = frames[cfg.benchmark]
    rows = []
    for symbol in cfg.symbols:
        f = add_features(frames[symbol], bench).copy()
        f["realized_vol"] = realized_vol(f["close"].to_numpy(), 20)
        f["vol_percentile_1y"] = f["realized_vol"].rolling(252, min_periods=60).rank(pct=True)
        for i in range(260, len(f) - HOLD_DAYS - 1):
            r = f.iloc[i]
            compressed = np.isfinite(r.get("vol_percentile_1y", np.nan)) and r.vol_percentile_1y < 0.25
            expanding_volume = np.isfinite(r.get("relative_volume", np.nan)) and r.relative_volume > 1.2
            if not (compressed and expanding_volume):
                continue
            entry_i, exit_i = i + 1, min(i + 1 + HOLD_DAYS - 1, len(f) - 1)
            rows.append({
                "signal_time": f.index[i], "symbol": symbol,
                "entry_time": f.index[entry_i], "exit_time": f.index[exit_i],
                "entry_spot": float(f.iloc[entry_i].open),
            })
    return pd.DataFrame(rows)


def price_real_straddles(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    rows = []
    n_priced, n_skipped = 0, 0
    t0 = time.time()
    for i, t in enumerate(out.itertuples()):
        r = real_straddle_trade(t.symbol, t.signal_time.date(), t.entry_time.date(), t.exit_time.date(),
                                 t.entry_spot, DTE_AT_ENTRY, 1.0)
        if r is None:
            rows.append({"net_return": np.nan})
            n_skipped += 1
        else:
            rows.append(r)
            n_priced += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(out)} priced ({n_priced} ok, {n_skipped} skipped), {time.time()-t0:.0f}s elapsed", flush=True)
    priced = pd.DataFrame(rows)
    out["net_return"] = priced["net_return"].to_numpy()
    print(f"Straddles: {n_priced} priced, {n_skipped} skipped", flush=True)
    return out


def summarize(trades: pd.DataFrame, label: str, cfg) -> dict:
    if trades.empty:
        return {"label": label, "trades_taken": 0}
    t = trades.dropna(subset=["net_return"]).copy()
    _, summary = simulate_single_position(t, cfg, SIZING)
    summary["label"] = label
    return summary


def main() -> None:
    cfg = _cfg()
    frames = load_csv_dir((*VOLATILE_UNIVERSE, cfg.benchmark), ROOT / "data/real")

    print("Building compressed-vol + expanding-volume signal pool...", flush=True)
    pool = build_signal_pool(frames, cfg)
    pool["signal_time"] = pd.to_datetime(pool["signal_time"], utc=True)
    pool["entry_time"] = pd.to_datetime(pool["entry_time"], utc=True)
    pool["exit_time"] = pd.to_datetime(pool["exit_time"], utc=True)
    print(f"{len(pool)} total setups, restricting to {REAL_DATA_START.date()} onward", flush=True)
    pool = pool[pool.signal_time >= REAL_DATA_START].reset_index(drop=True)
    print(f"{len(pool)} setups in real-data window", flush=True)

    if pool.empty:
        print("No setups in the real-data window -- nothing to price.")
        return

    priced = price_real_straddles(pool)
    summary = summarize(priced, "real_vol_breakout_straddle", cfg)
    print(json.dumps(summary, indent=2, default=str))

    out = ROOT / "outputs" / "equity_vol_breakout_real_data_retest"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")
    print("\nOriginal synthetic study (full universe/window, for reference, NOT the same window):")
    print("  5% spread: win 64.9%, total_return +342.5%")
    print("  10% spread: win 56.7%, total_return +216.9%")


if __name__ == "__main__":
    main()
