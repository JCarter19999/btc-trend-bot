"""Two follow-ups to the first order-flow diagnostic
(run_orderflow_signal_diagnostic.py), both aimed at finding a reachable
fast-paced regime rather than just re-confirming the sub-second effect:

1. Latency-haircut decay curve. The first diagnostic assumed zero reaction
   time -- capture literally the next bar's return. Real execution has
   signal-to-order latency. This resamples trades into fixed real-time
   (second-denominated) buckets so latency and holding period are both
   expressible in actual seconds, then sweeps (signal window, latency,
   holding period) to find whether *any* combination clears a realistic
   12-16bps round-trip cost floor -- the concrete fork this study exists to
   resolve: a reachable horizon exists, or Tier 1 is exhausted.

2. Regime-conditional check (cheap parallel thread). Does flow-imbalance
   predictive power get stronger when funding rate is in its extreme
   deciles, or when open interest is moving sharply? Uses the funding-rate
   (1yr) and open-interest (29-day cap) history already downloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc_trend_bot.orderflow_features import load_trades

ROUND_TRIP_COST_BPS_LOW = 12.0
ROUND_TRIP_COST_BPS_HIGH = 16.0


def build_second_bars(trades: pd.DataFrame) -> pd.DataFrame:
    """Fixed 1-second buckets -- makes latency/hold time directly
    interpretable in real seconds, unlike dollar bars (whose duration
    varies with the chosen threshold)."""
    t = trades.set_index("timestamp")
    bars = t.resample("1s").agg(
        close=("price", "last"),
        buy_volume=("signed_amount", lambda s: s[s > 0].sum()),
        sell_volume=("signed_amount", lambda s: -s[s < 0].sum()),
        n_trades=("price", "size"),
    )
    bars["close"] = bars["close"].ffill()
    bars["buy_volume"] = bars["buy_volume"].fillna(0.0)
    bars["sell_volume"] = bars["sell_volume"].fillna(0.0)
    bars["n_trades"] = bars["n_trades"].fillna(0)
    total = bars.buy_volume + bars.sell_volume
    bars["imbalance"] = np.where(total > 0, (bars.buy_volume - bars.sell_volume) / total, np.nan)
    return bars


def latency_decay_curve(bars: pd.DataFrame, windows_s: list[int], latencies_s: list[int], holds_s: list[int]) -> pd.DataFrame:
    close = bars["close"]
    rows = []
    for window in windows_s:
        buy_roll = bars.buy_volume.rolling(window).sum()
        sell_roll = bars.sell_volume.rolling(window).sum()
        n_roll = bars.n_trades.rolling(window).sum()
        total = buy_roll + sell_roll
        signal = (buy_roll - sell_roll) / total.replace(0, np.nan)
        for latency in latencies_s:
            for hold in holds_s:
                # Price at signal time (t), entry price at t+latency, exit price at t+latency+hold.
                entry_price = close.shift(-latency)
                exit_price = close.shift(-(latency + hold))
                realized_return = exit_price / entry_price - 1
                valid = pd.DataFrame({"signal": signal, "ret": realized_return, "n": n_roll}).dropna()
                if len(valid) < 100 or (valid["n"] == 0).all():
                    continue
                implied = np.sign(valid.signal) * valid.ret
                gross_bps = implied.mean() * 10000
                net_low = gross_bps - ROUND_TRIP_COST_BPS_LOW
                net_high = gross_bps - ROUND_TRIP_COST_BPS_HIGH
                rows.append({
                    "window_s": window, "latency_s": latency, "hold_s": hold, "n_obs": len(valid),
                    "win_rate": float((implied > 0).mean()), "gross_bps": gross_bps,
                    "net_bps_at_12bps_cost": net_low, "net_bps_at_16bps_cost": net_high,
                })
    return pd.DataFrame(rows)


def regime_conditional_check(bars: pd.DataFrame, funding: pd.DataFrame, oi: pd.DataFrame) -> None:
    window, latency, hold = 5, 1, 5  # a representative near-term point from the decay curve
    buy_roll = bars.buy_volume.rolling(window).sum()
    sell_roll = bars.sell_volume.rolling(window).sum()
    total = buy_roll + sell_roll
    signal = (buy_roll - sell_roll) / total.replace(0, np.nan)
    entry_price = bars["close"].shift(-latency)
    exit_price = bars["close"].shift(-(latency + hold))
    realized_return = exit_price / entry_price - 1
    frame = pd.DataFrame({"signal": signal, "ret": realized_return}).dropna()
    frame["implied_bps"] = np.sign(frame.signal) * frame.ret * 10000
    frame = frame.reset_index().rename(columns={"timestamp": "ts"})

    # Merge nearest-prior funding rate and OI level onto each second-bar (both are slow-moving context).
    funding_sorted = funding.sort_values("timestamp").rename(columns={"timestamp": "ts"})
    funding_sorted["ts"] = pd.to_datetime(funding_sorted["ts"], unit="ms", utc=True)
    frame = pd.merge_asof(frame.sort_values("ts"), funding_sorted[["ts", "funding_rate"]], on="ts", direction="backward")

    if len(oi):
        oi_sorted = oi.sort_values("timestamp").rename(columns={"timestamp": "ts"})
        oi_sorted["ts"] = pd.to_datetime(oi_sorted["ts"], unit="ms", utc=True)
        oi_sorted["oi_change"] = oi_sorted["open_interest_amount"].pct_change()
        frame = pd.merge_asof(frame.sort_values("ts"), oi_sorted[["ts", "oi_change"]], on="ts", direction="backward")

    print(f"\n=== Regime-conditional check (window={window}s, latency={latency}s, hold={hold}s) ===")
    print(f"Unconditional gross edge: {frame.implied_bps.mean():.3f} bps, n={len(frame):,}")

    if frame["funding_rate"].notna().sum() > 100:
        frame["funding_decile"] = pd.qcut(frame["funding_rate"], 10, labels=False, duplicates="drop")
        print("\nBy funding-rate decile (0=most negative/bearish funding, 9=most positive/bullish funding):")
        print(frame.groupby("funding_decile")["implied_bps"].agg(["mean", "count"]))
    else:
        print("Not enough funding-rate coverage in this window to decile.")

    if "oi_change" in frame.columns and frame["oi_change"].notna().sum() > 100:
        frame["oi_decile"] = pd.qcut(frame["oi_change"], 10, labels=False, duplicates="drop")
        print("\nBy OI-change decile (0=sharpest OI decline, 9=sharpest OI increase):")
        print(frame.groupby("oi_decile")["implied_bps"].agg(["mean", "count"]))
    else:
        print("Not enough OI coverage in this window to decile (OI history may not overlap trade data window).")


def main() -> None:
    trades_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data/orderflow/btcusdt_trades_14d.parquet")
    trades = load_trades(trades_path)
    print(f"{len(trades):,} trades, {trades.timestamp.min()} to {trades.timestamp.max()}")

    bars = build_second_bars(trades)
    print(f"{len(bars):,} one-second bars, {bars.n_trades.gt(0).mean()*100:.1f}% non-empty")

    print("\n=== Latency-haircut decay curve ===")
    curve = latency_decay_curve(
        bars,
        windows_s=[1, 5, 15, 30, 60, 300],
        latencies_s=[0, 1, 2, 5, 10],
        holds_s=[1, 5, 15, 30, 60, 300],
    )
    curve = curve.sort_values("net_bps_at_16bps_cost", ascending=False)
    pd.set_option("display.width", 160)
    print(curve.head(20).to_string(index=False))
    best = curve.iloc[0] if len(curve) else None
    if best is not None and best.net_bps_at_16bps_cost > 0:
        print(f"\n*** Crossover found: window={best.window_s}s latency={best.latency_s}s hold={best.hold_s}s "
              f"nets +{best.net_bps_at_16bps_cost:.2f}bps even at the higher 16bps cost assumption ***")
    else:
        print(f"\nNo combination tested clears the cost floor. Best net (16bps cost): "
              f"{best.net_bps_at_16bps_cost:.2f}bps" if best is not None else "no valid combinations")

    out = ROOT / "outputs" / "equity_orderflow_latency_study"  # reuse outputs/ convention
    out.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out / "latency_decay_curve.csv", index=False)
    print(f"\nFull curve written to {out / 'latency_decay_curve.csv'}")

    funding = pd.read_parquet(ROOT / "data/orderflow/btcusdt_funding_1y.parquet")
    oi_path = ROOT / "data/orderflow/btcusdt_oi_29d.parquet"
    oi = pd.read_parquet(oi_path) if oi_path.exists() else pd.DataFrame()
    regime_conditional_check(bars, funding, oi)


if __name__ == "__main__":
    main()
