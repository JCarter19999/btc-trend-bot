"""First honest look: does trade-flow imbalance (built from real tick data,
not candles) carry any forward-return predictive power at the dollar-bar
level? This is a diagnostic, not a strategy backtest -- no position sizing,
no exit rules yet. The question is narrower and more fundamental: is there
*any* signal here at all, above what a shuffled-label control would show.

Same discipline as every other finding tonight: report the honest number,
run a randomization control before believing a raw correlation, and don't
oversell a small sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc_trend_bot.orderflow_features import build_dollar_bars, load_trades, rolling_flow_imbalance


def diagnose(trades_path: Path, bars_target: int = 2000, imbalance_windows=(1, 5, 20, 50)) -> None:
    trades = load_trades(str(trades_path))
    print(f"{len(trades):,} trades, {trades.timestamp.min()} to {trades.timestamp.max()} "
          f"({(trades.timestamp.max()-trades.timestamp.min())})")

    dollars_per_bar = trades.cost.sum() / bars_target
    bars = build_dollar_bars(trades, dollars_per_bar)
    print(f"{len(bars)} dollar bars, avg duration {(bars.close_time - bars.open_time).mean()}")

    bars["forward_return"] = bars["return"].shift(-1)  # next bar's return, the thing we're trying to predict

    for window in imbalance_windows:
        bars[f"imbalance_{window}"] = rolling_flow_imbalance(bars, window) if window > 1 else bars["flow_imbalance"]

    print("\n=== Real correlation: rolling flow imbalance vs NEXT bar's return ===")
    real_results = {}
    for window in imbalance_windows:
        col = f"imbalance_{window}"
        valid = bars[[col, "forward_return"]].dropna()
        corr = valid[col].corr(valid["forward_return"])
        rank_corr = valid[col].corr(valid["forward_return"], method="spearman")
        real_results[window] = (corr, rank_corr, len(valid))
        print(f"window={window:3d} bars: n={len(valid):5d} pearson_corr={corr:+.4f} spearman_corr={rank_corr:+.4f}")

    print("\n=== Shuffled-label control (10 seeds): same correlation on permuted forward_return ===")
    print("(a real signal's correlation should sit clearly outside this null distribution)")
    for window in imbalance_windows:
        col = f"imbalance_{window}"
        valid = bars[[col, "forward_return"]].dropna()
        null_corrs = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            shuffled = rng.permutation(valid["forward_return"].to_numpy())
            null_corrs.append(np.corrcoef(valid[col], shuffled)[0, 1])
        real_corr = real_results[window][0]
        pct = float((np.array(null_corrs) < real_corr).mean() * 100)
        print(f"window={window:3d}: real={real_corr:+.4f}  null_mean={np.mean(null_corrs):+.4f} "
              f"null_std={np.std(null_corrs):.4f}  real_percentile_in_null={pct:.0f}")

    print("\n=== Simple decile check: does high imbalance -> higher forward return? ===")
    col = "imbalance_5"
    valid = bars[[col, "forward_return"]].dropna().copy()
    valid["decile"] = pd.qcut(valid[col], 10, labels=False, duplicates="drop")
    print(valid.groupby("decile")["forward_return"].agg(["mean", "count"]))


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/orderflow/btcusdt_trades_pilot.parquet"
    diagnose(path)
