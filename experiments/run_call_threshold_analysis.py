"""Task #4: does signal strength (relative_strength_20) predict when real
calls beat real shares, per the hypothesis "maybe only the strongest 5-10%
of signals justify the leverage." Pure post-hoc slicing of the raw trades
already priced with real ThetaData quotes in
run_equity_options_real_data_retest.py -- no new API calls.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_equity_options_real_data_retest import (  # noqa: E402
    CALL_SIZING, STOCK_SIZING, summarize,
)
from run_equity_real_data_walkforward import load_config  # noqa: E402

BUCKETS = {
    "bottom_50pct": (0.0, 0.50),
    "50-75pct": (0.50, 0.75),
    "75-90pct": (0.75, 0.90),
    "top_10pct": (0.90, 1.0),
}


def bucketed_comparison(trades: pd.DataFrame, cfg, label: str) -> dict:
    priced = trades.dropna(subset=["net_return"]).copy()
    ranks = priced["relative_strength_20"].rank(pct=True)
    out = {}
    print(f"\n=== {label} ({len(priced)} priced trades) ===")
    for name, (lo, hi) in BUCKETS.items():
        mask = (ranks > lo) & (ranks <= hi)
        bucket = priced[mask]
        if bucket.empty:
            continue
        stock_bucket = bucket.copy()
        stock_bucket["net_return"] = stock_bucket["stock_net_return"]
        call_summary = summarize(bucket, f"{label}_{name}_call", CALL_SIZING, cfg)
        stock_summary = summarize(stock_bucket, f"{label}_{name}_stock", STOCK_SIZING, cfg)
        out[name] = {"call": call_summary, "stock": stock_summary}
        print(f"  {name:12s} n={len(bucket):3d}  "
              f"call: win={call_summary.get('win_rate',float('nan')):.2f} PF={call_summary.get('profit_factor') or float('nan'):.2f} "
              f"total={call_summary.get('total_return',float('nan'))*100:7.1f}%   |   "
              f"stock: win={stock_summary.get('win_rate',float('nan')):.2f} PF={stock_summary.get('profit_factor') or float('nan'):.2f} "
              f"total={stock_summary.get('total_return',float('nan'))*100:7.1f}%   "
              f"call-stock spread={((call_summary.get('total_return') or 0)-(stock_summary.get('total_return') or 0))*100:+7.1f}pp")
    return out


def main() -> None:
    cfg_base = load_config(ROOT / "configs/real_data.yaml")
    symbols = ("TSLA", "COIN", "MSTR", "PLTR", "GME")
    cfg = dataclasses.replace(cfg_base, symbols=symbols, stop_atr=100.0, target_atr=100.0, max_hold_bars=10,
                               safety_enabled=False, hard_shutdown_drawdown=1.0, minimum_equity=0.0)

    src = ROOT / "outputs" / "equity_options_real_data_retest"
    results = {}
    for moneyness in (1.0, 1.05):
        trades = pd.read_parquet(src / f"raw_trades_moneyness{moneyness}.parquet")
        results[str(moneyness)] = bucketed_comparison(trades, cfg, f"moneyness{moneyness}")

    out = ROOT / "outputs" / "call_threshold_analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
