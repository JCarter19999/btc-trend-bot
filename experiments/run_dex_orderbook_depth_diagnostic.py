"""Tier 2 spot-check: does real L2 order-book depth imbalance (not trade tape)
carry forward-return predictive power, on the free 7-day Hyperliquid DEX
sample from Kaggle (adamatractor/dex-orderbook-data-5m)?

This is a different question from the one Tier 1 already closed
(`run_orderflow_signal_diagnostic.py`, `run_orderflow_latency_regime_study.py`):
that work measured Binance CEX trade-tape imbalance at sub-second-to-second
granularity and found a real but uncapturable edge (too small, decays with
any latency). This dataset is 5-minute bars on a DEX (Hyperliquid) — three
orders of magnitude coarser in time and a different venue entirely — so it
can only answer "is there a slower-horizon depth-imbalance signal here",
not re-test the fast-paced regime Tier 1 already exhausted.

Data-quality note, checked before trusting anything downstream: OHLC is
flat (open==high==low==close) on 100% of bars across all 24 instruments in
this sample -- each "bar" is a single mid-price snapshot, not true intrabar
aggregation. close_price still moves correctly bar-to-bar (verified), so it's
usable as a point-sampled price series for a returns test, but any
high/low-range feature would be dead weight and the "OHLCV" framing in the
dataset's own description overstates what's actually in the free sample.

Same discipline as every other finding in this project: report the honest
correlation, check it against a shuffled-label null before believing it,
don't oversell a 7-day sample.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FORWARD_HORIZONS_BARS = (1, 3, 12, 24)  # 5min, 15min, 1h, 2h at 5-minute bars
ROLLING_WINDOWS_BARS = (1, 3, 12)  # smoothing on the imbalance signal itself


def load_symbol(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    bid_cols = [f"bid_volume_level_{i}" for i in range(1, 11)]
    ask_cols = [f"ask_volume_level_{i}" for i in range(1, 11)]

    df["bid_l1"] = df["bid_volume_level_1"]
    df["ask_l1"] = df["ask_volume_level_1"]
    df["imbalance_l1"] = (df["bid_l1"] - df["ask_l1"]) / (df["bid_l1"] + df["ask_l1"])

    bid_sum10 = df[bid_cols].sum(axis=1)
    ask_sum10 = df[ask_cols].sum(axis=1)
    df["imbalance_10lvl"] = (bid_sum10 - ask_sum10) / (bid_sum10 + ask_sum10)

    df["mid"] = df["close_price"]
    for h in FORWARD_HORIZONS_BARS:
        df[f"fwd_ret_{h}"] = df["mid"].shift(-h) / df["mid"] - 1.0

    return df


def shuffled_null_percentile(x: np.ndarray, y: np.ndarray, real_corr: float, seeds: int = 10) -> tuple[float, float, float]:
    null_corrs = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(y)
        null_corrs.append(np.corrcoef(x, shuffled)[0, 1])
    null_corrs = np.array(null_corrs)
    pct = float((null_corrs < real_corr).mean() * 100)
    return float(null_corrs.mean()), float(null_corrs.std()), pct


def per_symbol_results(df: pd.DataFrame, symbol: str) -> list[dict]:
    rows = []
    for imb_name in ("imbalance_l1", "imbalance_10lvl"):
        for roll in ROLLING_WINDOWS_BARS:
            sig = df[imb_name].rolling(roll).mean() if roll > 1 else df[imb_name]
            for h in FORWARD_HORIZONS_BARS:
                valid = pd.DataFrame({"sig": sig, "fwd": df[f"fwd_ret_{h}"]}).dropna()
                if len(valid) < 30:
                    continue
                corr = valid["sig"].corr(valid["fwd"])
                null_mean, null_std, pct = shuffled_null_percentile(
                    valid["sig"].to_numpy(), valid["fwd"].to_numpy(), corr
                )
                rows.append(
                    dict(
                        symbol=symbol,
                        imbalance=imb_name,
                        roll_bars=roll,
                        fwd_horizon_bars=h,
                        n=len(valid),
                        pearson_corr=corr,
                        null_mean=null_mean,
                        null_std=null_std,
                        null_percentile=pct,
                        mean_fwd_ret_top_decile=float(
                            valid.assign(dec=pd.qcut(valid["sig"], 10, labels=False, duplicates="drop"))
                            .groupby("dec")["fwd"]
                            .mean()
                            .iloc[-1]
                        )
                        if valid["sig"].nunique() >= 10
                        else None,
                        mean_fwd_ret_bottom_decile=float(
                            valid.assign(dec=pd.qcut(valid["sig"], 10, labels=False, duplicates="drop"))
                            .groupby("dec")["fwd"]
                            .mean()
                            .iloc[0]
                        )
                        if valid["sig"].nunique() >= 10
                        else None,
                    )
                )
    return rows


def main(data_dir: Path, out_dir: Path) -> None:
    files = sorted(data_dir.glob("*_sample_7d.parquet"))
    if not files:
        raise SystemExit(f"No *_sample_7d.parquet files found in {data_dir}")

    print(f"Found {len(files)} instruments: {[f.stem.split('_')[0] for f in files]}\n")

    all_rows: list[dict] = []
    pooled_frames = []
    for f in files:
        symbol = f.stem.split("_")[0]
        df = load_symbol(f)
        rows = per_symbol_results(df, symbol)
        all_rows.extend(rows)

        # z-score the signals per-symbol before pooling, so one instrument's scale
        # doesn't dominate the pooled correlation
        pdf = df.copy()
        for col in ("imbalance_l1", "imbalance_10lvl"):
            std = pdf[col].std()
            pdf[col + "_z"] = (pdf[col] - pdf[col].mean()) / std if std > 0 else 0.0
        pooled_frames.append(pdf)

    results_df = pd.DataFrame(all_rows)

    print("=== BTC only, immediate (no smoothing) level-1 imbalance vs forward return ===")
    btc = results_df[(results_df.symbol == "BTC") & (results_df.imbalance == "imbalance_l1") & (results_df.roll_bars == 1)]
    print(btc[["fwd_horizon_bars", "n", "pearson_corr", "null_mean", "null_std", "null_percentile"]].to_string(index=False))

    print("\n=== BTC only, 10-level cumulative imbalance vs forward return (all rolling windows) ===")
    btc10 = results_df[(results_df.symbol == "BTC") & (results_df.imbalance == "imbalance_10lvl")]
    print(btc10[["roll_bars", "fwd_horizon_bars", "n", "pearson_corr", "null_percentile"]].to_string(index=False))

    print("\n=== Best |correlation| per instrument (any imbalance/roll/horizon combo) ===")
    best = results_df.loc[results_df.groupby("symbol")["pearson_corr"].apply(lambda s: s.abs().idxmax())]
    print(
        best[["symbol", "imbalance", "roll_bars", "fwd_horizon_bars", "n", "pearson_corr", "null_percentile"]]
        .sort_values("pearson_corr", key=lambda s: s.abs(), ascending=False)
        .to_string(index=False)
    )

    # Pooled, z-scored cross-instrument check for extra power given ~2000 rows/symbol
    pooled = pd.concat(pooled_frames, ignore_index=True)
    print("\n=== Pooled across all 24 instruments (z-scored per symbol), level-1 imbalance ===")
    pooled_rows = []
    for h in FORWARD_HORIZONS_BARS:
        valid = pooled[["imbalance_l1_z", f"fwd_ret_{h}"]].dropna()
        corr = valid["imbalance_l1_z"].corr(valid[f"fwd_ret_{h}"])
        null_mean, null_std, pct = shuffled_null_percentile(
            valid["imbalance_l1_z"].to_numpy(), valid[f"fwd_ret_{h}"].to_numpy(), corr
        )
        pooled_rows.append(
            dict(fwd_horizon_bars=h, n=len(valid), pearson_corr=corr, null_mean=null_mean, null_std=null_std, null_percentile=pct)
        )
        print(f"horizon={h:3d} bars: n={len(valid):6d} corr={corr:+.4f} null_mean={null_mean:+.4f} null_std={null_std:.4f} pct={pct:.0f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(
            dict(
                n_instruments=len(files),
                per_symbol_best=best.to_dict(orient="records"),
                pooled_level1=pooled_rows,
                data_quality_flat_ohlc_pct=100.0,
            ),
            fh,
            indent=2,
        )
    results_df.to_csv(out_dir / "full_results.csv", index=False)
    print(f"\nWrote {out_dir / 'summary.json'} and {out_dir / 'full_results.csv'}")


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/dex_orderbook_sample"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs/dex_orderbook_depth_diagnostic"
    main(data_dir, out_dir)
