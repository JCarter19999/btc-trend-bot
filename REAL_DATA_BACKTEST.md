# Real-data walk-forward backtest (v3.0)

## Scope

This runner downloads or imports adjusted OHLCV, builds features without using future rows, labels each signal using next-bar entry and deterministic ATR exits, trains direct-return Ridge on rolling chronological windows, purges overlapping labels, selects the highest predicted simultaneous symbol, and applies the v2.9 capital stoppages.

The runnable default is **daily data** because free yfinance intraday history is limited. For a long 30-minute backtest, export bars from a licensed provider into `data/real/{SYMBOL}.csv` and use `--provider csv`.

The stock route is a real-price backtest. The call overlay is intentionally disabled unless actual historical option-chain data is supplied. Current option chains or Black-Scholes reconstructions must not be presented as historical options performance.

## Required CSV schema

One file per symbol, named `AAPL.csv`, `MSFT.csv`, etc. Required columns:

```text
date,open,high,low,close,volume
```

Dates must identify the completed bar. Prices should be consistently adjusted for splits and dividends.

## Outputs

- `candidates.parquet`
- `folds.csv`
- `selected_trades.csv`
- `capital_path.csv`
- `results.json`

## Interpretation gates

Do not promote from historical testing unless performance survives:

1. SPY comparison.
2. 2020 crash, 2022 bear, and choppy subperiod review.
3. Higher slippage.
4. Nearby Ridge thresholds and ATR settings.
5. Per-symbol ablation.
6. Randomized-entry and simple-trend controls.
7. An untouched final period.
