# Synthetic AAPL + TSLA + MSFT Offshoot Results

The experiment uses 1,300 synthetic US trading sessions of 30-minute regular-session candles with overnight gaps, correlated market regimes, and symbol-specific volatility.

The BTC champion is unchanged. This is a separate equity research branch.

## Model comparison

| Features | Model | Selected trades | Mean selected return | Portfolio return | Max drawdown |
|---|---|---:|---:|---:|---:|
| Baseline | Ridge | 82 | 69.83 bps | 75.39% | 4.02% |
| Baseline + continuous geometry | Ridge | 91 | 57.15 bps | 66.45% | 4.03% |
| Baseline + continuous geometry | Histogram gradient boosting | 144 | 37.48 bps | 69.32% | 5.64% |

All strategies use 100% research allocation and enforce at most one open position across AAPL, TSLA, and MSFT.

## Buy-and-hold benchmarks over the same chronological test period

| Benchmark | Return | Max drawdown |
|---|---:|---:|
| AAPL buy-and-hold | 91.71% | 26.96% |
| TSLA buy-and-hold | 4.73% | 53.54% |
| MSFT buy-and-hold | 10.80% | 35.54% |
| Equal-weight basket | 35.74% | 23.71% |

## Interpretation

Continuous geometry increased opportunity capture, but did not improve average selected-trade expectancy relative to the baseline Ridge model in this synthetic path. The nonlinear tree achieved positive out-of-sample R² and selected more trades, but with lower per-trade expectancy and higher drawdown.

The baseline model beat the equal-weight basket while taking materially less drawdown, but did not beat the unusually strong AAPL buy-and-hold path. This is a synthetic capability test, not evidence of real stock-market profitability.
