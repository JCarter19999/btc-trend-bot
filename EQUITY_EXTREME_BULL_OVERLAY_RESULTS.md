# Synthetic Extreme-Bull Overlay Results

Configuration: 400 synthetic sessions, AAPL/TSLA/MSFT, 30-minute bars, four walk-forward folds, 5,000 Monte Carlo resamples.

| Strategy | Compounded return | Max drawdown | Probability final loss |
|---|---:|---:|---:|
| Stock only | 83.20% | 12.18% | 7.46% |
| Extreme-bull 10% overlay | 87.34% | 12.74% | 6.92% |
| Extreme-bull 20% overlay | 91.41% | 13.31% | 7.28% |
| Extreme-bull 30% overlay | 95.41% | 13.88% | 6.14% |
| Extreme-bull tiered overlay | **97.34%** | 13.03% | **6.06%** |

Only 2 of 123 stock trades passed every option-overlay gate. There were 12 extreme-bull stock signals, but ten failed the contract or edge requirements. One eligible call won strongly and one lost modestly. Their mean stock return was 2.20%; their mean option return was 14.31%.

The tiered overlay added 14.14 percentage points of compounded return versus stock-only while increasing observed maximum drawdown by 0.85 percentage points. This is promising but statistically weak because the incremental result is driven by only two overlay trades. The experiment validates selective behavior, not a reliable real-market edge.
