# Equity Hybrid Ranking + Return Gate + Option Optimizer — v2.5.0

## Scope

This experiment preserves direct-return Ridge as the absolute-edge champion, uses pairwise ranking only to choose among simultaneous AAPL, TSLA, and MSFT candidates, and then compares three ways to express the selected signal:

1. Stock only.
2. Long call only, with either fixed 25% premium allocation or fractional-Kelly sizing.
3. A dynamic stock-or-call decision and a stock-plus-call overlay.

The option engine generates synthetic chains across multiple expirations and deltas, applies symbol-specific volatility skew, earnings-related IV behavior, bid/ask spreads, open-interest and volume filters, and optimizes expected option utility. Monte Carlo trade-order resampling estimates drawdown and ruin probabilities.

## Backtest configuration

- Synthetic sessions: 400
- Symbols: AAPL, TSLA, MSFT
- Bar interval: 30 minutes, regular session
- Walk-forward folds: 4
- Absolute expected-return gate: 5 bps
- Pairwise ranking threshold: 0.55
- Monte Carlo simulations per expression: 5,000
- Options: long calls only; premium loss capped at 100%

## Main strategy results

| Expression | Trades | Mean trade return | Compounded return | Max drawdown | P(equity < 50%) | P(final loss) |
|---|---:|---:|---:|---:|---:|---:|
| stock only | 123 | 0.51% | 83.20% | 12.18% | 0.04% | 7.68% |
| stock plus call overlay | 123 | 0.50% | 79.02% | 12.18% | 0.06% | 8.98% |
| dynamic stock or call | 123 | 0.48% | 76.27% | 12.18% | 0.02% | 10.42% |
| call fractional kelly | 123 | 0.49% | 70.28% | 27.19% | 0.90% | 23.02% |
| call 25pct | 123 | 0.36% | 51.15% | 16.95% | 0.22% | 25.08% |


## Option-chain optimizer behavior

- Selected stock trades: 123
- Trades with an acceptable optimized call: 27 (22.0%)
- Option win rate among available contracts: 55.6%
- Mean realized option return among available contracts: 6.57%
- Median realized option return: 2.42%
- Median DTE: 14
- Median delta: 0.546
- Median quoted spread as a fraction of premium: 3.45%
- Median open interest: 1720
- Total full-premium losses: 0

The optimizer rejected roughly 78.0% of stock signals for options. This is expected: a favorable stock forecast does not imply that the available option premium, skew, spread, and decay produce positive expected value.

## Walk-forward interpretation

The stock signal was strongest in fold 1, weak and negative in fold 2, recovered in folds 3 and 4, and remained the best aggregate expression. Option availability also changed sharply by fold, including no acceptable contracts in fold 3. This shows why contract selection must be evaluated separately from directional prediction.

## Main conclusion

The hybrid system worked technically, but it did not manufacture extra edge from options. Stock-only produced the strongest result and lowest observed drawdown. Fractional-Kelly calls retained substantial upside but increased drawdown and probability of loss. The dynamic optimizer chose an option on only one trade, so its output mostly converged to stock-only.

This is a useful result rather than a failure: liquidity, volatility skew, and expected-value filtering prevented the options branch from blindly levering every stock signal. The next tuning target is the instrument-expression boundary—not the directional Ridge model. Future tests should enlarge the synthetic sample, calibrate the option utility function, and compare stock, call, and mixed exposures on equal risk rather than equal nominal allocation.
