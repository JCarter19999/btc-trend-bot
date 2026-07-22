# Equity Gap / Walk-Forward Synthetic Results

The next AAPL/TSLA/MSFT synthetic experiment added explicit overnight gap execution, quarterly earnings-like shocks, one/two/five-session holding windows, 5/10/15-bps selection hurdles, and four expanding chronological walk-forward folds.

## Best synthetic configuration

- Hold cap: two sessions (the five-session cap produced identical outcomes because adaptive exits closed every selected trade earlier)
- Features: baseline + continuous candlestick geometry
- Model: Ridge regression
- Acceptance hurdle: 5 bps predicted net return
- Positive walk-forward folds: 4 of 4
- Selected trades across folds: 574
- Mean selected expectancy: 44.95 bps
- Compounded fold return at 100% research allocation: 909.87%
- Worst fold return: 15.50%
- Largest fold drawdown: 20.11%
- Gap-through-stop exits: 15

The very large returns are a consequence of a deliberately planted synthetic cross-sectional edge and 100% allocation. They are capability-test outputs, not plausible forecasts.

## Geometry ablation

At the 5-bps hurdle and two-session cap:

- Geometry Ridge compounded fold return: 909.87%
- Baseline Ridge compounded fold return: 679.99%
- Geometry mean test R²: +0.0095
- Baseline mean test R²: -0.0074

Continuous geometry improved cross-sectional selection in this planted environment. Predictive R² remained very small, reinforcing that economic ranking can improve while point-return prediction remains weak.

## Threshold trade-off

For geometry Ridge, two-session hold:

- 5 bps: 574 trades, 44.95 bps mean expectancy, 909.87% compounded folds
- 10 bps: 527 trades, 49.01 bps mean expectancy, 775.53% compounded folds
- 15 bps: 484 trades, 55.19 bps mean expectancy, 680.95% compounded folds

Higher hurdles improved average trade quality but reduced opportunity count and total compounded return.

## Holding-window result

One-, two-, and five-session caps were nearly identical. The adaptive stop/target/momentum engine normally closed positions within the first session or early in the second. Extending the cap to five sessions added no benefit in this environment.

## Gap-risk result

Selected trades included both favorable and adverse gap exits:

- Gap-through-stop exits: 15 for the best configuration
- Gap-through-target exits: 16

The stop implementation fills at the next session open when price gaps below the active stop, preventing the unrealistic assumption that an overnight stop always fills at its requested level.

## Model comparison

Ridge outperformed the nonlinear tree economically. The tree selected more trades but had lower expectancy and lower compounded return. This synthetic edge was therefore sufficiently smooth for the regularized linear model, while the tree over-expanded coverage.

## Benchmark context

Across the broad benchmark interval:

- AAPL buy-and-hold: +34.53%, 43.66% max drawdown
- TSLA buy-and-hold: +478.70%, 57.04% max drawdown
- MSFT buy-and-hold: +44.10%, 47.99% max drawdown
- Equal-weight basket: +185.78%, 34.50% max drawdown

The synthetic trading strategy exceeded these returns, but that comparison is intentionally favorable because the market generator contains a planted, observable cross-sectional edge.
