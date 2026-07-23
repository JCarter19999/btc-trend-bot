# Broad-Universe Additive Overlay Results

## Configuration

- Fully synthetic 30-minute equity candles
- 450 sessions
- AAPL, MSFT, TSLA, NVDA
- Four expanding walk-forward folds
- Direct-return Ridge with continuous geometry
- One open stock position at a time
- Mandatory stock core leg
- Optional additive long-call overlay
- 2,000 Monte Carlo path simulations per route

## Aggregate results

| Strategy | Compounded return | Maximum drawdown | Probability of final loss |
|---|---:|---:|---:|
| Stock only | +151.19% | 9.67% | 5.40% |
| Stock + 5% call premium | +155.12% | 10.28% | 3.85% |
| Stock + 10% call premium | +158.84% | 10.90% | 4.45% |
| Stock + 20% call premium | +165.65% | 14.05% | 4.20% |
| Stock + 30% call premium | **+171.57%** | **18.22%** | 4.85% |
| Tiered overlay | +156.41% | 12.85% | 4.85% |

## Opportunity count

- 197 accepted stock trades
- 38 extreme-bull signals
- 7 option-overlay trades
- 3.55% of stock trades received a call
- Mean qualifying stock return: +0.26%
- Mean qualifying call return: +4.97%
- Qualifying call win rate: 42.86%
- Full-premium losses: 0

## Distribution

Qualifying calls appeared only in folds 1 and 4:

- Fold 1: 3 overlays, mean optimized-option return -4.11%
- Fold 4: 4 overlays, mean optimized-option return +9.80%

By symbol:

- AAPL: 3 qualifying overlays
- MSFT: 3 qualifying overlays
- NVDA: 1 qualifying overlay
- TSLA: 0 qualifying overlays

## Interpretation

The additive overlay increased synthetic compounded return at every tested fixed allocation. The improvement was monotonic because the seven qualifying calls had a positive aggregate return. However, the maximum drawdown also rose materially at 20% and 30% premium allocations.

The 5% overlay produced the best conservative trade-off in this sample: +3.93 percentage points over stock-only with only +0.61 percentage points of observed drawdown. The 30% overlay produced the most return, adding +20.38 percentage points, but nearly doubled maximum drawdown from 9.67% to 18.22%.

The option sample remains small and regime-concentrated. No overlays qualified in folds 2 or 3, and the first fold's qualifying calls were negative on average. The experiment supports keeping calls as an additive, independently gated leg, but does not establish 30% premium allocation as reliable.

All results are synthetic capability tests with a deliberately planted edge. They are not forecasts of live performance.
