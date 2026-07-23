# Cross-Sectional Ranking and Synthetic Long-Call Results

## Ranking result

The pairwise geometry ranker achieved a mean pairwise AUC of 0.605 across four expanding walk-forward folds. At a 0.65 rank-probability threshold it selected 582 stock trades, earned a mean 32.18 bps per selected trade, remained positive in all four folds, and compounded the independent fold returns to 520.61% at 100% research allocation.

This did not beat the prior direct-return Ridge configuration, which compounded to 909.87% in the same planted synthetic family. The ranking objective is therefore useful, but is a challenger rather than the new champion.

## Long-call overlay

The same ranked stock selections were converted into synthetic long calls. Premiums use Black-Scholes values with symbol-specific implied-volatility floors, earnings-like IV markup, bid/ask spread, theta decay, and IV changes between entry and exit.

At the 0.65 ranking threshold and 100% option allocation:

| Policy | Positive folds | Compounded fold return | Worst fold | Maximum drawdown |
|---|---:|---:|---:|---:|
| 14-session, ~0.35 delta | 3/4 | 1,672,476% | -54.83% | 93.63% |
| 30-session, ~0.40 delta | 4/4 | 3,772,889% | +37.38% | 76.21% |
| 45-session, ~0.50 delta | 4/4 | 905,807% | +75.92% | 62.75% |

These extreme returns are artifacts of a planted synthetic edge, repeated full-premium allocation, and compounding. They are capability tests, not forecasts.

Allocation sensitivity shows the risk more clearly. For the 30-session ~0.40-delta policy:

| Premium allocation | Compounded fold return | Worst fold | Maximum drawdown |
|---:|---:|---:|---:|
| 25% | 4,187.97% | +48.09% | 26.24% |
| 50% | 83,601.30% | +76.74% | 47.54% |
| 100% | 3,772,889.33% | +37.38% | 76.21% |

The shorter 14-session calls were the most fragile and suffered a negative fold and a 93.63% drawdown at full allocation. The 45-session calls produced lower synthetic upside but the best maximum drawdown among the full-allocation policies.

## Scope

Only purchased calls are modeled. The experiment does not include short option positions, assignment, exercise, dividends, early exercise, American-option pricing, volatility surfaces, strike-by-strike liquidity, contract adjustments, or real option-chain data.
