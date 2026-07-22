# Equity Capital-Constrained Overlay Results

## Primary assumptions

- Starting-capital grid: $50 to $10,000.
- Monthly contribution: $50.
- Stock leg: fractional shares; $0 online listed-stock commission.
- Option leg: whole contracts only; $0.65 per contract on entry and exit.
- Target option-premium budget: 30% of current equity.
- Variable regulatory and exchange fees are not separately modeled.

## Cash-safe route

| Initial capital | Total contributed | Ending equity | Net profit | Affordable option signals | Contracts | Max drawdown |
|---:|---:|---:|---:|---:|---:|---:|
| $50 | $650 | $1,303.61 | $653.61 | 0/7 | 0 | 6.32% |
| $100 | $700 | $1,429.20 | $729.20 | 0/7 | 0 | 6.32% |
| $250 | $850 | $1,805.99 | $955.99 | 0/7 | 0 | 6.32% |
| $500 | $1,100 | $2,433.96 | $1,333.96 | 0/7 | 0 | 6.32% |
| $1,000 | $1,600 | $4,513.32 | $2,913.32 | 3/7 | 3 | 6.32% |
| $2,500 | $3,100 | $8,906.34 | $5,806.34 | 5/7 | 9 | 8.25% |
| $5,000 | $5,600 | $14,881.58 | $9,281.58 | 7/7 | 17 | 12.08% |
| $10,000 | $10,600 | $28,683.09 | $18,083.09 | 7/7 | 36 | 14.30% |

## Additive-margin comparison

This route preserves stock notional equal to 100% of equity and adds calls on top. It requires margin or equivalent outside buying power. Margin interest is not modeled.

| Initial capital | Ending equity | Affordable option signals | Contracts | Max drawdown |
|---:|---:|---:|---:|---:|
| $50 | $1,303.61 | 0/7 | 0 | 6.32% |
| $100 | $1,429.20 | 0/7 | 0 | 6.32% |
| $250 | $1,805.99 | 0/7 | 0 | 6.32% |
| $500 | $2,433.96 | 0/7 | 0 | 6.32% |
| $1,000 | $4,569.01 | 3/7 | 3 | 6.32% |
| $2,500 | $9,029.29 | 5/7 | 9 | 8.10% |
| $5,000 | $14,931.93 | 7/7 | 17 | 12.81% |
| $10,000 | $28,673.96 | 7/7 | 34 | 15.05% |

## Interpretation

- Accounts beginning at $50–$500 never afforded a qualifying whole call within the 30% premium cap during this path; they behaved as fractional-stock accounts.
- The $1,000 starting account afforded 3 of 7 qualifying calls, with its first option trade late in the sample.
- $2,500 afforded 5 of 7; $5,000 and $10,000 afforded all 7.
- Whole-contract sizing is lumpy. More starting capital did not produce a perfectly monotonic return on contributed capital because each account purchased different contract counts at different times.
- The synthetic stock edge remains deliberately planted. These dollar outcomes are pipeline tests, not forecasts.