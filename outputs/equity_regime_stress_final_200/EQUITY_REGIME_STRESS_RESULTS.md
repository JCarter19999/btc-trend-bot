# Synthetic regime-stress results

Initial capital: **$2,500**. Monthly deposits: **$0**. Route: cash-safe whole-contract execution with a 30% call-premium budget.

| Scenario | Route | Ending equity | Net P/L | Max drawdown | Trades taken | Safety skips | Hard shutdown |
|---|---|---:|---:|---:|---:|---:|---|
| Mixed control | Unprotected | $3,659.67 | +$1,159.67 | 21.78% | 100 | 0 | No |
| Mixed control | Protected | **$4,241.78** | **+$1,741.78** | **8.75%** | 70 | 30 | No |
| Persistent bear | Unprotected | $2,258.85 | -$241.15 | 24.03% | 91 | 0 | No |
| Persistent bear | Protected | **$2,883.69** | **+$383.69** | **12.94%** | 63 | 28 | No |
| Choppy high volatility | Unprotected | **$2,600.35** | **+$100.35** | 8.76% | 15 | 0 | No |
| Choppy high volatility | Protected | $2,478.96 | -$21.04 | **2.93%** | 5 | 10 | No |
| Crash/recovery | Unprotected | **$2,691.61** | **+$191.61** | 7.09% | 30 | 0 | No |
| Crash/recovery | Protected | $2,525.28 | +$25.28 | 7.39% | 19 | 11 | No |
| Clustered tail losses | Unprotected | $261.73 | -$2,238.27 | **93.82%** | 100 | 0 | No |
| Clustered tail losses | Protected | **$2,817.90** | **+$317.90** | **31.74%** | 11 | 89 | No* |

\*The protected tail test stopped taking risk after reaching the safety state; the summary's hard-shutdown flag remains false because the drawdown pause and cooldown blocked further entries before the next loop crossed the explicit 35% hard-stop check.

## Interpretation

The safety layer substantially helped in the mixed and persistent-bear paths and prevented catastrophic loss in the deliberately hostile clustered-loss sequence. It was costly in choppy and recovery environments because it skipped profitable rebound trades. Therefore, stoppages are functioning, but the current regime gate is intentionally conservative and should remain a challenger until validated on historical data.

The stress test does not prove that the strategy will work live. It shows that the code can: skip long trades in weak regimes, pause after losing streaks, pause options independently, cap whole-contract premium exposure, and stop deploying capital before a synthetic loss cascade reaches total loss.
