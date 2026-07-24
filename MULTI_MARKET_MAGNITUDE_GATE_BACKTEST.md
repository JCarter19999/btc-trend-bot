# Multi-market magnitude gate — negative result

Tested whether blending tonight's four magnitude-correlation findings
(DAX's own move, Taiwan, EURUSD, USDJPY — every one showed a 100th-
percentile magnitude correlation with SPY's first-hour move) into a
single composite gate on the DAX-direction trade beats the gates already
in use. `experiments/run_multi_market_magnitude_gate_backtest.py`.

## Result (1bp round-trip cost, 462 trading days, 2023-09 to 2026-07)

| Variant | Trades | Win | Sharpe | Total return |
|---|---|---|---|---|
| Trade every day (no gate) | 462 | 53.7% | 2.19 | +23.7% |
| **DAX-only top-quartile gate (already live)** | 116 | 61.2% | **4.16** | +12.9% |
| DAX + Asia joint AND-gate (already known, per EUROPEAN_LEAD_ASIAN_MARKETS doc) | ~32 | n/a | **~5-6** | n/a |
| NEW: multi-market composite (DAX+Taiwan+EURUSD+USDJPY, avg percentile rank) top-quartile gate | 116 | 51.7% | 1.77 | +5.8% |
| NEW: non-DAX composite (Taiwan+EURUSD+USDJPY only) top-quartile gate | 116 | 56.9% | 2.08 | +6.8% |

Holds at every cost tier tested (1/2/5bps) — not a cherry-picked cost
level. The multi-market composite gate does still beat a random-direction
control on the same dates (91.2 percentile vs. 1000 shuffled seeds) — it
isn't noise — but it's clearly worse than either existing gate.

## Why blending hurt instead of helping

Averaging percentile ranks across DAX+Taiwan+EURUSD+USDJPY dilutes DAX's
own (strong, validated) magnitude signal with three weaker, noisier
proxies. The one gate that's actually known to work by *adding* a market
(DAX+Asia's joint AND-condition, not a blended average) uses Asia
specifically — the session immediately preceding Europe's own open,
temporally and thematically closest to the same overnight information
flow that drives DAX's pre-open move. Taiwan/EURUSD/USDJPY are more
indirect proxies for that channel (semiconductor-specific, and FX-carry-
specific respectively) and a simple average-then-threshold construction
doesn't reproduce what made the DAX+Asia AND-gate work.

## Honest verdict

**Tonight's Phase 3 search (Taiwan, bonds, FX, and this composite) did
not find a second independent directional edge.** Taiwan and USDJPY
showed real but weak/inconsistent directional correlations; EURUSD
showed none; bonds couldn't be tested at all on this data tier; and
blending their magnitude signals together underperforms what's already
validated and already live (DAX-only gate, or DAX+Asia specifically).
The only validated directional edge in this project remains the DAX
signal itself (optionally sharpened by Asia specifically, not a broader
multi-market blend). This is a real, disclosed negative result, not a
reason to keep iterating on this exact idea — the next genuine "second
edge" candidate needs a different mechanism, not a bigger blend of the
same magnitude-clustering pattern.
