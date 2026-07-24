# Puts and shorting revisited with real data: two very different answers

Joey asked to revisit puts/shorting now that real ThetaData access
exists. Two genuinely separate things got tested, with opposite results
-- worth keeping distinct rather than one blended "shorting" verdict.

## 1) European lead signal's short side (puts on DAX-down-move days) — real data confirms it works

Already covered in `EQUITY_OPTIONS_REAL_DATA_RETEST.md`'s rigor section.
Within the already-validated European lead signal, puts (short-DAX-signal
days) slightly OUTPERFORM calls: 0DTE shorts +83.8% (n=60) vs longs
+62.8% (n=56); 1DTE shorts +49.3% vs longs +39.4%. This mechanism is the
DAX-morning-weakness-predicts-SPY-weakness signal, independently
validated with an out-of-sample split before any options were involved.

## 2) The OLDER weakest-momentum put/short thesis — real data confirms it does NOT work

Different, separately-tested idea: short/put the single WEAKEST relative-
momentum candidate each day (the mirror image of the original `simple_trend`
long selector), same TSLA/COIN/MSTR/PLTR/GME universe, same 2021-06-01+
real-data window as every other re-test.

**Set expectations before running, not after**: this thesis's problem was
never established to be a data/pricing issue. The underlying short-STOCK
diagnostic (no options at all) was already negative long before any
options work started (`EQUITY_SHORT_STRATEGY_STUDY`: -1.64% mean return,
every symbol negative). Real data was not expected to rescue a selector
with no underlying edge.

| | Trades | Win rate | Profit factor | Total return |
|---|---|---|---|---|
| Short stock (same window, no options) | 41 | 46.3% | 0.67 | -100% |
| Real put, ATM | 33 | 21.2% | 0.37 | -100% |
| Real put, 5% OTM | 28 | 21.4% | 0.31 | -100% |
| *(reference)* Synthetic put, ATM, 5% spread | — | 28.9% | 1.01 | +1.7% |
| *(reference)* Synthetic put, ATM, 10% spread | — | 27.7% | 0.75 | -77.9% |

**Real data makes this thesis look WORSE than the synthetic version, not
better** — the opposite direction from the calls story. Consistent with
this being a genuinely bad signal rather than a mispriced one: real
premiums (reflecting the true volatility risk premium the synthetic
pricing understated) cost more than the old method assumed, and there's
no real directional edge underneath to offset that extra cost the way
there was for the European signal's puts.

## Verdict

**Shorting/puts don't have a universal answer** -- it depends entirely on
whether there's a real directional edge underneath, exactly as expected.
The European signal's short side works because the signal itself is real
(independently validated). The weakest-momentum short thesis doesn't work
with real data for the same reason it didn't work with synthetic data or
as a raw short-stock position: there was never a real edge to express,
only a bad selector. Confirms, doesn't overturn, the original short
verdict from earlier in the project.

Script: `experiments/run_equity_puts_real_data_retest.py`
