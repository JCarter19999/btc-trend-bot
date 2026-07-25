# Cross-market lead-lag deep dive: pushing the two unexploited threads

Extends `DAX_EDGE_EXPANSION_PROGRAM.md` and `TAIWAN_SEMICONDUCTOR_LEAD_LAG_STUDY.md`
— both flagged real, credible-looking findings that were never taken through
full rigor. This applies that rigor. Scripts run ad hoc against the cached
research venv; no new experiment file saved (see note at bottom).

## 1. EURUSD/DAX agreement — fails the shuffled-null control, despite looking strong as a screen

Reconstructed the exact condition from `DAX_EDGE_EXPANSION_PROGRAM.md`
(`experiments/run_dax_edge_expansion_program.py`'s `fx_conditioning` block):
among the 102 DAX-top-quartile-gated days, the 61 where EURUSD's pre-13:00-UTC
return agrees in sign with DAX's.

| | Value |
|---|---|
| Trades | 61 |
| Mean bps/trade (1bp cost) | 13.82 (matches doc exactly) |
| Sharpe | 6.15 |
| Cost stress: 2bp / 5bp | 12.82 / 9.82 bps — survives easily |
| **Shuffled-null (2,000 seeds, random 61-of-102 subset)** | null mean 10.30bps, std 3.37 — **real result sits at the 84.8th percentile** |

**Verdict: does not clear this project's bar.** 84.8th percentile is well
below the ~95-97th this project treats as real signal everywhere else
tonight. The apparent Sharpe improvement over the DAX-alone baseline (6.15
vs 3.95) is largely explained by variance reduction from trading a smaller
subset of the same 102 days — not by EURUSD agreement carrying real
incremental information. Survives cost stress fine, but that's a lower bar
than the shuffled-null it fails. **Not pursued to a real options backtest**
— doesn't clear the more fundamental test first.

## 2. Taiwan magnitude gating DAX→SPY — clean rejection

The Taiwan study's own proposed next step ("use Taiwan's move MAGNITUDE as
a filter on some other directional signal") tested directly: gate the
DAX-top-quartile/SPY signal with an ADDITIONAL requirement that Taiwan's
own session |return| is also in its expanding top quartile.

| | DAX alone (baseline) | DAX AND Taiwan-magnitude |
|---|---|---|
| Trades | 102 | 36 |
| Mean bps/trade | 10.41 | **8.73** (worse) |
| Sharpe | 3.95 | 2.63 |
| Shuffled-null percentile (2,000 seeds) | — | **36.4** (below median of random subsets) |
| OOS: first half / second half | — | 17.24 / **3.31** bps (steep decay) |

**Verdict: clean rejection.** Adding Taiwan magnitude as a confirming gate
makes the signal worse, not better — sits below the 50th percentile of a
random-subset null (i.e., a random 36-of-102 subset would typically do as
well or better), and decays sharply out of sample. Taiwan's magnitude
effect on SOXX/SMH (documented in the original study) does not transfer to
gating the DAX→SPY mechanism specifically. The original study's directional
finding (weak, contrarian, -0.12 to -0.14) was already known not to be
naively tradeable; this closes out the magnitude-as-gate idea for THIS
specific application too. Whether Taiwan magnitude gates something
SOXX/SMH-specific (a standalone momentum signal on the semiconductor ETFs
themselves, rather than the DAX/SPY mechanism) remains untested — flagged,
not pursued here due to time.

## 3. New-pair scan — not reached

Both priority threads consumed the full effort and both required real,
careful rigor (not assumed away). The new-pair scan (FTSE/CAC vs. other US
sectors, ASX/KOSPI earlier-Asian read, commodity/rate markets) was not
started. Flagged as the next open thread, not attempted.

## Bottom line

Neither of the two most promising-looking unexploited leads in this
project survives real testing. Both looked good as a screen (positive
both-halves OOS, decent Sharpe) and both failed the shuffled-null control
specifically — the same failure mode that's killed several other
candidates across this whole research effort. The DAX-alone signal (the
one already live-deployed) remains the only cross-market finding that has
survived every rigor pass applied to it. This is a real, disciplined
negative result, not a failure to find something — it closes out two
open threads cleanly rather than leaving them as unresolved "looks
promising" items.

Note: analysis run interactively against the cached `.venv`, reusing
`run_dax_edge_expansion_program.py`'s and
`run_taiwan_semiconductor_lead_lag_study.py`'s existing helper functions
(`build_dataset`, `_expanding_gate`, `_metrics`, `download_hourly`,
`european_pre_us_open_return`) rather than duplicating them into a new
script file, since both threads terminated in clean rejections rather than
results worth productionizing into a reusable pipeline.
