# Online Kalman-Filter Regression — Track B Alpha Candidate

Research-branch experiment. Not wired into the live paper deployment
(`equity_v2_4` main branch / systemd timer) and not a promotion-gate pass.

## Hypothesis

The live Ridge model (`configs/real_data.yaml`) retrains from scratch every
`step_bars` (126 trading days) on a trailing `train_bars` (756-day) window.
Between retrains, predictions across the whole 126-day test fold come from
one static coefficient vector, and coefficients can jump discontinuously at
each fold boundary. Does letting the coefficients update continuously (one
observation at a time, via a Kalman filter with random-walk coefficients)
instead change performance?

## Method

`src/btc_trend_bot/kalman_regression.py` implements the filter:
`beta_t = beta_{t-1} + w_t`, `y_t = x_t^T beta_t + v_t`. A label (a
candidate's realized `net_return`) is only folded into the filter once its
trade's `exit_time` has passed relative to the row currently being predicted
— causal by construction, no fold boundary needed. This means every
prediction in the run below is inherently out-of-sample relative to its own
training data, which is a stronger guarantee than the base runner's rolling
folds provide.

`experiments/run_equity_kalman_online.py` reuses the v3.0 pipeline (data
loading, features, ATR entry/exit, capital safety layer) unchanged from
`run_equity_real_data_walkforward.py` — only candidate selection changes.
Defaults used below, untuned: `process_var=1e-5`, `obs_var=1.0`,
`prior_var=1.0`, `warmup_updates=200`. Data: cached `data/real/*.csv`
(2018–2026 daily bars, AAPL/MSFT/NVDA/TSLA vs SPY, same universe as the live
config).

## Results

### Full sample (2018–2026)

| | Real labels | Shuffled labels (3 seeds) |
|---|---|---|
| Win rate | 54.6% | 48.8% / 48.8% / 46.6% |
| Total return | 4,155% | 300% / 511% / 209% |
| Max drawdown | 16.4% | 15.7–16.4% |
| Trades taken | 1,070 | ~1,300 |

The raw total-return figures are inflated by the same compounding artifact
this project's own `simple_trend` control documents (25%-of-equity
reinvestment into >1,000 sequential trades makes even a near-coin-flip win
rate compound into a large-looking percentage — see the shuffled-label
runs above, which have win rates indistinguishable from noise but still
show 2–6x). **Win rate is the metric that isn't compounding-distorted**, and
there the real run beats the label-shuffle noise floor by ~6–8 points,
consistently across 3 seeds. That's a real, repeatable separation from
noise — not proof of edge, but a legitimate first-pass signal.

### Year-by-year (real labels)

| Year | Trades | Win rate | Year return | Max DD in year |
|---|---|---|---|---|
| 2018 | 67 | 44.8% | −8.5% | 15.4% |
| 2019 | 134 | 59.7% | 85.2% | 7.9% |
| 2020 | 186 | 64.0% | 308.7% | 12.5% |
| 2021 | 132 | 54.5% | 46.2% | 14.5% |
| **2022** | **62** | **25.8%** | **−25.9%** | 16.4% |
| 2023 | 168 | 58.3% | 139.6% | 12.4% |
| 2024 | 145 | 56.6% | 77.2% | 9.9% |
| 2025 | 131 | 54.2% | 42.2% | 15.5% |
| 2026 | 45 | 35.6% | −6.0% | 15.2% |

**2022 is a clear weak spot** — win rate collapses to 25.8%, the year loses
money. 2018 is also negative. Plausible mechanism: with `process_var=1e-5`
the coefficient state adapts slowly, so a fast bear-market regime shift
leaves the filter's beta anchored to the prior bull-market fit for a while.
This is exactly the kind of finding the "2020 crash / 2022 bear / choppy
subperiod review" promotion gate exists to surface.

### Cost stress (5bps slippage each side vs. 2bps default)

Win rate 53.9% (vs. 54.6% baseline), max drawdown 15.6% — holds up
reasonably well under 2.5x transaction costs.

## Update: the label-shuffle control was too weak — corrected verdict

`experiments/run_equity_kalman_promotion_checks.py` ran the three checks
flagged as missing above. The result reverses the initial "encouraging"
reading:

### Random-selection control (the stronger null)

Label-shuffling breaks the feature/outcome relationship inside the model but
still lets the *model's own selection mechanism* pick winners. A
**random-selection control** — literally drawing a uniform-random candidate
per day from the same broad candidate pool, restricted to the same tradable
date window and passed through the same capital/safety layer — is the
control this codebase's base runner already ships (`--selection random`) for
exactly this reason, and it is the one that matters:

| | Kalman (real) | Random (3 seeds) |
|---|---|---|
| Win rate | 54.6% | 53.4% / 54.1% / 55.3% |
| Total return | 4,155% | 2,849% / 3,511% / 3,654% |

**The model is statistically indistinguishable from uninformed random
selection over the same window.** The win-rate gap over the *label-shuffle*
control (~47–49%) that looked meaningful in the first pass turns out to be
because shuffling labels is a weaker null than genuinely random selection —
shuffling still lets the fitted model's geometry (which candidates clear the
threshold at all, how the exit mechanics interact with the candidate pool)
leak into the result. Most of the apparent edge lives in the candidate
generation / ATR exit mechanics / position-sizing compounding, not in what
the Kalman filter predicts.

### Head-to-head vs. Ridge, matched date window

The earlier comparison (Kalman 41.5x vs. Ridge's reference-doc 7.9x) was not
apples-to-apples — Kalman starts trading in mid-2018 (needs only
`warmup_updates=200` raw candidate rows) while Ridge's first fold doesn't
start until it has a full `train_bars`-length (756-day) history, so Ridge
only starts trading in **April 2021**. Restricted to their shared window
(2021-04-06 to 2026-04-15):

| | Kalman | Ridge |
|---|---|---|
| Win rate | 53.0% | 54.1% |
| Total return | 562.8% | 689.4% |

Ridge modestly beats Kalman on both metrics on the identical window. No
edge from switching to online/continuous updating, at least at these
untuned filter settings.

### process_var sensitivity (2022)

Swept `1e-6` to `1e-2` (four orders of magnitude) — 2022 win rate stays
stuck in the 31–37% range and total return stays negative (−15% to −25%)
across the *entire* range. This rules out "coefficients just adapt too
slowly" as the explanation for the 2022 weakness; whatever is happening
there isn't fixed by faster/slower coefficient drift.

## Verdict

**Kill, for now.** This candidate does not clear the bar: no measurable edge
over a proper random-selection control, no improvement over the existing
Ridge baseline on a matched window, and a 2022 weakness that a four-order-of-
magnitude sweep of the one obviously relevant knob doesn't fix. Consistent
with the Track B framing this was scoped under — most candidates are
expected to fail this gate, and this process caught it before it went
anywhere near the random-selection or shuffle-label mistake nearly
overstating a result. Leaving the code and this writeup in place as a
documented negative result (and a reusable random-selection-control harness
for whatever candidate comes next), not deleting it.

## Next steps (Track B, not this candidate)

1. Try a genuinely different feature set or model family — per the
   Track A/B framing, Ridge (and now RLS/Kalman, which is mathematically a
   close cousin of Ridge) may both be exhausting what's extractable from this
   feature set; the next candidate should differ more structurally (tree
   ensembles, gradient boosting) rather than just changing how an
   already-similar linear model is fit.
2. Always run the random-selection control before the label-shuffle control
   in future candidates — it's the cheaper, stronger, more diagnostic test
   and should be the first gate, not an afterthought.
