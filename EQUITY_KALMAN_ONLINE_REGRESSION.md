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

## What this is not yet

Not a promotion-gate pass. Missing before this could be taken seriously as a
candidate: a `random`-selection control (distinct from label-shuffling — an
uninformed but real candidate pool, matching the base runner's `--selection
random`), a sweep over `process_var`/`obs_var`/`prior_var` (defaults were not
tuned — tuning now against this same data would be exactly the "optimize on
the holdout" mistake `CLAUDE.md` warns against), and a head-to-head run
against Ridge on an identical frozen data snapshot with matched-seed
comparison (like the AMZN-ablation doc's methodology).

## Next steps

1. Add a `random`-selection control to `run_equity_kalman_online.py`.
2. Investigate 2022: does raising `process_var` (faster-adapting
   coefficients) fix the regime-shift lag without degrading the stable years?
   Frame as a new hypothesis with its own out-of-sample check, not a tune.
3. Head-to-head vs. Ridge on the identical candidate pool/dates.
