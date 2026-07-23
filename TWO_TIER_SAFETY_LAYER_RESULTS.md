# Two-Tier Safety Layer -- Results

Companion to `TWO_TIER_SAFETY_LAYER.md`. Data: Coinbase 4h BTC-USD,
2018-01-01 through 2026-07-22 (18,729 bars), same frozen strategy signal as
production (`config/settings_production.yaml`), only the risk-control layer
varies. Scripts: `experiments/run_two_tier_safety_backtest.py`,
`run_two_tier_safety_sensitivity.py`, `run_two_tier_safety_oos_validation.py`.
Raw outputs: `outputs/two_tier_safety_backtest/`,
`outputs/two_tier_safety_sensitivity/`, `outputs/two_tier_safety_oos_validation/`.

## Full-history comparison (2018-2026)

| | Total return | Max drawdown | Sharpe | Calmar | Exposure |
|---|---:|---:|---:|---:|---:|
| Buy-and-hold BTC | 390% | -81.6% | 0.61 | 0.25 | 100% |
| Baseline (frozen, no breaker) | 619% | -38.5% | 1.03 | 0.67 | 46.7% |
| Two-tier safety (first-pass default) | 600% | -31.1% | 1.04 | 0.82 | 44.9% |

The active strategy, either variant, dominates buy-and-hold on both return and
risk. The safety layer's question is a refinement on top of an
already-working strategy, not a repair of a broken one.

**Paired moving-block bootstrap** (per-bar return difference, safety minus
baseline, same method as this project's own Variant B validation): mean
diff -2.0e-6/bar, 95% CI [-1.64e-5, +1.56e-5], **not distinguishable from
zero**. The ~19bp/year return difference over 8 years is noise.

**Where the drawdown benefit concentrates** -- 2022, the worst year in the
sample:

| Year | Baseline return | Safety return | Baseline max DD | Safety max DD |
|---|---:|---:|---:|---:|
| 2018 | -22.0% | -22.3% | -23.7% | -17.7% |
| 2019 | +86.8% | +80.7% | -25.2% | -15.5% |
| 2020 | +139.4% | +120.6% | -20.0% | -15.7% |
| 2021 | +27.0% | +35.9% | -22.4% | -15.9% |
| **2022** | **-28.7%** | **-22.6%** | **-38.1%** | **-15.5%** |
| 2023 | +63.9% | +58.5% | -38.5% | -15.6% |
| 2024 | +42.6% | +40.6% | -27.7% | -16.2% |

The hard-shutdown tier **never fired** at any tested threshold from 20% to
45% -- the soft pause alone kept realized drawdown under control. That means
the hard tier is currently unvalidated by real data; it would need a
synthetic crash scenario, not a historical backtest, to actually exercise
that code path.

## Parameter sensitivity (one-at-a-time from the default)

Two clear landmines and one inert zone, all worth knowing before trusting any
single point estimate:

- **`consecutive_loss_limit_bars=4` is a landmine**: return collapses to 225%
  (vs 600-690% elsewhere) -- a 4-bar losing streak happens constantly from
  noise, so the strategy sits out ~2.5 of the 8 years. Above ~12 bars the
  trigger goes **inert** (never fires; results become identical to the
  drawdown-pause-only case). Usable range: roughly 6-10 bars.
- **`drawdown_pause=0.10` is a landmine**: return 297%, drawdown -44.6% --
  *worse* than the no-breaker baseline on both counts. Neighboring values
  (0.075, 0.125) are fine. This is whipsaw sensitivity to the exact
  threshold, not a smooth relationship -- a warning against picking a single
  "best" value without checking neighbors.
- **`cooldown_bars` degrades past ~60 bars**: 90 and 120 bars (15-20 days)
  both cost real return (490%, 465%) from sitting out recoveries too long.
  10-45 bars all perform comparably.
- **`hard_shutdown_drawdown` is inert from 0.20 to 0.45** -- identical
  results at every value tested, because the soft pause already prevents
  drawdown from approaching even the tightest threshold.

Full grid: `outputs/two_tier_safety_sensitivity/sensitivity_results.csv`.

## Out-of-sample validation

Train = 2018-2021 (includes the 2018 crash, **not** 2022). Test = 2022-2026
(includes the 2022 bear + recovery -- the exact period motivating this whole
idea). Parameters were chosen by inspecting only the train-period sweep for a
*stable neighborhood*, not the best point, then frozen before test was
touched: `drawdown_pause=0.15`, `consecutive_loss_limit_bars=8`,
`cooldown_bars=30` (see `run_two_tier_safety_oos_validation.py` for the
reasoning trail against each parameter's train-only sweep).

| Train (2018-2021) | Return | Max DD | Sharpe | Calmar |
|---|---:|---:|---:|---:|
| Buy-and-hold | 243% | -81.6% | 0.79 | 0.44 |
| Baseline (no breaker) | **346%** | **-25.2%** | **1.53** | **1.80** |
| Two-tier safety (chosen) | 323% | -25.5% | 1.49 | 1.71 |

| Test (2022-2026, held out) | Return | Max DD | Sharpe | Calmar |
|---|---:|---:|---:|---:|
| Buy-and-hold | 41% | -67.3% | 0.41 | 0.12 |
| Baseline (no breaker) | 61% | -31.3% | 0.55 | 0.35 |
| Two-tier safety (chosen) | **70%** | **-28.7%** | **0.61** | **0.43** |

## Interpretation

On train, the chosen configuration is a near-wash -- marginally *worse* than
baseline on all three metrics. That is not flattering, but it is the
reassuring direction: an overfit configuration looks great in-sample and
degrades out-of-sample; this one did the opposite. On the held-out
2022-2026 window it never saw during selection, it beat the frozen no-breaker
baseline on return, drawdown, *and* Sharpe simultaneously, in exactly the
crash-and-recovery period this idea was motivated by. Both variants
dominated simple buy-and-hold throughout, in both periods.

This does not prove the layer will help the next crash -- one historical
out-of-sample period is one data point, and the hard-shutdown tier remains
completely unexercised by real data. It does show: the two-tier design is not
a fragile backtest artifact, it survives a genuine train/test split, and the
return cost when it doesn't help is statistically indistinguishable from
zero. Per `TWO_TIER_SAFETY_LAYER.md`, cost-stress and a synthetic crash
scenario are the next gates before any promotion, and nothing here is wired
into the frozen production path in the meantime.
