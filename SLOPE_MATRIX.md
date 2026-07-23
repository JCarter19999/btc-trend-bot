# Rolling-Slope Inversion Matrix v0.4

This is a research-only, spot BTC/cash matrix. It estimates the derivative of
trailing **log closes** and trades persistent slope inversions at 5m, 15m, 30m,
and 1h signal horizons. It never submits exchange orders.

## Mathematical signal

For the last `k` complete signal candles, fit a degree-1 or degree-2 polynomial
in time to log price. The signal is the fitted derivative at the newest candle.
For degree 2, the derivative is evaluated at the right endpoint; it is not a
centered Savitzky-Golay filter, so future candles are never used.

The derivative is normalized by realized one-candle log-return volatility:

```text
slope_score = endpoint_log_slope / rolling_log_return_std
```

A separate fit R² and derivative t-stat are retained as signal-quality fields.
The t-stat is used as a score, not as a claim that market returns satisfy iid OLS
assumptions.

## Caveats implemented

- Complete candles only; execution occurs at the next 5m open.
- Trailing endpoint fits only; no centered-window lookahead.
- Log prices instead of dollar-price slopes.
- Volatility-normalized slopes so thresholds scale across price regimes.
- Hysteresis: entry threshold is positive and exit threshold is negative.
- Persistence is counted only when a new signal-timeframe candle completes.
- 2-tick versus 3-tick 15m variants isolate confirmation lag.
- R² and slope t-stat gates reject poorly formed trends.
- Optional acceleration agreement for the quadratic candidate.
- A cost hurdle rejects slopes whose implied move over the expected hold does
  not exceed 1.25 times the modeled round-trip friction.
- Minimum holds, cooldowns, maximum holds, and emergency stops constrain churn.
- Gross and net shadow portfolios, cost sensitivity, folds, MAE/MFE, turnover,
  time in market, and inversion-forward-return diagnostics are produced.

## Research matrix

- `slope_5m_fast_3tick`: highest-tempo stress candidate.
- `slope_15m_fast_2tick`: primary higher-tempo candidate.
- `slope_15m_fast_3tick`: persistence/lag comparison.
- `slope_15m_quadratic_2tick`: endpoint derivative plus curvature agreement.
- `slope_30m_balanced_2tick`: slower intraday comparator.
- `slope_1h_reference_2tick`: reference only.
- Cash and buy-and-hold benchmarks.

The parameter set is intentionally small and predetermined. Do not select a
winner from one 10,000-bar run and then tune its thresholds against that same
sample.

## Run

```bash
docker compose -f compose.paper-5m.yaml build --no-cache

docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-paper-5m \
  -m pytest -q

docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-paper-5m \
  -m btc_trend_bot.slope_matrix \
  --config config/settings_slope_matrix.yaml \
  --bars 10000 \
  --output outputs/slope_matrix_10000
```

Then run 50,000 bars unchanged if the smoke test is operationally sound.

## Outputs

- `research_summary.json`
- `strategy_comparison.csv`
- `cost_sensitivity.csv`
- `chronological_folds.csv`
- `inversion_forward_returns.csv`
- `trade_episodes.csv`
- `transactions.csv`
- `signal_counts.csv`
- `slope_feature_frame.csv`
- `equity_curve.png`
- `drawdown_curve.png`
