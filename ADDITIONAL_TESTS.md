# Additional strategy tests

Version 0.4.0 keeps the successful long/flat, no-breaker configuration as the control and adds a mirrored long/short ablation.

## 1. Re-run the frozen long/flat control

```powershell
python -m btc_trend_bot.cli `
  --config config/settings_no_breaker.yaml `
  backtest `
  --data data/btc_usd_4h_coinbase.csv

New-Item -ItemType Directory -Force results\long_flat_no_breaker
Copy-Item outputs\* results\long_flat_no_breaker\
Copy-Item config\settings_no_breaker.yaml results\long_flat_no_breaker\settings.yaml
```

## 2. Run the mirrored long/short ablation

```powershell
python -m btc_trend_bot.cli `
  --config config/settings_long_short.yaml `
  backtest `
  --data data/btc_usd_4h_coinbase.csv

New-Item -ItemType Directory -Force results\long_short_no_breaker
Copy-Item outputs\* results\long_short_no_breaker\
Copy-Item config\settings_long_short.yaml results\long_short_no_breaker\settings.yaml
```

The short rule is deliberately symmetric: bearish EMA alignment plus negative trend strength or a downside breakout. No parameters were tuned specifically for shorts.

## New outputs

- `outputs/trades.csv`: entry, exit, side, duration, return, and P&L for both sizing arms.
- `outputs/yearly_performance.csv`: yearly return, volatility, Sharpe, and drawdown.
- `outputs/metrics.json`: separate bootstrap tests, trade summaries, and market-capture diagnostics.

## Decision rule

Do not adopt shorting merely because total return rises. It should improve at least one robust risk-adjusted measure without creating unacceptable short-side drawdowns, turnover, or concentration in a few trades. Compare fixed-size long/flat with fixed-size long/short first; continuous volatility scaling remains an ablation.
