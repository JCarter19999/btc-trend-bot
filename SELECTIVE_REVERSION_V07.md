# Selective Reversion Matrix v0.7

This release narrows the v0.6 result instead of adding more unrelated indicators.
It tests whether RSI(2)'s positive gross behavior survives after reducing turnover
with 4h/daily regime filters, volatility gates, two-tick recovery confirmation,
economic move hurdles, cooldowns, longer exits, and ATR risk controls.

Use `--compact` for 50k+ runs. It omits the very large feature/equity CSVs and
charts that exhausted the 2 GB server while retaining summaries, transactions,
episodes, signal counts, and cost sensitivity.

Recommended smoke test:

```bash
docker compose -f compose.paper-5m.yaml run --rm --entrypoint python btc-paper-5m \
  -m btc_trend_bot.selective_reversion_matrix \
  --config config/settings_selective_reversion.yaml \
  --bars 10000 --output outputs/selective_reversion_10000 --compact
```

Main validation:

```bash
docker compose -f compose.paper-5m.yaml run --rm --entrypoint python btc-paper-5m \
  -m btc_trend_bot.selective_reversion_matrix \
  --config config/settings_selective_reversion.yaml \
  --bars 50000 --output outputs/selective_reversion_50000 --compact
```
