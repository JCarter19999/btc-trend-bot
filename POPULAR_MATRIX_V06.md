# Popular strategy matrix v0.6

Research-only concurrent comparison of canonical intraday strategy families:
Donchian channel breakouts, EMA trend pullbacks, Bollinger/Keltner squeeze release,
RSI(2) mean reversion with a 4h trend filter, and MACD with ADX confirmation.

All strategies use completed candles and execute at the next 5m open. The same
fee/spread/slippage model, gross shadow account, cost grid, and chronological
folds are applied to every candidate. This module contains no authenticated or
live-order path.

Run:

```bash
docker compose -f compose.paper-5m.yaml run --rm --entrypoint python btc-paper-5m \
  -m btc_trend_bot.popular_matrix \
  --config config/settings_popular_matrix.yaml \
  --bars 10000 \
  --output outputs/popular_matrix_10000
```
