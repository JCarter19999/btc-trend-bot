# v0.6

Adds a research-only concurrent matrix of popular strategy families:

- 15m and 30m Donchian breakouts with ADX and ATR exits
- 5m pullback entries inside a 1h EMA/ADX trend
- 15m pullback entries inside a 4h EMA/ADX trend
- 15m Bollinger/Keltner squeeze release
- RSI(2) mean reversion with a 4h trend filter
- 15m MACD crossover with ADX and 1h trend filtering

Every strategy shares completed-candle features, next-5m-open execution, identical
cost assumptions, gross shadow accounting, chronological folds, and a cost grid.
No live trading, authenticated API, or Supabase path is included.
