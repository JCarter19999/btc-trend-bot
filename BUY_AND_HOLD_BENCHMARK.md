# Buy-and-Hold Benchmark

The synthetic exit-engine suite includes a buy-and-hold baseline over the same untouched final 20% candle interval.

The benchmark:

- buys BTC at the first test candle close,
- allocates the requested research fraction,
- marks equity on every subsequent candle,
- reports gross return,
- reports conservative realized-net return after the configured round-trip cost,
- reports candle-by-candle maximum drawdown.

Important: the delayed-edge synthetic generator has a persistent positive background drift to keep the frozen long-entry regime active. As a result, passive buy-and-hold receives an intentionally strong tailwind and is not a neutral profitability benchmark. Use this benchmark to verify comparison plumbing and exposure accounting; use real Binance data for the economically meaningful buy-and-hold comparison.
