# Adaptive Exit Engine v1

The frozen entry strategy remains `selective_long_5m_v1`. This engine changes only position management and therefore does not alter the champion entry definition.

## Priority order

1. Conservative same-bar collision: stop first.
2. Current hard or adaptive stop.
3. Profit target.
4. Breakeven stop update.
5. Chandelier ATR trailing-stop update.
6. Volatility-shock exit.
7. Momentum-decay exit.
8. EMA trend-reversal exit.
9. Maximum-hold time exit.

Hard barriers are evaluated before soft exits. Stop updates apply to subsequent candles, avoiding same-candle lookahead.

## Components

- **Initial stop:** entry minus the configured ATR multiple.
- **Profit target:** entry plus the configured ATR multiple.
- **Breakeven:** after a configurable favorable ATR excursion, raise the stop to entry plus a small offset.
- **Chandelier trailing:** after activation, trail below the recent high by a configurable ATR distance.
- **Momentum decay:** exit when recent close-to-close momentum falls below a configured threshold.
- **Trend reversal:** exit when a fast EMA of post-entry closes crosses below a slower EMA.
- **Volatility shock:** exit on an unusually large adverse candle range and a close below entry.
- **Time stop:** close any position still open after the maximum hold.

Each result records `exit_reason`, `bars_held`, MFE, MAE, whether breakeven/trailing armed, and the final dynamic stop.

## Dataset command

Adaptive exits are now the default:

```bash
docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python btc-v1 \
  -m btc_trend_bot.v1.cli build-dataset \
  --config config/v1.yaml \
  --candles data/parquet/btcusdt_5m.parquet \
  --output outputs/candidates_adaptive.parquet \
  --exit-engine adaptive
```

Use `--exit-engine fixed` to reproduce the original stop/target/time-only behavior.

## Synthetic experiment

```bash
docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python btc-v1 \
  experiments/run_exit_engine_experiments.py
```

Results are written to `outputs/exit_engine_experiments/`.
