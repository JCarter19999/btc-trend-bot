# Binance v1 Compose workflow

The Binance v1 research pipeline supports the same disposable Docker Compose workflow used by the legacy matrix backtests. The dedicated service is `btc-v1`.

## Build

```bash
docker compose -f compose.paper-5m.yaml build --no-cache btc-v1
```

## Download historical Binance candles

```bash
docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-v1 \
  -m btc_trend_bot.v1.cli \
  download \
  --config config/v1.yaml \
  --start 2020-01-01 \
  --output data/parquet/btcusdt_5m.parquet
```

## Build the candidate-trade dataset

```bash
docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-v1 \
  -m btc_trend_bot.v1.cli \
  build-dataset \
  --config config/v1.yaml \
  --candles data/parquet/btcusdt_5m.parquet \
  --output outputs/candidates.parquet
```

## Train a challenger

```bash
docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-v1 \
  -m btc_trend_bot.v1.cli \
  train \
  --config config/v1.yaml \
  --dataset outputs/candidates.parquet \
  --model-id challenger_2026_08_17 \
  --output models/challenger_2026_08_17
```

## Promote an eligible challenger

```bash
docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-v1 \
  -m btc_trend_bot.v1.cli \
  promote \
  --config config/v1.yaml \
  --model models/challenger_2026_08_17 \
  --approved-by YOUR_NAME \
  --reason "Passed validation, paper, and shadow gates"
```

## Why retain this pattern?

- Each run starts from a known image and dependency set.
- Host data and outputs persist through bind mounts.
- Research commands cannot accidentally leave a long-running container behind.
- The command maps closely to the existing `slope_matrix` workflow.
- Hetzner can use the same image and Compose file later.

The legacy command still works unchanged with the `btc-paper-5m` service. The new service only adds the storage mounts required by Binance datasets and model artifacts.
