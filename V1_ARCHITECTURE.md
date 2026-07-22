# Binance BTC/USDT Adaptive Paper Platform v1

## Frozen contract

- Global Binance spot, BTCUSDT, long/flat, no leverage.
- `selective_long_5m_v1` entry logic is immutable.
- ATR stop and target; maximum hold 144 five-minute bars.
- Same-candle stop/target collisions resolve stop-first and are flagged.
- Maximum allocation and BTC exposure are 10% of marked equity; one position.
- The calibrated trade-quality model accepts only when expected **net** return exceeds 5 bps.
- Sunday retraining occurs only after 50 newly matured candidate trades.
- The production champion never mutates. Challengers require all validation, paper, shadow, and manual-promotion gates.

## Storage

Parquet stores historical candles, features, candidate labels, folds, and research artifacts. SQLite stores operational state, decisions, scheduler activity, risk events, model registry, and promotion audit.

## First-run commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

btc-v1 download --start 2020-01-01 --output data/parquet/btcusdt_5m.parquet
btc-v1 build-dataset --candles data/parquet/btcusdt_5m.parquet --output data/parquet/candidates.parquet
btc-v1 train --dataset data/parquet/candidates.parquet --model-id challenger_YYYY_MM_DD --output models/challenger_YYYY_MM_DD
```

Promotion is intentionally blocked until every required gate in `model_manifest.json` is true:

```bash
btc-v1 promote --model models/challenger_YYYY_MM_DD --approved-by YOUR_NAME --reason "passed all gates"
```

## Important limitation

This repository supplies the research, data, model, risk, paper-ledger, scheduler, model-registry, and promotion foundations. Authenticated Binance order submission exists only as a thin client method and must remain disabled until testnet credentials, symbol-filter validation, order reconciliation, and prospective paper/shadow gates are complete.
