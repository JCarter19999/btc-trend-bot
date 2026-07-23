# Five synthetic market experiments

Run with 100% research allocation:

```bash
docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python btc-v1 \
  experiments/run_five_market_experiments.py \
  --allocation 1.0 \
  --bars 80000 \
  --output outputs/five_market_experiments
```

The experiment creates bull, bear, sideways, high-volatility, and regime-switching OHLCV paths. Each contains the same planted favorable/unfavorable candidate structure and uses the frozen entry logic, actual feature pipeline, ATR exits, adaptive exit engine, chronological model fitting, 29-bps round-trip costs, and buy-and-hold benchmark.

See `outputs/five_market_experiments/summary.csv`, `returns_pivot.csv`, and `drawdowns_pivot.csv`.

The synthetic paths are validation environments, not estimates of real BTC profitability.
