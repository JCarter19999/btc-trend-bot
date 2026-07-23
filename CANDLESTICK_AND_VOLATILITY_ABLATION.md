# Candlestick geometry and volatility-aware exits

The v1 feature pipeline now includes continuous candle geometry plus deterministic pattern flags. Named patterns are treated as features, never as assumed trading rules. The included ablation compares the original feature set against the expanded set across bull, bear, sideways, high-volatility, and regime-switching synthetic paths.

The adaptive exit engine now widens trailing distance, delays soft exits, loosens momentum-decay thresholds, and raises the volatility-shock threshold as the detected volatility factor increases. Hard stop and target priority are unchanged.

Run:

```bash
docker compose -f compose.paper-5m.yaml run --rm --entrypoint python btc-v1 \
  experiments/run_candlestick_volatility_ablation.py \
  --bars 80000 --allocation 1.0 \
  --output outputs/candlestick_volatility_ablation
```
