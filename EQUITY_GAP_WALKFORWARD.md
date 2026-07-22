# Equity Gap, Holding-Window, and Walk-Forward Experiment

This research-only experiment extends the AAPL/TSLA/MSFT synthetic branch with:

- quarterly earnings-like overnight shocks,
- explicit gap-through-stop and gap-through-target fills at the next session open,
- one-, two-, and five-session maximum holds,
- 5, 10, and 15 basis-point acceptance hurdles,
- baseline versus continuous candlestick geometry features,
- Ridge and histogram-gradient-boosting challengers,
- four expanding chronological walk-forward folds,
- one-position cross-sectional portfolio selection.

Run with:

```bash
docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-v1 \
  experiments/run_equity_gap_walkforward_experiment.py \
  --sessions 1300 \
  --output outputs/equity_gap_walkforward
```

This remains a planted synthetic capability test. It is not evidence of real AAPL, TSLA, or MSFT profitability.
