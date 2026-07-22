# v1.0 Hyperparameter Lab

## Original state goal

The immutable baseline is the best v0.9 result:

```yaml
exit_mode: time_capture
target_bps: 25
stop_bps: 65
minimum_hold_bars_5m: 0
maximum_hold_bars_5m: 144
```

It remains in every report under `original_goal_time144_t25_s65`. Tuning does not silently replace or rewrite it.

## What is faster now

The feature frame is downloaded and constructed once. The tuner then uses deterministic random search and successive halving:

1. All candidates run on the first 60% of scored history.
2. Only the top candidates advance to the next 20% validation period.
3. Only a small finalist set is opened on the final 20% test period.

This is considerably cheaper than evaluating every Cartesian combination over all 50,000 bars. It also makes selection overfit visible.

## How to observe overfitting

`overfit_report.json` compares:

- the candidate that looked best on training data;
- its later validation and test behavior;
- the candidate selected using validation;
- the fixed v0.9 baseline.

A dramatic score or return decline from train to test is the behavior you are curious about. Do not tune again using the reported test period; doing that converts the test set into another validation set.

## Exit families explored

- `time_capture`: fixed target, wide stop, and maximum holding time.
- `breakeven_lock`: wide initial stop, fixed target, then a causal profit lock once favorable excursion reaches an activation threshold.
- `activated_trailing`: wide initial stop and a price trail that begins only after a favorable move.

The v0.8 strict 15-minute entry template remains fixed.

## Quick search

```powershell
docker compose -f compose.paper-5m.yaml run --rm `
  --entrypoint python `
  btc-paper-5m `
  -m btc_trend_bot.hyperparameter_lab `
  --config config/settings_hyperparameter_lab.yaml `
  --bars 50000 `
  --trials 40 `
  --output outputs/hyperparameter_quick_50000
```

## Main search

```powershell
docker compose -f compose.paper-5m.yaml run --rm `
  --entrypoint python `
  btc-paper-5m `
  -m btc_trend_bot.hyperparameter_lab `
  --config config/settings_hyperparameter_lab.yaml `
  --bars 50000 `
  --trials 120 `
  --output outputs/hyperparameter_main_50000
```

## Rerun the selected champion as a normal exit lab

```powershell
docker compose -f compose.paper-5m.yaml run --rm `
  --entrypoint python `
  btc-paper-5m `
  -m btc_trend_bot.exit_lab `
  --config outputs/hyperparameter_main_50000/champion_exit_lab.yaml `
  --bars 50000 `
  --output outputs/champion_confirmation_50000 `
  --compact
```

## Option space after this search

1. Expand the historical sample. This is more valuable than continually widening the local parameter grid.
2. Use anchored walk-forward windows, selecting on prior history and evaluating only the next window.
3. Bootstrap complete trades to estimate uncertainty around average gross bps and break-even cost.
4. Tune entry rules only after the exit family stabilizes, and never tune entry and exit simultaneously on the same small sample.
5. Paper trade the frozen champion prospectively. That is the cleanest new-data test.
