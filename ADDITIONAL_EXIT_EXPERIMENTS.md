# Additional synthetic exit experiments

The experiment compares fixed 15-minute, 1-hour, 4-hour, and 1-day exits against three adaptive profiles using the same planted delayed-edge candle environment.

The balanced adaptive profile produced the strongest result in the initial run:

- Accepted expectancy: **+7.36 bps/trade**
- Acceptance rate: **27.6%**
- Median hold: **18 five-minute bars (90 minutes)**
- Accepted favorable fraction: **100%**
- Filtered maximum drawdown: **0.058%** in the synthetic test block

The fixed one-day engine accepted fewer trades and produced approximately **+2.18 bps/trade**, while the fixed shorter horizons accepted none. These are validation results for engine behavior, not evidence of a real BTC edge.

The next real-data sweep should tune exit parameters only inside walk-forward training/validation folds and compare the entire exit policy as a versioned challenger.

## Research-only 100% allocation rerun

The experiment runner accepts a research-only allocation override without changing `config/v1.yaml` production risk limits:

```bash
docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-v1 \
  experiments/run_exit_engine_experiments.py \
  --allocation 1.0 \
  --output outputs/exit_engine_experiments_alloc100
```

This override changes portfolio compounding and drawdown only. Candidate labels, model predictions, acceptance decisions, and mean return per accepted trade remain unchanged.

## Buy-and-hold benchmark

The experiment suite now includes a buy-and-hold benchmark over the same untouched final 20% candle interval. It buys BTC at the first test candle close, holds through the final candle, and reports both gross performance and conservative realized-net performance after the configured round-trip cost. The benchmark uses candle-by-candle mark-to-market equity for maximum drawdown.
