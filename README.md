# BTC Trend Bot — Binance Adaptive Paper Platform v2.0

The frozen Binance BTC/USDT architecture is documented in `V1_ARCHITECTURE.md`. Legacy research remains available, while the new path lives under `btc_trend_bot.v1`.

# BTC Trend Bot

A research-first, paper-trading-only Bitcoin trend system built from the reusable ideas in the `sp500-vol-estimator` project.

The strategy does **not** predict price or volatility. It uses:

- a long/flat time-series trend signal;
- a Donchian breakout confirmation;
- lagged realized volatility for position sizing;
- transaction fees and slippage;
- a volatility-shock exposure reduction;
- a portfolio drawdown circuit breaker;
- one-bar signal lag to prevent same-bar lookahead;
- a local paper account that advances only once per completed candle.

The default research timeframe is **4 hours** and the default public data source is Kraken through CCXT. No exchange keys are needed to download public candles. No live-order code is enabled.

## Architecture

```text
public OHLCV -> normalization/coverage checks -> features -> target position
             -> one-bar delayed execution -> fees/slippage -> risk controls
             -> metrics, equity curve, and paper-account state
```

## Quick start

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
python -m btc_trend_bot.cli demo
```

Download public BTC data and run the backtest:

```powershell
python -m btc_trend_bot.cli download
python -m btc_trend_bot.cli backtest
```

Advance the local paper account by one completed bar:

```powershell
python -m btc_trend_bot.cli paper-step
```

Run `paper-step` manually after a new four-hour candle closes. Later, Windows Task Scheduler can invoke the same command on a schedule.

## Outputs

- `data/btc_usd_4h.csv`: downloaded candles
- `outputs/backtest_bars.csv`: bar-by-bar research output
- `outputs/metrics.json`: strategy and benchmark metrics
- `outputs/equity_curve.png`: strategy versus BTC buy-and-hold
- `paper/paper_state.json`: local simulated account
- `paper/paper_trades.csv`: simulated fills

## Strategy definition

At the close of bar `t`:

1. Compute fast and slow EMAs.
2. Determine a trend direction.
3. Confirm upward trends with a recent Donchian breakout or positive normalized EMA spread.
4. Scale exposure using lagged realized volatility:

```text
target position = direction * min(max_position, target_vol / realized_vol)
```

5. Reduce exposure during a short-volatility shock.
6. Execute the resulting target on the next bar in the backtest.

The default mode is long/flat. This avoids pretending that a spot-style paper account can short Bitcoin without a proper margin and liquidation model.

## Research protocol

Do not tune parameters on the final holdout period. Treat changes to trend windows, breakout windows, costs, and risk limits as new hypotheses. Compare the strategy against:

- BTC buy-and-hold;
- fixed-size trend;
- realized-volatility-scaled trend;
- higher fee and slippage assumptions.

A positive total return alone is not evidence of skill. Review drawdown, turnover, exposure, benchmark-relative return, stability across years, and the block-bootstrap confidence interval.

## Important limitation

This repository is a research and paper-trading scaffold, not financial advice and not a production trading system. Public candle APIs can contain gaps or revisions. Paper fills are approximations. Do not connect real capital without independent review, exchange-specific margin modeling, reconciliation, monitoring, and kill switches.

## Continuous historical BTC-USD data

Kraken's recent OHLC endpoint cannot bridge the seam between a stale archive
and its rolling recent window. Version 0.3.0 can instead download paginated
Coinbase Exchange one-hour candles and resample complete groups into UTC-aligned
four-hour bars:

```powershell
python -m btc_trend_bot.cli download-coinbase-history
python -m btc_trend_bot.cli diagnose-gaps --data data/btc_usd_4h_coinbase.csv
python -m btc_trend_bot.cli backtest --data data/btc_usd_4h_coinbase.csv
```

See `DATA_REPAIR.md` for the complete workflow and data-quality gates.

## Research diagnostics (v0.4)

Each backtest now also writes `trades.csv` and `yearly_performance.csv`. The metrics JSON and console report include separate bootstrap tests, trade concentration, and market-capture diagnostics. See `ADDITIONAL_TESTS.md` for the frozen long/flat control and mirrored long/short experiment.

## Final short-overlay validation

Version 0.6.0 adds a frozen validation command for Variant B. It compares the selective short overlay directly with the no-breaker long/flat control using a paired moving-block bootstrap, 1x/2x/3x/5x execution-cost stress, calendar-year held-out scoring, and an optional second-exchange replication file.

```powershell
python -m btc_trend_bot.cli `
  validate-short-overlay `
  --control-config config/settings_no_breaker.yaml `
  --candidate-config config/settings_short_b_regime.yaml `
  --data data\btc_usd_4h_coinbase.csv
```

Optional replication:

```powershell
python -m btc_trend_bot.cli `
  validate-short-overlay `
  --control-config config/settings_no_breaker.yaml `
  --candidate-config config/settings_short_b_regime.yaml `
  --data data\btc_usd_4h_coinbase.csv `
  --replication-data data\btc_usd_4h_second_exchange.csv
```

Outputs are written under `outputs/validation/`.

## Selective reversion v0.8

The selective RSI research matrix now uses a two-stage setup state machine:

1. an oversold RSI event arms a setup;
2. a later recovery confirms the entry.

The recovery candle no longer has to remain oversold. The economic move hurdle
is reported diagnostically instead of blocking entries, the v0.6 control is
routed through the original `popular_matrix` RSI2 implementation, and
`gate_counts.csv` reports setup, recovery, regime-veto, volatility-veto, and
entry counts.


## Binance adaptive v1 Compose workflow

The new Binance pipeline can be run using the same disposable Compose pattern as the existing matrix backtests. Build the `btc-v1` service and invoke `python -m btc_trend_bot.v1.cli` with a subcommand:

```bash
docker compose -f compose.paper-5m.yaml build --no-cache btc-v1

docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-v1 \
  -m btc_trend_bot.v1.cli \
  build-dataset \
  --config config/v1.yaml \
  --candles data/parquet/btcusdt_5m.parquet \
  --output outputs/candidates.parquet
```

See `COMPOSE_V1_WORKFLOW.md` for download, training, and promotion commands.

## Synthetic equities offshoot

A separate AAPL + TSLA + MSFT 30-minute research experiment is documented in `EQUITY_THREE_SYMBOL_SYNTHETIC.md` and `EQUITY_OFFSHOOT_RESULTS.md`. It uses continuous candlestick geometry only and does not alter the frozen BTC strategy.

## v2.7 broad-universe additive overlay

The latest synthetic experiment expands the active universe to AAPL, MSFT, TSLA, and NVDA and implements true additive stock-plus-call accounting. See `EQUITY_BROAD_UNIVERSE_ADDITIVE_OVERLAY.md` and `EQUITY_BROAD_UNIVERSE_ADDITIVE_OVERLAY_RESULTS.md`.

## v2.9.0 regime stress finalization

The capital-constrained equity route now includes a safety challenger with drawdown pauses, loss-streak cooldowns, an option-loss pause, a long-only regime gate, and a hard shutdown. Run:

```bash
python experiments/run_equity_regime_stress_final.py \
  --initial-capital 2500 \
  --option-budget-fraction 0.30 \
  --output outputs/equity_regime_stress_final
```

See `EQUITY_REGIME_STRESS_FINAL.md` and the generated stress results for synthetic bear, choppy, crash/recovery, and clustered-loss tests.

## v3.0 real-equity walk-forward

See `REAL_DATA_BACKTEST.md`.

```bash
python experiments/run_equity_real_data_walkforward.py \
  --config configs/real_data.yaml \
  --provider yfinance \
  --output outputs/real_equity_walkforward
```
