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
