# CLAUDE.md

Guidance for Claude Code when working in this repository. Read this first.

## What this project is

A **research-first, paper/backtest-only** quantitative trading framework. The
current focus (v3.0) is a **long-only, cross-sectional equity strategy**
evaluated with **chronological walk-forward validation**. Expected forward
returns come from a Ridge regression; entries/exits are ATR-based; a safety
layer adds drawdown pauses, loss-streak cooldowns, and a hard shutdown.

It is **not** financial advice and **not** a live trading system. No live-order
code is enabled. Do not connect real capital.

### Important lineage / naming

The project began as a Bitcoin trend bot and grew into equity research, so names
are inconsistent — be careful:

- The **extracted top-level folder** is `equity_v2_4/` (this is the repo root).
- The **installed package** is still `btc-trend-bot` (import path
  `btc_trend_bot`), version **3.0.0**. Console scripts: `btc-trend-bot` and
  `btc-v1`.
- There are **two separate config directories** — do not mix them up:
  - `configs/` (plural) → `real_data.yaml`, the **equity v3.0** config.
  - `config/` (singular) → the legacy **BTC** `settings_*.yaml` and `v1.yaml`.
- The frozen BTC strategy and the equity experiments are independent. **Equity
  work must not alter the frozen BTC strategy.**

## Environment & setup

- **Python 3.11 or 3.12** (`pyproject.toml` requires `>=3.11`; Docker images use
  `python:3.12-slim`). Dependencies are pinned in `pyproject.toml`.
- Install editable, then run tests:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pytest -q
```

Key deps: numpy, pandas, scikit-learn, pyarrow, PyYAML, matplotlib, yfinance,
ccxt, coinbase-advanced-py, streamlit, joblib.

## Primary workflow — real-data equity walk-forward (v3.0)

Entry point: `experiments/run_equity_real_data_walkforward.py`.

```bash
# Default frozen run (downloads daily bars via yfinance)
python experiments/run_equity_real_data_walkforward.py \
  --config configs/real_data.yaml \
  --provider yfinance \
  --output outputs/real_equity_walkforward

# Convenience wrapper for the above:
./scripts/run_real_backtest.sh
```

CLI flags (argparse): `--config` (default `configs/real_data.yaml`),
`--provider {yfinance,csv}` (default `yfinance`), `--csv-dir` (default
`data/real`), `--output` (default `outputs/real_equity_walkforward`),
`--download-only`.

**Download once, then run reproducibly from cache** (freezes the input set):

```bash
# 1. Download only
python experiments/run_equity_real_data_walkforward.py \
  --config configs/real_data.yaml --provider yfinance --download-only

# 2. Run from cached CSVs
python experiments/run_equity_real_data_walkforward.py \
  --config configs/real_data.yaml --provider csv --csv-dir data/real \
  --output outputs/real_equity_walkforward_cached
```

### CSV provider schema

For licensed/broker-exported data, one file per symbol at
`data/real/{SYMBOL}.csv` (e.g. `AAPL.csv`, `SPY.csv`) with columns:

```
date,open,high,low,close,volume
```

Dates identify the **completed** bar. Prices must be **consistently split/dividend
adjusted** — mixing adjusted closes with unadjusted OHLC corrupts ATR, signals,
and trade returns. Free yfinance intraday history is limited, so the runnable
default is **daily** (`interval: 1d`); use licensed CSVs for long intraday runs.

### Outputs (written under `--output`)

`candidates.parquet`, `folds.csv`, `selected_trades.csv`, `capital_path.csv`,
`results.json`.

## Pipeline (how the runner works)

1. Download/import adjusted OHLCV (yfinance or CSV).
2. `add_features` — build features **without using future rows**.
3. `simulate_trade` — label each signal with **next-bar entry** and deterministic
   **ATR stop/target** exits.
4. `build_candidates` → `walk_forward` — train a `StandardScaler → Ridge`
   pipeline on rolling chronological windows; **purge overlapping labels**.
5. Cross-sectional selection: take the highest predicted simultaneous symbol.
6. `simulate_capital` — apply the v2.9 capital safety stoppages.
7. Write reports + benchmark (SPY) comparison.

Model: regularized Ridge predicting forward expected return. **Long-only, one
active position, chronological training, no future leakage.**

Features (`FEATURES` in the runner): returns (1/2/5/10/20), `ema_spread_atr`,
`ema_slope_atr`, `atr_pct`, `realized_vol_20`, `relative_volume`,
`volume_zscore`, `distance_20d_high/low_atr`, benchmark returns (5/20),
`relative_strength_20`, `market_above_ema50`, plus continuous candlestick
geometry columns (from `btc_trend_bot.v1.candlestick_geometry`).

### Default config (`configs/real_data.yaml`)

symbols `[AAPL, MSFT, NVDA, TSLA]`, benchmark `SPY`, start `2018-01-01`,
`interval: 1d`, `auto_adjust: true`, `initial_capital: 2500`,
`position_fraction: 0.25`, `return_threshold_bps: 10`, `max_hold_bars: 10`,
`train_bars: 756`, `test_bars: 126`, `step_bars: 126`, `purge_bars: 10`,
`ridge_alpha: 10`, `stop_atr: 1.35`, `target_atr: 2.15`,
`slippage_bps_each_side: 2`. Safety: drawdown_pause 0.15,
hard_shutdown_drawdown 0.35, consecutive_loss_limit 4, cooldown_trades 8,
minimum_equity 25.

`position_fraction` caps how much of current equity is committed as notional
per sequential trade (rest sits in cash, earning nothing) — without it, one
active position means 100% of equity on every trade, which compounds any
structural edge (or pure noise, per the label-shuffle/random-selection
controls) into wildly seed-dependent, unrealistic terminal equity.

## Non-negotiable guardrails

- **No lookahead.** Signal at bar `t`, entry on bar `t+1`. A test asserts
  `entry_time == index[i+1]` — never introduce same-bar entry.
- **Options overlay is disabled on real data** unless *actual historical
  option-chain data* is supplied. Do **not** present Black-Scholes / synthetic-IV
  reconstructions as historical options performance. The stock route must remain
  independently evaluable.
- **Don't tune on the final holdout.** Treat any change to windows, thresholds,
  costs, ATR, or risk limits as a **new hypothesis**, not an optimization.
- Positive total return alone is not evidence of skill — always review drawdown,
  turnover, exposure, benchmark-relative return, year-by-year stability, and
  bootstrap confidence intervals.
- Synthetic data validates **infrastructure, not edge**.

### Promotion gates (must survive all before shadow deployment)

SPY comparison; 2020 crash / 2022 bear / choppy subperiod review; higher
slippage; nearby Ridge-threshold and ATR settings; per-symbol ablation;
randomized-entry and simple-trend controls; an untouched final period.

### Recommended first runs

1. Default (frozen) config — change nothing.
2. Higher slippage (5 bps).
3. Higher Ridge threshold (20 bps).
4. Safety disabled.
5. 2022 bear-market analysis.
6. Symbol ablation.

## Tests

29 test files in `tests/`. Pytest is configured in `pyproject.toml`
(`testpaths=["tests"]`, `pythonpath=["src"]`, `addopts="-ra"`).

```bash
pytest -q                                 # all
pytest tests/test_real_data_walkforward.py -q   # equity v3.0 path
```

## Docker

```bash
# Build
docker compose -f compose.paper-5m.yaml build --no-cache btc-v1

# Real-data backtest in-container
docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python btc-v1 \
  experiments/run_equity_real_data_walkforward.py \
  --config configs/real_data.yaml --provider yfinance \
  --output outputs/real_equity_walkforward

# Tests in-container
docker compose -f compose.paper-5m.yaml run --rm --entrypoint pytest btc-v1 -q
```

Compose files: `compose.paper-5m.yaml` (services `btc-paper-5m`, `btc-v1`),
`compose.v1.yaml`, `compose.yaml`. Dockerfiles: `Dockerfile`,
`Dockerfile.paper-5m`. Compose mounts `./outputs`, `./runtime`, `./data`,
`./models`, `./config` — outputs persist to the host.

## Repository layout

```
equity_v2_4/                 # repo root
├── src/btc_trend_bot/       # installed package (import: btc_trend_bot)
│   ├── *.py                 # legacy BTC research modules (backtest, data, strategy, …)
│   └── v1/                  # Binance adaptive v1 pipeline + candlestick_geometry
├── experiments/             # runnable research scripts (real-data runner lives here)
├── configs/real_data.yaml   # EQUITY v3.0 config  (plural dir)
├── config/                  # LEGACY BTC settings_*.yaml + v1.yaml  (singular dir)
├── scripts/run_real_backtest.sh
├── tests/                   # 29 pytest files
├── dashboard/               # Streamlit apps
├── deploy/                  # systemd units/timers (BTC services)
├── data/  outputs/  results/  models/  runtime/
└── *.md                     # extensive per-experiment docs + result logs
```

Legacy CLI entry points (BTC lineage): `btc-trend-bot` (subcommands: `download`,
`download-coinbase-history`, `backtest`, `demo`, `paper-step`, `deploy-step`,
`deploy-status`, `deploy-halt`, `deploy-resume`, `validate-short-overlay`,
`diagnose-gaps`, `data-status`) and `btc-v1`.

## Deployment roadmap (not yet live)

Phase 1 historical walk-forward validation → Phase 2 **Hetzner shadow
deployment** (log hypothetical trades, verify fills/reconciliation, no live
orders) → Phase 3 small-capital live ($50–$250, fractional shares, options off)
→ Phase 4 scaled (~$2,500, option overlay enabled). Planned Schwab market-data
integration. See `LIVE_DEPLOYMENT.md` before any brokerage connection.

## Where to look for more

`REAL_DATA_BACKTEST.md` (v3.0 scope + gates), `README.md` / `README_V2_4.md`,
and the documentation bundle (`ARCHITECTURE.md`, `MODEL_CARD.md`,
`RESEARCH_LOG.md`, `LIVE_DEPLOYMENT.md`). Per-experiment `*.md` files at the root
document each research offshoot and its results.
