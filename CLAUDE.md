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

## Shadow paper deployment #4: European lead signal (2026-07-23, NOT promoted)

`experiments/run_european_signal_shadow_step.py` — a fourth Phase 2 shadow
deployment, but deliberately **not** in the table below and **not** on the
dashboard/rankings/daily email yet, per Joey's explicit instruction:
"only if positive we'd throw it onto the streamlit dashboard and include
it against the rankings and in the daily email summary digest." Check via
CLI only until it earns its way in:
```bash
python experiments/run_european_signal_shadow_step.py status
```

Signal: DAX's cumulative return from its own session open through the
last bar closed before US open (13:30 UTC) sets direction; two
eligibility filters (DAX move in its own top quartile; a 3-index Asian
magnitude composite in its top quartile), computed as **expanding
percentiles using only prior days** — bootstrapped from
`data/european_signal_percentile_seed.csv` (629 days of real historical
DAX/Asian data, 2023-09 to 2026-07, the same series validated in
`EUROPEAN_LEAD_US_FIRST_HOUR_STUDY.txt`/`_BACKTEST.txt` on the research
branch) so the threshold is meaningful from day one — but every trade's
entry/exit price is 100% live, real-time, forward-only starting
2026-07-24; nothing about the trade itself is historical or fabricated.
Trades SPY and QQQ, one hour, no live orders. Two systemd timers:
`european-signal-shadow-entry.{service,timer}` (13:32 UTC, logs entry
quotes) and `european-signal-shadow-exit.{service,timer}` (14:32 UTC,
logs exit + realized return at 1/2/5bps cost tiers).

**Scope note**: the research request (`EUROPEAN LEAD SIGNAL... SYNTHETIC
RESULTS + IMPLEMENTATION INSTRUCTIONS`, 2026-07-23) specified a much
larger 19-part system (10 statistical controls, 6 entry-delay scenarios,
options shadow overlay, full dashboard panel, unit tests, daylight-saving
audit reports). Built the essential core only — no-lookahead signal
computation, real quote-based entry/exit, cost-tier tracking, data-quality
flagging — and deferred the rest rather than build all of it at lower
quality. The "synthetic Monte Carlo sensitivity" results in that request
were not independently produced or verified in this session; the request
itself correctly frames them as not proof ("the real-data backtest and
live paper results remain the source of truth"), so they weren't relied
on here — this shadow deployment stands on the real-data backtest results
in `EUROPEAN_LEAD_US_FIRST_HOUR_BACKTEST.txt` alone.

**Promotion bar before this touches the dashboard, rankings, or daily
email** (informal, not the full 14-point gate list from that request):
enough completed trades to mean something (dozens, not a handful),
positive expectancy net of the 2bp cost tier, and the top-quartile arms
beating the daily arm the way the backtest predicted. Until then this is
CLI-only and absent from every other user-facing surface.

## Live paper deployments: three, running in parallel (2026-07-23)

Three independent live paper deployments run in parallel, each with its own
config and SQLite ledger — **they never share state**. `ARCHITECTURE.md`,
`MODEL_CARD.md`, `RESEARCH_LOG.md`, `LIVE_DEPLOYMENT.md` are still stale
references in the old "Where to look for more" section below (none exist in
the repo) — but the dashboard itself is now real: `/home/joey/btc-dashboard`
(separate project, served by `btc-dashboard.service` on :8501) has tabs for
all three, added 2026-07-23. Check via CLI too if needed:

| | Ridge (**live control arm**, not a candidate) | exit-regime / simple_trend (**primary stock**) | call options (**primary options**) |
|---|---|---|---|
| Deployment config | `config/settings_equity_paper_yfinance.yaml` | `config/settings_equity_paper_yfinance_simpletrend.yaml` | `config/settings_equity_paper_calls.yaml` |
| Strategy config | `config/schwab_paper_strategy.yaml` | `config/simple_trend_exit_regime_strategy.yaml` | `config/call_volatile_universe_strategy.yaml` |
| Ledger | `runtime/equity_yfinance_paper.sqlite3` | `runtime/equity_yfinance_paper_simpletrend.sqlite3` | `runtime/equity_call_paper.sqlite3` |
| systemd | `equity-paper-yfinance.{service,timer}` | `equity-paper-yfinance-simpletrend.{service,timer}` | `equity-paper-calls.{service,timer}` |
| Schedule | 22:30 UTC (after close) | 22:32 UTC (after close) | **19:00 UTC (during market hours)** |
| Universe | AAPL/MSFT/NVDA/TSLA | AAPL/MSFT/NVDA/TSLA | TSLA/COIN/MSTR/PLTR/GME (more volatile — see rationale below) |
| Selection | `ridge` | `simple_trend` | `simple_trend` |
| Expression | stock, ATR stop 1.35 / target 2.15 | stock, exit disabled (stop_atr/target_atr=100) — always full `max_hold_bars=10`, `time_exit` | 30-DTE ~5%-OTM call, real live bid/ask via yfinance, sold at 10-day hold |
| Status | halted 2026-07-23, **resumed the same day** as a live comparison arm | active since 2026-07-23 | active since 2026-07-23 |

### Intraday quote refresh (stock bots only)

The two stock bots only make trading decisions once/day (after-close daily
bar). Nothing refreshed a live price between those steps, so the dashboard
showed an entry price and nothing else until the next day's decision —
noticed and fixed 2026-07-23. `run_equity_paper_step.py refresh-quote`
(separate from `step`) updates `open_position.current_price`/
`current_price_at` only, never touches entry/exit/stop logic, on its own
timer: `equity-quote-refresh-simpletrend.{service,timer}` and
`equity-quote-refresh-ridge.{service,timer}`, every 15min, 13:00–21:00 UTC
weekdays. Dashboard shows current price + unrealized % on both stock tabs
now (previously only the calls tab had a live mark). The calls bot didn't
need this — it already re-marks `current_mark` on every `step()` call.

### Daily 7am automated checkup + email

`scripts/daily_checkup.py` (`equity-daily-checkup.{service,timer}`, 07:00
UTC daily) checks all three deployments' status, timer/service health, and
recent journalctl errors against known failure patterns (stuck
`pending_entry` > 1 day old, halts, stale/inactive timers, dashboard down),
self-heals exactly one thing (re-running `step()` to advance a stuck
`pending_entry` — the same fix applied manually the day this was built),
and emails a summary (HTML + plain text, with a rule-based "Claude's Hot
Take" section grounded in this doc's own validation history) to
josephbruno189@gmail.com, cc joseph.c.bruno@lmco.com.

**Deterministic Python, not an agentic session, deliberately.** First
attempt used a headless `claude -p` call, but (a) the workspace had never
passed Claude Code's interactive trust dialog, so its permission settings
were ignored and it hung with no TTY to answer a prompt, and (b) spawning a
second full-permission Claude Code instance from inside an already-running
session is correctly refused by the sandbox classifier — not something to
route around. A plain script covers the same checks without either problem
and is more predictable for an unattended job.

**Email delivery is SMTP, not the Gmail MCP connector** — that connector
(used interactively) can only create drafts, not send; useless unattended
(a test draft sat unsent in Gmail's Drafts folder). `scripts/send_email_smtp.py`
uses `smtplib` against Gmail with an app password from
`/home/joey/.config/btc-trend-bot/gmail_smtp.env` (not committed;
`GMAIL_SMTP_USER`/`GMAIL_SMTP_APP_PASSWORD`/`GMAIL_SMTP_TO`/`GMAIL_SMTP_CC`).
**Port 465 (implicit TLS) is blocked outbound on this VM; 587 (STARTTLS)
works** — confirmed by direct `/dev/tcp` testing, not assumed.

### Why the call deployment runs on a different schedule and universe

`experiments/run_equity_call_paper_step.py`, from the research branch's
`EQUITY_OPTIONS_DEEP_DIVE.md` (5% OTM / 5% spread synthetic call roughly
tied stock-only on return while beating it on profit factor and drawdown —
but the result was highly sensitive to the *assumed* bid-ask spread, which
that synthetic study couldn't pin down). This live deployment removes that
assumption entirely — it reads **real live option quotes** (`yfinance`
`option_chain()`) instead of Black-Scholes pricing, so it's the actual
resolution of that open question, not another backtest. Two consequences:
it **must run during market hours** (option bid/ask read `0.0` outside the
regular session, unlike the stock bots which only need the completed daily
bar — see the quote-source fallback logic in that module), and it runs on
a genuinely more volatile universe (TSLA/COIN/MSTR/PLTR/GME) since options
economics depend heavily on volatility and the original basket is unusually
low-vol. `safety_enabled`/`hard_shutdown_drawdown` matter differently here
too — a real bug was found and worked around in the backtest version
(`hard_shutdown_drawdown` ignores `safety_enabled`, see
`EQUITY_OPTIONS_DEEP_DIVE.md`); this live version uses a lighter,
premium-budget-scaled safety check (`_safety_allows_new_entry` in that
module) sized against the fixed $250-per-trade premium risk, not stock
notional.

Check either bot's real status with:
```bash
python experiments/run_equity_paper_step.py --config config/settings_equity_paper_yfinance.yaml status              # Ridge
python experiments/run_equity_paper_step.py --config config/settings_equity_paper_yfinance_simpletrend.yaml status   # simple_trend
```

**Ridge was resumed, not re-endorsed.** It failed re-validation (below) —
resuming it was a deliberate choice to keep collecting live paper-trading
data as an ongoing real-world comparison against the exit-regime strategy,
since it costs nothing (paper only) and gives an independent check on
whether the backtest reassessment holds up live. Its future trades are a
control-arm data point, not evidence it's viable again — don't let "it's
running and posting P&L" substitute for the validation it already failed.

### Why Ridge was originally halted (kept for the record)

`experiments/run_equity_paper_step.py status` against the Ridge config
showed `"halted": true` for several hours on 2026-07-23. That was deliberate,
not an outage. Full detail
in `EQUITY_KALMAN_ONLINE_REGRESSION.md` and `EQUITY_EXPECTANCY_MATRIX_FINDINGS.md`
on the `research` branch (`/home/joey/equity_v2_4_research`) — summary:

`simulate_capital` in `run_equity_real_data_walkforward.py` (the backtest
harness every promotion decision, including this model's original one, was
based on) has a real bug: it processes selected trades sequentially without
checking whether the previous trade's exit_time has passed, so 77.3% of
trades in practice overlap the prior one, implicitly running several
positions funded as if capital were free. (This bug does **not** affect
`run_equity_paper_step.py` itself — it correctly tracks one `open_position`
at a time — so the halt is about the *model's validity*, not a live-trading
integrity issue.) Re-running the promotion comparison with that fixed and
sizing decoupled from compounding: real-label Ridge lands in the 2nd–10th
percentile of a 50-seed random-selection distribution (negative-to-flat
expectancy per trade) — worse than most random controls, including its own
shuffled-label version. A follow-up decomposition found the pool's positive
expectancy (~50 bps/trade) comes entirely from medium-horizon drift in an
exceptional AAPL/MSFT/NVDA/TSLA bull run (buy-and-hold 2018–2026: +709% /
+396% / +4,208% / +1,650% vs. SPY +217%) — not from candidate selection
(mask-filtered vs. unrestricted pool: no difference), not from symbol/day
timing (Ridge underperforms random), and not from the ATR exit mechanics
(a naive fixed 10-day hold beats the ATR-managed exit by >2x on identical
entries).

Halted via `experiments/run_equity_paper_step.py --config
config/settings_equity_paper_yfinance.yaml halt --reason "..."` — this sets
a persistent flag in the SQLite ledger (`runtime/equity_yfinance_paper.sqlite3`)
that short-circuits `run_step()` entirely, including management of the one
`pending_entry` position that was open at halt time (AAPL, never filled).
Fully reversible: `... resume` clears the flag; the systemd timer keeps
firing on schedule either way (`halted` is just a fast no-op branch inside
`run_step()`, not a disabled timer). No ledger history was touched. Do not
`resume` without a candidate that beats the random/shuffled-label/fixed-hold
controls in the research-branch docs above by a margin outside their own
seed-to-seed variance — "looks profitable" was exactly what led to Ridge's
original promotion, and it wasn't sufficient.

### What replaced it and why

`simple_trend` (a deterministic momentum rule, not an ML model — see
`_select_fold_winners` in `run_equity_real_data_walkforward.py`) paired with
the ATR stop/target disabled (always exits at the fixed 10-day
`max_hold_bars`). Validated on the `research` branch
(`EQUITY_EXPECTANCY_MATRIX_FINDINGS.md`, "Update: candidate-edge / exit-edge
decomposition" section, plus a full-history extension of that check): full
2018–2026 sample, 188 single-position trades, **243.3 bps/trade expectancy,
58.0% win rate, profit factor 2.04**, bootstrap 95% CI on mean trade return
excludes zero, **99th percentile of a 100-seed random-selection
distribution** (vs. Ridge's 2nd–10th), robust to 2.5x slippage (237.2 bps),
and dramatically more stable year-by-year than Ridge (worst year −1.4% in
2018 and −0.9% in 2022, vs. Ridge's −25.9% in 2022 alone).

Important framing, not a caveat to bury: expectancy increased monotonically
in a hold-duration sensitivity check (5/10/15/20 days: 106.6 / 243.3 / 272.3
/ 452.7 bps) — that pattern means this should be understood as **medium-term
trend/beta exposure to a strong-momentum basket**, not a precisely-timed
edge at exactly 10 days. `max_hold_bars` was deliberately left at 10 (the
pre-existing project default) rather than cherry-picked from that sweep —
re-tuning it now against the same data that motivated the change would be
the exact "optimize on the holdout" mistake this whole exercise exists to
avoid.

**Options overlay (raised, deliberately deferred, not dropped):** if a
position is expected to be favorable for ~10 days, a 30-day call is a
reasonable-sounding way to express that with convexity. This project already
tested a synthetic-option-chain overlay
(`EQUITY_HYBRID_OPTION_OPTIMIZER_RESULTS.md`) and found it didn't help even
under favorable synthetic assumptions — stock-only had the best risk-adjusted
result, every options expression added drawdown/ruin risk without added
mean return, and the optimizer rejected ~78% of stock signals as lacking a
positive-EV contract. `CLAUDE.md`'s existing guardrail (below) already
forbids presenting Black-Scholes/synthetic-IV reconstructions as real
historical options performance, and buying a 30-day call to express a 10-day
thesis means *selling* (not exercising) at day 10, whose P&L depends on IV
movement a fixed-vol model can't honestly simulate. Real validation needs
actual historical option chains (bid/ask, OI, IV) — a data vendor
(ORATS/CBOE DataShop/Polygon.io/etc.), not a formula. Left out of this
deployment; tracked as a real next step once real options data is available,
not shelved.

## Deployment roadmap (not yet live)

Phase 1 historical walk-forward validation → Phase 2 **shadow deployment** (log
hypothetical trades, verify fills/reconciliation, no live orders) → Phase 3
small-capital live ($50–$250, fractional shares, options off) → Phase 4 scaled
(~$2,500, option overlay enabled).

Phase 2 is implemented in `experiments/run_equity_paper_step.py` (`step` /
`status` / `halt` / `resume` subcommands) and runs today against **yfinance**
as an interim live-data source (`config/settings_equity_paper_yfinance.yaml`,
no credentials required) while Schwab API access is pending. Once approved,
the Schwab path (`config/settings_schwab_equity_paper.yaml`,
`SCHWAB_API_KEY`/`SCHWAB_APP_SECRET`) is a drop-in swap — same state machine,
ledger schema, and safety rules, only the OHLCV fetch differs; the two use
separate SQLite ledgers (`runtime/equity_yfinance_paper.sqlite3` vs
`runtime/equity_schwab_paper.sqlite3`) so switching doesn't lose history.
Daily scheduling: `deploy/equity-paper-yfinance.{service,timer}` (systemd,
weekdays 22:30 UTC, runs the project `.venv` directly — no Docker rebuild
needed). It never submits a live order.

Every `run_step()` also mirrors a summary row (status, open position, completed-
trade count, simulated equity/return/drawdown) to Supabase table
`equity_paper_runs` (schema: `sql/003_equity_paper_runs.sql`, must be applied by
hand in the Supabase SQL Editor — no DDL access from here), via
`btc_trend_bot.paper_lab.SupabaseOutbox`'s durable local outbox — the same
upload/retry mechanism the BTC 5-minute paper lab uses. Reads
`SUPABASE_URL`/`SUPABASE_SECRET_KEY` from the environment (already present in
`/home/joey/.config/btc-trend-bot/telemetry.env`, loaded by the systemd
service); silently a no-op if unset. This is best-effort remote monitoring
only — the SQLite ledger stays the sole source of truth for trading decisions,
and a Supabase outage never blocks the step. The Streamlit dashboard
(`btc-dashboard.service`, tab "📈 Equity Paper (yfinance)") currently reads the
local SQLite ledger directly rather than this Supabase mirror.

## Where to look for more

`REAL_DATA_BACKTEST.md` (v3.0 scope + gates), `README.md` / `README_V2_4.md`,
and the documentation bundle (`ARCHITECTURE.md`, `MODEL_CARD.md`,
`RESEARCH_LOG.md`, `LIVE_DEPLOYMENT.md`). Per-experiment `*.md` files at the root
document each research offshoot and its results.
