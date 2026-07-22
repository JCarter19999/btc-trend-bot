# BTC 5-Minute Paper Lab v0.1.0

This experimental branch adds a high-turnover, five-minute paper-trading lab to
the current BTC Trend Bot v1.0 codebase. It does **not** modify or replace the
running four-hour service and contains no path that submits an exchange order.

## What the experiment compares

Every portfolio starts from the same simulated cash value and inception candle:

| Strategy ID | Rule |
|---|---|
| `cash_5m` | Hold cash; zero-risk control |
| `buy_hold_5m` | Buy BTC at inception and hold; required market benchmark |
| `candle_follow_1bar_5m` | One green candle moves to BTC; one red candle moves to cash |
| `candle_run_2bar_5m` | Two consecutive green candles move to BTC; two red candles move to cash |
| `candle_run_2bar_fee_aware_5m` | Two-bar rule, but entries must clear a full estimated round-trip cost hurdle plus volume/body/trend filters |

The system is spot-only. A bearish decision means cash, not a short position.

For each strategy, the engine maintains two synchronized shadow accounts:

- **net account** — includes configured fee, spread, and slippage assumptions;
- **gross account** — follows the same targets with zero transaction costs.

`gross_equity - equity` is displayed as transaction-cost drag. This makes the
high-turnover strategy useful even if it performs badly: it quantifies exactly
how much apparent edge was consumed by execution costs.

## Added files

```text
src/btc_trend_bot/paper_lab.py
config/settings_paper_5m.yaml
Dockerfile.paper-5m
compose.paper-5m.yaml
deploy/btc-paper-5m.service
deploy/btc-paper-5m.timer
sql/002_paper_lab.sql
tests/test_paper_lab.py
dashboard/2_Paper_Lab.py
PAPER_LAB_5M.md
```

The original production files and `btc-trend-bot.timer` are unchanged.

---

# Deployment

## 1. Verify the current four-hour bot first

On Hetzner:

```bash
systemctl status btc-trend-bot.timer --no-pager
systemctl list-timers btc-trend-bot.timer
sudo journalctl -u btc-trend-bot.service -n 30 -l --no-pager
```

Do not change the existing production directory while installing this branch.

## 2. Create an isolated Git worktree and branch

```bash
cd /home/joey/btc-trend-bot
git status --short
git fetch --all --prune

git worktree add \
  -b experiment/5m-paper-lab \
  /home/joey/btc-paper-5m \
  HEAD
```

Confirm the separation:

```bash
git worktree list
```

Expected shape:

```text
/home/joey/btc-trend-bot  ... [main]
/home/joey/btc-paper-5m   ... [experiment/5m-paper-lab]
```

The running four-hour service continues to use `/home/joey/btc-trend-bot`.

## 3. Copy the overlay onto the experimental worktree

From Windows PowerShell, upload the overlay ZIP:

```powershell
scp "$HOME\Downloads\btc-paper-5m-overlay-v0.1.0.zip" `
  joey@178.104.237.151:/home/joey/
```

On Hetzner:

```bash
cd /home/joey/btc-paper-5m
unzip -o /home/joey/btc-paper-5m-overlay-v0.1.0.zip

git status --short
```

Only the files listed in **Added files** should appear.

## 4. Enter the real Coinbase fee assumption

Open the five-minute config:

```bash
nano /home/joey/btc-paper-5m/config/settings_paper_5m.yaml
```

The supplied placeholder is deliberately conservative:

```yaml
fee_bps_per_side: 60.0
slippage_bps_per_side: 5.0
assumed_spread_bps_per_side: 1.0
```

`60.0` basis points is `0.60%` for one side of a transaction. Replace it with
the **taker fee shown in your own Coinbase Advanced account**. Do not reduce the
assumption merely to make a high-frequency strategy look viable.

The historical test uses the configured spread assumption because OHLCV candles
do not contain a bid/ask spread. Live paper steps use the public ticker's actual
bid and ask, then add the configured slippage.

## 5. Create the Supabase paper tables

In the Supabase SQL Editor, paste and run:

```text
sql/002_paper_lab.sql
```

It creates:

- `paper_portfolio_snapshots`
- `paper_trades`
- `paper_latest_status`
- `paper_strategy_performance`

No anonymous or authenticated-user RLS policy is added. The existing server-side
Supabase secret key is used by the private service and dashboard.

Verify in Supabase:

```sql
select * from public.paper_latest_status;
```

The view will be empty until the first paper run.

## 6. Build the separate image

```bash
cd /home/joey/btc-paper-5m

docker compose -f compose.paper-5m.yaml build
```

Confirm the two images are distinct:

```bash
docker image ls | grep -E 'btc-trend-bot|btc-paper-5m'
```

Expected image names include:

```text
btc-trend-bot   1.0.0
btc-paper-5m    0.1.0
```

## 7. Run every unit test

```bash
cd /home/joey/btc-paper-5m

docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-paper-5m \
  -m pytest -q
```

The supplied package passed **44 tests** before delivery, including the original
v1.0 suite and the new paper-lab tests.

## 8. Run an initial five-minute research sample

Ten thousand five-minute bars represent approximately five weeks of continuous
market time:

```bash
cd /home/joey/btc-paper-5m

docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-paper-5m \
  -m btc_trend_bot.paper_lab \
  --config config/settings_paper_5m.yaml \
  research \
  --bars 10000 \
  --output outputs/paper_5m_research
```

Outputs:

```text
outputs/paper_5m_research/conditional_continuation.csv
outputs/paper_5m_research/paper_research_equity.csv
outputs/paper_5m_research/paper_research_trades.csv
outputs/paper_5m_research/paper_research_summary.json
```

`conditional_continuation.csv` directly tests:

```text
P(next candle continues | at least 1 same-direction candle)
P(next candle continues | at least 2 same-direction candles)
...
```

The historical simulator generates a signal only after a candle is complete and
executes at the following candle's open. It does not trade at the same close used
to form the signal.

## 9. Run one live-market paper step manually

Load the existing Supabase telemetry variables into the shell:

```bash
set -a
source /home/joey/.config/btc-trend-bot/telemetry.env
set +a
```

Run:

```bash
cd /home/joey/btc-paper-5m
docker compose -f compose.paper-5m.yaml run --rm btc-paper-5m
```

This uses public Coinbase market data. It does not load the Coinbase trading key
and cannot submit an order.

Check local simulated accounts:

```bash
docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-paper-5m \
  -m btc_trend_bot.paper_lab \
  --config config/settings_paper_5m.yaml \
  status
```

Run the same step again. It should report that the latest completed candle was
already processed rather than duplicating trades.

## 10. Confirm Supabase received the first rows

```sql
select
  strategy_id,
  bar_timestamp,
  equity,
  gross_equity,
  cost_drag,
  trade_count,
  signal,
  reason
from public.paper_latest_status
order by strategy_id;
```

Five strategy rows should appear.

## 11. Install the isolated five-minute timer

```bash
sudo cp /home/joey/btc-paper-5m/deploy/btc-paper-5m.service \
  /etc/systemd/system/

sudo cp /home/joey/btc-paper-5m/deploy/btc-paper-5m.timer \
  /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable --now btc-paper-5m.timer
```

Verify both timers:

```bash
systemctl list-timers btc-trend-bot.timer btc-paper-5m.timer
```

The original bot should remain scheduled at six four-hour times. The paper lab
should be scheduled at:

```text
:02, :07, :12, :17, :22, :27, :32, :37, :42, :47, :52, :57 UTC
```

It runs roughly two minutes after each five-minute candle boundary.

Check status and logs:

```bash
systemctl status btc-paper-5m.timer --no-pager
sudo journalctl -u btc-paper-5m.service -n 100 -l --no-pager
```

Expected timer state:

```text
active (waiting)
```

## 12. Add the dashboard page

The existing dashboard is at `/home/joey/btc-dashboard`.

```bash
mkdir -p /home/joey/btc-dashboard/pages

cp /home/joey/btc-paper-5m/dashboard/2_Paper_Lab.py \
  /home/joey/btc-dashboard/pages/2_Paper_Lab.py

sudo systemctl restart btc-dashboard.service
sudo systemctl status btc-dashboard.service --no-pager
```

Open the existing private Tailscale dashboard. Streamlit will show a new page
named **Paper Lab**.

The page includes:

- all five portfolio values;
- buy-and-hold comparison;
- net versus frictionless gross equity;
- cumulative transaction-cost drag;
- drawdowns;
- target BTC exposure;
- fees, spread, slippage, turnover, and trade count;
- recent simulated trades;
- CSV export.

## 13. Commit the experimental branch

```bash
cd /home/joey/btc-paper-5m

git add \
  src/btc_trend_bot/paper_lab.py \
  config/settings_paper_5m.yaml \
  Dockerfile.paper-5m \
  compose.paper-5m.yaml \
  deploy/btc-paper-5m.service \
  deploy/btc-paper-5m.timer \
  sql/002_paper_lab.sql \
  tests/test_paper_lab.py \
  dashboard/2_Paper_Lab.py \
  PAPER_LAB_5M.md

git commit -m "Add isolated five-minute paper trading lab"
git push -u origin experiment/5m-paper-lab
```

---

# Operations

## Run the paper lab immediately

```bash
sudo systemctl start btc-paper-5m.service
sudo journalctl -u btc-paper-5m.service -n 100 -l --no-pager
```

## Watch live service logs

```bash
sudo journalctl -u btc-paper-5m.service -f
```

## Check local storage and outbox

```bash
ls -lh /home/joey/btc-paper-5m/runtime/paper_5m.sqlite3

du -h /home/joey/btc-paper-5m/runtime/paper_5m.sqlite3

wc -l /home/joey/btc-paper-5m/runtime/paper_5m_supabase_outbox.jsonl
```

Local SQLite records every five-minute portfolio mark. Supabase is intentionally
sampled every fifteen minutes plus every trade to control row growth.

## Check service resource use

```bash
systemctl show btc-paper-5m.service \
  -p CPUUsageNSec -p MemoryCurrent -p MemoryPeak

docker stats --no-stream
```

The service is one-shot, so no paper container remains running between candles.

## Stop only the five-minute experiment

```bash
sudo systemctl disable --now btc-paper-5m.timer
```

This does not affect:

```text
btc-trend-bot.timer
btc-trend-bot.service
btc-dashboard.service
```

## Reset the paper portfolios

This permanently deletes only the local five-minute paper experiment:

```bash
cd /home/joey/btc-paper-5m

docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-paper-5m \
  -m btc_trend_bot.paper_lab \
  --config config/settings_paper_5m.yaml \
  reset --yes
```

Do not reset after the formal comparison period begins unless every strategy is
being restarted from the same candle.

## Remove the experiment completely

```bash
sudo systemctl disable --now btc-paper-5m.timer
sudo rm -f /etc/systemd/system/btc-paper-5m.service
sudo rm -f /etc/systemd/system/btc-paper-5m.timer
sudo systemctl daemon-reload

docker image rm btc-paper-5m:0.1.0

cd /home/joey/btc-trend-bot
git worktree remove /home/joey/btc-paper-5m
```

Supabase tables are left intact for historical analysis. Drop them only after
exporting any data you want to retain.

---

# Safety properties

- No live-order function is imported or called by `paper_lab.py`.
- The paper compose file does not load the production `.env` Coinbase key.
- The existing four-hour Docker image, timer, state database, and config are unchanged.
- Strategy accounts are independent and share the same market timestamps and cost assumptions.
- SQLite uses a primary key on `(strategy_id, bar_timestamp)` to prevent duplicate processing.
- A process lock prevents overlapping timer executions.
- Missed candles are replayed sequentially at the following candle open.
- If an outage exceeds the configured lookback, the service fails loudly rather than silently skipping candles.
- Supabase failure does not fail or erase local paper state; records remain in a durable outbox.
- The Streamlit page remains available only through the existing Tailscale network.

## v0.2 lower-turnover revision

After the initial high-turnover research, use `PAPER_LAB_5M_V02.md` before
installing Supabase tables or enabling the timer. The original strategies remain
available in `config/settings_paper_5m_stress.yaml` as diagnostic controls.
