# BTC Trend Bot v1.0 Deployment Runbook

This runbook deploys the frozen long/cash BTC trend strategy to Coinbase Advanced Trade using a small Ubuntu VPS. Start in dry-run mode, then move to $500 live only after every gate passes.

## Safety model

- One strategy, one Coinbase portfolio, one execution process.
- Spot BTC-USD only; no leverage, margin, derivatives, transfers, or withdrawals.
- Every order has a deterministic client order ID so the same candle cannot create a second order.
- A persistent SQLite kill switch blocks future orders after a rejected order or failed reconciliation.
- A filesystem lock prevents overlapping processes.
- Live mode requires both `deployment.mode: live` and `LIVE_TRADING_ACK=I_UNDERSTAND_THIS_PLACES_REAL_ORDERS`.

## 1. Coinbase account and API keys

1. Secure the Coinbase account with a unique password, authenticator or hardware-key 2FA, and account alerts.
2. Create a dedicated Coinbase portfolio for this bot and initially fund it with exactly the amount approved for the deployment stage.
3. In Coinbase Developer Platform, create a CDP API key limited to the bot portfolio.
4. For dry-run verification, use view permission only.
5. For live trading, create a separate key with view and trade permissions. Do not grant transfer or withdrawal permission.
6. Save the key name and private key once. They cannot safely be recovered from source control, chat, screenshots, or shell history.

Coinbase's Advanced Trade API uses the v3 brokerage endpoints. The official Python SDK handles JWT authentication and supports market-order helpers. Coinbase's Advanced Trade sandbox is static and mocked, so use it only to inspect response shapes—not to validate fills, balances, or market behavior.

## 2. Create the VPS

Recommended baseline:

- Ubuntu 24.04 LTS
- 1-2 vCPU
- 1-2 GB RAM
- 20 GB disk
- Static public IPv4 if available
- Automatic snapshots or provider backups

The strategy runs once after each four-hour candle, so compute requirements are minimal. Reliability, backups, and alerting matter more than CPU.

## 3. Secure the server

Log in as the provider-created administrative user, then run:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y ca-certificates curl git ufw unattended-upgrades
sudo adduser trader
sudo usermod -aG sudo trader
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow OpenSSH
sudo ufw enable
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

Copy your SSH public key to the `trader` account, verify a new login works, and then disable password authentication and root SSH login in `/etc/ssh/sshd_config`:

```text
PasswordAuthentication no
PermitRootLogin no
```

Restart SSH only after verifying key access:

```bash
sudo systemctl restart ssh
```

## 4. Install Docker

Use Docker's official Ubuntu repository or your provider's supported Docker package. Verify:

```bash
docker --version
docker compose version
sudo systemctl enable --now docker
sudo usermod -aG docker trader
```

Log out and back in so the group change applies.

## 5. Install the release

Copy or clone the v1.0 repository into `/opt/btc-trend-bot`:

```bash
sudo mkdir -p /opt/btc-trend-bot
sudo chown trader:trader /opt/btc-trend-bot
cd /opt/btc-trend-bot
# copy the release contents here, or git clone your private repository into this directory
mkdir -p runtime data outputs paper
chmod 700 runtime
```

Build the container:

```bash
docker compose build
```

## 6. Configure secrets

Create `/opt/btc-trend-bot/.env`:

```bash
umask 077
cp .env.example .env
nano .env
chmod 600 .env
```

Populate:

```text
COINBASE_API_KEY=organizations/ORG_ID/apiKeys/KEY_ID
COINBASE_API_SECRET=YOUR_PRIVATE_KEY_OR_BASE64_SECRET
BTC_BOT_WEBHOOK_URL=OPTIONAL_WEBHOOK
```

Leave `LIVE_TRADING_ACK` absent during dry-run validation. Never commit `.env`.

For stronger secret handling later, replace the `.env` file with a provider secret manager or systemd credentials. At the $500 stage, a root/trader-readable `0600` file on a hardened VPS is an acceptable starting point, but it is not the final institutional design.

## 7. Review the production configuration

Open `config/settings_production.yaml` and verify:

```yaml
deployment:
  mode: dry_run
  product_id: BTC-USD
  allocation_cap: 0.98
  cash_reserve: 10.0
  min_notional: 10.0
  rebalance_tolerance: 0.01
```

Do not alter the frozen strategy parameters while validating infrastructure. Research improvements belong on a separate branch and require a fresh backtest and validation decision before promotion.

## 8. Run tests on the server

```bash
docker build -t btc-trend-bot:1.0.0 .
docker run --rm btc-trend-bot:1.0.0 python -m pytest -q
```

If the runtime image does not include test files, run the supplied tests before packaging or use a separate CI test target. The delivered release passed 33 tests.

## 9. Verify dry-run connectivity

The production dry-run reads actual Coinbase balances but does not submit an order.

```bash
docker compose run --rm btc-trend-bot btc-trend-bot \
  --config config/settings_production.yaml deploy-status

docker compose run --rm btc-trend-bot btc-trend-bot \
  --config config/settings_production.yaml deploy-step
```

Expected behavior:

- The process downloads completed four-hour BTC/USD candles.
- It computes the frozen target.
- It reads Coinbase balances.
- It records a dry-run order or no-op in `runtime/production.sqlite3`.
- Re-running on the same candle returns `bar already processed`.
- No real Coinbase order appears.

Inspect state:

```bash
docker compose run --rm btc-trend-bot btc-trend-bot \
  --config config/settings_production.yaml deploy-status
sqlite3 runtime/production.sqlite3 'select * from runs order by id desc limit 10;'
sqlite3 runtime/production.sqlite3 'select * from orders order by created_at desc limit 10;'
```

## 10. Install the timer

Copy the supplied units:

```bash
sudo cp deploy/btc-trend-bot.service /etc/systemd/system/
sudo cp deploy/btc-trend-bot.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now btc-trend-bot.timer
systemctl list-timers btc-trend-bot.timer
```

The timer runs at 00:05, 04:05, 08:05, 12:05, 16:05, and 20:05 UTC—five minutes after each four-hour candle boundary. The application itself is idempotent, so a delayed or repeated invocation does not intentionally duplicate an order.

View logs:

```bash
journalctl -u btc-trend-bot.service -n 100 --no-pager
journalctl -u btc-trend-bot.service -f
```

## 11. Dry-run acceptance gates

Stay in dry-run until all are true across enough closed candles to exercise both no-op and order-producing behavior:

- Every scheduled run occurs or the persistent timer catches up after reboot.
- Candle timestamps match the intended UTC four-hour schedule.
- Signals match an independently run local instance.
- Repeated execution on the same candle produces no duplicate action.
- Coinbase balances are read correctly.
- Webhook alerts arrive.
- VPS reboot recovery is verified.
- Network interruption produces an error without corrupting state.
- Manual halt and resume commands work.
- SQLite database is backed up and restorable.
- No secrets appear in logs, Git history, screenshots, or support messages.

Test the kill switch:

```bash
docker compose run --rm btc-trend-bot btc-trend-bot \
  --config config/settings_production.yaml deploy-halt --reason 'operator test'
docker compose run --rm btc-trend-bot btc-trend-bot \
  --config config/settings_production.yaml deploy-status
docker compose run --rm btc-trend-bot btc-trend-bot \
  --config config/settings_production.yaml deploy-resume
```

## 12. Enable $500 live trading

1. Stop the timer.
2. Back up `runtime/production.sqlite3`.
3. Replace the view-only key with the dedicated view+trade key.
4. Confirm transfer/withdrawal permissions are disabled.
5. Fund the dedicated portfolio with approximately $500 plus the intended cash reserve.
6. Change `deployment.mode` from `dry_run` to `live`.
7. Add the explicit acknowledgment to `.env`:

```text
LIVE_TRADING_ACK=I_UNDERSTAND_THIS_PLACES_REAL_ORDERS
```

8. Run one manual status command, then restart the timer:

```bash
sudo systemctl stop btc-trend-bot.timer
docker compose run --rm btc-trend-bot btc-trend-bot \
  --config config/settings_production.yaml deploy-status
sudo systemctl start btc-trend-bot.timer
```

Do not manually force an entry merely because BTC appears down or due for a rebound. The live system should enter only when the validated signal says long.

## 13. Live reconciliation and failure behavior

After a submitted order, v1.0 checks Coinbase balances again. If the order is rejected or balances do not change, the bot records the event, sets the persistent halt flag, sends an alert, and refuses future trading until reviewed.

When halted:

```bash
docker compose run --rm btc-trend-bot btc-trend-bot \
  --config config/settings_production.yaml deploy-status
```

Review Coinbase order history, the `orders` and `runs` tables, network/API errors, and actual balances. Resume only after explaining the discrepancy:

```bash
docker compose run --rm btc-trend-bot btc-trend-bot \
  --config config/settings_production.yaml deploy-resume
```

Never clear a halt merely to make the alert disappear.

## 14. Backups and updates

Back up daily:

- `runtime/production.sqlite3`
- `config/settings_production.yaml`
- Git commit/release identifier
- Systemd unit files

Do not back up `.env` into an unencrypted general-purpose location. Store API secrets in a password manager or encrypted secret store.

Before any release update:

1. Stop the timer.
2. Back up state.
3. Review the diff.
4. Run all tests.
5. Build a versioned image.
6. Run one dry-run cycle with a view-only key where practical.
7. Confirm the strategy parameters did not change unintentionally.
8. Restart the timer and verify the next run.

## 15. Scaling policy

Capital increases should require positive evidence, not merely elapsed time or recent profit. At each stage, evaluate:

- Correct signal timing
- Zero duplicate orders
- Expected fees and slippage
- Reliable reconciliation
- No unexplained state divergence
- No discretionary overrides
- Ability to tolerate the historically observed drawdown in dollar terms

The planned progression—$500, then $1,000, $2,000, and eventually $10,000—does not create a meaningful BTC market-capacity issue. The dominant scaling risks at those amounts are operational error and human behavior, not market impact.

## Official references

- Coinbase Advanced Trade endpoint overview: https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/rest-api
- Coinbase official Python SDK: https://github.com/coinbase/coinbase-advanced-py
- Coinbase API authentication overview: https://docs.cdp.coinbase.com/get-started/authentication/overview
- Coinbase Advanced Trade sandbox: https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/sandbox
