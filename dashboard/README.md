# BTC / Equity Operations Dashboard

Streamlit app with three tabs, each reading a different local data source on
the deployment host:

- **BTC Bot** -- Supabase-backed telemetry for the frozen production BTC bot
  (`scripts/run_scheduled_bot.py` in this repo).
- **Equity Paper (yfinance)** -- reads the SQLite ledger written by
  `equity_v2_4/experiments/run_equity_paper_step.py` directly
  (`/home/joey/equity_v2_4/runtime/equity_yfinance_paper.sqlite3`).
- **Two-Tier Safety (BTC)** -- reads the local state/trades/run-log files
  written by `experiments/run_two_tier_safety_paper_step.py` in this repo
  (`paper/two_tier_safety_*`).

The equity- and two-tier-safety tabs use **hardcoded absolute paths** into
sibling project checkouts on the same host (see the `EQUITY_*` and
`TWO_TIER_*` path constants near the top of `dashboard.py`) -- this app is
not portable as-is to a machine where those projects live somewhere else;
update those constants first.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml with the real Supabase URL + service-role key
streamlit run dashboard.py --server.address=127.0.0.1 --server.port=8501
```

`secrets.toml` is gitignored -- never commit the populated file, it carries a
key that bypasses Supabase row-level security.

## Deployment

Runs as `btc-dashboard.service` (systemd) on the production host, bound to
the Tailscale interface. See the running unit for exact ExecStart/ordering
(`systemctl cat btc-dashboard.service`) -- it is not currently tracked as a
file in this repo.
