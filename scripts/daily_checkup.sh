#!/bin/bash
# Daily 7am unattended checkup -- see daily_checkup.py for the actual logic
# and why this is deterministic Python rather than a nested agentic session.
set -euo pipefail
cd /home/joey/equity_v2_4
source .venv/bin/activate
set -a; source /home/joey/.config/btc-trend-bot/gmail_smtp.env; set +a
python scripts/daily_checkup.py
