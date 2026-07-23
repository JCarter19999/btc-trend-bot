#!/bin/bash
# Daily 7am unattended checkup: an agentic Claude Code session diagnoses the
# three live paper-trading deployments (see daily_checkup_prompt.txt for what
# it checks and is/isn't allowed to fix), writes its findings to two files,
# then this script sends them via SMTP -- the agent itself has no send
# capability, only draft-creation via the Gmail MCP connector, which is
# useless unattended (see 2026-07-23 session: draft sat unsent).
set -euo pipefail
cd /home/joey/equity_v2_4
source .venv/bin/activate

rm -f runtime/daily_checkup_subject.txt runtime/daily_checkup_body.txt

claude -p "$(cat scripts/daily_checkup_prompt.txt)" \
  --dangerously-skip-permissions \
  --allowedTools "Bash,Read,Write"

if [[ ! -f runtime/daily_checkup_subject.txt || ! -f runtime/daily_checkup_body.txt ]]; then
  echo "Checkup agent did not write the expected output files -- sending a failure alert instead." >&2
  set -a; source /home/joey/.config/btc-trend-bot/gmail_smtp.env; set +a
  echo "The daily checkup agent ran but did not produce runtime/daily_checkup_subject.txt / runtime/daily_checkup_body.txt as expected. Check the equity_v2_4 host directly." | \
    python scripts/send_email_smtp.py --subject "Equity/BTC daily check -- FAILED TO RUN"
  exit 1
fi

set -a; source /home/joey/.config/btc-trend-bot/gmail_smtp.env; set +a
python scripts/send_email_smtp.py \
  --subject "$(cat runtime/daily_checkup_subject.txt)" \
  --body-file runtime/daily_checkup_body.txt
