#!/bin/bash
# Auto-restarting wrapper for the BTC 60-day trade pull. This VM has only
# 1.9GB RAM and runs several other services concurrently (dashboard, paper
# bots) -- occasional OOM kills during a long download are expected here,
# not necessarily a bug in the downloader itself (confirmed via dmesg: two
# separate OOM events, different PIDs, one after the part-file checkpoint
# fix already landed). Since download_trades_range() checkpoints to small
# part files and resumes cheaply from the last one, the right response to
# an OOM kill is "restart," not "stop." This loop does that automatically
# instead of needing a human to notice and relaunch it each time.
set -uo pipefail
cd /home/joey/equity_v2_4_research
source .venv/bin/activate

START=$(date -u -d '60 days ago' +%Y-%m-%dT00:00:00Z)

while true; do
  END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "=== $(date -u) : starting/resuming pull, target end $END ==="
  python3 -c "
import sys; sys.path.insert(0,'src')
from btc_trend_bot.orderflow_data import download_trades_range
download_trades_range('BTC/USDT', '$START', '$END', 'data/orderflow/btcusdt_trades_60d.parquet')
"
  exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "=== $(date -u) : completed successfully ==="
    break
  fi
  echo "=== $(date -u) : exited with code $exit_code, restarting in 15s ==="
  sleep 15
done
