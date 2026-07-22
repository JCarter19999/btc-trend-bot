#!/usr/bin/env bash
set -euo pipefail
python experiments/run_equity_real_data_walkforward.py \
  --config configs/real_data.yaml \
  --provider yfinance \
  --output outputs/real_equity_walkforward
