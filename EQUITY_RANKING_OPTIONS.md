# Cross-Sectional Ranking and Long-Call Options Research

This branch compares a pairwise cross-sectional ranking objective against the earlier direct-return regression approach. At each signal timestamp, the ranker learns which candidate should outperform another candidate rather than attempting to predict the exact numerical return.

The options overlay is research-only and buys calls on the selected AAPL, TSLA, or MSFT candidate. It models Black-Scholes premium, symbol-specific implied-volatility floors, earnings-like IV premium, bid/ask spread, theta decay, changing IV, and a maximum loss of the premium paid.

Policies:

- 14-session call, approximately 0.35 delta
- 30-session call, approximately 0.40 delta
- 45-session call, approximately 0.50 delta

No short options, naked writing, spreads, assignment, or early exercise are included in v1. Long calls were chosen because their loss is capped at premium while still providing the high-risk convex exposure requested.

Run:

```bash
docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-v1 \
  experiments/run_equity_ranking_options_experiment.py \
  --sessions 1300 \
  --rank-threshold 0.57 \
  --output outputs/equity_rank_options
```
