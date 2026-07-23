# Hybrid Equity and Options Experiment

Run:

```bash
python experiments/run_equity_hybrid_option_optimizer.py --sessions 400 --ruin-simulations 5000 --output outputs/equity_hybrid_option_optimizer_final
```

The experiment uses direct-return Ridge as the absolute-return gate, pairwise ranking for simultaneous candidates, synthetic option-chain generation, volatility skew, liquidity filtering, contract optimization, stock/call expression selection, fractional-Kelly sizing, and Monte Carlo probability-of-ruin analysis.
