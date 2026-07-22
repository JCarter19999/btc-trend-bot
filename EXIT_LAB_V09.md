# v0.9 Fixed-Entry Exit Laboratory

This research module freezes the v0.8 `rsi2_selective_15m_strict` entry logic and varies only exit behavior.

Variants cover the current control, fixed profit/stop exits, profit-activated trailing exits, RSI recovery exits, and target-plus-time exits. The report adds average realized gross basis points, average MFE basis points, MFE capture ratio, best-five-trade concentration, cost sensitivity, and five chronological folds.

Run:

```powershell
docker compose -f compose.paper-5m.yaml run --rm `
  --entrypoint python `
  btc-paper-5m `
  -m btc_trend_bot.exit_lab `
  --config config/settings_exit_lab.yaml `
  --bars 50000 `
  --output outputs/exit_lab_v09_50000 `
  --compact
```

Research only. No exchange authentication or order submission is included.
