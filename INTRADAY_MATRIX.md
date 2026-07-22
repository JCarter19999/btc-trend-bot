# Intraday strategy matrix v0.3

This release adds a **research-only** strategy matrix. It does not change the
running four-hour production bot, submit orders, create Supabase tables, or start
the five-minute timer.

The purpose is to compare genuinely different intraday hypotheses before any
candidate is promoted into persistent paper trading.

## Strategies

| Strategy ID | Hypothesis |
|---|---|
| `cash_5m` | Cash benchmark |
| `buy_hold_5m` | Required BTC buy-and-hold benchmark |
| `candle_run_2bar_fee_aware_5m` | Prior high-turnover candle-run control |
| `breakout_5m_1h_regime` | Five-minute volatility breakout inside bullish 1h/4h context |
| `vwap_reversion_5m` | Oversold rolling-VWAP deviation with recovery confirmation |
| `momentum_1h_immediate` | One-hour momentum regime, immediate execution |
| `momentum_1h_5m_entry` | Same one-hour regime and exit, with delayed five-minute entry timing |

The last two strategies isolate whether five-minute entry timing adds value to
the same slower signal.

## Causality

- Signals use only completed five-minute candles.
- Orders are simulated at the **next five-minute candle open**.
- One-hour and four-hour candles are created only after every required source
  candle has completed.
- Incomplete higher-timeframe groups are discarded.
- The same starting timestamp and capital are used for every strategy.

## Default market and costs

```yaml
market:
  exchange: binanceus
  symbol: BTC/USD
  timeframe: 5m

costs:
  fee_bps_per_side: 2.0
  slippage_bps_per_side: 5.0
  assumed_spread_bps_per_side: 1.0
```

This is an eight-basis-point all-in assumption per transaction side and a
sixteen-basis-point modeled round trip.

## Install the overlay

Upload the overlay from Windows PowerShell:

```powershell
scp "$HOME\Downloads\btc-paper-5m-v0.3.0-overlay.zip" `
  joey@178.104.237.151:/home/joey/
```

On Hetzner:

```bash
cd /home/joey/btc-paper-5m
unzip -o /home/joey/btc-paper-5m-v0.3.0-overlay.zip
```

Review the added files:

```bash
git status --short
```

## Rebuild and test

```bash
cd /home/joey/btc-paper-5m

docker compose -f compose.paper-5m.yaml build

docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-paper-5m \
  -m pytest -q
```

Expected test result:

```text
58 passed
```

## Run the 10,000-bar smoke test

```bash
docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-paper-5m \
  -m btc_trend_bot.intraday_matrix \
  --config config/settings_intraday_matrix.yaml \
  --bars 10000 \
  --output outputs/intraday_matrix_10000
```

This produces:

- `research_summary.json`
- `strategy_comparison.csv`
- `strategy_equity.csv`
- `transactions.csv`
- `trade_episodes.csv`
- `signal_counts.csv`
- `cost_sensitivity.csv`
- `chronological_folds.csv`
- `feature_frame.csv`
- `equity_curve.png`
- `drawdown_curve.png`

Inspect the compact comparison:

```bash
column -s, -t \
  outputs/intraday_matrix_10000/strategy_comparison.csv \
  | less -S
```

Or inspect the JSON:

```bash
cat outputs/intraday_matrix_10000/research_summary.json
```

## Important metrics

Prioritize:

- `net_return_pct`
- `net_excess_return_vs_buy_hold`
- `max_drawdown`
- `transaction_count`
- `round_trips_closed`
- `turnover_multiple`
- `gross_break_even_all_in_bps_per_side`
- `cost_to_break_even_multiple`
- `average_holding_bars`
- `average_gross_return_per_round_trip`
- `average_net_return_per_round_trip`
- `round_trip_win_rate`
- `average_mae_pct`
- `average_mfe_pct`

A strategy with positive gross return but a break-even execution cost below the
assumed eight basis points per side remains economically invalid.

## Cost sensitivity

`cost_sensitivity.csv` reruns every strategy at total all-in costs of:

```text
3, 5, 8, 10, and 12 basis points per side
```

This changes only execution cost. It does not retune strategy thresholds for
each cost level.

## Chronological folds

`chronological_folds.csv` divides the scored sample into five chronological
periods and restarts each portfolio with fresh capital in each period. These are
stability slices, not optimized training folds.

A candidate should not be promoted because one fold or one 35-day period is
strong. Look for consistent behavior across most periods.

## Run the larger validation

After reviewing the 10,000-bar smoke test:

```bash
docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-paper-5m \
  -m btc_trend_bot.intraday_matrix \
  --config config/settings_intraday_matrix.yaml \
  --bars 50000 \
  --output outputs/intraday_matrix_50000
```

Fifty thousand five-minute bars are approximately 174 days. The public download
may take several minutes because CCXT paginates the exchange history.

## Promotion rule

Do not install Supabase tables or enable the five-minute timer yet. Promote a
candidate into persistent paper trading only after it shows:

1. Positive net performance at the base eight-basis-point cost assumption.
2. A meaningful reduction in turnover versus the candle-run control.
3. No dependence on a single chronological fold.
4. Reasonable behavior under nearby cost assumptions.
5. Small parameter perturbations that do not reverse the result.
6. A useful comparison against buy and hold, not merely against cash.

## Commit

```bash
cd /home/joey/btc-paper-5m

git add \
  src/btc_trend_bot/intraday_matrix.py \
  config/settings_intraday_matrix.yaml \
  tests/test_intraday_matrix.py \
  INTRADAY_MATRIX.md \
  CHANGES_5M_PAPER_LAB_V03.md

git commit -m "Add intraday strategy research matrix"
git push
```
