# Five-minute paper lab v0.2: lower-turnover swing candidates

The first 10,000-bar test produced a useful rejection:

- the one-candle strategy had no gross edge;
- the two-candle strategy had a positive gross result but required roughly 77x
  turnover in 34.7 days;
- Coinbase Intro-tier taker costs overwhelmed the signal;
- a full round-trip fee hurdle on the initial 5–10 minute run produced no trades.

Version 0.2 does **not** abandon five-minute data. It changes its role:

> Five-minute candles trigger entry and help confirm deterioration, while a slower
> one-hour momentum / six-hour regime controls whether the portfolio changes state.

This aims to preserve the informational value of a short run without paying for
hundreds of complete account rotations.

## Default candidates

- `cash_5m`
- `buy_hold_5m`
- `candle_swing_balanced_5m`
- `candle_swing_strict_5m`
- `candle_swing_slow_exit_5m`

The original stress tests remain in:

```text
config/settings_paper_5m_stress.yaml
```

## Apply this v0.2 overlay

From `/home/joey/btc-paper-5m`:

```bash
unzip -o /home/joey/btc-paper-5m-v0.2.0-overlay.zip

grep -A10 '^features_5m:' config/settings_paper_5m.yaml
grep -A80 '^paper_strategies:' config/settings_paper_5m.yaml
```

Build and test:

```bash
docker compose -f compose.paper-5m.yaml build --no-cache

docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-paper-5m \
  -m pytest -q
```

## Research before creating Supabase tables

Keep the prior output intact and use a new directory:

```bash
docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-paper-5m \
  -m btc_trend_bot.paper_lab \
  --config config/settings_paper_5m.yaml \
  research \
  --bars 10000 \
  --output outputs/paper_5m_swing_10000
```

Inspect:

```bash
cat outputs/paper_5m_swing_10000/paper_research_summary.json

column -s, -t \
  outputs/paper_5m_swing_10000/conditional_horizon_returns.csv \
  | less -S
```

The summary now includes:

```text
gross_break_even_all_in_bps_per_side
assumed_all_in_bps_per_side
cost_to_break_even_multiple
```

A candidate is not economically viable when its gross break-even cost is far below
126 bps per side, even if its gross return is positive.

## Longer validation

A 34.7-day sample is exploratory, not enough to promote a strategy. After the
10,000-bar smoke test succeeds, run:

```bash
docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-paper-5m \
  -m btc_trend_bot.paper_lab \
  --config config/settings_paper_5m.yaml \
  research \
  --bars 50000 \
  --output outputs/paper_5m_swing_50000
```

Fifty thousand five-minute bars represent roughly 174 days. Do not proceed to
Supabase deployment merely because one candidate wins the 10,000-bar sample.

## Deployment gate

Proceed to the Supabase and timer steps only when at least one swing candidate:

1. trades materially less than the original ~9 executions per day;
2. remains positive after the 120 bps taker fee assumption;
3. is not dependent on one brief sample;
4. has acceptable drawdown relative to buy-and-hold;
5. has a gross break-even cost reasonably close to the modeled all-in cost.

The paper engine still submits no Coinbase orders.
