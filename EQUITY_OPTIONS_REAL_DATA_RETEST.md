# Real ThetaData re-test: calls and the volatility breakout straddle

Follow-up to every "not promotion-grade without real spread data" verdict
in this project's options work. Joey purchased ThetaData's Options Value
tier ($40/mo, 4yr history, 1-min granularity) specifically to resolve
this. Two studies re-tested so far with real quotes: the original calls
deep dive, and the volatility breakout straddle (previously the strongest
synthetic finding of the night, and Joey's stated top priority to
re-check).

**Data window, confirmed empirically not assumed:** real quotes return
correctly from 2021 onward; 2020 and earlier return "no data" under this
subscription tier (2019 returns an explicit `PERMISSION_DENIED` requiring
a higher tier). All re-tests below are restricted to signals from
2021-06-01 onward — roughly half the original 2018+ sample, stated
plainly rather than hidden.

## Method discipline: same signals, same window, only pricing differs

Every comparison here holds the signal, universe, dates, and hold period
identical between the old (Black-Scholes-off-realized-vol) and new (real
ThetaData bid/ask) pricing — the only thing that changes is how the
option leg is priced. This is deliberate: it's the only way a difference
in outcome can be attributed to data quality rather than a confound.

**A methodology trap caught before trusting the first comparison**: raw
trade counts between synthetic and real runs are NOT directly comparable
in this project's `simulate_single_position` — a candidate with no real
quote available gets dropped from the sequence entirely before the
single-position overlap filter runs, and a removed candidate can't
"block" a neighboring one anymore. This mechanically inflates the real
run's trades-taken count relative to synthetic in a way that has nothing
to do with pricing quality. Verified this directly (checked that 0 of 924
signals were floor-rejected by the synthetic pricer, ruling out the
"synthetic silently drops most signals" theory before it got repeated as
a finding) before trusting anything downstream. Per-trade expectancy
doesn't have this artifact and is the fairer metric throughout.

## 1) Calls deep dive

| | Synthetic (old, same window) | Real (ThetaData) |
|---|---|---|
| ATM: expectancy/trade | -27.3% | **-4.4%** |
| ATM: profit factor | 0.43 | **0.88** |
| ATM: total return | -100% | -48.5% |
| 5% OTM: expectancy/trade | -26.6% | -22.4% |
| 5% OTM: profit factor | 0.47 | **0.24** |

Mixed, not a clean "real data vindicates calls" story. ATM real pricing
is dramatically less bad than synthetic implied (nearly breakeven PF vs.
a clean loser) — a real chunk of the original "not a clean nail"
uncertainty was genuinely a data artifact. But real ATM calls still lost
money overall in this window, and 5% OTM real data is actually *worse*
by profit factor than synthetic suggested despite a higher win rate.
5% OTM real sample is thin (29 trades) — held loosely.

**Reading it honestly**: real data resolves the uncertainty in the
direction of "still not a winner, but less badly wrong than the
synthetic pricing implied." Not promotion-grade either way.

## 2) Volatility breakout straddle — the decisive result

This was flagged as the strongest synthetic finding of the whole project
(+342.5% total return, 64.9% win rate, full 2018+ window) and Joey's
explicit top priority to re-check with real data.

| | Real (2021-06-01+) | Synthetic, SAME window | Synthetic, ORIGINAL full window (2018+) |
|---|---|---|---|
| Trades | 66 | 72 | (full sample) |
| Win rate | 27.3% | 30.6% | 64.9% |
| Profit factor | **0.36** | 0.62 | — |
| Total return | **-83.5%** | -40.0% | **+342.5%** |

**Two separate effects, both pointing the same direction:**

1. **Period effect**: even under the OLD synthetic pricing, the
   2021-06-01+ window alone is already a loser (-40% total return, PF
   0.62) — nowhere near the +342.5% headline number. That number's
   strength was concentrated in 2018–mid-2021, exactly the period this
   subscription tier can't verify with real quotes.
2. **Pricing effect**: holding the window constant, real quotes make it
   meaningfully worse than synthetic (-83.5% vs -40.0%, PF 0.36 vs
   0.62). Consistent with the volatility risk premium bias flagged in
   every options doc all along — a straddle is a pure long-vega bet with
   no directional payoff to fall back on, making it the single
   structure most exposed to synthetic pricing understating what a real
   premium costs.

## Verdict

**Calls**: still not a winner, but real data narrows the uncertainty
in a mildly encouraging direction (ATM specifically).

**Volatility breakout straddle**: does not survive. The good years are
unverifiable at this subscription tier, and the verifiable years are a
real loser under both pricing methods, with real data making it *worse*
than synthetic already suggested. This reverses the night's most
promising options finding — a clean, decisive result, and the strongest
validation yet of why every prior options doc withheld promotion pending
real data.

## What's still queued

- European lead signal (SPY shares vs. 0DTE/1DTE/0.40-delta options, +
  QQQ) with real quotes
- Long-call threshold analysis: does calls-vs-shares performance depend
  on signal strength percentile
- Tail hedge true carrying cost with real SPY put quotes
- P&L decomposition (delta/IV/theta/spread) for completed real trades --
  core math built and validated (`implied_greeks.py`, solves implied vol
  from real prices via Black-Scholes inversion rather than paying for
  ThetaData's $80/mo Standard tier's direct Greeks feed)

Scripts: `experiments/run_equity_options_real_data_retest.py`,
`experiments/run_equity_vol_breakout_real_data_retest.py`,
`experiments/run_equity_options_real_vs_synthetic_comparison.py`
