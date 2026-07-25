# ThetaData vs. yfinance stock data: a provenance check

Every walk-forward / `simple_trend` / regime-classifier result in this
project has been built entirely on yfinance daily bars
(`configs/real_data.yaml`, `auto_adjust: true`, cached in `data/real/`).
The project already pays for ThetaData (Options Value tier) for the
options-side real-data retests. Question: does ThetaData's own stock data
agree with the cached yfinance data, and does it resolve the
"yfinance intraday history is limited" constraint noted in `CLAUDE.md`?

Script: `experiments/run_thetadata_vs_yfinance_stock_check.py`. Read-only —
never modifies `data/real/` or `configs/real_data.yaml`.

## Step 0: subscription boundary, confirmed empirically before trusting anything

The current ThetaData subscription is **Options Value** — it does not
include a paid stock-data tier. Bisected directly against
`stock_history_eod` (not assumed):

| Window | Requires |
|---|---|
| before ~2021 | STANDARD subscription (blocked) |
| 2021-01 .. 2023-05-31 | VALUE subscription (blocked) |
| **2023-06-01 onward** | **accessible under the account's FREE stock tier** |

Boundary day pinned exactly: 2023-05-31 blocked, 2023-06-01 OK.

**Intraday equity data (`stock_history_ohlc`, `stock_history_trade`,
`stock_history_quote`, `stock_snapshot_ohlc`) is blocked unconditionally at
every subscription level tested — no free-tier carve-out at all**,
confirmed at three dates spanning 2023-2026. **Correction to an assumption
made before this check ran**: this subscription does NOT resolve
`CLAUDE.md`'s "free yfinance intraday history is limited" constraint. Any
real equity intraday work (an order-flow tick study, more granular entry
timing, etc.) still needs either a ThetaData VALUE/STANDARD stock upgrade
or a different vendor — it is not already-paid-for infrastructure.

Practical consequence: this project's cached history starts 2018-01-01.
Only the most recent ~3 of those ~8 years (2023-06-01 onward) can actually
be cross-checked against ThetaData under the current subscription — stated
plainly rather than silently comparing a partial window and implying full
coverage.

One more API detail worth recording: `stock_history_eod` caps at 365 days
per call (`INVALID_ARGUMENT` above that) — `option_history_eod` never hit
this limit in prior work. Chunked accordingly.

## Steps 1-3: EOD close-price comparison, 2023-06-01 to 2026-07-22

AAPL, MSFT, NVDA, TSLA, SPY — 787 trading days each side.

| Symbol | Matched dates | Dates only in one source | Close exact-match rate | Mean pct diff | Max pct diff |
|---|---|---|---|---|---|
| AAPL | 787/787 | 0 | 0.3% | 0.66% | 1.40% |
| MSFT | 787/787 | 0 | 0.1% | 1.20% | 2.37% |
| NVDA | 787/787 | 0 | 0.4% | **294.59%** | **902.05%** |
| TSLA | 787/787 | 0 | 8.4% | **0.0000%** | 0.0012% |
| SPY | 787/787 | 0 | 0.1% | 1.92% | 4.18% |

**Calendar alignment is perfect** — 787/787 matched on every symbol, zero
dates present in one source and not the other. No holiday-handling
mismatch, no missing-day problem.

**The price differences are not a data-quality bug — they're a
diagnosed, mechanical adjustment-methodology difference**, confirmed
directly rather than assumed:

- **TSLA pays no dividend and had no split in this window** — the two
  sources agree almost exactly (mean diff 0.0000%). This is the clean
  control case: when there's nothing to adjust for, the two sources
  match.
- **AAPL/MSFT/SPY pay regular dividends, no split in this window** —
  mean diffs of 0.66%/1.20%/1.92%, roughly tracking each symbol's
  dividend yield (SPY's diversified-but-still-dividend-paying profile
  shows the largest gap of the three). Checked directly for AAPL: the pct
  diff **shrinks monotonically moving toward the present** (1.40% in June
  2023 -> 0.87% in May 2024) — exactly the signature of yfinance's
  retroactive dividend adjustment (`auto_adjust=True` discounts historical
  prices by all dividends paid between that date and the series' most
  recent date, so the adjustment shrinks as the historical date gets
  closer to "now"). ThetaData's EOD close is not dividend-adjusted this
  way.
- **NVDA had a 10:1 split on 2024-06-10, inside this window** — this is
  what blows up the mean/max diff. Direct before/after check:

  | Date | yfinance close (split-adjusted) | ThetaData close (raw) | pct diff |
  |---|---|---|---|
  | 2024-06-05 | 122.23 | 1224.40 | 9.02x |
  | 2024-06-07 | 120.68 | 1208.88 | 9.02x |
  | **2024-06-10 (split date)** | 121.58 | 121.79 | **0.0017%** |
  | 2024-06-11 | 120.71 | 120.91 | 0.0016% |
  | 2024-06-14 | 131.66 | 131.88 | 0.0016% |

  Confirms directly: ThetaData's `stock_history_eod` does **not**
  retroactively split-adjust historical closes the way yfinance does.
  Before the split it reports raw pre-split-era prices (~10x yfinance's
  adjusted number, matching the 10:1 ratio exactly); after the split, the
  two agree to within rounding.

## Verdict

**The cached yfinance data is not shown to be wrong, and nothing published
in this project needs revisiting** — every backtest here has used
`auto_adjust=True` yfinance data consistently throughout its own history,
so there's no internal inconsistency in what's already been run.

**The real, forward-looking finding**: ThetaData's raw stock EOD data is
adjusted differently (materially less, or not at all, for dividends and
splits) than the yfinance data this project's pipeline assumes.
`CLAUDE.md` already warns "mixing adjusted closes with unadjusted OHLC
corrupts ATR, signals, and trade returns" as a general caution — this is
now an empirically demonstrated instance of exactly that risk, not just a
theoretical one. **If ThetaData stock EOD data is ever pulled into this
project's pipeline (to extend history, cross-validate, or replace
yfinance for any symbol), it must be adjusted to match methodology first**
— a naive substitution around any split date would silently introduce a
~9x price discontinuity into ATR/return calculations, and every dividend
payer would carry a small, compounding, systematic bias.

**Intraday**: does not resolve the yfinance-intraday-history-limitation
problem under the current subscription. A VALUE or STANDARD stock-tier
upgrade (or a different vendor) would be needed before any equity
intraday work — e.g. an order-flow tick study — could use ThetaData
instead of yfinance's free intraday endpoint.

Outputs: `outputs/thetadata_stock_crosscheck/summary.json`.
