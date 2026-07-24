# Macro-calendar decomposition: does DAX→SPY concentrate around scheduled events?

The third leg of Joey's vertical-understanding pivot (after sector and
market-state — `EDGE_DECOMPOSITION_SECTOR_AND_MARKET_STATE.md`). Tests
Possibility 1 ("Europe processes scheduled macro information first,
that's why it leads") against Possibility 2 ("general risk-transmission
effect, not event-specific"). `experiments/run_macro_calendar_decomposition_study.py`.

## Data sourcing — the part this test explicitly needed

The original FOMC check (`EUROPEAN_LEAD_US_FIRST_HOUR_BACKTEST.txt`)
logged its own honest gap: *"ECB meeting dates weren't checked with the
same precision — couldn't get a complete verified date list."* This
session closed most of that gap via direct lookups against primary
sources:

- **FOMC**: federalreserve.gov official calendar. Full, high-confidence,
  all 29 decision dates 2023–2026.
- **ECB**: cross-referenced against ecb.europa.eu press-release URLs.
  High-confidence for **2023–2024 only** (16 dates). 2025–2026 explicitly
  **excluded, not guessed** — available sources disagreed or were
  incomplete (a March 2025 meeting couldn't be confirmed; a "June 3–5
  2025" entry didn't match ECB's normal single-Thursday pattern; 2026
  Jan–Jul dates weren't found). The ON/OFF comparison window for ECB is
  restricted to 2023-09→2024-12 so the OFF bucket isn't contaminated by
  an unchecked period.
- **Earnings weeks**: real reported-earnings dates for AAPL/MSFT/NVDA via
  Alpha Vantage's `EARNINGS` endpoint (already validated real data in
  this project), 36 dates, 2023-08→2026-05. "Earnings week" = the ISO
  week containing any of them.
- **Payroll Fridays**: first Friday of each month — the standard BLS
  Employment Situation convention. Computed, not individually verified
  against rare holiday-driven BLS exceptions.
- **CPI: not tested.** Checked BLS, usinflationcalculator.com,
  investing.com, FRED/ALFRED, and tradingeconomics.com — every source
  either blocked the fetch (403/404) or only covered a partial window
  (mostly 2025-2026, missing 2023-2024 entirely). A real, disclosed gap,
  not a fabricated calendar.

**Methodology note**: comparison uses the ungated daily direction signal
(not the top-quartile arm) — matches the original FOMC check exactly,
and necessary for sample size: gating first (tried it, discarded) leaves
only 1–16 trades in the ON bucket per event type, since "scheduled event
day" and "top-quartile DAX move day" are independently rare conditions
whose overlap is too sparse to read anything from in a 2.7-year window.

## Results (1bp cost)

| Event type | ON: n, win, mean, Sharpe | OFF: n, win, mean, Sharpe |
|---|---|---|
| FOMC decision days | 15, 46.7%, 0.21bps, **0.13** | 447, 53.9%, 4.80bps, **2.25** |
| ECB decision days (2023-24 only) | 7, 42.9%, -4.29bps, **-2.23** | 199, 55.3%, 5.76bps, **3.00** |
| Mega-cap earnings weeks | 71, 56.3%, 7.16bps, **3.91** | 391, 53.2%, 4.20bps, **1.93** |
| Nonfarm payroll Fridays | 17, 64.7%, 25.74bps, **7.39** | 445, 53.3%, 3.85bps, **1.89** |

## Reading it honestly — not a single yes/no answer

**Central bank decision days (FOMC, ECB) do NOT strengthen the edge —
if anything they weaken or reverse it**, consistent with (and for FOMC,
directly replicating) the original ad hoc check. Plausible mechanism:
both events announce **later the same day** (FOMC ~2pm ET, well after
the US first-hour window this signal targets) — traders positioning
ahead of an afternoon rate decision plausibly swamps whatever
information Europe's morning session carried, diluting the signal
rather than sharpening it.

**Earnings weeks and payroll Fridays show a REAL, stronger edge** — and
these events are released or already fully known **before or at market
open**, not competing with the signal later in the day. Payroll numbers
specifically drop at 8:30am ET, before the 9:30am open — by the time
this signal's window trades, the number is already priced, so there's
no same-day competing catalyst; if anything a big payroll surprise
plausibly reinforces whatever directional conviction Europe's morning
session already carried.

**Sample-size honesty**: ECB (n=7) and payrolls (n=17) are modest
samples — real signals given the effect sizes and the directional
consistency with the larger-sample FOMC/earnings results, but not
independently as strong evidence as the European lead signal's own
462-trade, out-of-sample-tested validation. Treat this as **suggestive
of a timing-of-competing-information mechanism**, not as four
independently promotion-grade findings.

## Net verdict

**Neither Possibility 1 nor Possibility 2 cleanly wins — the honest
answer is more specific than either.** The edge isn't a "scheduled
European information" story (central bank days don't help, they hurt).
It isn't purely "general risk transmission" either (earnings/payroll
days show a real, non-random difference). The better-supported
mechanism: **the edge is strongest when nothing competes with it later
the same trading day.** Central bank decisions create a same-day
afternoon catalyst that dilutes the morning signal; earnings and
payrolls are resolved before or at the open, leaving the morning
cross-market signal as the dominant catalyst for that day. This is a
genuinely new piece of understanding, not previously tested in this
project, and closes the "reliable historical event calendar" gap that
was explicitly flagged as blocking this test before now.
