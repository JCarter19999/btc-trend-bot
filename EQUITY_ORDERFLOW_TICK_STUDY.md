# Equity order-flow (tick data) study — blocked at the data-access layer

Directive: build the equity-market equivalent of `BTC_ORDERFLOW_STUDY.md`
(real tick-level trade tape -> information-driven bars -> flow-imbalance
signal -> shuffled-null diagnostic -> gross-edge-vs-cost-floor verdict),
now that ThetaData is already a paid vendor in this project. **Blocked
before any real data could be pulled — a genuine subscription-tier
ceiling, confirmed directly, not assumed.**

## What was checked

The $40/mo "Options Value" tier already purchased for this project's real
options re-tests (`EQUITY_OPTIONS_REAL_DATA_RETEST.md` and others) does
**not** cover stock-side tick data. ThetaData gates stock endpoints on
their own separate Free/Value/Standard/Pro ladder, independent of the
options subscription. Verified directly against the live API
(2026-07-24, `experiments/check_thetadata_stock_tick_access.py`):

| Endpoint | Result |
|---|---|
| `stock_history_eod` | **OK** — works today |
| `stock_list_symbols` | **OK** — works today |
| `stock_history_quote` | `PERMISSION_DENIED`: "requiring a **value** subscription" |
| `stock_history_trade` | `PERMISSION_DENIED`: "requiring a **standard** subscription" |
| `stock_history_trade_quote` | `PERMISSION_DENIED`: "requiring a **standard** subscription" |

This is the same shape of finding as the tail-hedge dead-end in
`EQUITY_OPTIONS_REAL_DATA_RETEST.md` section 4: a real, root-caused data-tier
ceiling, not a bug in this project's code and not something to route around
with synthetic data. The account has free-tier stock access (enough for
daily EOD bars, which is why `run_equity_real_data_walkforward.py` could in
principle cross-check against ThetaData's own EOD series) but nothing at
tick/quote granularity — exactly what an order-flow / flow-imbalance study
needs, since the whole method depends on classifying individual trades as
buy- or sell-initiated against the prevailing quote.

## Why no pipeline code was built beyond the capability probe

The directive asked for a downloader module, a feature-building adaptation,
and a diagnostic script. All three would be **untestable against real
data** right now — writing them anyway would be exactly the "half-finished
implementation" this project's standards call out: code that can't be run,
can't be verified, and sits as dead weight until (if) a subscription
upgrade happens. `BTC_ORDERFLOW_STUDY.md`'s methodology (`orderflow_data.py`,
`orderflow_features.py`) is generic enough to adapt quickly once real
equity tick data is actually reachable — `orderflow_features.py`'s
`build_dollar_bars`/`flow_imbalance`/`rolling_flow_imbalance` only need a
DataFrame with `timestamp`, `price`, `amount`/`cost`, and a buy/sell `side`
column, which is exchange-agnostic. The missing piece is entirely the data
feed, not the analysis method.

What was built instead: `experiments/check_thetadata_stock_tick_access.py`,
a small, verified-working probe that re-checks all five endpoints on
demand — rerun it after any subscription change to see immediately whether
this study has become buildable.

## What would unblock this

ThetaData's stock data uses the same tier names as their options
data (Free / Value / Standard / Pro), but it is a **separate
subscription** from the Options Value tier already purchased — upgrading
the options plan would not add stock tick access. `stock_history_trade`
and `stock_history_trade_quote` both require at least a **Standard** stock
subscription; `stock_history_quote` alone needs only **Value**. Exact
current pricing wasn't confirmed here (thetadata.net/pricing renders tier
names via JS without the dollar figures loading in a plain fetch) — stated
plainly as unconfirmed rather than guessed; check
`https://www.thetadata.net/pricing` directly or the account's own upgrade
page before purchasing.

## Verdict

**Inconclusive by data-access ceiling, not by evidence** — the same
category of honest dead-end as the tail-hedge study, not a disguised
negative result. No claim is made here about whether an equity flow-imbalance
signal exists, is capturable, or clears costs — that question was never
reachable. If a Standard-tier stock subscription is purchased, rerun
`check_thetadata_stock_tick_access.py` to confirm access, then build the
downloader/features/diagnostic trio using `BTC_ORDERFLOW_STUDY.md`'s
method directly (same shuffled-null discipline, same "translate correlation
into gross bps vs. a realistic cost floor" decisive test — note equity's
$0-commission/tighter-spread cost floor should be independently derived,
not assumed equal to BTC's 12-16bps figure).

Script: `experiments/check_thetadata_stock_tick_access.py`.
