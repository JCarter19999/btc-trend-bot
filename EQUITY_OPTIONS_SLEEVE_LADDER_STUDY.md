# Options sleeve ladder: debit spreads, tail hedge, volatility breakout

Follow-up to Joey's "HIGH-RISK/HIGH-REWARD OPTIONS PORTFOLIO GAMBIT" brief
(2026-07-23) — a detailed 8-structure research ladder with an explicit
priority order. This tests the three structures that ladder puts first
that hadn't already been tested: **bull call debit spreads**, a **tail
hedge**, and a **volatility-breakout straddle**. Selective long calls
(the #1 priority) were already tested in `EQUITY_OPTIONS_DEEP_DIVE.md` —
reused here, not re-run. Calendars/diagonals/butterflies/short-premium are
explicitly **out of scope tonight**, matching the brief's own stated
priority order ("should be studied after simpler defined-risk structures" /
"what to avoid initially").

**Same hard framing as every options doc in this project**: Black-Scholes
priced off trailing REALIZED volatility, not real implied vol or real
bid/ask. This systematically UNDERPRICES real options (volatility risk
premium) — generous to every structure here, not neutral. Per
`CLAUDE.md`'s guardrail: **nothing in this document is promotion-grade.**
It validates infrastructure and gives a directional read, not a green
light for real capital. Real historical option-chain data (a paid vendor —
ORATS/CBOE DataShop/Polygon.io) is the actual prerequisite for promotion,
exactly as the brief's own "Data Requirements" section says.

## A real bug found and fixed before trusting any of this

First run produced obvious nonsense: a tail-hedge backtest reporting an
**average winning trade of +35,574,300%**, and debit spreads showing
losses beyond -1000% on individual trades. Stopped immediately rather than
report it. Two distinct causes, both fixed in `options_pricing.py`:

1. **No minimum tradable premium.** Black-Scholes correctly computes a
   45-DTE, 10%-OTM SPY put at ~$0.003 on a $400 spot when realized vol is
   10% (a realistic calm-market input) — mathematically correct, but no
   real market quotes options at a fraction of a cent. Dividing a later
   normal-sized premium by that near-zero entry price produces absurd
   percentage returns. Added `MIN_TRADABLE_PREMIUM = $0.05`; trades priced
   below it are now marked untradeable (`net_return = NaN`), not floored
   and kept.
2. **Debit spreads could report losses beyond -100%.** A defined-risk
   spread's max loss is capped at the net debit paid, by construction —
   but the entry/exit bid-ask markup is applied with *opposite* sign to
   the long vs. short leg, which can flip their ordering when both legs
   are tiny (both far OTM) and produce `net_credit < 0`. That's a modeling
   artifact of the markup, not a real payoff a defined-risk spread can
   produce. Clamped `net_return` at -100% to match the actual economic
   invariant.

Verified the fix by inspecting raw per-trade premiums and returns
directly, not just trusting the aggregate stats moved back into a
plausible range. Both bugs are documented in `options_pricing.py`'s
comments so they don't get silently reintroduced.

## 1) Bull call debit spreads — narrow loses, wide barely wins, and that's the real finding

Same `simple_trend` long signal as the calls deep dive (TSLA/COIN/MSTR/
PLTR/GME), 30-DTE spread sold at the same 10-day exit as the stock thesis.

| Spread width (long → short strike) | Trades | Win rate | Expectancy | Profit factor | Total return |
|---|---|---|---|---|---|
| 1.00 → 1.05 (5pt) | 24 | 12.5% | -4409 bps | 0.07 | -100% |
| 1.00 → 1.10 (10pt) | 70 | 35.7% | -1495 bps | 0.54 | -100% |
| 1.00 → 1.20 (20pt) | 162 | 37.7% | -637 bps | 0.82 | -100% |
| **1.00 → 1.30 (30pt)** | 192 | 37.5% | **+150 bps** | **1.04** | **+28.9%** |
| **1.00 → 1.50 (50pt)** | 192 | 38.0% | **+961 bps** | **1.27** | **+184.5%** |

**Narrow spreads — the ones that actually deliver the structure's stated
benefit (lower premium cost, lower theta burden) — lose money here.**
Checked the raw trades to understand why, not just the headline number:
with only a 10-day hold out of a 30-DTE contract, both legs still carry
substantial extrinsic value at exit — the position hasn't converged
toward intrinsic value the way it would by expiration. A narrow spread's
net debit is thin, so the round-trip spread-crossing cost on **two legs**
(4 total crossings vs. an outright call's 2) eats a disproportionate share
of it, and even a *directionally correct but small* stock move often isn't
enough to overcome that. Only once the short strike is wide enough (30-50
points out) that the position starts behaving like an outright call with
a small subsidy — not really "spreading" the risk anymore — does it turn
positive. **This somewhat defeats the structural purpose of using a spread
in the first place**, which is worth being direct about rather than
picking the one favorable row and calling it a win.

## 2) Tail hedge — modest positive expectancy, real crash payoff, meaningful annual drag

Recurring 45-DTE, 10%-OTM SPY put, re-entered every 21 trading days, sized
at 2% of capital/month (mid of the brief's 1-3% range), tested against the
**actual deployed 4-stock `simple_trend` universe's own crash windows**
(a hedge should be evaluated against what it protects, not the volatile
options sandbox).

| | Value |
|---|---|
| Trades | 50 (all takeable — see note below) |
| Win rate | 20.0% |
| Profit factor | 2.96 |
| Total return (hedge sleeve alone, 8.3yr) | +147.1% |
| Avg winner | +1110% |
| Avg loser | -93.6% |
| Annual cost | 24% of the hedge sleeve's own budget (2%/month × 12) |

**2018 Q4** (SPY -13.8%): hedge trades averaged +301%. **2020 COVID**
(SPY -17.4%): hedge averaged +570%. **2022 bear** (SPY -18.2%, slower
decline): hedge averaged +98%. The hedge does what a hedge is supposed to
do — pays off specifically when the thing it protects falls hard, small
persistent cost otherwise (the classic 80%+ of months lose ~94-100% of a
small premium, occasional large payoff funds the rest).

One implementation note worth flagging: evaluating this initially through
the *stock strategy's* safety layer (drawdown-pause, loss-cooldown) caused
it to self-pause during normal strings of small hedge losses — exactly
the wrong behavior for insurance, which is supposed to keep paying premium
through quiet periods. Re-ran with that layer removed for the hedge
sleeve's own evaluation (kept everywhere else). Full detail in the
script's comments.

## 3) Volatility breakout straddle — the strongest result of the three, with a real mechanism

Signal: trailing 20-day realized vol in its own bottom quartile (1-year
lookback) + `relative_volume > 1.2` (volume expanding) — "coiling then
waking up." ATM straddle, 30 DTE, sold at 10-day exit, same volatile
universe.

| Spread assumption | Trades | Win rate | Expectancy | Total return |
|---|---|---|---|---|
| 5% | 97 | 64.9% | +3531 bps | +342.5% |
| 10% | 97 | 56.7% | +2236 bps | +216.9% |

Checked the raw trades: the signal genuinely precedes volatility
expansion — entry realized vol and exit realized vol were inspected
directly, and vol roughly doubles from entry to exit in the median case
(e.g. 30% → 47%, 35% → 63%), which is exactly the mechanism a
volatility-breakout thesis needs. Realized move exceeded the vol implied
at entry on ~40% of trades — not most, but the straddle still profits on
net because it benefits from vol expansion happening mid-hold even on
trades where the *directional* move alone wouldn't have cleared the
implied bar (unlike the debit spread, a long straddle's short leg doesn't
exist to fight that effect). This is the one structure of the three where
the underlying mechanism (vol compression → expansion) is directly
visible in the data, not just a P&L number — the most credible of the
three findings, still bounded by every caveat above.

## Portfolio-level synthesis (the "asymmetric basket" framing)

Per-trade payoff shape, and expected growth **of total account capital**
(not the option's own % return — a -100% loss on a 10%-of-capital premium
allocation only costs the account 10%, so log-growth has to be computed
on the capital-weighted return, not the raw option return; an earlier
pass at this got `-infinity` from taking `log(1 + r)` directly on trades
that lost 100% of premium, which is the wrong basis and worth naming as a
mistake caught before it reached this doc):

| Structure | Win rate | Payoff ratio (avg win / avg loss) | % near-total loss | Expected log growth/trade (portfolio) |
|---|---|---|---|---|
| Long call, 5% OTM, 5% spread | 39.5% | **3.50** | 10.1% | **+3.00%** |
| Debit spread, wide (1.0→1.5) | 39.7% | 2.11 | 5.9% | +0.73% |
| Debit spread, narrow (1.0→1.1) | 28.3% | 0.78 | 5.8% | **-2.85%** |

This is the venture-capital payoff shape the brief describes — low-ish
win rate, big payoff ratio, most losses capped and small — for the
outright call and the wide spread. The narrow spread inverts it: payoff
ratio below 1 means even its wins don't clear its losses on average, and
that shows up directly as negative expected log growth, not just a bad
Sharpe. This table is the clearest single piece of evidence for **why
selective long calls are the right first structure** (as the brief's own
priority order already guessed) and why a spread only helps once it's
wide enough to stop functioning like a real spread.

## What was deliberately not done tonight

Calendars/diagonals, butterflies, and every short-premium structure —
consistent with the brief's own priority order and explicit "what to
avoid initially" list. Also not attempted: real historical bid/ask data
acquisition (still the actual bottleneck for promotion, per every options
doc in this project) and combining these into one blended monthly sleeve
equity curve (each structure here was evaluated independently, as the
brief itself asked for before combining).

## Recommended path forward

1. **Selective long calls remain the strongest, best-evidenced structure**
   — highest payoff ratio, positive log growth, and it's the one already
   partially de-risked by the live call-options paper deployment reading
   real bid/ask quotes going forward.
2. **The volatility-breakout straddle is the most interesting new
   finding** — a visible, checkable mechanism (vol compression →
   expansion), not just a favorable backtest number. Worth a second pass
   with real IV data before anything else, since it's the one structure
   here whose edge concept is genuinely distinct from the existing
   momentum thesis.
3. **Debit spreads, as tested, don't earn a place in the sleeve** — narrow
   ones lose, wide ones just approximate a call with extra complexity.
   Not worth pursuing further without a different exit rule (e.g. holding
   closer to expiration, which the current "10-day exit of a 30-day
   thesis" framing deliberately doesn't do, since that framing comes from
   matching the stock strategy's own hold period).
4. **Tail hedge is cheap insurance with a real historical track record in
   this dataset** — reasonable to consider pairing with any live capital
   deployment regardless of what else ships, on its own merits, independent
   of the other three structures.
5. **Do not promote any of this toward live capital.** Everything here is
   Black-Scholes-off-realized-vol, per this project's hard guardrail. The
   single highest-leverage next step, stated in every options doc in this
   project and still true, is real historical bid/ask data — not more
   synthetic backtesting.
