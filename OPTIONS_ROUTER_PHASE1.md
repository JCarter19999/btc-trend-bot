# Options strategy router: Phase 1 (CPU-only, rule-based)

Built from a user-supplied 26-section spec for an intraday options
strategy-routing agent (observe 9:30-10:30 ET, classify the regime, route to
a defined-risk options structure, manage exits). AI/ML mode was explicitly
out of scope for this pass — everything here is deterministic arithmetic,
no trained models.

## What's in scope vs. deferred

**Built**: first-hour feature engineering (free yfinance data — SPX/SPY/QQQ/
IWM/VIX 5-minute bars), a 7-class deterministic regime classifier, regime
persistence at 15/30/60/90 minutes (free, from the dense underlying price
path), a strategy router with an expectancy-proxy comparison across
candidates, all 6 option structures (call/put debit spread, call/put credit
spread, long straddle, long strangle) priced from real SPXW/XSP 0DTE bid/ask,
a binary exit policy (hold-to-close vs. exit-at-noon), risk-based position
sizing, a backtest engine producing a per-day decision log, baselines for
every structure, and a promotion-controls suite (window-overlap assertion,
day-clustered t-test, negative control, vol-matched benchmark, drop-one-month
sensitivity).

**Deferred** (needs option quotes sampled *between* 10:30 and the close, not
just at two anchor points): true slope-based dynamic exits, MFE/MAE and
time-to-target/stop labels, the target-before-stop probability question.

## Code

`src/btc_trend_bot/options_router/` (features, regimes, structures, router,
risk, exit_policy, backtest, reporting) · `experiments/run_options_router_
backtest.py`, `run_xsp_strangle_daily_test.py`, `run_xsp_straddle_1dte_test.py`
· `configs/options_router.yaml` (spec-realistic $5,000/2%-per-trade sizing),
`configs/options_router_relaxed_risk.yaml` (30%/trade, testing only) ·
`tests/test_options_router_*.py` (17 tests: regime classification,
structure payoff math, no-lookahead invariants, end-to-end smoke test).

Data: real SPXW/XSP 0DTE quotes fetched via `experiments/fetch_spx_0dte_
options.py` (parameterized for any OPRA parent-symbology root/window,
cost-gated before every spend) into `data/opra_{spx,xsp}[_exit]/` —
gitignored (see `.gitignore`), regenerable for ~$1.50/underlying/window.

## Results (2026-05-06 to 2026-07-31, ~59-60 trading days)

### Full router, default $5,000/2%-per-trade sizing (SPX)

**Zero trades.** Every structure sizes to zero contracts — full-size SPX
0DTE contracts (100x multiplier) cost $500-2,500+ of max loss per contract at
these strike widths, and a $100 risk budget can't fit one. This matches the
spec's own section-16 caveat that SPX may be too large for a $5k account.

### Full router, relaxed risk (30%/trade, testing only — SPX)

6 trades cleared sizing (debit spreads only; credit spreads and straddles/
strangles still too large even at 30%). Mean return -15.6%, and it **lost to
a uniform-random structure choice** (-15.6% vs. -4.1% mean) on the negative
control. n=6 is far too thin to conclude anything beyond "no evidence of
value yet."

### XSP baselines — buy the same structure every day, no regime filter

| Structure | n | win rate | mean return | median return |
|---|---|---|---|---|
| Long straddle (ATM), hold to close | 59 | 30.5% | -10.6% | -17.0% |
| Long straddle, dynamic exit (20%/-50%) | 59 | 39.0% | -10.6% | -13.3% |
| Long strangle (7.5pt OTM), hold to close | 59 | 6.8% | -23.1% | -100% |
| Long strangle, dynamic exit | 59 | 8.5% | -42.2% | -76.2% |
| 1DTE straddle (enter today, expire next day) | 59 | 35.6% | -4.5% | -21.9% |

Findings:
- **Straddle beats strangle** on every metric — narrower OTM legs need a
  much smaller move to pay off at all, so the strangle's cheaper entry isn't
  worth its near-total-loss modal outcome (median -100%).
- **A fixed profit target actively hurts a long-convexity structure.** The
  strangle's single best day (+3,387% to close) got capped at +115% by the
  20% noon profit target — that one day is most of the gap between hold-to-
  close and dynamic-exit performance. The entire economic case for buying
  premium is the rare tail winner; a fixed cap truncates it.
- **1DTE is less bad than 0DTE** — mean return more than doubles (-4.5% vs.
  -10.6%), profit factor rises from 0.67 to 0.86 — but median is worse
  (-21.9% vs. -17.0%) and the tail win is smaller (2.4x vs. 3.4x). Giving the
  position a full extra day to realize a move softens 0DTE's uniquely harsh
  same-day decay, but doesn't flip the sign.

### Bottom line

No structure or exit policy tested here shows evidence of edge on this
sample. Buying premium (straddle/strangle) blindly loses to theta decay, as
expected; the regime router hasn't been shown to beat random structure
choice; and SPX-sized contracts don't fit a realistic small-account risk
budget at all — XSP is the sizing-realistic instrument if this is pursued
further. n≈59 days is also small; nothing here rules out a real effect that
would need a longer sample to detect, but nothing here supports one either.
