# Broad-Universe Additive Stock + Call Overlay Experiment

Version 2.7.0 expands the synthetic equity experiment from three to four liquid names (AAPL, MSFT, TSLA, NVDA) and from 400 to 450 sessions. Direct-return Ridge remains the champion model. At each timestamp, the candidate with the highest calibrated predicted return is selected, subject to the absolute return hurdle and one-position constraint.

The stock leg is mandatory for every accepted signal. A call is added only when the signal is an extreme-bull setup and an optimized synthetic contract passes expected-return, probability-of-profit, spread, open-interest, and liquidity requirements.

This version also corrects the earlier overlay accounting. The additive return is:

`stock return + option premium allocation × option return`

The full stock exposure is retained; the option premium is additional risk capital.

Run:

```bash
python experiments/run_equity_broad_universe_additive_overlay.py \
  --sessions 450 \
  --ruin-simulations 2000 \
  --output outputs/equity_broad_universe_additive_overlay
```
