# Equity Extreme-Bull Call Overlay

This experiment preserves the direct-return Ridge stock strategy as the default expression. A long-call overlay is considered only when all of the following are true at signal time:

1. At least four of five forward-available momentum conditions indicate an extreme bull regime.
2. Ridge predicts at least 40 bps net return.
3. The existing cross-sectional ranker selects the symbol.
4. The optimized call has at least 2% expected return.
5. Estimated option probability of profit is at least 50%.
6. The contract passes spread, volume, open-interest, premium, skew, and utility filters.

The five regime measurements are 26-bar return, 13-bar return, 8-bar return, EMA slope normalized by ATR, and three-bar close progression. No future return is used to activate the overlay.

The tiered policy allocates 15% of active-trading capital to call premium for a 4/5 regime score and 30% for a 5/5 score. The remainder stays in the stock trade. The call overlay is never required for the underlying stock trade to proceed.
