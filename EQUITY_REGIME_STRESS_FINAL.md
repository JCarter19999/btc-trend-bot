# Equity Regime Stress and Safety Finalization (v2.9.0)

This experiment finalizes the capital-constrained long-only stock core plus selective 30% long-call overlay for an intended $2,500 initial active account.

## Safety layer

- 15% drawdown pause
- 35% hard drawdown shutdown
- four consecutive losing trades trigger an eight-opportunity cooldown
- two losing option overlays trigger a ten-opportunity option pause
- long-only regime gate requiring 26-bar return above -1% and EMA slope/ATR above -0.20
- minimum live equity of $25
- stock leg uses remaining cash after whole-contract premium in cash-safe mode
- Schwab assumptions remain $0 listed-stock commission and $0.65 per option contract per transaction

## Stress environments

- mixed control: 200 synthetic sessions
- persistent bear: 200 synthetic sessions
- high-volatility chop: 120 synthetic sessions
- crash and recovery: 120 synthetic sessions
- clustered tail-loss injection: 100 control trades with repeated -12%, -8%, and -6% loss clusters and total option-premium losses

The shorter high-volatility scenarios were used because full option-chain optimization is computationally expensive. These tests are synthetic capability tests, not return forecasts.
