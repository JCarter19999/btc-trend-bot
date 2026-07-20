# What was cannibalized from `sp500-vol-estimator`

This project preserves the strongest architectural lessons from the prior SPY volatility/options experiment while replacing the instrument and strategy assumptions.

## Preserved concepts

- configuration-driven research runs;
- normalized provider-neutral market data;
- explicit coverage and duplicate checks;
- no-lookahead tests;
- delayed execution after signal formation;
- realistic transaction-cost deductions;
- chronological equity and drawdown accounting;
- hard drawdown breaker;
- benchmark comparisons;
- moving-block bootstrap inference;
- separate historical research and paper-account state;
- outputs that can be audited bar by bar.

## Deliberately removed

- VIX and VIX-term-structure inputs;
- implied-versus-realized volatility edge;
- `buy_vol` and `sell_vol` signals;
- SPY option-chain normalization;
- put-credit-spread selection;
- option Greeks and Black-Scholes repricing;
- contract multipliers and spread-width risk;
- NYSE trading-calendar assumptions;
- IBKR execution.

## New hypothesis

The initial hypothesis is that a long/flat BTC trend signal can improve drawdown and risk-adjusted performance relative to BTC buy-and-hold after fees and slippage. Lagged realized volatility is used only to size exposure, not to predict direction.

The package also runs a fixed-size trend arm. The volatility-scaled version should not be retained merely because it is more sophisticated; it must outperform the fixed-size arm on predeclared risk and robustness criteria.

## Historical-data correction in v0.2.0

The original REST-only downloader could retrieve only Kraken's recent OHLC window, producing a 720-bar smoke test rather than a multi-year research sample. Version 0.2.0 adds a first-class importer for Kraken's downloadable OHLCVT archive and changes recent REST downloads into merge-safe incremental updates.
