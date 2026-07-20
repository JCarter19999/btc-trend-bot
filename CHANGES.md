# Changes

## 0.2.0

- Added direct import of Kraken's complete OHLCVT ZIP without extracting it.
- Added support for `XBTUSD`, `BTCUSD`, and `XXBTZUSD` archive pair aliases.
- Added headerless seven-column archive parsing and tolerant eight-column REST-style parsing.
- Added safe timestamp-based merging of archive and recent REST candles.
- Changed `download` so it updates an existing history file instead of overwriting it.
- Added `data-status` coverage reporting.
- Added direct ZIP, direct CSV, and extracted-directory import paths.
- Added five archive and merge tests; the suite now contains 16 tests.
- Added `HISTORICAL_DATA.md` with the complete Windows workflow.

## 0.1.1

- Replaced deprecated generic NumPy timedelta construction with explicit units.

## 0.1.0

- Initial research and paper-trading scaffold.
