# Changes

## 0.2.1

- Fixed a NumPy/Pandas deprecation warning in `timeframe_to_minutes`.
- Replaced division by `pd.Timedelta(minutes=1)` with explicit seconds-to-minutes conversion.
- Verified all 16 tests with deprecation warnings treated as errors.

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

## v0.3.0

- Add paginated Coinbase Exchange hourly history download and deterministic UTC resampling to four-hour candles.
- Add `download-coinbase-history` command with a separate default data file.
- Add `diagnose-gaps` command and remove the prior diagnostic-script timedelta/timezone warnings.
- Add a research data-quality gate for configured start coverage, total missing rate, and largest contiguous gap.
- Add `--allow-data-gaps` for explicitly diagnostic backtests only.
- Add four Coinbase/gap tests; strict suite now contains 20 tests.

## 0.4.0

- Added trade-level attribution and `outputs/trades.csv`.
- Added year-by-year strategy comparison in `outputs/yearly_performance.csv`.
- Added separate block bootstrap tests for fixed-size returns and excess returns.
- Added upside/downside capture, beta, correlation, and arithmetic alpha diagnostics.
- Added frozen no-breaker long/flat and mirrored long/short research configurations.
- Added fixed-size position and turnover columns to the bar-level output.

## 0.4.0

- Added trade-level attribution and `outputs/trades.csv`.
- Added year-by-year strategy comparison in `outputs/yearly_performance.csv`.
- Added separate block bootstrap tests for fixed-size returns and excess returns.
- Added upside/downside capture, beta, correlation, and arithmetic alpha diagnostics.
- Added frozen no-breaker long/flat and mirrored long/short research configurations.
- Added fixed-size position and turnover columns to the bar-level output.
