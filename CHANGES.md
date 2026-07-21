## 0.5.0

- Added three predeclared selective-short overlay configurations.
- Added asymmetric +100% long / -50% short position sizing.
- Added a declining 1200-bar EMA bear-regime filter.
- Added a stateful 2.5 ATR short trailing stop with re-entry blocking.
- Added selective-short tests; strict suite now contains 27 tests.

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

## 0.6.0

- Added paired moving-block bootstrap of Variant B minus the frozen long/flat control.
- Added 1x, 2x, 3x, and 5x fee/slippage sensitivity comparisons.
- Added frozen calendar-year validation slices while preserving prior indicator history.
- Added optional cross-exchange replication using a second normalized OHLCV CSV.
- Added `validate-short-overlay` CLI command and machine-readable validation outputs.

## 1.0.0

- Added Coinbase Advanced Trade execution adapter using the official Python SDK.
- Added deterministic client order IDs and duplicate-order protection.
- Added SQLite audit/state store, persistent kill switch, and overlap lock.
- Added dry-run/live deployment modes with a second live-trading acknowledgment gate.
- Added balance reconciliation and fail-closed behavior after rejected or unreconciled orders.
- Added webhook alerts, Docker packaging, systemd timer units, and production configuration.
- Added production decision tests; full suite now contains 33 passing tests.
- Frozen production strategy remains fixed-size long/cash with no permanent drawdown breaker.
