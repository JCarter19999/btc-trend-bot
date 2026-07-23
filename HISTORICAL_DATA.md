# Kraken historical OHLCVT workflow

Kraken's public REST OHLC endpoint is useful for recent updates but returns only a limited recent window. For a multi-year backtest, use Kraken's downloadable OHLCVT archive first, then use the REST downloader to append recent completed candles.

Official Kraken support page:

```text
https://support.kraken.com/articles/360047124832-downloadable-historical-ohlcvt-open-high-low-close-volume-trades-data
```

On that page, choose **Single ZIP File** under **Complete Data**. Kraken's archive includes headerless CSV files for 1, 5, 15, 30, 60, 240, 720, and 1440-minute intervals. For the default four-hour strategy, the importer looks for `XBTUSD_240.csv` and also recognizes common Kraken aliases such as `XXBTZUSD_240.csv`.

## Recommended storage

The complete archive can be large. Keep it outside the Git repository, for example:

```text
C:\Users\<you>\Downloads\Kraken_OHLCVT.zip
```

The importer reads the matching CSV directly inside the ZIP. You do not need to extract the entire archive.

## Import the complete archive

From the activated project virtual environment:

```powershell
python -m btc_trend_bot.cli import-kraken-history `
    --archive "$HOME\Downloads\Kraken_OHLCVT.zip"
```

The command:

1. Finds the correct pair and interval inside the archive.
2. Converts Unix timestamps to UTC.
3. Validates OHLC and volume values.
4. Removes data before `market.start` in `config/settings.yaml`.
5. Merges with an existing local CSV if one exists.
6. Preserves existing recent REST rows on overlapping timestamps.
7. Writes the result to `market.data_path`.

Use an explicit pair override only if the archive uses an unexpected pair name:

```powershell
python -m btc_trend_bot.cli import-kraken-history `
    --archive "$HOME\Downloads\Kraken_OHLCVT.zip" `
    --pair XBTUSD
```

Use `--replace` only when you intentionally want to discard the existing local candle file:

```powershell
python -m btc_trend_bot.cli import-kraken-history `
    --archive "$HOME\Downloads\Kraken_OHLCVT.zip" `
    --replace
```

## Inspect coverage

```powershell
python -m btc_trend_bot.cli data-status
```

Expected output should show many thousands of four-hour candles and a date range beginning near the configured start date.

Kraken notes that its archive omits intervals in which no trades occurred. That behavior can create legitimate gaps for thinly traded pairs. For BTC/USD at four-hour resolution, large gaps should still be investigated before trusting a backtest.

## Append recent REST candles

After importing the archive:

```powershell
python -m btc_trend_bot.cli download
```

Unlike the earlier version of this project, `download` no longer overwrites the historical CSV. It downloads recent completed candles and merges them into the existing file. Recent rows take precedence on overlapping timestamps.

Run the coverage check again:

```powershell
python -m btc_trend_bot.cli data-status
```

## Run the full backtest

Archive the original 720-bar smoke test before overwriting outputs:

```powershell
New-Item -ItemType Directory -Force results\kraken_720_bar_smoke_test
Copy-Item outputs\* results\kraken_720_bar_smoke_test\ -ErrorAction SilentlyContinue
```

Then run:

```powershell
python -m btc_trend_bot.cli backtest
```

Review:

```text
outputs\metrics.json
outputs\equity_curve.png
outputs\backtest_bars.csv
```

Do not tune strategy parameters merely because the first multi-year result is weak. First add year-by-year attribution, trade-level reporting, explicit development/validation/holdout splits, and cost sensitivity.

## Quarterly updates

Kraken also publishes quarterly incremental ZIP files. Import each update using the same command:

```powershell
python -m btc_trend_bot.cli import-kraken-history `
    --archive "C:\path\to\quarterly-update.zip"
```

The importer merges new rows by timestamp and leaves the complete local history in one normalized CSV.
