# Repairing the 2026 Kraken Gap

The prior combined Kraken dataset had 5,103 observed four-hour bars and 483
missing bars. Of those, 480 formed one continuous gap from 2026-01-01 through
2026-03-21. This is the seam between a historical archive ending on
2025-12-31 and Kraken's recent REST window beginning approximately 720
four-hour observations before the current date.

Do not forward-fill this 80-day gap and do not tune the strategy against the
partial dataset. Version 0.3.0 builds a separate, continuous Coinbase BTC-USD
history by downloading one-hour candles in <=300-candle pages and resampling
them into complete four-hour candles.

## Download full Coinbase history

```powershell
python -m btc_trend_bot.cli download-coinbase-history
```

Default output:

```text
data/btc_usd_4h_coinbase.csv
```

The command uses the configured market start (`2018-01-01T00:00:00Z`) and
stops at the latest completed hour. It intentionally does not overwrite the
Kraken file.

## Validate coverage

```powershell
python -m btc_trend_bot.cli data-status `
    --data data/btc_usd_4h_coinbase.csv

python -m btc_trend_bot.cli diagnose-gaps `
    --data data/btc_usd_4h_coinbase.csv
```

The normal backtest now enforces:

- first candle near the configured research start;
- missing-candle rate no greater than 0.5%;
- no contiguous gap longer than six four-hour bars.

These thresholds live under `data_quality` in `config/settings.yaml`.

## Run the repaired baseline

```powershell
python -m btc_trend_bot.cli backtest `
    --data data/btc_usd_4h_coinbase.csv
```

A backtest that fails the gate stops before features or P&L are calculated.
For a deliberately diagnostic run only:

```powershell
python -m btc_trend_bot.cli backtest `
    --data data/btc_usd_4h.csv `
    --allow-data-gaps
```

Do not use an override run for parameter selection or performance claims.
