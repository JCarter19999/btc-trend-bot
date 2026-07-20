# Windows, VS Code, virtual environment, and Git setup

## 1. Extract and open the project

Extract the ZIP to a normal development directory, for example:

```text
C:\Users\<you>\Desktop\btc-trend-bot
```

In VS Code, choose **File -> Open Folder** and select `btc-trend-bot`.

## 2. Verify Python

Use Python 3.11 or 3.12.

```powershell
py --list
py -3.12 --version
```

## 3. Create the virtual environment

From the VS Code terminal:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once for your user account:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then close and reopen the terminal and activate again.

## 4. Select the interpreter in VS Code

Press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and select:

```text
.venv\Scripts\python.exe
```

## 5. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 6. Verify the installation

```powershell
python -c "import btc_trend_bot; print(btc_trend_bot.__version__)"
pytest -q
python -m btc_trend_bot.cli demo
```

## 7. Download public BTC candles

```powershell
python -m btc_trend_bot.cli download
```

The default is Kraken `BTC/USD`, four-hour candles. No credentials are required for public OHLCV data.

## 8. Run the backtest

```powershell
python -m btc_trend_bot.cli backtest
```

Inspect:

```text
outputs\metrics.json
outputs\equity_curve.png
outputs\backtest_bars.csv
```

## 9. Advance the local paper bot

```powershell
python -m btc_trend_bot.cli paper-step
```

The command refuses to process the same candle twice. The simulated account is stored under `paper\`.

Reset the paper account:

```powershell
Remove-Item paper\paper_state.json -ErrorAction SilentlyContinue
Remove-Item paper\paper_trades.csv -ErrorAction SilentlyContinue
python -m btc_trend_bot.cli paper-step
```

## 10. Initialize Git

```powershell
git init
git branch -M main
git status
git add .
git commit -m "Initial BTC trend research and paper trading scaffold"
```

## 11. Create a new GitHub repository

Create an empty repository on GitHub named `btc-trend-bot`. Do not initialize it with a README, `.gitignore`, or license because those files already exist locally.

Then connect and push:

```powershell
git remote add origin git@github.com:JCarter19999/btc-trend-bot.git
git push -u origin main
```

HTTPS alternative:

```powershell
git remote add origin https://github.com/JCarter19999/btc-trend-bot.git
git push -u origin main
```

Verify:

```powershell
git remote -v
git log --oneline --decorate -5
```

## 12. Normal development workflow

```powershell
git status
git switch -c feature/funding-filter
# make changes
pytest -q
git add .
git commit -m "Add funding-aware trend filter"
git push -u origin feature/funding-filter
```
