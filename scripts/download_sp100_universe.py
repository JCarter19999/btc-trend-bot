"""One-off download of the S&P 100 universe (yfinance daily bars,
2018-01-01 to present) for the multi-position portfolio study. Caches to
data/real_sp100/{SYMBOL}.csv, same schema as data/real/ (date,open,high,
low,close,volume) so it's a drop-in for load_csv_dir.

Symbol list pulled from Wikipedia's S&P 100 constituent table (2026-07-23),
101 tickers (Alphabet's dual share classes both count). Known caveat, not
solved here: this is TODAY's membership projected back to 2018 --
survivorship bias (delisted/removed constituents aren't included), and
several of these tickers didn't exist under this name/ticker for the whole
window (META was FB until 2022, GEV spun off from GE in 2024, PLTR IPO'd
2020) -- those just get shorter history, not an error.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_equity_real_data_walkforward import download_yfinance  # noqa: E402

SP100_SYMBOLS = (
    "AAPL,ABBV,ABT,ACN,ADBE,AMAT,AMD,AMGN,AMT,AMZN,AVGO,AXP,BA,BAC,BKNG,BLK,BMY,BNY,BRK-B,C,CAT,CL,CMCSA,COF,"
    "COP,COST,CRM,CSCO,CVS,CVX,DE,DHR,DIS,DUK,EMR,FDX,GD,GE,GEV,GILD,GM,GOOG,GOOGL,GS,HD,HON,IBM,INTC,INTU,ISRG,"
    "JNJ,JPM,KO,LIN,LLY,LMT,LOW,LRCX,MA,MCD,MDLZ,MDT,META,MMM,MO,MRK,MS,MSFT,MU,NEE,NFLX,NKE,NOW,NVDA,ORCL,PEP,"
    "PFE,PG,PLTR,PM,QCOM,RTX,SBUX,SCHW,SO,SPG,T,TMO,TMUS,TSLA,TXN,UBER,UNH,UNP,UPS,USB,V,VZ,WFC,WMT,XOM"
).split(",")
BENCHMARK = "SPY"


def main() -> None:
    out_dir = ROOT / "data" / "real_sp100"
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = tuple(dict.fromkeys((*SP100_SYMBOLS, BENCHMARK)))
    print(f"Downloading {len(symbols)} symbols (S&P 100 + benchmark), 2018-01-01 to present...")
    frames = download_yfinance(symbols, "2018-01-01", None, "1d")
    ok, empty = 0, []
    for symbol, frame in frames.items():
        if frame.empty:
            empty.append(symbol)
            continue
        frame = frame.drop(columns="symbol")
        frame.index.name = "date"
        frame.to_csv(out_dir / f"{symbol}.csv")
        ok += 1
    print(f"Saved {ok}/{len(symbols)} symbols to {out_dir}")
    if empty:
        print(f"No data returned for: {empty}")


if __name__ == "__main__":
    main()
