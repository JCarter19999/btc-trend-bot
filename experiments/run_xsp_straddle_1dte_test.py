"""Buy an XSP straddle at 10:30 ET on the contract expiring the NEXT trading
day (1 DTE at entry) instead of same-day (0DTE), hold to that expiration's
close, every day, no regime filter.

No new data needed: the entry snapshot's parent symbology (data/opra_xsp/)
already pulls every live expiry in the chain at that instant, we've just
only ever filtered it down to expiry == trade_date (0DTE) before. This
filters to expiry == next_trading_date instead, using the same real 10:30
ET bid/ask.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from btc_trend_bot.options_router.features import build_first_hour_features, fetch_confirmation_data  # noqa: E402
from btc_trend_bot.options_router.structures import load_chain, long_straddle, payoff_at_close, pnl_and_return  # noqa: E402

ENTRY_DIR = ROOT / "data" / "opra_xsp"
OUT_DIR = ROOT / "outputs" / "xsp_straddle_1dte_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    entry_days = sorted(pd.to_datetime(Path(f).stem) for f in glob.glob(str(ENTRY_DIR / "*.parquet")))

    bars = fetch_confirmation_data([], underlying="^GSPC")
    feature_df = build_first_hour_features(bars, underlying="^GSPC").set_index("date")
    trading_days = sorted(feature_df.index)

    rows = []
    for d in entry_days:
        if d not in feature_df.index:
            continue
        idx = trading_days.index(d)
        if idx + 1 >= len(trading_days):
            continue
        d1 = trading_days[idx + 1]
        if (d1 - d).days > 4:  # guard against a data gap masquerading as "next day"
            continue

        xsp_spot = float(feature_df.loc[d, "first_hour_close"]) / 10.0
        xsp_close_1dte = float(feature_df.loc[d1, "day_close"]) / 10.0

        entry_c = load_chain(ENTRY_DIR / f"{d:%Y-%m-%d}.parquet", d1, "C")
        entry_p = load_chain(ENTRY_DIR / f"{d:%Y-%m-%d}.parquet", d1, "P")
        structure = long_straddle(entry_c, entry_p, xsp_spot)
        if structure is None:
            continue

        payoff = payoff_at_close(structure, xsp_close_1dte)
        pnl, ret = pnl_and_return(structure, payoff)
        rows.append({
            "entry_date": d.strftime("%Y-%m-%d"),
            "expiry_date": d1.strftime("%Y-%m-%d"),
            "debit": structure.debit,
            "strike": structure.legs[0].strike,
            "xsp_spot": xsp_spot,
            "xsp_close_at_expiry": xsp_close_1dte,
            "return": ret,
        })

    trades = pd.DataFrame(rows)
    trades.to_csv(OUT_DIR / "trades.csv", index=False)

    s = trades["return"].dropna()
    summary = {"n": 0} if s.empty else {
        "n": int(len(s)),
        "win_rate": float((s > 0).mean()),
        "mean_return": float(s.mean()),
        "median_return": float(s.median()),
        "profit_factor": float(s[s > 0].sum() / -s[s < 0].sum()) if (s < 0).any() else float("inf"),
        "worst_trade": float(s.min()),
        "best_trade": float(s.max()),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
