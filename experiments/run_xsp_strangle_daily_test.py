"""Buy an XSP strangle (5-10 points OTM each side) or straddle (ATM, same
strike) every day, no regime filter -- see what happens, hold to close vs.
exit at the noon-ET snapshot.

Uses real XSP 0DTE quotes from data/opra_xsp/ (10:30 entry) and
data/opra_xsp_exit/ (noon exit). XSP spot is proxied as SPX/10 (XSP's
redemption value is literally defined as 1/10 the SPX index level, and it
tracks tightly intraday), using the same no-lookahead SPX 5-minute bars as
the rest of this project's first-hour signal work.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from btc_trend_bot.options_router.backtest import available_days  # noqa: E402
from btc_trend_bot.options_router.exit_policy import evaluate_exit  # noqa: E402
from btc_trend_bot.options_router.features import build_first_hour_features, fetch_confirmation_data  # noqa: E402
from btc_trend_bot.options_router.structures import (  # noqa: E402
    load_chain, long_straddle, long_strangle, payoff_at_close, pnl_and_return, price_exit_from_chain,
)

ENTRY_DIR = ROOT / "data" / "opra_xsp"
EXIT_DIR = ROOT / "data" / "opra_xsp_exit"
OUT_DIR = ROOT / "outputs" / "xsp_strangle_daily_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH_POINTS = 7.5  # strangle: 5-10 XSP points OTM each side

EXIT_CFG = {"exit_policy": {"mode": "noon_if_profit_target_met", "profit_target_pct": 0.20, "hard_stop_pct": -0.50}}

BUILDERS = {
    "strangle": lambda c, p, spot: long_strangle(c, p, spot, WIDTH_POINTS),
    "straddle": lambda c, p, spot: long_straddle(c, p, spot),
}


def main() -> int:
    days = available_days(ENTRY_DIR, EXIT_DIR)
    print(f"{len(days)} days with both entry and exit XSP snapshots")

    bars = fetch_confirmation_data([], underlying="^GSPC")
    feature_df = build_first_hour_features(bars, underlying="^GSPC").set_index("date")

    rows = []
    for d in days:
        if d not in feature_df.index:
            continue
        row = feature_df.loc[d]
        xsp_spot = float(row["first_hour_close"]) / 10.0
        xsp_close = float(row["day_close"]) / 10.0

        entry_c = load_chain(ENTRY_DIR / f"{d:%Y-%m-%d}.parquet", d, "C")
        entry_p = load_chain(ENTRY_DIR / f"{d:%Y-%m-%d}.parquet", d, "P")
        exit_c = load_chain(EXIT_DIR / f"{d:%Y-%m-%d}.parquet", d, "C")
        exit_p = load_chain(EXIT_DIR / f"{d:%Y-%m-%d}.parquet", d, "P")

        for kind, build in BUILDERS.items():
            structure = build(entry_c, entry_p, xsp_spot)
            if structure is None:
                continue
            noon_value = price_exit_from_chain(structure, {"C": exit_c, "P": exit_p})
            close_payoff = payoff_at_close(structure, xsp_close)
            close_pnl, close_ret = pnl_and_return(structure, close_payoff)
            exit_decision = evaluate_exit(structure, noon_value, close_payoff, EXIT_CFG)

            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "kind": kind,
                "debit": structure.debit,
                "call_strike": structure.legs[0].strike,
                "put_strike": structure.legs[1].strike,
                "xsp_spot": xsp_spot,
                "xsp_close": xsp_close,
                "hold_to_close_return": close_ret,
                "dynamic_exit_return": exit_decision.ret,
                "dynamic_exit_reason": exit_decision.exit_reason,
            })

    trades = pd.DataFrame(rows)
    trades.to_csv(OUT_DIR / "trades.csv", index=False)

    def summarize(s: pd.Series) -> dict:
        s = s.dropna()
        if s.empty:
            return {"n": 0}
        wins = s > 0
        return {
            "n": int(len(s)),
            "win_rate": float(wins.mean()),
            "mean_return": float(s.mean()),
            "median_return": float(s.median()),
            "profit_factor": float(s[s > 0].sum() / -s[s < 0].sum()) if (s < 0).any() else float("inf"),
            "worst_trade": float(s.min()),
            "best_trade": float(s.max()),
        }

    summary = {"n_days": int(len(days)), "width_points": WIDTH_POINTS}
    for kind in BUILDERS:
        sub = trades[trades["kind"] == kind]
        summary[kind] = {
            "hold_to_close": summarize(sub["hold_to_close_return"]),
            "dynamic_exit_noon_or_close": summarize(sub["dynamic_exit_return"]),
            "dynamic_exit_reason_counts": sub["dynamic_exit_reason"].value_counts().to_dict() if not sub.empty else {},
        }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
