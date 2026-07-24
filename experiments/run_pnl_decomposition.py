"""Task #6: apply the validated P&L decomposition
(`implied_greeks.decompose_option_pnl`) to the European signal's actual
completed real-quote SPY 0DTE/1DTE trades -- answers "did this trade lose
because the signal was wrong, or because I overpaid for volatility,"
splitting each trade's P&L into underlying-move / IV-change / theta /
residual(gamma+everything else).

The saved raw trades only kept option-side fields (entry_ask, exit_bid,
strike, expiration, right) -- entry/exit SPY spot aren't in the parquet,
so they're reconstructed here from the same already-cached SPY 60m bars
(data/intraday/SPY_60m.csv, no new API call): entry_spot = that day's
13:30 UTC bar's open, exit_spot = the same bar's close (the bar spans
exactly the traded window, 13:30-14:30 UTC / 9:30-10:30 ET -- confirmed
against thetadata_intraday_pricing.real_first_hour_option_trade's own
entry/exit windows).

DTE handled at intraday (fractional-year) resolution, not calendar days
-- 0DTE trades expire the same calendar day, so integer day-counting
would give entry_dte_years == exit_dte_years == 0 and make every 0DTE
implied-vol solve fail. Standard US equity option expiration cutoff
(16:00 ET / 20:00 UTC) used as the expiration timestamp.
"""

from __future__ import annotations

import json
import sys
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btc_trend_bot.implied_greeks import decompose_option_pnl  # noqa: E402
from run_european_lead_us_first_hour_study import download_hourly  # noqa: E402

ENTRY_UTC = dtime(13, 30)
EXPIRY_CUTOFF_UTC = dtime(20, 0)  # 16:00 ET


def main() -> None:
    spy = download_hourly("SPY")
    entry_bars = spy[spy.index.time == ENTRY_UTC].copy()
    entry_bars["date"] = entry_bars.index.date
    entry_bars = entry_bars.set_index("date")

    out = ROOT / "outputs" / "pnl_decomposition"
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for dte in (0, 1):
        trades = pd.read_parquet(
            ROOT / "outputs" / "european_signal_options_real_data_retest" / f"raw_trades_{dte}dte.parquet")
        n_decomposed, n_skipped = 0, 0
        for trade_date, row in trades.iterrows():
            d = trade_date.date()
            if d not in entry_bars.index:
                n_skipped += 1
                continue
            bar = entry_bars.loc[d]
            entry_spot, exit_spot = float(bar["open"]), float(bar["close"])
            entry_dt = pd.Timestamp.combine(d, ENTRY_UTC).tz_localize("UTC")
            exit_dt = entry_dt + pd.Timedelta(hours=1)
            expiry_dt = pd.Timestamp.combine(row["expiration"], EXPIRY_CUTOFF_UTC).tz_localize("UTC")
            entry_dte_years = (expiry_dt - entry_dt).total_seconds() / (365 * 86400)
            exit_dte_years = (expiry_dt - exit_dt).total_seconds() / (365 * 86400)
            days_held = (exit_dt - entry_dt).total_seconds() / 86400

            dec = decompose_option_pnl(
                entry_price=row["entry_ask"], exit_price=row["exit_bid"],
                entry_spot=entry_spot, exit_spot=exit_spot, strike=row["strike"],
                entry_dte_years=entry_dte_years, exit_dte_years=exit_dte_years,
                right=row["right"], days_held=days_held)
            if dec is None:
                n_skipped += 1
                continue
            dec.update({"dte": dte, "trade_date": str(d), "right": row["right"],
                        "strike": row["strike"], "entry_spot": entry_spot, "exit_spot": exit_spot,
                        "net_return": row["net_return"]})
            all_rows.append(dec)
            n_decomposed += 1
        print(f"{dte}DTE: {n_decomposed} decomposed, {n_skipped} skipped (no matching spot bar or IV unsolvable)", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_parquet(out / "decomposed_trades.parquet")

    print("\n=== P&L decomposition, mean $ contribution per trade (as % of total |P&L|) ===")
    for dte in (0, 1):
        d = df[df["dte"] == dte]
        if d.empty:
            continue
        total_abs = d["total_pnl"].abs().sum()
        summary = {
            "dte": dte, "n_trades": len(d),
            "mean_total_pnl": float(d["total_pnl"].mean()),
            "mean_underlying_move_pnl": float(d["underlying_move_pnl"].mean()),
            "mean_iv_change_pnl": float(d["iv_change_pnl"].mean()),
            "mean_theta_pnl": float(d["theta_pnl"].mean()),
            "mean_residual_gamma_pnl": float(d["residual_gamma_pnl"].mean()),
            "pct_of_|pnl|_from_underlying_move": float(d["underlying_move_pnl"].abs().sum() / total_abs * 100),
            "pct_of_|pnl|_from_iv_change": float(d["iv_change_pnl"].abs().sum() / total_abs * 100),
            "pct_of_|pnl|_from_theta": float(d["theta_pnl"].abs().sum() / total_abs * 100),
            "pct_of_|pnl|_from_residual_gamma": float(d["residual_gamma_pnl"].abs().sum() / total_abs * 100),
            "mean_entry_iv": float(d["entry_iv"].mean()), "mean_exit_iv": float(d["exit_iv"].mean()),
            "win_rate": float((d["total_pnl"] > 0).mean()),
            "loser_mean_theta_pnl": float(d.loc[d["total_pnl"] < 0, "theta_pnl"].mean()) if (d["total_pnl"] < 0).any() else None,
            "loser_mean_underlying_move_pnl": float(d.loc[d["total_pnl"] < 0, "underlying_move_pnl"].mean()) if (d["total_pnl"] < 0).any() else None,
        }
        print(json.dumps(summary, indent=2))
        (out / f"summary_{dte}dte.json").write_text(json.dumps(summary, indent=2))

    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
