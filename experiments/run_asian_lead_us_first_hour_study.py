"""Extends the DAX/FTSE -> SPY first-hour lead-lag finding to Asian
markets. Asian sessions close well before European ones even open (Nikkei
~06:00 UTC, Hang Seng/Shanghai ~06:30-07:30 UTC, vs. European open at
07:00 UTC and US open at 13:30 UTC) -- their FULL session is finished and
known long before US open, unlike Europe's which is still running when the
US opens. Uses full session open-to-close return rather than the
partial-session-before-cutoff logic Europe needed.

Also builds a combined Asia+Europe signal (does adding an even-earlier
region help beyond DAX alone, or is Asia's information already subsumed
by the time Europe has traded on it too?).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_european_lead_us_first_hour_study import (  # noqa: E402
    DATA_DIR, US_OPEN_UTC, N_SHUFFLE, download_hourly, us_first_hour_return, analyze,
)
from run_european_lead_us_first_hour_backtest import evaluate, random_control  # noqa: E402

ASIAN_SYMBOLS = {"^N225": "Nikkei_225", "^HSI": "Hang_Seng", "000001.SS": "Shanghai_Composite"}


def full_session_return(df: pd.DataFrame) -> pd.Series:
    """Whole-day open-to-close return, for sessions that finish well before
    US open (unlike Europe's, which is still trading at 13:30 UTC)."""
    df = df.copy()
    df["date"] = df.index.date
    out = {}
    for date, day in df.groupby("date"):
        day = day.sort_index()
        if len(day) < 2:
            continue
        out[date] = float(day["close"].iloc[-1] / day["open"].iloc[0] - 1)
    return pd.Series(out, name="asia_full_session_return")


def main() -> None:
    us_target = us_first_hour_return(download_hourly("SPY"))
    dax_df = download_hourly("^GDAXI")
    from run_european_lead_us_first_hour_study import european_pre_us_open_return
    dax_signal = european_pre_us_open_return(dax_df).rename("dax")

    print("=== Individual Asian markets vs. SPY first hour ===")
    asian_signals = {}
    for symbol, label in ASIAN_SYMBOLS.items():
        df = download_hourly(symbol)
        sig = full_session_return(df)
        asian_signals[label] = sig
        r = analyze(sig.rename("eu_pre_open_return"), us_target, label)
        print(f"\n{label}: n={r.get('n_days')}, dir_corr={r.get('directional_correlation')}, "
              f"dir_pctile={r.get('directional_corr_percentile_vs_shuffled')}, "
              f"mag_corr={r.get('magnitude_correlation')}, mag_pctile={r.get('magnitude_corr_percentile_vs_shuffled')}")

    print("\n=== Combined Asia (Nikkei+HSI) + Europe (DAX) signal ===")
    combined = pd.concat([asian_signals["Nikkei_225"].rename("nikkei"),
                           asian_signals["Hang_Seng"].rename("hsi"),
                           dax_signal], axis=1).dropna()
    combined_signal = combined.mean(axis=1).rename("eu_pre_open_return")
    r = analyze(combined_signal, us_target, "Nikkei+HSI+DAX_average")
    print(f"n={r.get('n_days')}, dir_corr={r.get('directional_correlation')}, "
          f"dir_pctile={r.get('directional_corr_percentile_vs_shuffled')}, "
          f"mag_corr={r.get('magnitude_correlation')}, mag_pctile={r.get('magnitude_corr_percentile_vs_shuffled')}")

    print("\n=== Backtest: combined signal vs. DAX-only, 2bp cost ===")
    df_combined = pd.concat([combined_signal.rename("eu_pre_open_return"), us_target], axis=1).dropna()
    df_combined.index = pd.to_datetime(df_combined.index)
    df_combined["direction"] = np.sign(df_combined["eu_pre_open_return"])
    abs_sig = df_combined["eu_pre_open_return"].abs()
    q75 = abs_sig.quantile(0.75)
    for name, direction in [("combined_every_day", df_combined["direction"]),
                             ("combined_top_quartile", df_combined["direction"].where(abs_sig >= q75, 0))]:
        res = evaluate(df_combined, direction, 2.0, name)
        print(f"{name}: trades={res.get('trades')} win={res.get('win_rate'):.3f} "
              f"mean={res.get('mean_return_bps'):.2f}bps sharpe={res.get('sharpe_annualized'):.2f}")

    import json
    out = ROOT / "outputs" / "asian_lead_us_first_hour_study"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"note": "see stdout capture for full results"}, indent=2))


if __name__ == "__main__":
    main()
