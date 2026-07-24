"""Quick pivot check, Phase 3 candidate D (FX transmission): EURUSD/USDJPY
overnight move -> SPY first hour, plus independence vs. the DAX signal.
Bonds (candidate C) couldn't be tested this way -- TLT/IEF/^TNX only have
US-session hourly bars via yfinance (13:30-19:30 UTC), no pre-US-open
data at this data tier. FX trades 24/5 so the same european_pre_us_open_return
signal window (day-start through last bar before 13:30 UTC) applies
unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_european_lead_us_first_hour_study import (  # noqa: E402
    download_hourly, european_pre_us_open_return, us_first_hour_return,
)
from run_taiwan_semiconductor_lead_lag_study import analyze  # noqa: E402

spy = download_hourly("SPY")
dax = download_hourly("^GDAXI")
target = us_first_hour_return(spy)
dax_signal = european_pre_us_open_return(dax)

for ticker, label in [("EURUSD=X", "EURUSD"), ("USDJPY=X", "USDJPY")]:
    fx = download_hourly(ticker)
    sig = european_pre_us_open_return(fx)
    r = analyze(sig, target, f"{label}_-> SPY_first_hour")
    print(f"\n{label}: n={r.get('n_days', r.get('n'))} "
          f"dir_corr={r.get('directional_correlation')} (pctile={r.get('directional_corr_percentile_vs_shuffled')}) "
          f"mag_corr={r.get('magnitude_correlation')} (pctile={r.get('magnitude_corr_percentile_vs_shuffled')})")
    joined = pd.concat([sig.rename("fx"), dax_signal.rename("dax")], axis=1).dropna()
    corr = float(np.corrcoef(joined["fx"], joined["dax"])[0, 1]) if len(joined) >= 30 else None
    print(f"  independence vs DAX: n={len(joined)}, correlation={corr}")
