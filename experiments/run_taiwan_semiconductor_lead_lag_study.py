"""Phase 3 candidate A (Joey's ranking, 2026-07-24): does Taiwan's own
session carry real, no-lookahead information about US semiconductor
stocks' reaction later that day? Same lead-lag methodology as
EUROPEAN_LEAD_US_FIRST_HOUR_STUDY.txt, deliberately reused unchanged
where it applies -- this is testing a genuinely different information
transmission channel (Taiwan chip-supply-chain fundamentals -> US
semiconductor sector), not a new technical indicator.

Why Taiwan specifically, not "Asia" generally: this project already
tested broad Asian markets (Nikkei/HSI/Shanghai) as a predictor of SPY
and found a real but OPPOSITE-sign (contrarian) relationship -- a general
risk-sentiment signal, not a sector-specific one. Taiwan is different:
TSMC alone produces the large majority of the world's leading-edge chips,
so a Taiwan-session move plausibly carries sector-specific supply-chain/
fundamental information that a broad "risk-on/risk-off" signal wouldn't.
Testing both a broad-market proxy (^TWII, the Taiwan Weighted Index) and
the single-stock proxy (TSM, Taiwan Semiconductor's US-listed ADR, which
also trades in Taipei) against two US semiconductor targets (SOXX, SMH).

Independence check (the actual point of Phase 3): correlates this
signal against the ALREADY-VALIDATED DAX pre-US-open signal on the same
dates. A second edge that's highly correlated with the first isn't
really a second edge -- it's the same risk factor wearing a different
ticker.

No lookahead: Taiwan's session (09:00-13:30 Taipei = 01:00-05:30 UTC)
closes a full 8 hours before US open (13:30 UTC) and well before DAX's
own session even opens (07:00 UTC) -- everything used here is known
before a DAX trader, let alone a US trader, could act on it.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_european_lead_us_first_hour_study import (  # noqa: E402
    download_hourly, european_pre_us_open_return, us_first_hour_return,
)

N_SHUFFLE = 1000


def analyze(signal: pd.Series, target: pd.Series, label: str) -> dict:
    joined = pd.concat([signal.rename("signal"), target.rename("target")], axis=1).dropna()
    if len(joined) < 30:
        return {"label": label, "n": len(joined), "note": "too few overlapping days"}

    sig, tgt = joined["signal"], joined["target"]
    dir_corr = float(np.corrcoef(sig, tgt)[0, 1])
    mag_corr = float(np.corrcoef(sig.abs(), tgt.abs())[0, 1])

    rng = np.random.default_rng(0)
    shuffled_dir = np.array([np.corrcoef(sig.to_numpy(), rng.permutation(tgt.to_numpy()))[0, 1] for _ in range(N_SHUFFLE)])
    shuffled_mag = np.array([np.corrcoef(sig.abs().to_numpy(), rng.permutation(tgt.abs().to_numpy()))[0, 1] for _ in range(N_SHUFFLE)])
    dir_percentile = float((shuffled_dir < dir_corr).mean() * 100)
    mag_percentile = float((shuffled_mag < mag_corr).mean() * 100)

    return {
        "label": label, "n_days": len(joined),
        "date_range": f"{min(joined.index)} to {max(joined.index)}",
        "directional_correlation": dir_corr,
        "directional_corr_percentile_vs_shuffled": dir_percentile,
        "magnitude_correlation": mag_corr,
        "magnitude_corr_percentile_vs_shuffled": mag_percentile,
    }


def main() -> None:
    print("Downloading/loading hourly bars (Taiwan proxies, US semis, DAX for independence check)...", flush=True)
    twii = download_hourly("^TWII")
    tsm = download_hourly("TSM")
    soxx = download_hourly("SOXX")
    smh = download_hourly("SMH")
    dax = download_hourly("^GDAXI")

    twii_signal = european_pre_us_open_return(twii).rename("eu_pre_open_return")
    tsm_signal = european_pre_us_open_return(tsm).rename("eu_pre_open_return")
    dax_signal = european_pre_us_open_return(dax).rename("eu_pre_open_return")
    print(f"TWII session observations: {len(twii_signal)}, TSM: {len(tsm_signal)}, DAX (reference): {len(dax_signal)}", flush=True)

    results = []
    for target_df, target_label in [(soxx, "SOXX"), (smh, "SMH")]:
        target = us_first_hour_return(target_df)
        for sig, sig_label in [(twii_signal, "TWII"), (tsm_signal, "TSM")]:
            results.append(analyze(sig, target, f"{sig_label}_signal_-> {target_label}_first_hour"))

    print("\n" + "=" * 70)
    print("LEAD-LAG RESULTS")
    for r in results:
        print("\n" + "-" * 70)
        for k, v in r.items():
            print(f"{k}: {v}")

    # Independence check: correlate Taiwan signals against the already-validated DAX signal
    print("\n" + "=" * 70)
    print("INDEPENDENCE CHECK (Taiwan signal vs. already-validated DAX signal, same dates)")
    independence = {}
    for sig, sig_label in [(twii_signal, "TWII"), (tsm_signal, "TSM")]:
        joined = pd.concat([sig.rename("taiwan"), dax_signal.rename("dax")], axis=1).dropna()
        corr = float(np.corrcoef(joined["taiwan"], joined["dax"])[0, 1]) if len(joined) >= 30 else None
        independence[sig_label] = {"n_overlapping_days": len(joined), "correlation_with_dax_signal": corr}
        print(f"{sig_label} vs DAX: n={len(joined)}, correlation={corr}")

    out_dir = ROOT / "outputs" / "taiwan_semiconductor_lead_lag_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({"lead_lag": results, "independence_check": independence}, indent=2, default=str))
    print(f"\nWritten to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
