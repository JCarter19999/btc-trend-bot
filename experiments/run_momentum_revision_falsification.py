"""Falsification test for the Momentum (simple_trend) mechanism card
(MECHANISM_CARDS.md). Two competing mechanism claims for the same observed
edge:

  (A) Joey's hypothesis: persistence caused by slow institutional
      repositioning -- if real, the edge should be measurably stronger
      around periods when institutions are actively repositioning (visible
      via real analyst rating changes/revisions), not just whenever
      relative-strength happens to look strong.
  (B) This project's own more skeptical existing characterization
      (EQUITY_EXIT_REGIME_SIMPLE_TREND.md): "rotate into whichever of four
      historically exceptional growth stocks has the strongest relative
      momentum... not a stock-picking or market-timing edge" -- i.e. this
      may just be "we found 4 stocks that went up a lot," with no real
      repositioning-lag mechanism underneath the price-momentum symptom.

This test tries to discriminate between them using REAL analyst-revision
history (yfinance `Ticker.upgrades_downgrades`, Action in {'up','down'} --
genuine rating changes, not 'main'/'reit' reiterations or 'init' coverage
starts), joined to `simple_trend`'s real per-trade returns from the
walk-forward harness (same method as the regime classifier's Check 1).

Coverage confirmed empirically before trusting anything downstream:
AAPL/MSFT/NVDA upgrades_downgrades history goes back to 2012-2016; TSLA only
to 2019-04-25 -- stated plainly, not silently assumed to match the others.
`Ticker.earnings_dates` free-tier depth is only ~25 quarters (roughly
2020-onward) -- too short to usefully split the full 2018-2026 sample, so
this test uses the revision-event signal (fuller history) as primary and
does not attempt an earnings-surprise split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_equity_real_data_walkforward import build_candidates, load_config, load_csv_dir, walk_forward  # noqa: E402

N_NULL_SEEDS = 1000
REVISION_WINDOW_DAYS = 10  # "active repositioning" = a real up/down rating change within the last N trading days


def shuffled_null_percentile(labels: np.ndarray, values: np.ndarray, real_stat: float, seeds: int = N_NULL_SEEDS):
    rng_stats = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(labels)
        if shuffled.sum() in (0, len(labels)):
            continue
        rng_stats.append(values[shuffled].mean() - values[~shuffled].mean())
    rng_stats = np.array(rng_stats)
    pct = float((rng_stats < real_stat).mean() * 100)
    return float(rng_stats.mean()), float(rng_stats.std()), pct


def fetch_revision_dates(symbol: str) -> tuple[pd.DatetimeIndex, dict]:
    t = yf.Ticker(symbol)
    ud = t.upgrades_downgrades
    diag = {"symbol": symbol, "total_rows": int(len(ud)), "coverage_start": str(ud.index.min().date()),
            "coverage_end": str(ud.index.max().date())}
    revisions = ud[ud["Action"].isin(["up", "down"])]
    diag["n_revision_events"] = int(len(revisions))
    dates = pd.DatetimeIndex(revisions.index.tz_localize(None).normalize().unique())
    return dates, diag


def main() -> None:
    cfg = load_config(ROOT / "configs" / "real_data.yaml")
    all_symbols = tuple(dict.fromkeys((*cfg.symbols, cfg.benchmark)))
    frames = load_csv_dir(all_symbols, ROOT / "data" / "real")

    print("Fetching real analyst-revision history (yfinance upgrades_downgrades) per symbol...")
    revision_dates_by_symbol: dict[str, pd.DatetimeIndex] = {}
    diagnostics = []
    for sym in cfg.symbols:
        dates, diag = fetch_revision_dates(sym)
        revision_dates_by_symbol[sym] = dates
        diagnostics.append(diag)
        print(f"  {sym}: {diag['n_revision_events']} real up/down revisions, "
              f"{diag['coverage_start']} to {diag['coverage_end']}")

    print("\nRunning simple_trend walk-forward over cached 2018-2026 real daily data...")
    candidates = build_candidates(frames, cfg.benchmark, cfg)
    _, winners = walk_forward(candidates, cfg, selection="simple_trend")
    winners = winners.copy()
    winners["signal_date"] = pd.to_datetime(winners["signal_time"]).dt.tz_localize(None).dt.normalize()

    def active_revision(row) -> bool:
        sym = row["symbol"] if "symbol" in row else None
        if sym is None or sym not in revision_dates_by_symbol:
            return False
        dates = revision_dates_by_symbol[sym]
        if len(dates) == 0:
            return False
        d = row["signal_date"]
        window_start = d - pd.Timedelta(days=REVISION_WINDOW_DAYS)
        return bool(((dates >= window_start) & (dates <= d)).any())

    symbol_col = "symbol" if "symbol" in winners.columns else None
    print(f"\nColumns in winners table: {winners.columns.tolist()}")
    if symbol_col is None:
        print("WARNING: no 'symbol' column in winners table -- cannot join per-symbol revision dates. "
              "Falling back to ANY-symbol-revised-recently as a market-wide proxy (weaker test, stated plainly).")
        all_dates = pd.DatetimeIndex(sorted(set().union(*[set(d) for d in revision_dates_by_symbol.values()])))

        def active_revision_market(row) -> bool:
            d = row["signal_date"]
            window_start = d - pd.Timedelta(days=REVISION_WINDOW_DAYS)
            return bool(((all_dates >= window_start) & (all_dates <= d)).any())

        winners["active_revision"] = winners.apply(active_revision_market, axis=1)
    else:
        winners["active_revision"] = winners.apply(active_revision, axis=1)

    n_active = int(winners["active_revision"].sum())
    n_quiet = int((~winners["active_revision"]).sum())
    mean_active = float(winners.loc[winners["active_revision"], "net_return"].mean()) if n_active else float("nan")
    mean_quiet = float(winners.loc[~winners["active_revision"], "net_return"].mean()) if n_quiet else float("nan")
    real_diff = mean_active - mean_quiet
    null_mean, null_std, pct = shuffled_null_percentile(
        winners["active_revision"].to_numpy(), winners["net_return"].to_numpy(), real_diff)

    print(f"\n=== Falsification check: does simple_trend perform better around active analyst revisions? ===")
    print(f"trades: total={len(winners)} active_revision={n_active} quiet={n_quiet}")
    print(f"mean net_return | active revision (±{REVISION_WINDOW_DAYS}d): {mean_active*100:6.2f}%")
    print(f"mean net_return | quiet:                          {mean_quiet*100:6.2f}%")
    print(f"real diff: {real_diff*100:+.2f}pp  null_mean={null_mean*100:+.2f}pp null_std={null_std*100:.2f}pp "
          f"percentile={pct:.0f}")

    out_dir = ROOT / "outputs" / "momentum_revision_falsification"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump({
            "revision_data_diagnostics": diagnostics,
            "used_per_symbol_join": symbol_col is not None,
            "revision_window_days": REVISION_WINDOW_DAYS,
            "n_trades": len(winners), "n_active_revision": n_active, "n_quiet": n_quiet,
            "mean_return_active_revision": mean_active, "mean_return_quiet": mean_quiet,
            "real_diff": real_diff, "null_mean": null_mean, "null_std": null_std, "null_percentile": pct,
        }, fh, indent=2, default=str)
    print(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
