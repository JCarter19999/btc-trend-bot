"""Diagnostic (not a strategy backtest): does ThetaData's real open-interest
history (`option_history_open_interest`, never called anywhere else in this
project) give advance warning of the illiquid/not-yet-listed contracts that
caused the tail-hedge study's 69% skip rate and the strangle studies'
60-68% skip rates?

Three checks:
1. Liquid baseline -- OI on a handful of the SPY 0DTE call contracts from
   the fully-successful European-lead real-data retest (116/116 priced),
   to confirm the API call itself is being used correctly before trusting
   anything below.
2. Tail hedge -- reconstruct the exact 61 cycles from
   run_tail_hedge_real_data_retest.py (45-DTE, 10%-OTM SPY puts), pull each
   contract's OI history, and check whether OI as-of-entry-date would have
   flagged the ones that failed to price *before* attempting the trade.
3. Strangle skip spot-check (read-only) -- a handful of skipped candidates
   from outputs/short_strangle_chop_backtest_realized_vol/chop_gated_trades.csv,
   re-deriving their intended contract the same way the strangle script
   does, then checking OI.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btc_trend_bot.options_pricing import realized_vol  # noqa: E402
from btc_trend_bot.thetadata_pricing import (  # noqa: E402
    find_nearest_expiration, find_nearest_strike, get_client, real_put_trade,
)
from run_equity_options_real_data_retest import REAL_DATA_START  # noqa: E402
from run_equity_real_data_walkforward import load_config, load_csv_dir  # noqa: E402

HEDGE_DTE = 45
HEDGE_HOLD = 21
HEDGE_MONEYNESS = 0.90

STRANGLE_DTE = 30
STRANGLE_CALL_MONEYNESS = 1.05
STRANGLE_PUT_MONEYNESS = 0.95


def fetch_oi_series(symbol: str, expiration, strike: float, right: str, start_date, end_date) -> pd.DataFrame:
    client = get_client()
    try:
        oi = client.option_history_open_interest(
            symbol=symbol, expiration=expiration, strike=f"{strike:.2f}", right=right.upper(),
            start_date=start_date, end_date=end_date,
        )
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})
    if oi is None or oi.empty:
        return pd.DataFrame()
    return oi


def oi_as_of(oi_df: pd.DataFrame, as_of_date, date_col_candidates=("timestamp", "date", "report_date", "start_date")) -> float | None:
    if oi_df.empty or "error" in oi_df.columns:
        return None
    date_col = next((c for c in date_col_candidates if c in oi_df.columns), None)
    if date_col is None:
        return None
    d = oi_df.copy()
    d[date_col] = pd.to_datetime(d[date_col]).dt.tz_localize(None).dt.normalize().dt.date
    d = d[d[date_col] <= as_of_date]
    if d.empty:
        return None
    oi_col = "open_interest" if "open_interest" in d.columns else next(
        (c for c in d.columns if "interest" in c.lower()), None
    )
    if oi_col is None:
        return None
    return float(d.sort_values(date_col).iloc[-1][oi_col])


def check_liquid_baseline() -> list[dict]:
    print("\n=== Check 1: liquid baseline (SPY 0DTE calls, 116/116 priced originally) ===")
    df = pd.read_parquet(ROOT / "outputs/european_signal_options_real_data_retest/raw_trades_0dte.parquet")
    sample = df.head(5)
    rows = []
    for entry_date, row in sample.iterrows():
        entry_date = pd.Timestamp(entry_date).date()
        expiration = pd.Timestamp(row["expiration"]).date()
        strike, right = float(row["strike"]), row["right"]
        oi_df = fetch_oi_series("SPY", expiration, strike, right, entry_date - timedelta(days=5), entry_date)
        oi_val = oi_as_of(oi_df, entry_date)
        print(f"  {entry_date} SPY {expiration} {strike}{right[0].upper()}: OI as-of-entry={oi_val} "
              f"(raw rows={len(oi_df)}, cols={list(oi_df.columns)})")
        rows.append({"entry_date": str(entry_date), "expiration": str(expiration), "strike": strike,
                      "right": right, "oi_as_of_entry": oi_val, "raw_rows": len(oi_df)})
    return rows


def reconstruct_tail_hedge_cycles():
    cfg_base = load_config(ROOT / "configs/real_data.yaml")
    frames = load_csv_dir((cfg_base.benchmark,), ROOT / "data/real")
    spy = frames[cfg_base.benchmark].copy()
    spy["realized_vol"] = realized_vol(spy["close"].to_numpy(), 20)
    spy.index = pd.to_datetime(spy.index, utc=True)
    start_idx = spy.index.searchsorted(REAL_DATA_START)

    cycles = []
    i = max(60, start_idx)
    while i < len(spy) - HEDGE_HOLD - 1:
        signal_date = spy.index[i].date()
        entry_i, exit_i = i + 1, min(i + 1 + HEDGE_HOLD - 1, len(spy) - 1)
        entry_date, exit_date = spy.index[entry_i].date(), spy.index[exit_i].date()
        entry_spot = float(spy.iloc[entry_i].open)
        cycles.append(dict(signal_date=signal_date, entry_date=entry_date, exit_date=exit_date, entry_spot=entry_spot))
        i += HEDGE_HOLD
    return cfg_base, cycles


def check_tail_hedge() -> list[dict]:
    print("\n=== Check 2: tail hedge (45-DTE, 10%-OTM SPY puts, 61 cycles) ===")
    cfg_base, cycles = reconstruct_tail_hedge_cycles()
    print(f"Reconstructed {len(cycles)} cycles (original study: 61 cycles, 19 priced, 42 skipped)")

    # PASS 1: pricing only, no OI calls interleaved -- an earlier attempt that
    # interleaved OI fetches with pricing calls silently under-priced trades
    # that price fine in isolation (verified directly: cycle 1 above prices
    # successfully standalone but reported priced=False when OI calls ran in
    # the same loop). Root cause not fully confirmed (likely request-rate
    # throttling silently swallowed by this module's broad except-Exception
    # convention), but the fix -- separating the passes -- is confirmed to
    # resolve it, so pricing here should be trustworthy on its own.
    print("Pass 1/2: pricing all 61 cycles (no OI calls yet)...")
    priced_rows = []
    for n, c in enumerate(cycles, 1):
        symbol = cfg_base.benchmark
        expiration = find_nearest_expiration(symbol, c["signal_date"], HEDGE_DTE)
        if expiration is None or expiration <= c["exit_date"]:
            priced_rows.append(dict(**c, expiration=None, strike=None, priced=False,
                                     note="no valid expiration found (lifetime list)"))
            continue
        target_strike = c["entry_spot"] * HEDGE_MONEYNESS
        strike = find_nearest_strike(symbol, expiration, target_strike)
        if strike is None:
            priced_rows.append(dict(**c, expiration=str(expiration), strike=None, priced=False,
                                     note="no strike list at all for this expiration"))
            continue
        trade = real_put_trade(symbol, c["signal_date"], c["entry_date"], c["exit_date"], c["entry_spot"],
                                HEDGE_DTE, HEDGE_MONEYNESS)
        priced_rows.append(dict(**c, expiration=str(expiration), strike=strike, priced=trade is not None))
        print(f"  [{n}/{len(cycles)}] signal={c['signal_date']} strike={strike} exp={expiration} priced={trade is not None}")
        time.sleep(0.15)

    n_priced = sum(r["priced"] for r in priced_rows)
    print(f"Pass 1 result: {n_priced}/{len(priced_rows)} priced (compare to original study's 19/61)")

    # PASS 2: OI, as an independent pass over the same reconstructed contracts.
    print("Pass 2/2: fetching OI history for each cycle's contract...")
    rows = []
    for n, r in enumerate(priced_rows, 1):
        if r["strike"] is None:
            rows.append(dict(**r, oi_at_signal=None, oi_at_entry=None, earliest_oi_date=None, oi_raw_rows=0))
            continue
        expiration = pd.Timestamp(r["expiration"]).date()
        oi_df = fetch_oi_series(cfg_base.benchmark, expiration, r["strike"], "put",
                                 r["signal_date"] - timedelta(days=10), r["exit_date"])
        oi_at_signal = oi_as_of(oi_df, r["signal_date"])
        oi_at_entry = oi_as_of(oi_df, r["entry_date"])
        earliest_oi_date = None
        if not oi_df.empty and "error" not in oi_df.columns:
            date_col = next((col for col in ("timestamp", "date", "report_date", "start_date") if col in oi_df.columns), None)
            if date_col:
                earliest_oi_date = str(pd.to_datetime(oi_df[date_col]).dt.tz_localize(None).min().date())
        rows.append(dict(**r, oi_at_signal=oi_at_signal, oi_at_entry=oi_at_entry,
                          earliest_oi_date=earliest_oi_date, oi_raw_rows=len(oi_df)))
        print(f"  [{n}/{len(priced_rows)}] signal={r['signal_date']} strike={r['strike']} priced={r['priced']} "
              f"OI@signal={oi_at_signal} OI@entry={oi_at_entry} earliest_OI_date={earliest_oi_date}")
        time.sleep(0.15)
    return rows


def check_strangle_skips() -> list[dict]:
    print("\n=== Check 3: strangle skip spot-check (read-only, realized-vol pass) ===")
    csv_path = ROOT / "outputs/short_strangle_chop_backtest_realized_vol/chop_gated_trades.csv"
    df = pd.read_csv(csv_path)
    skipped = df[df["expiration"].isna()].copy()
    print(f"{len(skipped)} skipped candidates in this file, spot-checking up to 8")

    rows = []
    for _, row in skipped.head(8).iterrows():
        signal_date = pd.Timestamp(row["signal_time"]).date()
        entry_date = pd.Timestamp(row["entry_time"]).date()
        entry_spot = float(row["entry_spot"])

        expiration = find_nearest_expiration("SPY", signal_date, STRANGLE_DTE)
        if expiration is None:
            rows.append(dict(signal_date=str(signal_date), entry_date=str(entry_date),
                              note="no expiration found"))
            continue
        call_strike = find_nearest_strike("SPY", expiration, entry_spot * STRANGLE_CALL_MONEYNESS)
        put_strike = find_nearest_strike("SPY", expiration, entry_spot * STRANGLE_PUT_MONEYNESS)

        entry = dict(signal_date=str(signal_date), entry_date=str(entry_date), expiration=str(expiration),
                      call_strike=call_strike, put_strike=put_strike)
        for right, strike in (("call", call_strike), ("put", put_strike)):
            if strike is None:
                entry[f"{right}_oi_at_entry"] = None
                continue
            oi_df = fetch_oi_series("SPY", expiration, strike, right, entry_date - timedelta(days=10), entry_date)
            entry[f"{right}_oi_at_entry"] = oi_as_of(oi_df, entry_date)
            entry[f"{right}_oi_raw_rows"] = len(oi_df)
        print(f"  signal={signal_date} exp={expiration} call={call_strike}(OI={entry.get('call_oi_at_entry')}) "
              f"put={put_strike}(OI={entry.get('put_oi_at_entry')})")
        rows.append(entry)
    return rows


def main() -> None:
    out_dir = ROOT / "outputs" / "option_oi_liquidity_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = check_liquid_baseline()
    hedge_rows = check_tail_hedge()
    strangle_rows = check_strangle_skips()

    hedge_df = pd.DataFrame(hedge_rows)
    n_priced = int(hedge_df["priced"].sum()) if "priced" in hedge_df else 0
    n_total = len(hedge_df)
    print(f"\n=== Tail hedge summary ===\n{n_priced}/{n_total} priced with real quotes")

    if "priced" in hedge_df.columns:
        priced_oi = hedge_df.loc[hedge_df["priced"], "oi_at_entry"].dropna()
        skipped_oi = hedge_df.loc[~hedge_df["priced"], "oi_at_entry"].dropna()
        print(f"Priced trades: OI@entry available for {len(priced_oi)}/{hedge_df['priced'].sum()}, "
              f"mean={priced_oi.mean() if len(priced_oi) else float('nan')}")
        print(f"Skipped trades: OI@entry available for {len(skipped_oi)}/{(~hedge_df['priced']).sum()}, "
              f"mean={skipped_oi.mean() if len(skipped_oi) else float('nan')}")

        # Would a naive "OI@entry > 0" pre-trade gate have recovered a
        # materially different set than the 19 that happened to price?
        gate_pass = hedge_df["oi_at_entry"].fillna(0) > 0
        print(f"\nOI>0-at-entry gate would pass: {int(gate_pass.sum())}/{n_total} cycles")
        print(f"Overlap with actually-priced set: {int((gate_pass & hedge_df['priced']).sum())}")
        print(f"Gate passes but trade was NOT priced (OI present, quote still missing): "
              f"{int((gate_pass & ~hedge_df['priced']).sum())}")
        print(f"Gate fails but trade WAS priced (OI missing/zero despite a real quote existing): "
              f"{int((~gate_pass & hedge_df['priced']).sum())}")

    with open(out_dir / "summary.json", "w") as fh:
        json.dump({
            "liquid_baseline": baseline_rows,
            "tail_hedge_n_total": n_total,
            "tail_hedge_n_priced": n_priced,
            "tail_hedge_rows": hedge_rows,
            "strangle_skip_spotcheck": strangle_rows,
        }, fh, indent=2, default=str)
    hedge_df.to_csv(out_dir / "tail_hedge_oi_check.csv", index=False)
    print(f"\nWrote {out_dir / 'summary.json'} and {out_dir / 'tail_hedge_oi_check.csv'}")


if __name__ == "__main__":
    main()
