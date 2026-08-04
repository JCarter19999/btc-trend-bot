"""Fetch real SPXW 0DTE option-chain BBO snapshots for the first-hour
call/put signal test (see run_spx_first_hour_0dte_test.py).

One request per trading day, priced with `metadata.get_cost` before any
download (Databento bills by bytes delivered, not calendar span -- see
[[opra-options-data-lane]] memory). A 5-minute window is the billing floor
for cbbo-1m, so we take the full 5 minutes even though only the first
snapshot is used.

Window: 14:30-14:35 UTC = 10:30-10:35am ET during EDT (daylight saving),
which covers the whole date range this script is run against
(2026-05-06 - 2026-07-31, entirely inside EDT). This is the "first hour
just closed" decision point for the signal.

Parent symbology (SPXW.OPT) pulls the whole 0DTE chain for that day, which
is what strike selection needs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import databento as db
import pandas as pd

ENV_PATH = Path("/home/joey/.config/databento/databento.env")
DATASET = "OPRA.PILLAR"
SCHEMA = "cbbo-1m"


def load_key() -> str:
    if os.environ.get("DATABENTO_API_KEY"):
        return os.environ["DATABENTO_API_KEY"]
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("DATABENTO_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no DATABENTO_API_KEY")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "data" / "opra_spx"))
    ap.add_argument("--window-start", default="14:30", help="UTC HH:MM window start")
    ap.add_argument("--window-end", default="14:35", help="UTC HH:MM window end")
    ap.add_argument("--symbols", nargs="+", default=["SPXW.OPT"], help="parent symbology root(s), e.g. XSP.OPT")
    ap.add_argument("--max-spend", type=float, required=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    WINDOW_START, WINDOW_END = args.window_start, args.window_end
    SYMBOLS = args.symbols

    client = db.Historical(load_key())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    days = list(pd.bdate_range(args.start, args.end, tz="UTC"))
    todo = [d for d in days if not (outdir / f"{d:%Y-%m-%d}.parquet").exists()]
    print(f"{len(days)} weekdays in range, {len(todo)} not yet fetched")
    if not todo:
        print("nothing to do")
        return 0

    sample = todo[: min(5, len(todo))]
    unit_costs = []
    for d in sample:
        try:
            unit_costs.append(client.metadata.get_cost(
                dataset=DATASET, schema=SCHEMA, symbols=SYMBOLS, stype_in="parent",
                start=f"{d:%Y-%m-%d}T{WINDOW_START}", end=f"{d:%Y-%m-%d}T{WINDOW_END}",
            ))
        except db.common.error.BentoClientError:
            continue
    unit = max(unit_costs) if unit_costs else 0.031  # fallback: observed unit cost from prior run
    projected = unit * len(todo)
    print(f"unit cost ${unit:.5f} (max of {len(sample)} sampled days)")
    print(f"projected total for {len(todo)} days: ${projected:.2f}")
    print(f"ceiling: ${args.max_spend:.2f}")

    if projected > args.max_spend:
        print("ABORT: projected spend exceeds --max-spend")
        return 1
    if not args.execute:
        print("priced only; re-run with --execute to download")
        return 0

    spent = 0.0
    for i, d in enumerate(todo, 1):
        day = f"{d:%Y-%m-%d}"
        try:
            cost = client.metadata.get_cost(
                dataset=DATASET, schema=SCHEMA, symbols=SYMBOLS, stype_in="parent",
                start=f"{day}T{WINDOW_START}", end=f"{day}T{WINDOW_END}",
            )
        except db.common.error.BentoClientError as e:
            print(f"{day}: no symbols resolve (likely a market holiday), skipping: {e}")
            continue
        if spent + cost > args.max_spend:
            print(f"ABORT at {day}: would exceed --max-spend ({spent + cost:.2f} > {args.max_spend})")
            return 1
        try:
            data = client.timeseries.get_range(
                dataset=DATASET, schema=SCHEMA, symbols=SYMBOLS, stype_in="parent",
                start=f"{day}T{WINDOW_START}", end=f"{day}T{WINDOW_END}",
            )
            df = data.to_df()
        except Exception as e:
            print(f"{day}: ERROR {e}")
            continue
        spent += cost
        df.to_parquet(outdir / f"{day}.parquet")
        print(f"[{i}/{len(todo)}] {day}: {len(df)} rows, ${cost:.5f} (spent ${spent:.2f})")

    print(f"done, total spent ${spent:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
