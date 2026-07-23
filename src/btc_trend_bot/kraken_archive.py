from __future__ import annotations

import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable

import pandas as pd

from btc_trend_bot.data import REQUIRED_COLUMNS, normalize_ohlcv, timeframe_to_minutes

# Kraken's downloadable OHLCVT CSVs are headerless and contain:
# timestamp, open, high, low, close, volume, trades.
KRAKEN_ARCHIVE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trades",
]


def _compact_symbol(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def kraken_pair_candidates(symbol: str, pair_override: str | None = None) -> set[str]:
    """Return plausible Kraken archive pair names for a unified symbol.

    Kraken historically uses XBT for Bitcoin and sometimes exposes internal
    asset-code variants such as XXBTZUSD. The archive normally uses XBTUSD.
    """
    if pair_override:
        return {_compact_symbol(pair_override)}

    compact = _compact_symbol(symbol)
    candidates = {compact}
    if compact.startswith("BTC"):
        quote = compact[3:]
        candidates.update({f"XBT{quote}", f"XXBTZ{quote}"})
    elif compact.startswith("XBT"):
        quote = compact[3:]
        candidates.update({f"BTC{quote}", f"XXBTZ{quote}"})
    return candidates


def _member_pair_and_interval(name: str) -> tuple[str, int] | None:
    basename = PurePosixPath(name).name
    match = re.fullmatch(r"(.+)_([0-9]+)\.csv", basename, flags=re.IGNORECASE)
    if not match:
        return None
    return _compact_symbol(match.group(1)), int(match.group(2))


def find_kraken_archive_member(
    names: Iterable[str],
    symbol: str,
    timeframe: str,
    pair_override: str | None = None,
) -> str:
    interval = timeframe_to_minutes(timeframe)
    candidates = kraken_pair_candidates(symbol, pair_override=pair_override)
    matches: list[str] = []
    available_for_pair: set[int] = set()

    for name in names:
        parsed = _member_pair_and_interval(name)
        if parsed is None:
            continue
        pair, member_interval = parsed
        if pair not in candidates:
            continue
        available_for_pair.add(member_interval)
        if member_interval == interval:
            matches.append(name)

    if not matches:
        available = ", ".join(str(value) for value in sorted(available_for_pair)) or "none"
        raise FileNotFoundError(
            "No Kraken archive CSV matched "
            f"symbol={symbol!r}, timeframe={timeframe!r}, candidates={sorted(candidates)}. "
            f"Available intervals for matching pair names: {available}."
        )
    if len(matches) > 1:
        # Prefer the shortest path, which is normally the canonical top-level file.
        matches.sort(key=lambda value: (len(PurePosixPath(value).parts), len(value), value))
    return matches[0]


def _read_archive_csv(handle: BinaryIO, source_name: str) -> pd.DataFrame:
    raw = pd.read_csv(handle, header=None)
    if raw.shape[1] == 7:
        raw.columns = KRAKEN_ARCHIVE_COLUMNS
    elif raw.shape[1] == 8:
        # Tolerate REST-shaped exports that include VWAP before volume/count.
        raw.columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "vwap",
            "volume",
            "trades",
        ]
    else:
        raise ValueError(
            f"Kraken OHLCVT member {source_name!r} has {raw.shape[1]} columns; "
            "expected 7 archive columns or 8 REST-style columns."
        )

    timestamp_numeric = pd.to_numeric(raw["timestamp"], errors="coerce")
    if timestamp_numeric.isna().any():
        raise ValueError(f"Kraken archive member {source_name!r} contains invalid timestamps.")

    # Archive timestamps are Unix seconds. Also tolerate millisecond exports.
    unit = "ms" if float(timestamp_numeric.abs().max()) >= 10_000_000_000 else "s"
    raw["timestamp"] = pd.to_datetime(timestamp_numeric, unit=unit, utc=True)
    return raw[REQUIRED_COLUMNS].copy()


def load_kraken_ohlcvt_archive(
    archive_path: str | Path,
    symbol: str,
    timeframe: str,
    pair_override: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load one Kraken OHLCVT CSV from a ZIP, directory, or direct CSV path."""
    source = Path(archive_path)
    if not source.exists():
        raise FileNotFoundError(f"Kraken archive path not found: {source}")

    if source.is_dir():
        candidates = [str(path.relative_to(source)).replace("\\", "/") for path in source.rglob("*.csv")]
        member = find_kraken_archive_member(
            candidates, symbol=symbol, timeframe=timeframe, pair_override=pair_override
        )
        with (source / Path(member)).open("rb") as handle:
            frame = _read_archive_csv(handle, member)
    elif source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            member = find_kraken_archive_member(
                archive.namelist(), symbol=symbol, timeframe=timeframe, pair_override=pair_override
            )
            with archive.open(member) as handle:
                frame = _read_archive_csv(handle, member)
    elif source.suffix.lower() == ".csv":
        member = source.name
        with source.open("rb") as handle:
            frame = _read_archive_csv(handle, member)
    else:
        raise ValueError("Kraken history source must be a .zip file, .csv file, or extracted directory.")

    normalized, _ = normalize_ohlcv(frame, timeframe=timeframe)
    return normalized, member
