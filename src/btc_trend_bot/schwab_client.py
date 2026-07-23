"""Schwab market-data client wrapper for equity paper trading.

Data-only: this module never places an order. Schwab's API has no paper-trading
support, so real Schwab market data is combined with locally simulated fills
elsewhere (experiments/run_equity_paper_step.py) -- this module's only job is
authenticating and fetching daily OHLCV bars.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Protocol, Sequence

import pandas as pd

NormalizeFn = Callable[[pd.DataFrame, str], pd.DataFrame]


class SchwabResponse(Protocol):
    status_code: int
    text: str

    def json(self) -> dict: ...


class SchwabClientProtocol(Protocol):
    def get_price_history_every_day(
        self,
        symbol: str,
        *,
        start_datetime: datetime | None = None,
        end_datetime: datetime | None = None,
        need_extended_hours_data: bool | None = None,
        need_previous_close: bool | None = None,
    ) -> SchwabResponse: ...


def load_schwab_client(token_path: Path, api_key: str, app_secret: str) -> "SchwabClientProtocol":
    """Load a client from a previously-minted token file. Refreshes automatically
    when the access token expires, but Schwab refresh tokens themselves hard-expire
    after 7 days -- callers must re-run bootstrap_token_interactive weekly."""
    from schwab.auth import client_from_token_file
    return client_from_token_file(str(token_path), api_key, app_secret)


def bootstrap_token_interactive(token_path: Path, api_key: str, app_secret: str, callback_url: str) -> None:
    """One-time (and weekly-recurring, per Schwab's 7-day refresh-token expiry),
    human-run OAuth login -- opens a browser. Never call this from step/systemd."""
    from schwab.auth import client_from_login_flow
    client_from_login_flow(api_key, app_secret, callback_url, str(token_path))


def fetch_schwab_ohlcv(
    client: SchwabClientProtocol,
    symbols: Sequence[str],
    lookback_days: int,
    normalize_fn: NormalizeFn,
) -> dict[str, pd.DataFrame]:
    """Fetch trailing daily OHLCV for each symbol and normalize it to the same
    dict[str, DataFrame] shape run_equity_real_data_walkforward.download_yfinance()
    returns, via the caller-supplied normalize_fn (== that module's _normalize)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        response = client.get_price_history_every_day(
            symbol, start_datetime=start, end_datetime=end, need_previous_close=False,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Schwab price history failed for {symbol}: HTTP {response.status_code}: {response.text}")
        candles = response.json().get("candles", [])
        if not candles:
            raise RuntimeError(f"Schwab returned no candles for {symbol}")
        raw = pd.DataFrame(candles)
        raw["datetime"] = pd.to_datetime(raw["datetime"], unit="ms", utc=True)
        raw = raw.set_index("datetime")
        out[symbol] = normalize_fn(raw, symbol)
    return out
