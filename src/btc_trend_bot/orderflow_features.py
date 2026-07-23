"""Information-driven bars and trade-flow features built from raw tick data
(btc_trend_bot.orderflow_data), instead of fixed-time candles.

Fixed-time candles (even 5-minute ones) sample uniformly in clock time
regardless of how much actually happened -- a quiet Sunday and a violent
news-driven hour get equal representation. Information-driven bars (Lopez de
Prado's terminology) sample by activity instead: a new bar closes once a
fixed amount of *volume* or *dollar turnover* has traded, so bars are dense
in dollar terms rather than dense in wall-clock terms. This is the
structural difference from candlesticks this module exists to provide, not
just a faster clock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_trades(path: str) -> pd.DataFrame:
    trades = pd.read_parquet(path)
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], unit="ms", utc=True)
    trades["signed_amount"] = np.where(trades["side"] == "buy", trades["amount"], -trades["amount"])
    trades["signed_cost"] = np.where(trades["side"] == "buy", trades["cost"], -trades["cost"])
    return trades.sort_values("timestamp").reset_index(drop=True)


def build_volume_bars(trades: pd.DataFrame, volume_per_bar: float) -> pd.DataFrame:
    """Each bar closes once `volume_per_bar` BTC has traded (either side)."""
    cum_volume = trades["amount"].cumsum()
    bar_id = (cum_volume // volume_per_bar).astype(int)
    return _aggregate_bars(trades, bar_id)


def build_dollar_bars(trades: pd.DataFrame, dollars_per_bar: float) -> pd.DataFrame:
    """Each bar closes once `dollars_per_bar` USDT has traded (either side).
    More stable than volume bars across large price regime changes, since a
    fixed BTC-volume threshold represents wildly different dollar risk at
    $20k BTC vs $100k BTC."""
    cum_cost = trades["cost"].cumsum()
    bar_id = (cum_cost // dollars_per_bar).astype(int)
    return _aggregate_bars(trades, bar_id)


def _aggregate_bars(trades: pd.DataFrame, bar_id: pd.Series) -> pd.DataFrame:
    grouped = trades.groupby(bar_id)
    bars = grouped.agg(
        open_time=("timestamp", "first"),
        close_time=("timestamp", "last"),
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("amount", "sum"),
        dollar_volume=("cost", "sum"),
        n_trades=("price", "size"),
        buy_volume=("signed_amount", lambda s: s[s > 0].sum()),
        sell_volume=("signed_amount", lambda s: -s[s < 0].sum()),
    ).reset_index(drop=True)
    # Trade-flow imbalance: (buy - sell) / total, in [-1, 1]. This is the
    # core order-flow feature -- who was actually crossing the spread more
    # aggressively within this bar, not just where price ended up.
    bars["flow_imbalance"] = (bars.buy_volume - bars.sell_volume) / (bars.buy_volume + bars.sell_volume)
    bars["return"] = bars.close.pct_change()
    bars["avg_trade_size"] = bars.volume / bars.n_trades
    return bars.drop(bars.index[-1])  # last bar is likely partial/incomplete


def rolling_flow_imbalance(bars: pd.DataFrame, window: int) -> pd.Series:
    """Multi-bar rolling imbalance -- smooths single-bar noise, still an
    order-flow feature (not a price feature) since it's built from
    buy/sell volume, not price action."""
    buy = bars.buy_volume.rolling(window).sum()
    sell = bars.sell_volume.rolling(window).sum()
    return (buy - sell) / (buy + sell)
