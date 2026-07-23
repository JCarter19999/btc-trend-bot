from __future__ import annotations

import math

import numpy as np
import pandas as pd


def extract_trades(
    bars: pd.DataFrame,
    position_col: str,
    return_col: str,
    equity_col: str,
    label: str,
    bars_per_day: float = 6.0,
) -> pd.DataFrame:
    """Extract non-zero directional episodes from a backtest.

    Exit execution costs on a flat transition are assigned to the trade that just
    ended. Direct long/short flips begin a new episode on the flip bar.
    """
    required = {"timestamp", "close", position_col, return_col, equity_col}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"Trade extraction missing columns: {sorted(missing)}")

    work = bars.reset_index(drop=True)
    position = work[position_col].fillna(0.0).astype(float)
    sign = np.sign(position).astype(int)
    records: list[dict[str, object]] = []
    start: int | None = None
    active_sign = 0

    def append_trade(start_index: int, end_index: int, side_sign: int) -> None:
        segment = work.iloc[start_index : end_index + 1]
        active = segment.loc[np.sign(segment[position_col].fillna(0.0)) == side_sign]
        if active.empty:
            return
        entry_index = int(active.index[0])
        last_active_index = int(active.index[-1])
        trade_return = float((1.0 + segment[return_col].fillna(0.0)).prod() - 1.0)
        entry_equity = float(work.loc[start_index - 1, equity_col]) if start_index > 0 else float(
            work.loc[start_index, equity_col] / (1.0 + work.loc[start_index, return_col])
        )
        exit_equity = float(work.loc[end_index, equity_col])
        records.append(
            {
                "strategy": label,
                "side": "long" if side_sign > 0 else "short",
                "entry_timestamp": work.loc[entry_index, "timestamp"],
                "last_active_timestamp": work.loc[last_active_index, "timestamp"],
                "exit_timestamp": work.loc[end_index, "timestamp"],
                "entry_close": float(work.loc[entry_index, "close"]),
                "exit_close": float(work.loc[last_active_index, "close"]),
                "active_bars": int(len(active)),
                "calendar_bars": int(end_index - start_index + 1),
                "active_days": float(len(active) / bars_per_day),
                "trade_return": trade_return,
                "pnl": float(exit_equity - entry_equity),
                "winning_trade": bool(trade_return > 0),
            }
        )

    for index, current_sign in enumerate(sign):
        current_sign = int(current_sign)
        if active_sign == 0 and current_sign != 0:
            start = index
            active_sign = current_sign
        elif active_sign != 0:
            if current_sign == 0:
                assert start is not None
                append_trade(start, index, active_sign)
                start = None
                active_sign = 0
            elif current_sign != active_sign:
                assert start is not None
                append_trade(start, index - 1, active_sign)
                start = index
                active_sign = current_sign

    if active_sign != 0 and start is not None:
        append_trade(start, len(work) - 1, active_sign)

    columns = [
        "strategy", "side", "entry_timestamp", "last_active_timestamp", "exit_timestamp",
        "entry_close", "exit_close", "active_bars", "calendar_bars", "active_days",
        "trade_return", "pnl", "winning_trade",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def summarize_trades(trades: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty:
        return {"trades": 0}
    winners = trades.loc[trades["trade_return"] > 0, "trade_return"]
    losers = trades.loc[trades["trade_return"] < 0, "trade_return"]
    gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    gross_loss = abs(float(trades.loc[trades["pnl"] < 0, "pnl"].sum()))
    sorted_pnl = trades["pnl"].sort_values(ascending=False)
    total_positive_pnl = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    return {
        "trades": int(len(trades)),
        "wins": int((trades["trade_return"] > 0).sum()),
        "losses": int((trades["trade_return"] < 0).sum()),
        "win_rate": float((trades["trade_return"] > 0).mean()),
        "average_trade_return": float(trades["trade_return"].mean()),
        "median_trade_return": float(trades["trade_return"].median()),
        "average_win": float(winners.mean()) if len(winners) else float("nan"),
        "average_loss": float(losers.mean()) if len(losers) else float("nan"),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "average_active_days": float(trades["active_days"].mean()),
        "largest_winner": float(trades["trade_return"].max()),
        "largest_loser": float(trades["trade_return"].min()),
        "top_5_profit_contribution": (
            float(sorted_pnl.head(5).sum() / total_positive_pnl)
            if total_positive_pnl > 0 else float("nan")
        ),
    }


def yearly_performance(
    bars: pd.DataFrame,
    return_columns: dict[str, str],
    bars_per_year: int,
) -> pd.DataFrame:
    timestamps = pd.to_datetime(bars["timestamp"], utc=True)
    records: list[dict[str, float | int | str]] = []
    for year, index in bars.groupby(timestamps.dt.year).groups.items():
        for label, column in return_columns.items():
            returns = bars.loc[index, column].fillna(0.0).astype(float)
            equity = (1.0 + returns).cumprod()
            volatility = float(returns.std(ddof=0) * math.sqrt(bars_per_year))
            sharpe = (
                float(returns.mean() / returns.std(ddof=0) * math.sqrt(bars_per_year))
                if returns.std(ddof=0) > 0 else float("nan")
            )
            drawdown = float((equity / equity.cummax() - 1.0).min())
            records.append(
                {
                    "year": int(year),
                    "strategy": label,
                    "bars": int(len(returns)),
                    "return": float(equity.iloc[-1] - 1.0),
                    "annualized_volatility": volatility,
                    "sharpe": sharpe,
                    "max_drawdown": drawdown,
                    "positive_bar_rate": float((returns > 0).mean()),
                }
            )
    return pd.DataFrame.from_records(records)


def market_capture(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    bars_per_year: int,
) -> dict[str, float]:
    strategy = strategy_returns.fillna(0.0).astype(float)
    benchmark = benchmark_returns.fillna(0.0).astype(float)
    up = benchmark > 0
    down = benchmark < 0
    upside_denominator = float(benchmark.loc[up].sum())
    downside_denominator = float(benchmark.loc[down].sum())
    variance = float(benchmark.var(ddof=0))
    beta = float(np.cov(strategy, benchmark, ddof=0)[0, 1] / variance) if variance > 0 else float("nan")
    alpha_bar = float(strategy.mean() - beta * benchmark.mean()) if not np.isnan(beta) else float("nan")
    return {
        "upside_capture": float(strategy.loc[up].sum() / upside_denominator) if upside_denominator else float("nan"),
        "downside_capture": float(strategy.loc[down].sum() / downside_denominator) if downside_denominator else float("nan"),
        "beta": beta,
        "annualized_arithmetic_alpha": alpha_bar * bars_per_year,
        "correlation": float(strategy.corr(benchmark)),
    }
