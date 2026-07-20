from __future__ import annotations

import math

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    running_peak = equity.cummax()
    return float((equity / running_peak - 1.0).min())


def annualized_return(equity: pd.Series, bars_per_year: int) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return float("nan")
    years = max((len(equity) - 1) / bars_per_year, 1 / bars_per_year)
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series, bars_per_year: int) -> float:
    return float(returns.std(ddof=0) * math.sqrt(bars_per_year))


def sharpe_ratio(returns: pd.Series, bars_per_year: int) -> float:
    volatility = returns.std(ddof=0)
    if volatility == 0 or np.isnan(volatility):
        return float("nan")
    return float(returns.mean() / volatility * math.sqrt(bars_per_year))


def sortino_ratio(returns: pd.Series, bars_per_year: int) -> float:
    downside = returns.where(returns < 0, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    if downside_deviation == 0:
        return float("nan")
    return float(returns.mean() / downside_deviation * math.sqrt(bars_per_year))


def summarize_series(
    equity: pd.Series,
    returns: pd.Series,
    bars_per_year: int,
    exposure: float,
    turnover: float,
) -> dict[str, float]:
    annual_return = annualized_return(equity, bars_per_year)
    drawdown = max_drawdown(equity)
    return {
        "starting_equity": float(equity.iloc[0]),
        "ending_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "annualized_return": annual_return,
        "annualized_volatility": annualized_volatility(returns, bars_per_year),
        "sharpe": sharpe_ratio(returns, bars_per_year),
        "sortino": sortino_ratio(returns, bars_per_year),
        "max_drawdown": drawdown,
        "calmar": float(annual_return / abs(drawdown)) if drawdown < 0 else float("nan"),
        "exposure": float(exposure),
        "total_turnover": float(turnover),
        "positive_bar_rate": float((returns > 0).mean()),
    }


def summarize_backtest(bars: pd.DataFrame, bars_per_year: int) -> dict[str, dict[str, float]]:
    strategy = summarize_series(
        bars["equity"],
        bars["strategy_return"],
        bars_per_year,
        exposure=float((bars["held_position"].abs() > 1e-12).mean()),
        turnover=float(bars["turnover"].sum()),
    )
    benchmark = summarize_series(
        bars["benchmark_equity"],
        bars["benchmark_return"],
        bars_per_year,
        exposure=1.0,
        turnover=0.0,
    )
    return {"strategy": strategy, "buy_and_hold": benchmark}


def moving_block_bootstrap_mean_ci(
    returns: pd.Series,
    n_samples: int,
    block_bars: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float | int | bool]:
    values = returns.dropna().to_numpy(dtype=float)
    n = len(values)
    if n < 2 or block_bars < 1:
        return {"n": n, "mean": float(values.mean()) if n else 0.0}

    block_bars = min(block_bars, n)
    starts = np.arange(0, n - block_bars + 1)
    rng = np.random.default_rng(seed)
    means = np.empty(n_samples, dtype=float)
    blocks_needed = math.ceil(n / block_bars)

    for sample in range(n_samples):
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        boot = np.concatenate([values[start : start + block_bars] for start in selected])[:n]
        means[sample] = boot.mean()

    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "n": n,
        "mean_bar_return": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_positive": float((means > 0).mean()),
        "distinguishable_from_zero": bool(low > 0),
    }
