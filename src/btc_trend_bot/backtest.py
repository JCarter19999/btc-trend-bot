from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    bars: pd.DataFrame
    breaker_timestamp: pd.Timestamp | None


def run_backtest(frame: pd.DataFrame, cfg: dict) -> BacktestResult:
    required = {"timestamp", "close", "simple_return", "target_position"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Backtest input missing columns: {sorted(missing)}")

    initial_cash = float(cfg["initial_cash"])
    fee_rate = float(cfg["fee_bps_per_turnover"]) / 10_000.0
    slippage_rate = float(cfg["slippage_bps_per_turnover"]) / 10_000.0
    breaker_limit = float(cfg["max_drawdown_breaker"])

    out = frame.copy().reset_index(drop=True)
    desired_held_position = out["target_position"].shift(1).fillna(0.0)

    equity = initial_cash
    peak = initial_cash
    previous_position = 0.0
    halted = False
    breaker_timestamp: pd.Timestamp | None = None
    records: list[dict[str, float | bool]] = []

    for index, row in out.iterrows():
        position = 0.0 if halted else float(desired_held_position.iloc[index])
        asset_return = 0.0 if pd.isna(row["simple_return"]) else float(row["simple_return"])
        turnover = abs(position - previous_position)
        execution_cost = turnover * (fee_rate + slippage_rate)
        funding_rate = float(row.get("funding_rate", 0.0) or 0.0)
        funding_cost = position * funding_rate
        gross_return = position * asset_return
        net_return = gross_return - execution_cost - funding_cost
        if net_return <= -1.0:
            raise RuntimeError("A bar produced a return <= -100%; inspect data and leverage assumptions.")

        equity *= 1.0 + net_return
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0

        if not halted and breaker_limit > 0 and drawdown <= -breaker_limit:
            halted = True
            breaker_timestamp = pd.Timestamp(row["timestamp"])

        records.append(
            {
                "held_position": position,
                "turnover": turnover,
                "gross_strategy_return": gross_return,
                "execution_cost_return": execution_cost,
                "funding_cost_return": funding_cost,
                "strategy_return": net_return,
                "equity": equity,
                "drawdown": drawdown,
                "breaker_active": halted,
            }
        )
        previous_position = position

    record_frame = pd.DataFrame(records)
    out = pd.concat([out, record_frame], axis=1)
    out["benchmark_return"] = out["simple_return"].fillna(0.0)
    out["benchmark_equity"] = initial_cash * (1.0 + out["benchmark_return"]).cumprod()
    return BacktestResult(bars=out, breaker_timestamp=breaker_timestamp)
