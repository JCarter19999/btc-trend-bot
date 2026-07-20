from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from btc_trend_bot.analytics import extract_trades
from btc_trend_bot.metrics import moving_block_bootstrap_mean_ci, summarize_series
from btc_trend_bot.pipeline import run_research


def _write_temp_config(cfg: dict[str, Any]) -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    try:
        yaml.safe_dump(cfg, handle, sort_keys=False)
        return handle.name
    finally:
        handle.close()


def _run_cfg(cfg: dict[str, Any], data_path: str, allow_data_gaps: bool):
    path = _write_temp_config(cfg)
    try:
        return run_research(path, data_path=data_path, allow_data_gaps=allow_data_gaps)
    finally:
        Path(path).unlink(missing_ok=True)


def _fixed_summary(bars: pd.DataFrame, bars_per_year: int) -> dict[str, float]:
    returns = bars["fixed_trend_return"].fillna(0.0).astype(float)
    equity = 10_000.0 * (1.0 + returns).cumprod()
    position = bars["fixed_trend_position"].fillna(0.0).astype(float)
    turnover = bars["fixed_trend_turnover"].fillna(0.0).astype(float)
    return summarize_series(
        equity=equity,
        returns=returns,
        bars_per_year=bars_per_year,
        exposure=float((position.abs() > 1e-12).mean()),
        turnover=float(turnover.sum()),
    )


def _short_summary(bars: pd.DataFrame) -> dict[str, float | int]:
    trades = extract_trades(
        bars,
        "fixed_trend_position",
        "fixed_trend_return",
        "fixed_trend_equity",
        "fixed_size",
    )
    shorts = trades.loc[trades["side"] == "short"]
    if shorts.empty:
        return {"trades": 0, "wins": 0, "win_rate": 0.0, "total_pnl": 0.0, "average_return": 0.0}
    return {
        "trades": int(len(shorts)),
        "wins": int((shorts["trade_return"] > 0).sum()),
        "win_rate": float((shorts["trade_return"] > 0).mean()),
        "total_pnl": float(shorts["pnl"].sum()),
        "average_return": float(shorts["trade_return"].mean()),
    }


def _comparison_row(label: str, control_bars: pd.DataFrame, candidate_bars: pd.DataFrame, bars_per_year: int) -> dict[str, Any]:
    control = _fixed_summary(control_bars, bars_per_year)
    candidate = _fixed_summary(candidate_bars, bars_per_year)
    short = _short_summary(candidate_bars)
    return {
        "segment": label,
        "control_cagr": control["annualized_return"],
        "candidate_cagr": candidate["annualized_return"],
        "cagr_delta": candidate["annualized_return"] - control["annualized_return"],
        "control_sharpe": control["sharpe"],
        "candidate_sharpe": candidate["sharpe"],
        "sharpe_delta": candidate["sharpe"] - control["sharpe"],
        "control_max_drawdown": control["max_drawdown"],
        "candidate_max_drawdown": candidate["max_drawdown"],
        "drawdown_improvement": candidate["max_drawdown"] - control["max_drawdown"],
        "control_calmar": control["calmar"],
        "candidate_calmar": candidate["calmar"],
        "calmar_delta": candidate["calmar"] - control["calmar"],
        "candidate_short_trades": short["trades"],
        "candidate_short_pnl": short["total_pnl"],
        "candidate_short_average_return": short["average_return"],
    }


def validate_short_overlay(
    control_config_path: str,
    candidate_config_path: str,
    data_path: str,
    output_dir: str = "outputs/validation",
    replication_data_path: str | None = None,
    allow_data_gaps: bool = False,
    cost_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0),
) -> dict[str, Any]:
    from btc_trend_bot.config import load_config

    control_cfg = load_config(control_config_path)
    candidate_cfg = load_config(candidate_config_path)
    bars_per_year = int(candidate_cfg["backtest"]["bars_per_year"])

    control_result, _ = run_research(control_config_path, data_path=data_path, allow_data_gaps=allow_data_gaps)
    candidate_result, _ = run_research(candidate_config_path, data_path=data_path, allow_data_gaps=allow_data_gaps)

    aligned = control_result.bars[["timestamp", "fixed_trend_return"]].merge(
        candidate_result.bars[["timestamp", "fixed_trend_return"]],
        on="timestamp",
        suffixes=("_control", "_candidate"),
        validate="one_to_one",
    )
    difference = aligned["fixed_trend_return_candidate"] - aligned["fixed_trend_return_control"]
    bootstrap_kwargs = {
        "n_samples": int(candidate_cfg["backtest"]["bootstrap_samples"]),
        "block_bars": int(candidate_cfg["backtest"]["bootstrap_block_bars"]),
        "seed": int(candidate_cfg["backtest"]["random_seed"]),
    }
    paired_bootstrap = moving_block_bootstrap_mean_ci(difference, **bootstrap_kwargs)

    cost_rows: list[dict[str, Any]] = []
    base_control_fee = float(control_cfg["backtest"]["fee_bps_per_turnover"])
    base_control_slippage = float(control_cfg["backtest"]["slippage_bps_per_turnover"])
    base_candidate_fee = float(candidate_cfg["backtest"]["fee_bps_per_turnover"])
    base_candidate_slippage = float(candidate_cfg["backtest"]["slippage_bps_per_turnover"])
    for multiplier in cost_multipliers:
        control_cost_cfg = copy.deepcopy(control_cfg)
        candidate_cost_cfg = copy.deepcopy(candidate_cfg)
        control_cost_cfg["backtest"]["fee_bps_per_turnover"] = base_control_fee * multiplier
        control_cost_cfg["backtest"]["slippage_bps_per_turnover"] = base_control_slippage * multiplier
        candidate_cost_cfg["backtest"]["fee_bps_per_turnover"] = base_candidate_fee * multiplier
        candidate_cost_cfg["backtest"]["slippage_bps_per_turnover"] = base_candidate_slippage * multiplier
        control_cost_result, _ = _run_cfg(control_cost_cfg, data_path, allow_data_gaps)
        candidate_cost_result, _ = _run_cfg(candidate_cost_cfg, data_path, allow_data_gaps)
        row = _comparison_row(
            f"{multiplier:g}x_costs", control_cost_result.bars, candidate_cost_result.bars, bars_per_year
        )
        row["cost_multiplier"] = multiplier
        cost_rows.append(row)

    timestamps = pd.to_datetime(candidate_result.bars["timestamp"], utc=True)
    walk_rows: list[dict[str, Any]] = []
    for year in range(2021, int(timestamps.dt.year.max()) + 1):
        mask = timestamps.dt.year == year
        if int(mask.sum()) < 2:
            continue
        row = _comparison_row(
            str(year),
            control_result.bars.loc[mask].reset_index(drop=True),
            candidate_result.bars.loc[mask].reset_index(drop=True),
            bars_per_year,
        )
        row["bars"] = int(mask.sum())
        walk_rows.append(row)

    replication_rows: list[dict[str, Any]] = []
    if replication_data_path:
        replication_control, _ = run_research(
            control_config_path, data_path=replication_data_path, allow_data_gaps=allow_data_gaps
        )
        replication_candidate, _ = run_research(
            candidate_config_path, data_path=replication_data_path, allow_data_gaps=allow_data_gaps
        )
        replication_rows.append(
            _comparison_row(
                Path(replication_data_path).name,
                replication_control.bars,
                replication_candidate.bars,
                bars_per_year,
            )
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cost_rows).to_csv(output / "cost_sensitivity.csv", index=False)
    pd.DataFrame(walk_rows).to_csv(output / "walk_forward_years.csv", index=False)
    pd.DataFrame(replication_rows).to_csv(output / "cross_exchange_replication.csv", index=False)

    full_sample = _comparison_row("full_sample", control_result.bars, candidate_result.bars, bars_per_year)
    summary = {
        "control_config": control_config_path,
        "candidate_config": candidate_config_path,
        "data_path": data_path,
        "replication_data_path": replication_data_path,
        "full_sample": full_sample,
        "paired_bootstrap_candidate_minus_control": paired_bootstrap,
        "cost_sensitivity": cost_rows,
        "walk_forward_years": walk_rows,
        "cross_exchange_replication": replication_rows,
    }
    with (output / "validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    return summary
