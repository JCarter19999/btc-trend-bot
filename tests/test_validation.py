from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from btc_trend_bot.synthetic import generate_synthetic_ohlcv
from btc_trend_bot.validation import validate_short_overlay


def _write_config(path: Path, mode: str, short_regime_filter: bool) -> None:
    cfg = {
        "market": {
            "exchange": "kraken",
            "symbol": "BTC/USD",
            "timeframe": "4h",
            "start": "2018-01-01T00:00:00Z",
            "data_path": str(path.parent / "data.csv"),
            "max_download_bars": 50000,
        },
        "data_quality": {
            "require_configured_start": False,
            "max_missing_rate": 0.01,
            "max_single_gap_bars": 6,
        },
        "strategy": {
            "mode": mode,
            "fast_ema_bars": 12,
            "slow_ema_bars": 30,
            "breakout_bars": 20,
            "atr_bars": 12,
            "realized_vol_bars": 12,
            "long_vol_bars": 30,
            "short_regime_ema_bars": 60,
            "target_annual_vol": 0.35,
            "min_position": 0.0,
            "max_position": 1.0,
            "long_position_size": 1.0,
            "short_position_size": 0.5,
            "trend_strength_atr": 0.1,
            "short_regime_filter": short_regime_filter,
            "short_trailing_stop_atr": 0.0,
            "vol_shock_ratio": 1.8,
            "vol_shock_multiplier": 0.5,
        },
        "backtest": {
            "initial_cash": 10000.0,
            "fee_bps_per_turnover": 6.0,
            "slippage_bps_per_turnover": 4.0,
            "max_drawdown_breaker": 0.0,
            "bars_per_year": 2190,
            "bootstrap_samples": 50,
            "bootstrap_block_bars": 12,
            "random_seed": 42,
        },
        "paper": {
            "initial_cash": 10000.0,
            "state_path": "paper/state.json",
            "trades_path": "paper/trades.csv",
            "lookback_bars": 100,
        },
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def test_validation_writes_expected_outputs(tmp_path: Path) -> None:
    frame = generate_synthetic_ohlcv(rows=2500, seed=7)
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)
    control = tmp_path / "control.yaml"
    candidate = tmp_path / "candidate.yaml"
    _write_config(control, "long_flat", False)
    _write_config(candidate, "selective_short", True)
    output = tmp_path / "validation"

    summary = validate_short_overlay(
        str(control),
        str(candidate),
        str(data_path),
        output_dir=str(output),
        cost_multipliers=(1.0, 2.0),
    )

    assert "paired_bootstrap_candidate_minus_control" in summary
    assert len(summary["cost_sensitivity"]) == 2
    assert (output / "validation_summary.json").exists()
    assert (output / "cost_sensitivity.csv").exists()
    assert (output / "walk_forward_years.csv").exists()
    costs = pd.read_csv(output / "cost_sensitivity.csv")
    assert list(costs["cost_multiplier"]) == [1.0, 2.0]


def test_validation_replication_path(tmp_path: Path) -> None:
    frame = generate_synthetic_ohlcv(rows=1800, seed=11)
    primary = tmp_path / "primary.csv"
    replication = tmp_path / "replication.csv"
    frame.to_csv(primary, index=False)
    frame.assign(open=frame["open"] * 1.001, high=frame["high"] * 1.001, low=frame["low"] * 1.001, close=frame["close"] * 1.001).to_csv(replication, index=False)
    control = tmp_path / "control.yaml"
    candidate = tmp_path / "candidate.yaml"
    _write_config(control, "long_flat", False)
    _write_config(candidate, "selective_short", True)

    summary = validate_short_overlay(
        str(control),
        str(candidate),
        str(primary),
        output_dir=str(tmp_path / "out"),
        replication_data_path=str(replication),
        cost_multipliers=(1.0,),
    )

    assert len(summary["cross_exchange_replication"]) == 1
