import hashlib
import pandas as pd
from .features import FEATURE_COLUMNS
from .types import Candidate

STRATEGY_VERSION = "selective_long_5m_v1"

def candidate_mask(features: pd.DataFrame, cfg: dict) -> pd.Series:
    strategy = cfg["strategy"]
    n = int(strategy["entry_run_bars"])
    all_green = (features["close"] > features["open"]).astype(int).rolling(n).sum().eq(n)
    return features["feature_valid"] & all_green & (features["entry_run_return_bps"] >= float(strategy["min_run_return_bps"])) & (features["relative_volume"] >= float(strategy["min_relative_volume"])) & (features["body_fraction"] >= float(strategy["min_body_fraction"])) & (features["regime_spread_bps"] >= float(strategy["entry_regime_spread_bps"])) & (features["momentum_bps"] >= float(strategy["entry_momentum_bps"]))

def make_candidate(row: pd.Series, cfg: dict) -> Candidate:
    entry = float(row["close"])
    atr = float(row["atr"])
    exit_cfg = cfg["exit"]
    raw_id = f"{STRATEGY_VERSION}|{cfg['system']['symbol']}|{row['open_time'].isoformat()}"
    return Candidate(hashlib.sha256(raw_id.encode()).hexdigest()[:24], row["open_time"].to_pydatetime(), STRATEGY_VERSION, cfg["system"]["symbol"], entry, atr, entry - float(exit_cfg["stop_atr_multiple"]) * atr, entry + float(exit_cfg["target_atr_multiple"]) * atr, int(exit_cfg["max_hold_bars"]), {name: float(row[name]) for name in FEATURE_COLUMNS}, {"entry_reason": "frozen_selective_long"})

def generate_candidates(features: pd.DataFrame, cfg: dict) -> list[Candidate]:
    return [make_candidate(row, cfg) for _, row in features.loc[candidate_mask(features, cfg)].iterrows()]
