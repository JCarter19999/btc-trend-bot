from dataclasses import asdict
import pandas as pd
from .types import Candidate

def round_trip_cost_return(cost_cfg: dict) -> float:
    total_bps = 2 * (float(cost_cfg["fee_bps_per_side"]) + float(cost_cfg["spread_bps_per_side"]) + float(cost_cfg["slippage_bps_per_side"])) + float(cost_cfg.get("latency_bps_round_trip", 0))
    return total_bps / 10000.0

def mature_candidate(candidate: Candidate, future: pd.DataFrame, cost_cfg: dict) -> dict:
    rows = future[future["open_time"] > pd.Timestamp(candidate.timestamp)].head(candidate.max_hold_bars)
    if rows.empty:
        return {**asdict(candidate), "label_status": "PENDING"}
    exit_price = float(rows.iloc[-1]["close"])
    reason = "time_exit"
    ambiguous = False
    maturity = rows.iloc[-1]["open_time"]
    for _, row in rows.iterrows():
        stop = float(row["low"]) <= candidate.stop_price
        target = float(row["high"]) >= candidate.target_price
        if stop and target:
            exit_price, reason, ambiguous, maturity = candidate.stop_price, "stop_loss", True, row["open_time"]
            break
        if stop:
            exit_price, reason, maturity = candidate.stop_price, "stop_loss", row["open_time"]
            break
        if target:
            exit_price, reason, maturity = candidate.target_price, "profit_target", row["open_time"]
            break
    gross = exit_price / candidate.entry_price - 1
    net = gross - round_trip_cost_return(cost_cfg)
    return {**asdict(candidate), "label_status": "MATURED", "maturity_timestamp": maturity, "exit_price": exit_price, "exit_reason": reason, "ambiguous_same_bar": ambiguous, "target_hit_first": reason == "profit_target", "gross_return": gross, "net_return": net, "positive_net": int(net > 0)}


def mature_candidate_adaptive(candidate, future, cost_cfg, exit_cfg):
    from .exit_engine import ExitEngineConfig, run_exit_engine
    return run_exit_engine(candidate, future, cost_cfg, ExitEngineConfig.from_dict(exit_cfg, candidate.max_hold_bars))
