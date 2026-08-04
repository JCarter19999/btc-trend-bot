"""Phase-1 exit policy: binary choice between exiting at the noon-ET snapshot
(real bid/ask) and holding to the cash-settled close (real intrinsic value)
-- the only two post-entry anchor points backed by real quotes right now.

`ExitDecision` deliberately matches the exit-related fields of spec section
25's per-decision schema, so Phase 2 can swap in the true slope/drawdown/
regime-persistence controller (spec section 14) behind the same interface
without changing callers in backtest.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from .structures import OptionStructure, pnl_and_return

VALID_MODES = ("always_hold", "always_noon", "noon_if_profit_target_met")


@dataclass
class ExitDecision:
    exit_at: str          # "noon" or "close"
    exit_reason: str
    exit_value: float | None
    pnl: float | None
    ret: float | None
    data_quality_warning: str | None = None


def evaluate_exit(structure: OptionStructure, noon_exit_value: float | None,
                   close_payoff: float, cfg: dict) -> ExitDecision:
    ep = cfg["exit_policy"]
    mode = ep["mode"]
    if mode not in VALID_MODES:
        raise ValueError(f"unknown exit_policy.mode {mode!r}, expected one of {VALID_MODES}")

    close_pnl, close_ret = pnl_and_return(structure, close_payoff)

    if mode == "always_hold":
        return ExitDecision("close", "time_exit", close_payoff, close_pnl, close_ret)

    if noon_exit_value is None:
        return ExitDecision("close", "time_exit", close_payoff, close_pnl, close_ret,
                             data_quality_warning="noon exit quote missing, held to close by fallback")

    noon_pnl, noon_ret = pnl_and_return(structure, noon_exit_value)

    if mode == "always_noon":
        return ExitDecision("noon", "forced_noon_exit", noon_exit_value, noon_pnl, noon_ret)

    # noon_if_profit_target_met
    if noon_ret >= ep["profit_target_pct"]:
        return ExitDecision("noon", "profit_target_met", noon_exit_value, noon_pnl, noon_ret)
    if noon_ret <= ep["hard_stop_pct"]:
        return ExitDecision("noon", "hard_stop_hit", noon_exit_value, noon_pnl, noon_ret)
    return ExitDecision("close", "time_exit", close_payoff, close_pnl, close_ret)
