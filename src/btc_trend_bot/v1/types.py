from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

class DecisionAction(StrEnum):
    REJECT = "REJECT"
    ACCEPT_FULL_SIZE = "ACCEPT_FULL_SIZE"

class KillMode(StrEnum):
    NORMAL = "NORMAL"
    HALT_NEW_ENTRIES = "HALT_NEW_ENTRIES"
    CANCEL_OPEN_ORDERS = "CANCEL_OPEN_ORDERS"
    FLATTEN_AND_HALT = "FLATTEN_AND_HALT"
    SERVICE_DISABLED = "SERVICE_DISABLED"

@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    timestamp: datetime
    strategy_version: str
    symbol: str
    entry_price: float
    atr: float
    stop_price: float
    target_price: float
    max_hold_bars: int
    features: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ModelDecision:
    candidate_id: str
    model_id: str
    probability_target_first: float
    expected_net_return: float
    safety_margin_return: float
    action: DecisionAction

@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason_code: str
    checks: dict[str, bool]
