"""Deterministic risk management (spec section 16): position sizing off
account value, daily/weekly loss limits, consecutive-loss cooldown, single
open position. No data dependency -- pure state machine, driven day by day
by the backtest engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CONTRACT_MULTIPLIER = 100  # SPX options are $100 x index points per contract


@dataclass
class RiskManager:
    account_value: float
    max_risk_pct_per_trade: float
    max_daily_loss_pct: float
    max_weekly_loss_pct: float
    consecutive_loss_cooldown: int
    cooldown_trades: int

    current_week: object = None
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    consecutive_losses: int = 0
    cooldown_remaining: int = 0
    open_position: bool = False

    def max_planned_loss_dollars(self) -> float:
        return self.account_value * self.max_risk_pct_per_trade

    def contracts_for(self, max_loss_per_contract_points: float) -> int:
        if max_loss_per_contract_points <= 0:
            return 0
        dollars_per_contract = max_loss_per_contract_points * CONTRACT_MULTIPLIER
        return max(0, int(self.max_planned_loss_dollars() // dollars_per_contract))

    def roll_day(self, trade_date) -> None:
        week = trade_date.isocalendar()[:2]
        if week != self.current_week:
            self.current_week = week
            self.weekly_pnl = 0.0
        self.daily_pnl = 0.0

    def can_trade(self) -> tuple[bool, str]:
        if self.open_position:
            return False, "one open strategy at a time"
        if self.cooldown_remaining > 0:
            return False, f"cooldown active ({self.cooldown_remaining} trades remaining after consecutive losses)"
        if self.daily_pnl <= -self.max_daily_loss_pct * self.account_value:
            return False, "daily loss limit reached"
        if self.weekly_pnl <= -self.max_weekly_loss_pct * self.account_value:
            return False, "weekly loss limit reached"
        return True, ""

    def record_trade(self, pnl_dollars: float) -> None:
        self.daily_pnl += pnl_dollars
        self.weekly_pnl += pnl_dollars
        self.open_position = False
        if pnl_dollars < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.consecutive_loss_cooldown:
                self.cooldown_remaining = self.cooldown_trades
        else:
            self.consecutive_losses = 0
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1


def build_risk_manager(cfg: dict) -> RiskManager:
    rc = cfg["risk"]
    return RiskManager(
        account_value=rc["account_value"],
        max_risk_pct_per_trade=rc["max_risk_pct_per_trade"],
        max_daily_loss_pct=rc["max_daily_loss_pct"],
        max_weekly_loss_pct=rc["max_weekly_loss_pct"],
        consecutive_loss_cooldown=rc["consecutive_loss_cooldown"],
        cooldown_trades=rc["cooldown_trades"],
    )
