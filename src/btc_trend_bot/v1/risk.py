from dataclasses import dataclass
from .types import KillMode, RiskDecision

@dataclass
class RiskState:
    equity: float
    day_start_equity: float
    week_start_equity: float
    peak_equity: float
    open_positions: int = 0
    btc_exposure: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    kill_mode: KillMode = KillMode.NORMAL

class RiskEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def evaluate(self, state: RiskState, requested_notional: float, spread_bps: float, stale_candles: int, clock_drift_seconds: float) -> RiskDecision:
        r = self.cfg
        checks = {
            "kill_switch_ok": state.kill_mode == KillMode.NORMAL,
            "allocation_ok": requested_notional <= state.equity * float(r["max_allocation_fraction"]),
            "exposure_ok": state.btc_exposure + requested_notional <= state.equity * float(r["max_btc_exposure_fraction"]),
            "open_positions_ok": state.open_positions < int(r["max_open_positions"]),
            "daily_loss_ok": (state.day_start_equity - state.equity) / max(state.day_start_equity, 1e-9) < float(r["daily_loss_fraction"]),
            "weekly_loss_ok": (state.week_start_equity - state.equity) / max(state.week_start_equity, 1e-9) < float(r["weekly_loss_fraction"]),
            "drawdown_ok": (state.peak_equity - state.equity) / max(state.peak_equity, 1e-9) < float(r["max_drawdown_fraction"]),
            "trade_count_ok": state.trades_today < int(r["max_trades_per_day"]),
            "loss_streak_ok": state.consecutive_losses < int(r["max_consecutive_losses"]),
            "spread_ok": spread_bps <= float(r["max_spread_bps"]),
            "data_fresh": stale_candles <= int(r["max_stale_candles"]),
            "clock_ok": abs(clock_drift_seconds) <= float(r["max_clock_drift_seconds"]),
        }
        failed = next((name for name, passed in checks.items() if not passed), None)
        return RiskDecision(failed is None, "APPROVED" if failed is None else failed.upper(), checks)
