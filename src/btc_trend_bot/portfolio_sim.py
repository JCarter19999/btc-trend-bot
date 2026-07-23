"""Single-position-correct portfolio simulator with pluggable sizing modes.

`run_equity_real_data_walkforward.simulate_capital` processes every selected
trade sequentially by signal_time and hands each one a slice of "current
equity," without checking whether the *previous* trade's exit_time has
actually passed. Since candidates are generated for most days and a fresh
winner is picked per day, roughly 3 in 4 "sequential" trades in practice
start while the prior trade is still open (see
EQUITY_KALMAN_ONLINE_REGRESSION.md's overlap audit: 77.3% of trades start
before the previous one's exit_time, at a median 5 days early against a
10-day max hold). That means the existing simulator implicitly runs several
overlapping positions at once, each drawing a fresh notional slice as though
capital were freely available — a materially different (much more
optimistic) strategy than the "long-only, one active position" model
`CLAUDE.md` describes, and than what the live paper-step engine
(`run_equity_paper_step.py`, which tracks a real `open_position` singleton
and only evaluates new entries when nothing is open) actually does.

This module fixes that: a candidate is skipped entirely -- exactly as a real
single-position trader would have to skip it -- if its entry_time is before
the previous *taken* trade's exit_time. It also decouples position sizing
from the 25%-of-compounding-equity default so selector quality can be judged
without compounding path-dependence dominating the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SizingMode:
    name: str
    kind: str  # "fixed_notional" | "fixed_fractional" | "vol_targeted"
    value: float
    # fixed_notional: value is a dollar amount, constant, no compounding.
    # fixed_fractional: value is a fraction of *current* equity (compounds).
    # vol_targeted: value is a fraction of equity risked per trade; position
    #   size = risk_dollars / stop_distance_pct, where stop_distance_pct is
    #   derived from cfg.stop_atr * atr_pct (the same ATR stop used for exits).


def _notional(equity: float, sizing: SizingMode, trade: pd.Series, cfg) -> float:
    if sizing.kind == "fixed_notional":
        return sizing.value
    if sizing.kind == "fixed_fractional":
        return equity * sizing.value
    if sizing.kind == "vol_targeted":
        atr_pct = float(trade.get("atr_pct", np.nan))
        stop_distance_pct = cfg.stop_atr * atr_pct
        if not np.isfinite(stop_distance_pct) or stop_distance_pct <= 0:
            return 0.0
        risk_dollars = equity * sizing.value
        return risk_dollars / stop_distance_pct
    raise ValueError(f"Unknown sizing kind: {sizing.kind}")


def simulate_single_position(trades: pd.DataFrame, cfg, sizing: SizingMode) -> tuple[pd.DataFrame, dict]:
    trades = trades.sort_values("signal_time").reset_index(drop=True)
    equity = cfg.initial_capital
    peak = equity
    next_available: pd.Timestamp | None = None
    cooldown = 0
    dd_cooldown = 0
    losses = 0
    hard = False
    rows = []
    skipped_overlap = 0

    for _, t in trades.iterrows():
        if next_available is not None and t.entry_time < next_available:
            skipped_overlap += 1
            continue  # a single-position trader physically could not take this

        dd = 1 - equity / peak
        reason = ""
        if hard or equity < cfg.minimum_equity or dd >= cfg.hard_shutdown_drawdown:
            hard = True
            reason = "hard_shutdown"
        elif cfg.safety_enabled and cooldown > 0:
            cooldown -= 1
            reason = "loss_cooldown"
        elif cfg.safety_enabled and (dd_cooldown > 0 or dd >= cfg.drawdown_pause):
            if dd_cooldown == 0:
                dd_cooldown = cfg.cooldown_trades
            dd_cooldown -= 1
            reason = "drawdown_pause"
            if dd_cooldown == 0:
                peak = equity
        if reason:
            rows.append({"signal_time": t.signal_time, "symbol": t.symbol, "trade_taken": False,
                         "skip_reason": reason, "ending_equity": equity, "drawdown": dd})
            continue

        notional = _notional(equity, sizing, t, cfg)
        pnl = notional * float(t.net_return)
        start = equity
        equity = max(0.0, start + pnl)
        peak = max(peak, equity)
        dd = 1 - equity / peak
        if pnl < 0:
            losses += 1
            if cfg.safety_enabled and losses >= cfg.consecutive_loss_limit:
                cooldown = cfg.cooldown_trades
                losses = 0
        else:
            losses = 0
        next_available = t.exit_time
        rows.append({"signal_time": t.signal_time, "symbol": t.symbol, "trade_taken": True,
                     "skip_reason": "", "starting_equity": start, "notional": notional,
                     "trade_return": float(t.net_return), "trade_pnl": pnl,
                     "ending_equity": equity, "drawdown": dd})

    path = pd.DataFrame(rows)
    taken = path[path.trade_taken] if len(path) else path
    wins = taken.loc[taken.trade_pnl > 0, "trade_pnl"] if len(taken) else pd.Series(dtype=float)
    losses_s = taken.loc[taken.trade_pnl < 0, "trade_pnl"] if len(taken) else pd.Series(dtype=float)
    returns = taken["trade_return"] if len(taken) else pd.Series(dtype=float)

    summary = {
        "sizing": sizing.name,
        "initial_capital": cfg.initial_capital,
        "ending_equity": equity,
        "net_profit": equity - cfg.initial_capital,
        "total_return": equity / cfg.initial_capital - 1,
        "max_drawdown": float(path.drawdown.max()) if len(path) else 0.0,
        "candidate_selected_trades": int(len(trades)),
        "trades_taken": int(len(taken)),
        "trades_skipped_safety": int((~path.trade_taken).sum()) if len(path) else 0,
        "trades_skipped_overlap": int(skipped_overlap),
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "profit_factor": float(wins.sum() / abs(losses_s.sum())) if len(losses_s) and losses_s.sum() != 0 else np.nan,
        "mean_return_per_trade": float(returns.mean()) if len(returns) else np.nan,
        "median_return_per_trade": float(returns.median()) if len(returns) else np.nan,
        "avg_win_return": float(returns[returns > 0].mean()) if (returns > 0).any() else np.nan,
        "avg_loss_return": float(returns[returns < 0].mean()) if (returns < 0).any() else np.nan,
        "expectancy_bps": float(returns.mean() * 10000) if len(returns) else np.nan,
        "total_pnl_dollars": float(taken.trade_pnl.sum()) if len(taken) else 0.0,
        "hard_stopped": hard,
    }
    return path, summary
