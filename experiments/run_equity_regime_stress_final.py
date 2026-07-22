from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))

import run_equity_broad_universe_additive_overlay as broad
from run_equity_capital_constrained_overlay import (
    CapitalConfig,
    affordable_contracts,
    option_round_trip_pnl,
    _monthly_deposits_between,
    SCHWAB_FRACTIONAL_MINIMUM,
    OPTION_MULTIPLIER,
    SCHWAB_OPTION_CONTRACT_FEE,
)


@dataclass(frozen=True)
class SafetyConfig:
    drawdown_pause: float = 0.15
    hard_shutdown_drawdown: float = 0.35
    consecutive_loss_limit: int = 4
    cooldown_trades: int = 8
    option_loss_limit: int = 2
    option_pause_trades: int = 10
    minimum_equity: float = 25.0
    require_positive_trend: bool = True


def regime_profile(name: str):
    def profile(session: int, sessions: int) -> tuple[float, float]:
        phase = session / max(sessions, 1)
        if name == 'mixed_control':
            return broad.regime_for_session_original(session, sessions)
        if name == 'persistent_bear':
            return (-0.00075 if phase < 0.70 else -0.00035, 1.45 if phase < 0.70 else 1.15)
        if name == 'choppy_high_vol':
            cycle = (session // 12) % 2
            return (0.00028 if cycle == 0 else -0.00028, 1.75)
        if name == 'crash_recovery':
            if phase < 0.25:
                return -0.00135, 2.35
            if phase < 0.48:
                return -0.00025, 1.70
            if phase < 0.72:
                return 0.00065, 1.55
            return 0.00035, 1.05
        raise ValueError(name)
    return profile


def simulate_with_safety(trades: pd.DataFrame, capital: CapitalConfig, safety: SafetyConfig | None) -> tuple[pd.DataFrame, dict]:
    ordered = trades.sort_values('signal_time').copy()
    ordered['signal_time'] = pd.to_datetime(ordered['signal_time'], utc=True)
    equity = float(capital.initial_capital)
    total_deposits = equity
    prior_time = None
    peak = equity
    max_dd = 0.0
    consecutive_losses = 0
    cooldown = 0
    option_losses = 0
    option_pause = 0
    hard_stopped = False
    skipped_safety = 0
    rows = []

    for _, trade in ordered.iterrows():
        t = pd.Timestamp(trade.signal_time)
        months = _monthly_deposits_between(prior_time, t, capital.monthly_deposit)
        deposit = months * capital.monthly_deposit
        equity += deposit
        total_deposits += deposit
        prior_time = t
        peak = max(peak, equity)
        current_dd = 1.0 - equity / peak if peak else 1.0

        if safety and (hard_stopped or equity < safety.minimum_equity or current_dd >= safety.hard_shutdown_drawdown):
            hard_stopped = True
            skipped_safety += 1
            rows.append({'signal_time': t, 'symbol': trade.symbol, 'deposit': deposit, 'starting_equity': equity,
                         'ending_equity': equity, 'trade_taken': False, 'skip_reason': 'hard_shutdown', 'drawdown': current_dd,
                         'stock_return': float(trade.net_return), 'option_contracts': 0, 'option_pnl_after_fees': 0.0})
            continue

        trend_ok = True
        if safety and safety.require_positive_trend:
            trend_ok = float(trade.get('return_26', 0.0)) > -0.01 and float(trade.get('ema_slope_atr', 0.0)) > -0.20
        paused = safety and (cooldown > 0 or current_dd >= safety.drawdown_pause or not trend_ok)
        if paused:
            skipped_safety += 1
            reason = 'loss_cooldown' if cooldown > 0 else ('drawdown_pause' if current_dd >= safety.drawdown_pause else 'regime_gate')
            if cooldown > 0: cooldown -= 1
            if option_pause > 0: option_pause -= 1
            rows.append({'signal_time': t, 'symbol': trade.symbol, 'deposit': deposit, 'starting_equity': equity,
                         'ending_equity': equity, 'trade_taken': False, 'skip_reason': reason, 'drawdown': current_dd,
                         'stock_return': float(trade.net_return), 'option_contracts': 0, 'option_pnl_after_fees': 0.0})
            continue

        starting = equity
        qualified = bool(trade.get('option_overlay_eligible', False)) and not (safety and option_pause > 0)
        contracts = affordable_contracts(starting, float(trade.get('entry_fill', np.nan)), capital.option_budget_fraction) if qualified else 0
        option_open_cost = (contracts * (float(trade.get('entry_fill', 0.0)) * OPTION_MULTIPLIER + SCHWAB_OPTION_CONTRACT_FEE)) if contracts else 0.0
        stock_notional = max(0.0, starting - option_open_cost) if capital.mode == 'cash_safe' else starting
        if 0 < stock_notional < SCHWAB_FRACTIONAL_MINIMUM: stock_notional = 0.0
        stock_pnl = stock_notional * float(trade.net_return)
        option_pnl = option_round_trip_pnl(float(trade.get('entry_fill', np.nan)), float(trade.get('realized_option_return', 0.0)), contracts) if contracts else 0.0
        trade_pnl = stock_pnl + option_pnl
        equity = max(0.0, starting + trade_pnl)
        peak = max(peak, equity)
        dd = 1.0 - equity / peak if peak else 1.0
        max_dd = max(max_dd, dd)

        if trade_pnl < 0:
            consecutive_losses += 1
            if safety and consecutive_losses >= safety.consecutive_loss_limit:
                cooldown = safety.cooldown_trades
                consecutive_losses = 0
        else:
            consecutive_losses = 0
        if contracts:
            if option_pnl < 0:
                option_losses += 1
                if safety and option_losses >= safety.option_loss_limit:
                    option_pause = safety.option_pause_trades
                    option_losses = 0
            else:
                option_losses = 0
        if option_pause > 0: option_pause -= 1

        rows.append({'signal_time': t, 'symbol': trade.symbol, 'deposit': deposit, 'starting_equity': starting,
                     'ending_equity': equity, 'trade_taken': True, 'skip_reason': '', 'drawdown': dd,
                     'stock_return': float(trade.net_return), 'stock_pnl': stock_pnl, 'option_contracts': contracts,
                     'option_pnl_after_fees': option_pnl, 'trade_pnl': trade_pnl})

    path = pd.DataFrame(rows)
    taken = path[path.trade_taken] if len(path) else path
    summary = {
        'initial_capital': capital.initial_capital,
        'monthly_deposit': capital.monthly_deposit,
        'safety_enabled': safety is not None,
        'ending_equity': equity,
        'total_contributed': total_deposits,
        'net_profit': equity - total_deposits,
        'return_on_contributed_capital': (equity-total_deposits)/total_deposits if total_deposits else np.nan,
        'max_drawdown': max_dd,
        'candidate_trades': len(path),
        'trades_taken': int(path.trade_taken.sum()) if len(path) else 0,
        'trades_skipped_by_safety': skipped_safety,
        'option_contracts': int(path.option_contracts.sum()) if len(path) else 0,
        'hard_stopped': hard_stopped,
        'minimum_equity': float(path.ending_equity.min()) if len(path) else equity,
        'loss_trade_rate': float((taken.trade_pnl < 0).mean()) if len(taken) and 'trade_pnl' in taken else np.nan,
    }
    return path, summary


def run_scenario(name: str, sessions: int, seed: int, threshold: float):
    broad.regime_for_session = regime_profile(name)
    frames, _ = broad.generate_broad_market(sessions, seed)
    candidates = broad.build_candidates(frames, max_bars=26)
    folds, trades = broad.run_fast_walk_forward(candidates, threshold)
    trades = broad.add_overlay_fields(trades)
    trades['stress_scenario'] = name
    return folds, trades


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sessions', type=int, default=320)
    p.add_argument('--seed', type=int, default=12017)
    p.add_argument('--initial-capital', type=float, default=2500.0)
    p.add_argument('--monthly-deposit', type=float, default=0.0)
    p.add_argument('--option-budget-fraction', type=float, default=0.30)
    p.add_argument('--return-threshold-bps', type=float, default=5.0)
    p.add_argument('--output', default='outputs/equity_regime_stress_final')
    p.add_argument('--scenarios', default='mixed_control,persistent_bear,choppy_high_vol,crash_recovery')
    args = p.parse_args()
    out = Path(args.output)
    if not out.is_absolute(): out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    broad.regime_for_session_original = broad.regime_for_session
    broad.SYMBOLS = ('AAPL','MSFT','TSLA','NVDA')
    scenarios = [x.strip() for x in args.scenarios.split(',') if x.strip()]
    all_summaries=[]
    all_trades=[]
    safety = SafetyConfig()
    capital = CapitalConfig(args.initial_capital, args.monthly_deposit, args.option_budget_fraction, 'cash_safe')

    for i, scenario in enumerate(scenarios):
        folds, trades = run_scenario(scenario, args.sessions, args.seed + i*997, args.return_threshold_bps)
        all_trades.append(trades)
        folds.to_csv(out / f'{scenario}_folds.csv', index=False)
        trades.to_csv(out / f'{scenario}_selected_trades.csv', index=False)
        for label, config in [('unprotected', None), ('protected', safety)]:
            path, summary = simulate_with_safety(trades, capital, config)
            summary.update({'scenario':scenario, 'route':label, 'selected_stock_trades':len(trades),
                            'overlay_eligible':int(trades.option_overlay_eligible.sum()) if len(trades) else 0,
                            'mean_stock_return':float(trades.net_return.mean()) if len(trades) else np.nan})
            all_summaries.append(summary)
            path.to_csv(out / f'{scenario}_{label}_capital_path.csv', index=False)

    summary_df=pd.DataFrame(all_summaries)
    summary_df.to_csv(out/'stress_summary.csv', index=False)
    pd.concat(all_trades, ignore_index=True).to_csv(out/'all_stress_selected_trades.csv', index=False)
    metadata={'configuration':vars(args),'safety':asdict(safety),'results':summary_df.to_dict(orient='records')}
    (out/'results.json').write_text(json.dumps(metadata, indent=2, default=str))
    print(summary_df[['scenario','route','ending_equity','net_profit','max_drawdown','trades_taken','trades_skipped_by_safety','option_contracts','hard_stopped']].to_string(index=False))

if __name__=='__main__':
    main()
