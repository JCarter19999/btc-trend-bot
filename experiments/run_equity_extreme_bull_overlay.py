from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))

from run_equity_gap_walkforward_experiment import build_candidates, generate_market_with_shocks
from run_equity_hybrid_option_optimizer import path_metrics, probability_of_ruin, run_walk_forward


def add_bull_regime_fields(trades: pd.DataFrame) -> pd.DataFrame:
    """Apply fixed, forward-available momentum conditions; no future return is used."""
    out = trades.copy()
    conditions = pd.DataFrame({
        'return_26_strong': out['return_26'] >= 0.060,
        'return_13_strong': out['return_13'] >= 0.040,
        'return_8_strong': out['return_8'] >= 0.030,
        'ema_slope_strong': out['ema_slope_atr'] >= 0.90,
        'close_progress_strong': out['three_bar_close_progress'] >= 1.80,
    }, index=out.index)
    out['bull_regime_score'] = conditions.sum(axis=1).astype(int)
    out['extreme_bull_regime'] = out['bull_regime_score'] >= 4
    out['bull_option_eligible'] = (
        out['extreme_bull_regime']
        & out['realized_option_return'].notna()
        & (out['predicted_net_return'] >= 0.0040)
        & (out['expected_option_return'] >= 0.020)
        & (out['option_probability_profit'] >= 0.50)
    )
    return out


def overlay_fraction(score: pd.Series, eligible: pd.Series, mode: str) -> np.ndarray:
    if mode == 'fixed_10':
        return np.where(eligible, 0.10, 0.0)
    if mode == 'fixed_20':
        return np.where(eligible, 0.20, 0.0)
    if mode == 'fixed_30':
        return np.where(eligible, 0.30, 0.0)
    if mode == 'tiered':
        return np.where(eligible, np.where(score >= 5, 0.30, 0.15), 0.0)
    raise ValueError(f'Unknown mode: {mode}')


def apply_overlay(trades: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray]:
    fractions = overlay_fraction(trades['bull_regime_score'], trades['bull_option_eligible'], mode)
    option_returns = trades['realized_option_return'].fillna(0.0).to_numpy(float)
    stock_returns = trades['net_return'].to_numpy(float)
    combined = (1.0 - fractions) * stock_returns + fractions * option_returns
    return combined, fractions


def summarize(trades: pd.DataFrame, ruin_simulations: int, seed: int) -> pd.DataFrame:
    strategies: dict[str, np.ndarray] = {'stock_only': trades['net_return'].to_numpy(float)}
    for mode in ('fixed_10', 'fixed_20', 'fixed_30', 'tiered'):
        strategies[f'extreme_bull_{mode}'], _ = apply_overlay(trades, mode)

    rows = []
    for i, (name, returns) in enumerate(strategies.items()):
        metrics = path_metrics(returns)
        ruin = probability_of_ruin(returns, ruin_simulations, seed + i * 31)
        rows.append({
            'strategy': name,
            'trades': len(returns),
            'mean_trade_return': float(np.mean(returns)),
            'compounded_return': metrics['return'],
            'max_drawdown': metrics['max_drawdown'],
            **ruin,
        })
    return pd.DataFrame(rows).sort_values('compounded_return', ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', type=int, default=400)
    parser.add_argument('--seed', type=int, default=9381)
    parser.add_argument('--return-threshold-bps', type=float, default=5.0)
    parser.add_argument('--rank-threshold', type=float, default=0.55)
    parser.add_argument('--ruin-simulations', type=int, default=5000)
    parser.add_argument('--output', default='outputs/equity_extreme_bull_overlay')
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    frames, shocks = generate_market_with_shocks(args.sessions, args.seed)
    candidates = build_candidates(frames, max_bars=26)
    folds, trades = run_walk_forward(candidates, args.return_threshold_bps, args.rank_threshold)
    trades = add_bull_regime_fields(trades)
    summary = summarize(trades, args.ruin_simulations, args.seed)

    for mode in ('fixed_10', 'fixed_20', 'fixed_30', 'tiered'):
        returns, fractions = apply_overlay(trades, mode)
        trades[f'{mode}_overlay_fraction'] = fractions
        trades[f'{mode}_return'] = returns

    eligible = trades[trades['bull_option_eligible']].copy()
    regime_summary = {
        'selected_stock_trades': int(len(trades)),
        'extreme_bull_signals': int(trades['extreme_bull_regime'].sum()),
        'option_overlay_eligible': int(trades['bull_option_eligible'].sum()),
        'overlay_eligibility_rate': float(trades['bull_option_eligible'].mean()),
        'eligible_mean_stock_return': float(eligible['net_return'].mean()) if len(eligible) else None,
        'eligible_mean_option_return': float(eligible['realized_option_return'].mean()) if len(eligible) else None,
        'eligible_option_win_rate': float((eligible['realized_option_return'] > 0).mean()) if len(eligible) else None,
        'eligible_symbols': eligible['symbol'].value_counts().to_dict(),
        'eligible_folds': eligible['fold'].value_counts().sort_index().to_dict(),
    }

    folds.to_csv(output / 'fold_results.csv', index=False)
    trades.to_csv(output / 'selected_trades_with_bull_overlay.csv', index=False)
    summary.to_csv(output / 'strategy_summary.csv', index=False)
    shocks.to_csv(output / 'earnings_like_shocks.csv', index=False)
    (output / 'results.json').write_text(json.dumps({
        'configuration': vars(args),
        'regime_summary': regime_summary,
        'strategy_summary': summary.to_dict(orient='records'),
    }, indent=2, default=str))
    print(summary.to_string(index=False))
    print(json.dumps(regime_summary, indent=2, default=str))


if __name__ == '__main__':
    main()
