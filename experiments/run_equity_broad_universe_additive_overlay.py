from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from run_equity_three_symbol_experiment import BARS_PER_SESSION, SymbolSpec, trading_index, regime_for_session
from run_equity_gap_walkforward_experiment import build_candidates, fit_model, GEOMETRY_EXTENDED
import run_equity_hybrid_option_optimizer as hybrid

SYMBOLS = ('AAPL', 'MSFT', 'TSLA', 'NVDA')
SPECS = {
    'AAPL': SymbolSpec(145.0, 1.00, 0.0032, 0.0080, 1.00),
    'MSFT': SymbolSpec(265.0, 0.88, 0.0028, 0.0070, 0.92),
    'TSLA': SymbolSpec(220.0, 1.45, 0.0060, 0.0160, 1.25),
    'NVDA': SymbolSpec(190.0, 1.35, 0.0054, 0.0140, 1.20),
    'AMZN': SymbolSpec(120.0, 1.12, 0.0041, 0.0100, 1.04),
    'META': SymbolSpec(185.0, 1.08, 0.0043, 0.0110, 1.08),
    'GOOGL': SymbolSpec(105.0, 0.96, 0.0035, 0.0085, 0.98),
    'AMD': SymbolSpec(92.0, 1.32, 0.0058, 0.0150, 1.18),
}

@dataclass(frozen=True)
class ShockSpec:
    frequency_sessions: int
    positive_probability: float
    mean_abs_gap: float
    sigma_gap: float

SHOCKS = {
    'AAPL': ShockSpec(63, 0.56, 0.035, 0.018),
    'MSFT': ShockSpec(63, 0.58, 0.030, 0.015),
    'TSLA': ShockSpec(63, 0.52, 0.065, 0.035),
    'NVDA': ShockSpec(63, 0.55, 0.060, 0.030),
    'AMZN': ShockSpec(63, 0.54, 0.045, 0.023),
    'META': ShockSpec(63, 0.55, 0.050, 0.026),
    'GOOGL': ShockSpec(63, 0.57, 0.038, 0.019),
    'AMD': ShockSpec(63, 0.53, 0.065, 0.034),
}

OFFSETS = {symbol: 8 + 7 * i for i, symbol in enumerate(SYMBOLS)}


def generate_broad_market(sessions: int, seed: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = trading_index('2017-01-03', sessions)
    market_returns = np.zeros(len(index))
    for i in range(len(index)):
        session = i // BARS_PER_SESSION
        drift, vol_factor = regime_for_session(session, sessions)
        opening = i % BARS_PER_SESSION == 0
        sigma = 0.0024 * vol_factor * (1.4 if opening else 1.0)
        market_returns[i] = drift / BARS_PER_SESSION + rng.normal(0.0, sigma)

    shock_rows: list[dict] = []
    shock_maps: dict[str, dict[int, float]] = {}
    for symbol in SYMBOLS:
        spec = SHOCKS[symbol]
        mapping: dict[int, float] = {}
        for session in range(OFFSETS[symbol], sessions, spec.frequency_sessions):
            positive = rng.random() < spec.positive_probability
            magnitude = abs(rng.normal(spec.mean_abs_gap, spec.sigma_gap))
            gap = magnitude if positive else -magnitude
            mapping[session] = gap
            shock_rows.append({'symbol': symbol, 'session': session, 'gap_return': gap, 'positive': positive})
        shock_maps[symbol] = mapping

    frames: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        spec = SPECS[symbol]
        previous_close = spec.start_price
        latent_edge = 0.0
        rows = []
        for i, timestamp in enumerate(index):
            session = i // BARS_PER_SESSION
            bar = i % BARS_PER_SESSION
            opening = bar == 0
            _, vol_factor = regime_for_session(session, sessions)
            routine_gap = rng.normal(0.0, spec.overnight_sigma * vol_factor) if opening else 0.0
            event_gap = shock_maps[symbol].get(session, 0.0) if opening else 0.0
            open_price = previous_close * math.exp(routine_gap + event_gap)
            if bar in (2, 7) and rng.random() < 0.17:
                quality = rng.normal()
                latent_edge = spec.edge_scale * 0.0048 * quality
            else:
                latent_edge *= 0.88
            shock_vol = 1.8 if session in shock_maps[symbol] else 1.0
            idio = rng.normal(0.0, spec.intraday_sigma * vol_factor * 0.72 * shock_vol)
            ret = spec.beta * market_returns[i] + idio + latent_edge
            close = open_price * math.exp(ret)
            base_range = abs(ret) + spec.intraday_sigma * vol_factor * 0.55 * shock_vol
            upper_share = np.clip(0.50 - np.sign(latent_edge) * 0.16 + rng.normal(0.0, 0.10), 0.08, 0.92)
            extra = open_price * base_range
            high = max(open_price, close) + extra * upper_share
            low = min(open_price, close) - extra * (1.0 - upper_share)
            volume_level = 11.55 if symbol in {'TSLA', 'AMD'} else 12.05
            volume = ((1.0 + 0.40 * abs(latent_edge) / 0.002)
                      * (1.0 + 0.55 * (opening or bar == 12))
                      * (2.2 if session in shock_maps[symbol] else 1.0)
                      * rng.lognormal(volume_level, 0.32))
            rows.append((timestamp, open_price, high, low, close, volume, session, bar, event_gap))
            previous_close = close
        frames[symbol] = pd.DataFrame(rows, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume', 'session', 'bar_in_session', 'event_gap'
        ])
    return frames, pd.DataFrame(shock_rows)



def run_fast_walk_forward(candidates: pd.DataFrame, return_threshold_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Direct-return Ridge champion; cross-sectional winner is max predicted return."""
    times = np.sort(candidates.signal_time.unique())
    boundaries = [int(len(times) * x) for x in (0.40, 0.55, 0.70, 0.85, 1.00)]
    folds, trade_frames = [], []
    for fold in range(4):
        train_end = times[boundaries[fold]]
        test_end = times[boundaries[fold + 1] - 1]
        train_all = candidates[candidates.signal_time < train_end].copy()
        test = candidates[(candidates.signal_time >= train_end) & (candidates.signal_time <= test_end)].copy()
        if len(train_all) < 150 or len(test) < 25:
            continue
        cut = int(len(train_all) * 0.80)
        train, calibration = train_all.iloc[:cut].copy(), train_all.iloc[cut:].copy()
        ridge = fit_model(train, GEOMETRY_EXTENDED, 'ridge')
        cal_raw = ridge.predict(calibration[GEOMETRY_EXTENDED + ['symbol']])
        slope, intercept = np.polyfit(cal_raw, calibration.net_return.to_numpy(float), 1)
        test['predicted_net_return'] = intercept + slope * ridge.predict(test[GEOMETRY_EXTENDED + ['symbol']])
        residual_std = float(np.std(calibration.net_return.to_numpy(float) - (intercept + slope * cal_raw), ddof=1))
        expected_hold_sessions = float(np.clip(train.bars_held.median() / 13.0, 0.25, 3.0))
        winners = test.sort_values(['signal_time','predicted_net_return'], ascending=[True,False]).groupby('signal_time', as_index=False).head(1)
        winners = winners[winners.predicted_net_return > return_threshold_bps / 10000.0]
        accepted=[]
        next_free=pd.Timestamp.min.tz_localize('UTC')
        for _, row in winners.sort_values('signal_time').iterrows():
            if pd.Timestamp(row.signal_time) < next_free:
                continue
            accepted.append(row)
            next_free = pd.Timestamp(row.exit_time)
        selected=pd.DataFrame(accepted)
        optimized=[]
        for _, row in selected.iterrows():
            best=hybrid.optimize_contract(row, residual_std, expected_hold_sessions)
            rec=row.to_dict()
            if best is None:
                rec.update({'strike':np.nan,'dte':np.nan,'delta':np.nan,'iv':np.nan,'open_interest':np.nan,'option_volume':np.nan,'spread_fraction':np.nan,'expected_option_return':np.nan,'option_probability_profit':np.nan,'option_downside_semivariance':np.nan,'option_cvar10':np.nan,'option_utility':np.nan,'entry_fill':np.nan,'realized_option_return':np.nan})
            else:
                rec.update(best); rec.update(hybrid.realized_option_return(row,best))
            optimized.append(rec)
        expressed=pd.DataFrame(optimized)
        if len(expressed):
            expressed['fold']=fold+1; trade_frames.append(expressed)
        folds.append({'fold':fold+1,'train_candidates':len(train),'calibration_candidates':len(calibration),'test_candidates':len(test),'selected_trades':len(expressed),'residual_std_bps':residual_std*10000,'expected_hold_sessions':expected_hold_sessions,'mean_predicted_return_bps':expressed.predicted_net_return.mean()*10000 if len(expressed) else np.nan,'mean_realized_stock_bps':expressed.net_return.mean()*10000 if len(expressed) else np.nan,'option_available_rate':expressed.realized_option_return.notna().mean() if len(expressed) else np.nan})
    return pd.DataFrame(folds), pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

def add_overlay_fields(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    checks = pd.DataFrame({
        'return_26_strong': out['return_26'] >= 0.055,
        'return_13_strong': out['return_13'] >= 0.035,
        'return_8_strong': out['return_8'] >= 0.025,
        'ema_slope_strong': out['ema_slope_atr'] >= 0.80,
        'close_progress_strong': out['three_bar_close_progress'] >= 1.55,
    }, index=out.index)
    out['bull_regime_score'] = checks.sum(axis=1).astype(int)
    out['extreme_bull_regime'] = out['bull_regime_score'] >= 4
    out['option_overlay_eligible'] = (
        out['extreme_bull_regime']
        & out['realized_option_return'].notna()
        & (out['predicted_net_return'] >= 0.0035)
        & (out['expected_option_return'] >= 0.015)
        & (out['option_probability_profit'] >= 0.48)
        & (out['spread_fraction'] <= 0.08)
        & (out['open_interest'] >= 250)
    )
    return out


def overlay_fraction(trades: pd.DataFrame, mode: str) -> np.ndarray:
    eligible = trades['option_overlay_eligible'].to_numpy(bool)
    if mode == 'fixed_05': return np.where(eligible, 0.05, 0.0)
    if mode == 'fixed_10': return np.where(eligible, 0.10, 0.0)
    if mode == 'fixed_20': return np.where(eligible, 0.20, 0.0)
    if mode == 'fixed_30': return np.where(eligible, 0.30, 0.0)
    if mode == 'tiered':
        score = trades['bull_regime_score'].to_numpy(int)
        predicted = trades['predicted_net_return'].to_numpy(float)
        utility = trades['option_utility'].fillna(-np.inf).to_numpy(float)
        fraction = np.where(score >= 5, 0.20, 0.10)
        fraction = np.where((score >= 5) & (predicted >= 0.008) & (utility >= 0.02), 0.30, fraction)
        return np.where(eligible, fraction, 0.0)
    raise ValueError(mode)


def additive_returns(trades: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray]:
    fractions = overlay_fraction(trades, mode)
    stock = trades['net_return'].to_numpy(float)
    option = trades['realized_option_return'].fillna(0.0).to_numpy(float)
    # True additive exposure: keep the full stock return and add premium-at-risk exposure.
    return stock + fractions * option, fractions


def summarize(trades: pd.DataFrame, simulations: int, seed: int) -> pd.DataFrame:
    strategies = {'stock_only': trades['net_return'].to_numpy(float)}
    for mode in ('fixed_05', 'fixed_10', 'fixed_20', 'fixed_30', 'tiered'):
        strategies[f'additive_{mode}'], _ = additive_returns(trades, mode)
    rows = []
    for i, (name, returns) in enumerate(strategies.items()):
        metrics = hybrid.path_metrics(returns)
        ruin = hybrid.probability_of_ruin(returns, simulations, seed + 101 * i)
        rows.append({'strategy': name, 'trades': len(returns), 'mean_trade_return': float(np.mean(returns)),
                     'win_rate': float(np.mean(returns > 0)), 'compounded_return': metrics['return'],
                     'max_drawdown': metrics['max_drawdown'], **ruin})
    return pd.DataFrame(rows).sort_values('compounded_return', ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', type=int, default=500)
    parser.add_argument('--seed', type=int, default=9381)
    parser.add_argument('--return-threshold-bps', type=float, default=5.0)
    parser.add_argument('--rank-threshold', type=float, default=0.55)
    parser.add_argument('--ruin-simulations', type=int, default=5000)
    parser.add_argument('--output', default='outputs/equity_broad_universe_additive_overlay')
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute(): output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    hybrid.SYMBOL_IV_FLOOR.update({'NVDA': 0.34, 'AMZN': 0.28, 'META': 0.30, 'GOOGL': 0.24, 'AMD': 0.40})
    hybrid.SYMBOL_LIQUIDITY.update({'NVDA': 1.20, 'AMZN': 1.08, 'META': 1.02, 'GOOGL': 1.00, 'AMD': 0.92})

    frames, shocks = generate_broad_market(args.sessions, args.seed)
    candidates = build_candidates(frames, max_bars=26)
    folds, trades = run_fast_walk_forward(candidates, args.return_threshold_bps)
    trades = add_overlay_fields(trades)
    summary = summarize(trades, args.ruin_simulations, args.seed)

    for mode in ('fixed_05', 'fixed_10', 'fixed_20', 'fixed_30', 'tiered'):
        ret, frac = additive_returns(trades, mode)
        trades[f'{mode}_overlay_fraction'] = frac
        trades[f'{mode}_additive_return'] = ret

    eligible = trades[trades['option_overlay_eligible']].copy()
    symbol_summary = trades.groupby('symbol').agg(
        selected_trades=('symbol', 'size'),
        mean_stock_return=('net_return', 'mean'),
        overlay_eligible=('option_overlay_eligible', 'sum'),
        mean_option_return=('realized_option_return', 'mean'),
    ).reset_index()
    fold_overlay = trades.groupby('fold').agg(
        selected_trades=('fold', 'size'),
        mean_stock_return=('net_return', 'mean'),
        extreme_bull_signals=('extreme_bull_regime', 'sum'),
        overlay_eligible=('option_overlay_eligible', 'sum'),
        mean_option_return=('realized_option_return', 'mean'),
    ).reset_index()

    regime_summary = {
        'symbols': list(SYMBOLS),
        'sessions': args.sessions,
        'selected_stock_trades': int(len(trades)),
        'extreme_bull_signals': int(trades['extreme_bull_regime'].sum()),
        'option_overlay_eligible': int(trades['option_overlay_eligible'].sum()),
        'overlay_eligibility_rate': float(trades['option_overlay_eligible'].mean()),
        'eligible_mean_stock_return': float(eligible['net_return'].mean()) if len(eligible) else None,
        'eligible_mean_option_return': float(eligible['realized_option_return'].mean()) if len(eligible) else None,
        'eligible_option_win_rate': float((eligible['realized_option_return'] > 0).mean()) if len(eligible) else None,
        'eligible_full_premium_losses': int((eligible['realized_option_return'] <= -0.999).sum()),
        'eligible_symbols': eligible['symbol'].value_counts().to_dict(),
        'eligible_folds': eligible['fold'].value_counts().sort_index().to_dict(),
    }

    folds.to_csv(output / 'fold_results.csv', index=False)
    trades.to_csv(output / 'selected_trades_with_additive_overlay.csv', index=False)
    summary.to_csv(output / 'strategy_summary.csv', index=False)
    symbol_summary.to_csv(output / 'symbol_summary.csv', index=False)
    fold_overlay.to_csv(output / 'fold_overlay_summary.csv', index=False)
    shocks.to_csv(output / 'earnings_like_shocks.csv', index=False)
    (output / 'results.json').write_text(json.dumps({
        'configuration': vars(args), 'regime_summary': regime_summary,
        'strategy_summary': summary.to_dict(orient='records'),
        'symbol_summary': symbol_summary.to_dict(orient='records'),
        'fold_overlay_summary': fold_overlay.to_dict(orient='records'),
    }, indent=2, default=str))
    print(summary.to_string(index=False))
    print(json.dumps(regime_summary, indent=2, default=str))

if __name__ == '__main__':
    main()
