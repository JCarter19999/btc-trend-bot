from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from run_equity_gap_walkforward_experiment import (
    GEOMETRY_EXTENDED,
    build_candidates,
    fit_model,
    generate_market_with_shocks,
)
from run_equity_ranking_options_experiment import fit_pairwise_ranker, make_pairwise_training, rank_scores


@dataclass(frozen=True)
class ChainConfig:
    dtes: tuple[int, ...] = (14, 21, 30, 45, 60)
    target_deltas: tuple[float, ...] = (0.25, 0.35, 0.45, 0.55, 0.65)
    min_open_interest: int = 100
    min_volume: int = 20
    max_spread_fraction: float = 0.12
    min_premium: float = 0.15
    rate: float = 0.04


CHAIN = ChainConfig()
SYMBOL_IV_FLOOR = {'AAPL': 0.22, 'MSFT': 0.20, 'TSLA': 0.38}
SYMBOL_LIQUIDITY = {'AAPL': 1.15, 'MSFT': 1.00, 'TSLA': 0.90}


def stable_uniform(*parts: object) -> float:
    digest = hashlib.sha256('|'.join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], 'big') / 2**64


def bs_call(spot: float, strike: float, tau: float, sigma: float, rate: float = 0.04) -> float:
    if tau <= 0:
        return max(spot - strike, 0.0)
    sigma = max(float(sigma), 1e-6)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)
    return spot * norm.cdf(d1) - strike * math.exp(-rate * tau) * norm.cdf(d2)


def bs_delta(spot: float, strike: float, tau: float, sigma: float, rate: float = 0.04) -> float:
    if tau <= 0:
        return float(spot > strike)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * tau) / (sigma * math.sqrt(tau))
    return float(norm.cdf(d1))


def strike_for_delta(spot: float, tau: float, sigma: float, delta: float, rate: float = 0.04) -> float:
    d1 = norm.ppf(np.clip(delta, 0.02, 0.98))
    raw = spot / math.exp(d1 * sigma * math.sqrt(tau) - (rate + 0.5 * sigma * sigma) * tau)
    increment = 2.5 if spot < 250 else 5.0
    return round(raw / increment) * increment


def base_realized_iv(row: pd.Series) -> float:
    rv = max(float(row.get('realized_vol_13', 0.01)), 1e-5) * math.sqrt(13 * 252)
    return max(SYMBOL_IV_FLOOR.get(str(row.symbol), 0.25), rv)


def skewed_iv(row: pd.Series, strike: float, dte: int) -> float:
    spot = float(row.entry_price)
    log_moneyness = math.log(strike / spot)
    base = base_realized_iv(row)
    # Equity-style left skew plus a mild smile and term structure.
    left_skew = 0.42 * max(-log_moneyness, 0.0)
    right_smile = 0.16 * max(log_moneyness, 0.0)
    curvature = 1.9 * log_moneyness * log_moneyness
    term = 0.035 * math.sqrt(30.0 / dte)
    event = 0.10 if float(row.get('shock_volume_flag', 0.0)) > 0 else 0.0
    return float(np.clip(base + left_skew + right_smile + curvature + term + event, 0.12, 1.60))


def liquidity_fields(row: pd.Series, strike: float, dte: int, delta: float, premium: float) -> dict:
    symbol = str(row.symbol)
    spot = float(row.entry_price)
    moneyness = abs(math.log(strike / spot))
    term_peak = math.exp(-abs(dte - 30) / 38.0)
    delta_peak = math.exp(-abs(delta - 0.50) / 0.24)
    base = SYMBOL_LIQUIDITY.get(symbol, 1.0) * term_peak * delta_peak * math.exp(-4.0 * moneyness)
    jitter = 0.72 + 0.56 * stable_uniform(symbol, row.signal_time, strike, dte)
    oi = int(max(1, 2600 * base * jitter))
    volume = int(max(0, 0.11 * oi * (0.7 + stable_uniform('v', symbol, row.signal_time, strike, dte))))
    spread_fraction = 0.012 + 0.16 / math.sqrt(max(oi, 1)) + 0.065 * moneyness + 0.018 * (14 / dte)
    spread_fraction *= 1.0 + 0.35 * (symbol == 'TSLA')
    spread_fraction = float(np.clip(spread_fraction, 0.008, 0.30))
    return {'open_interest': oi, 'option_volume': volume, 'spread_fraction': spread_fraction, 'premium_mid': premium}


def generate_chain(row: pd.Series, config: ChainConfig = CHAIN) -> pd.DataFrame:
    spot = float(row.entry_price)
    contracts = []
    for dte in config.dtes:
        tau = dte / 252.0
        base_iv = base_realized_iv(row)
        for requested_delta in config.target_deltas:
            strike = strike_for_delta(spot, tau, base_iv, requested_delta, config.rate)
            iv = skewed_iv(row, strike, dte)
            premium = bs_call(spot, strike, tau, iv, config.rate)
            delta = bs_delta(spot, strike, tau, iv, config.rate)
            liq = liquidity_fields(row, strike, dte, delta, premium)
            rec = {
                'symbol': str(row.symbol), 'signal_time': row.signal_time, 'spot': spot,
                'strike': strike, 'dte': dte, 'iv': iv, 'delta': delta, **liq,
            }
            rec['liquid'] = (
                rec['open_interest'] >= config.min_open_interest
                and rec['option_volume'] >= config.min_volume
                and rec['spread_fraction'] <= config.max_spread_fraction
                and premium >= config.min_premium
            )
            contracts.append(rec)
    return pd.DataFrame(contracts)


def contract_expectation(
    row: pd.Series,
    contract: pd.Series,
    residual_std: float,
    expected_hold_sessions: float,
    scenario_count: int = 81,
) -> dict:
    mean_return = float(row.predicted_net_return)
    horizon_scale = math.sqrt(max(expected_hold_sessions, 0.25) / 2.0)
    sigma_return = float(np.clip(residual_std * horizon_scale, 0.004, 0.12))
    quantiles = (np.arange(scenario_count) + 0.5) / scenario_count
    scenario_returns = mean_return + sigma_return * norm.ppf(quantiles)
    entry_mid = float(contract.premium_mid)
    half_spread = float(contract.spread_fraction) / 2.0
    entry_fill = entry_mid * (1.0 + half_spread)
    tau1 = max((float(contract.dte) - expected_hold_sessions) / 252.0, 0.0)
    expected_exit_iv = max(0.10, float(contract.iv) * (0.95 - 0.55 * max(mean_return, 0.0)))
    exit_values = []
    for ret in scenario_returns:
        spot1 = float(row.entry_price) * max(0.05, 1.0 + ret)
        # Adverse scenarios receive a modest IV expansion; favorable scenarios experience mild crush.
        iv1 = expected_exit_iv * (1.0 + 0.9 * max(-ret, 0.0) - 0.25 * max(ret, 0.0))
        exit_mid = bs_call(spot1, float(contract.strike), tau1, max(iv1, 0.10), CHAIN.rate)
        exit_fill = max(exit_mid * (1.0 - half_spread), 0.0)
        exit_values.append(max(exit_fill / max(entry_fill, 1e-9) - 1.0, -1.0))
    values = np.asarray(exit_values)
    expected = float(values.mean())
    downside = float(np.mean(np.square(np.minimum(values, 0.0))))
    prob_profit = float(np.mean(values > 0))
    cvar10 = float(np.mean(np.sort(values)[: max(1, scenario_count // 10)]))
    liquidity_penalty = 0.20 * float(contract.spread_fraction) + 2.0 / math.sqrt(float(contract.open_interest))
    utility = expected - 0.24 * downside + 0.08 * cvar10 - liquidity_penalty
    return {
        'expected_option_return': expected,
        'option_downside_semivariance': downside,
        'option_probability_profit': prob_profit,
        'option_cvar10': cvar10,
        'option_utility': utility,
        'expected_exit_iv': expected_exit_iv,
        'entry_fill': entry_fill,
        'expected_hold_sessions': expected_hold_sessions,
    }


def optimize_contract(row: pd.Series, residual_std: float, expected_hold_sessions: float) -> dict | None:
    chain = generate_chain(row)
    liquid = chain[chain.liquid].copy()
    if liquid.empty:
        return None
    scored = []
    for _, contract in liquid.iterrows():
        scored.append({**contract.to_dict(), **contract_expectation(row, contract, residual_std, expected_hold_sessions)})
    table = pd.DataFrame(scored)
    # Avoid contracts that are only attractive because of a tiny premium or very low win probability.
    eligible = table[(table.option_probability_profit >= 0.42) & (table.expected_option_return > 0.0)]
    if eligible.empty:
        return None
    best = eligible.sort_values(['option_utility', 'expected_option_return'], ascending=False).iloc[0]
    return best.to_dict()


def realized_option_return(row: pd.Series, contract: dict) -> dict:
    elapsed = max(float(row.bars_held) / 13.0, 1.0 / 13.0)
    tau1 = max((float(contract['dte']) - elapsed) / 252.0, 0.0)
    adverse = max(-float(row.mae), 0.0)
    favorable = max(float(row.mfe), 0.0)
    event_crush = 0.14 if float(row.get('shock_volume_flag', 0.0)) > 0 else 0.0
    exit_iv = max(0.10, float(contract['iv']) * (0.97 + min(adverse * 7.0, 0.30) - min(favorable * 2.2, 0.10) - event_crush))
    exit_mid = bs_call(float(row.exit_price), float(contract['strike']), tau1, exit_iv, CHAIN.rate)
    half_spread = float(contract['spread_fraction']) / 2.0
    exit_fill = max(exit_mid * (1.0 - half_spread), 0.0)
    ret = max(exit_fill / max(float(contract['entry_fill']), 1e-9) - 1.0, -1.0)
    return {'realized_option_return': ret, 'option_exit_fill': exit_fill, 'realized_exit_iv': exit_iv}


def group_pairwise_scores(model, group: pd.DataFrame, features: list[str]) -> np.ndarray:
    if len(group) == 1:
        return np.ones(1)
    scores = np.zeros(len(group), dtype=float)
    for i in range(len(group)):
        probs = []
        left = group.iloc[i]
        for j in range(len(group)):
            if i == j:
                continue
            right = group.iloc[j]
            rec = {f: float(left[f]) - float(right[f]) for f in features}
            rec['left_symbol'] = str(left.symbol)
            rec['right_symbol'] = str(right.symbol)
            x = pd.DataFrame([rec])
            probs.append(float(model.predict_proba(x[features + ['left_symbol', 'right_symbol']])[:, 1][0]))
        scores[i] = float(np.mean(probs))
    return scores


def select_hybrid_candidates(test: pd.DataFrame, return_threshold: float, rank_threshold: float) -> pd.DataFrame:
    winners = []
    for _, group in test.groupby('signal_time', sort=True):
        group = group.sort_values(['rank_score', 'predicted_net_return'], ascending=False)
        row = group.iloc[0]
        if float(row.predicted_net_return) > return_threshold and float(row.rank_score) >= rank_threshold:
            winners.append(row)
    ranked = pd.DataFrame(winners)
    if ranked.empty:
        return ranked
    accepted = []
    next_free = pd.Timestamp.min.tz_localize('UTC')
    for _, row in ranked.sort_values('signal_time').iterrows():
        if pd.Timestamp(row.signal_time) < next_free:
            continue
        accepted.append(row)
        next_free = pd.Timestamp(row.exit_time)
    return pd.DataFrame(accepted)


def kelly_fraction(expected_return: float, downside_semivariance: float, cap: float = 0.50) -> float:
    variance_proxy = max(downside_semivariance * 2.0, 1e-4)
    raw = max(expected_return, 0.0) / variance_proxy
    return float(np.clip(0.25 * raw, 0.05, cap))


def expression_returns(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in selected.iterrows():
        stock_ret = float(row.net_return)
        opt_ret = float(row.realized_option_return) if pd.notna(row.realized_option_return) else np.nan
        opt_expected = float(row.expected_option_return) if pd.notna(row.expected_option_return) else -np.inf
        opt_fraction = kelly_fraction(opt_expected, float(row.option_downside_semivariance), 0.50) if np.isfinite(opt_expected) else 0.0
        # Stock-only is the stable active strategy; option-only is intentionally aggressive.
        stock_only = stock_ret
        call_25 = 0.25 * opt_ret if np.isfinite(opt_ret) else 0.0
        call_kelly = opt_fraction * opt_ret if np.isfinite(opt_ret) else 0.0
        # Dynamic selector uses options only when expected convex return clearly exceeds stock edge.
        use_option = np.isfinite(opt_ret) and opt_expected > max(0.12, 5.0 * float(row.predicted_net_return))
        dynamic = call_kelly if use_option else stock_ret
        # Combined expression keeps stock exposure and layers a capped premium position on the strongest signals.
        overlay_fraction = min(opt_fraction, 0.30) if use_option else 0.0
        stock_weight = 1.0 - overlay_fraction
        stock_plus_call = stock_weight * stock_ret + overlay_fraction * opt_ret if np.isfinite(opt_ret) else stock_ret
        rows.append({
            **row.to_dict(), 'stock_return_expr': stock_only, 'call_25_return_expr': call_25,
            'call_kelly_return_expr': call_kelly, 'dynamic_return_expr': dynamic,
            'stock_plus_call_return_expr': stock_plus_call, 'option_fraction': opt_fraction,
            'overlay_fraction': overlay_fraction, 'dynamic_used_option': bool(use_option),
        })
    return pd.DataFrame(rows)


def path_metrics(returns: np.ndarray) -> dict:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for ret in returns:
        equity *= max(1.0 + float(ret), 1e-9)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
    return {'return': equity - 1.0, 'max_drawdown': max_dd, 'final_equity': equity}


def probability_of_ruin(returns: np.ndarray, simulations: int, seed: int) -> dict:
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return {}
    rng = np.random.default_rng(seed)
    below50 = below25 = below10 = 0
    final_losses = 0
    max_dds = []
    finals = []
    for _ in range(simulations):
        sample = rng.choice(returns, size=len(returns), replace=True)
        # Stress execution and model error with occasional additional losses.
        noise = rng.normal(-0.0004, 0.0030, size=len(sample))
        tail_mask = rng.random(len(sample)) < 0.015
        noise[tail_mask] -= rng.uniform(0.03, 0.15, size=int(tail_mask.sum()))
        stressed = np.maximum(sample + noise, -0.999)
        equity = 1.0
        peak = 1.0
        min_equity = 1.0
        dd = 0.0
        for ret in stressed:
            equity *= max(1.0 + ret, 1e-9)
            peak = max(peak, equity)
            min_equity = min(min_equity, equity)
            dd = max(dd, (peak - equity) / peak)
        below50 += min_equity < 0.50
        below25 += min_equity < 0.25
        below10 += min_equity < 0.10
        final_losses += equity < 1.0
        max_dds.append(dd)
        finals.append(equity)
    return {
        'simulations': simulations,
        'p_equity_below_50pct': below50 / simulations,
        'p_equity_below_25pct': below25 / simulations,
        'p_equity_below_10pct': below10 / simulations,
        'p_final_loss': final_losses / simulations,
        'median_final_equity': float(np.median(finals)),
        'p05_final_equity': float(np.quantile(finals, 0.05)),
        'median_max_drawdown': float(np.median(max_dds)),
        'p95_max_drawdown': float(np.quantile(max_dds, 0.95)),
    }


def run_walk_forward(candidates: pd.DataFrame, return_threshold_bps: float, rank_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = GEOMETRY_EXTENDED
    times = np.sort(candidates.signal_time.unique())
    boundaries = [int(len(times) * x) for x in (0.40, 0.55, 0.70, 0.85, 1.00)]
    fold_rows = []
    trade_frames = []
    for fold in range(4):
        train_end = times[boundaries[fold]]
        test_end = times[boundaries[fold + 1] - 1]
        train_all = candidates[candidates.signal_time < train_end].copy()
        test = candidates[(candidates.signal_time >= train_end) & (candidates.signal_time <= test_end)].copy()
        if len(train_all) < 150 or len(test) < 25:
            continue
        cut = int(len(train_all) * 0.80)
        train = train_all.iloc[:cut].copy()
        calibration = train_all.iloc[cut:].copy()
        ridge = fit_model(train, features, 'ridge')
        cal_raw = ridge.predict(calibration[features + ['symbol']])
        slope, intercept = np.polyfit(cal_raw, calibration.net_return.to_numpy(float), 1)
        test['predicted_net_return'] = intercept + slope * ridge.predict(test[features + ['symbol']])
        residual_std = float(np.std(calibration.net_return.to_numpy(float) - (intercept + slope * cal_raw), ddof=1))
        expected_hold_sessions = float(np.clip(train.bars_held.median() / 13.0, 0.25, 3.0))

        ranker = fit_pairwise_ranker(train_all, features)
        # Vectorized pairwise-ranking score against a neutral CASH reference. The direct-return
        # Ridge model remains the absolute gate; this score is only used to resolve simultaneous candidates.
        test['rank_score'] = rank_scores(ranker, test, features)
        selected = select_hybrid_candidates(test, return_threshold_bps / 10_000.0, rank_threshold)

        pair_x, pair_y = make_pairwise_training(test, features)
        auc = float('nan')
        if len(pair_y) and len(np.unique(pair_y)) == 2:
            auc = roc_auc_score(pair_y, ranker.predict_proba(pair_x[features + ['left_symbol', 'right_symbol']])[:, 1])

        optimized = []
        for _, row in selected.iterrows():
            best = optimize_contract(row, residual_std, expected_hold_sessions)
            rec = row.to_dict()
            if best is None:
                rec.update({
                    'strike': np.nan, 'dte': np.nan, 'delta': np.nan, 'iv': np.nan,
                    'open_interest': np.nan, 'option_volume': np.nan, 'spread_fraction': np.nan,
                    'expected_option_return': np.nan, 'option_probability_profit': np.nan,
                    'option_downside_semivariance': np.nan, 'option_cvar10': np.nan,
                    'entry_fill': np.nan, 'realized_option_return': np.nan,
                })
            else:
                rec.update(best)
                rec.update(realized_option_return(row, best))
            optimized.append(rec)
        expressed = expression_returns(pd.DataFrame(optimized)) if optimized else pd.DataFrame()
        if len(expressed):
            expressed['fold'] = fold + 1
            trade_frames.append(expressed)

        fold_rec = {
            'fold': fold + 1, 'train_candidates': len(train), 'calibration_candidates': len(calibration),
            'test_candidates': len(test), 'selected_trades': len(expressed), 'pairwise_auc': auc,
            'residual_std_bps': residual_std * 10000, 'expected_hold_sessions': expected_hold_sessions,
            'mean_predicted_return_bps': expressed.predicted_net_return.mean() * 10000 if len(expressed) else np.nan,
            'mean_realized_stock_bps': expressed.net_return.mean() * 10000 if len(expressed) else np.nan,
            'option_available_rate': expressed.realized_option_return.notna().mean() if len(expressed) else np.nan,
            'mean_option_return': expressed.realized_option_return.mean() if len(expressed) else np.nan,
            'median_option_dte': expressed.dte.median() if len(expressed) else np.nan,
            'median_option_delta': expressed.delta.median() if len(expressed) else np.nan,
            'median_spread_pct': expressed.spread_fraction.median() if len(expressed) else np.nan,
            'mean_open_interest': expressed.open_interest.mean() if len(expressed) else np.nan,
        }
        for col in ['stock_return_expr', 'call_25_return_expr', 'call_kelly_return_expr', 'dynamic_return_expr', 'stock_plus_call_return_expr']:
            metrics = path_metrics(expressed[col].to_numpy(float)) if len(expressed) else {'return': np.nan, 'max_drawdown': np.nan}
            fold_rec[f'{col}_portfolio_return'] = metrics['return']
            fold_rec[f'{col}_max_drawdown'] = metrics['max_drawdown']
        fold_rows.append(fold_rec)
    return pd.DataFrame(fold_rows), pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', type=int, default=1300)
    parser.add_argument('--seed', type=int, default=9381)
    parser.add_argument('--return-threshold-bps', type=float, default=5.0)
    parser.add_argument('--rank-threshold', type=float, default=0.55)
    parser.add_argument('--ruin-simulations', type=int, default=5000)
    parser.add_argument('--output', default='outputs/equity_hybrid_option_optimizer')
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    frames, shocks = generate_market_with_shocks(args.sessions, args.seed)
    candidates = build_candidates(frames, max_bars=26)
    folds, trades = run_walk_forward(candidates, args.return_threshold_bps, args.rank_threshold)

    strategies = {
        'stock_only': 'stock_return_expr',
        'call_25pct': 'call_25_return_expr',
        'call_fractional_kelly': 'call_kelly_return_expr',
        'dynamic_stock_or_call': 'dynamic_return_expr',
        'stock_plus_call_overlay': 'stock_plus_call_return_expr',
    }
    summaries = []
    ruin_payload = {}
    for name, column in strategies.items():
        returns = trades[column].to_numpy(float)
        metrics = path_metrics(returns)
        ruin = probability_of_ruin(returns, args.ruin_simulations, args.seed + len(name))
        ruin_payload[name] = ruin
        summaries.append({
            'strategy': name,
            'trades': len(returns),
            'mean_trade_return': float(np.mean(returns)),
            'median_trade_return': float(np.median(returns)),
            'win_rate': float(np.mean(returns > 0)),
            'compounded_return': metrics['return'],
            'max_drawdown': metrics['max_drawdown'],
            **ruin,
        })
    summary = pd.DataFrame(summaries).sort_values('compounded_return', ascending=False)

    contract_summary = {
        'contracts_optimized': int(trades.realized_option_return.notna().sum()),
        'option_availability_rate': float(trades.realized_option_return.notna().mean()),
        'median_dte': float(trades.dte.median()),
        'dte_counts': trades.dte.value_counts(dropna=True).sort_index().to_dict(),
        'median_delta': float(trades.delta.median()),
        'median_spread_fraction': float(trades.spread_fraction.median()),
        'median_open_interest': float(trades.open_interest.median()),
        'mean_expected_option_return': float(trades.expected_option_return.mean()),
        'mean_realized_option_return': float(trades.realized_option_return.mean()),
        'option_win_rate': float((trades.loc[trades.realized_option_return.notna(), 'realized_option_return'] > 0).mean()),
        'total_premium_losses': int((trades.realized_option_return <= -0.999).sum()),
        'dynamic_option_usage_rate': float(trades.dynamic_used_option.mean()),
        'mean_overlay_fraction': float(trades.overlay_fraction.mean()),
        'symbol_counts': trades.symbol.value_counts().to_dict(),
    }

    folds.to_csv(output / 'fold_results.csv', index=False)
    trades.to_csv(output / 'selected_trades_with_contracts.csv', index=False)
    summary.to_csv(output / 'strategy_summary.csv', index=False)
    candidates.to_csv(output / 'candidates.csv', index=False)
    shocks.to_csv(output / 'earnings_like_shocks.csv', index=False)
    payload = {
        'configuration': vars(args),
        'contract_summary': contract_summary,
        'strategy_summary': summary.to_dict(orient='records'),
        'folds': folds.to_dict(orient='records'),
        'probability_of_ruin': ruin_payload,
    }
    (output / 'results.json').write_text(json.dumps(payload, indent=2, default=str))
    print('\nSTRATEGY SUMMARY')
    print(summary.to_string(index=False))
    print('\nCONTRACT SUMMARY')
    print(json.dumps(contract_summary, indent=2, default=str))
    print('\nFOLDS')
    print(folds.to_string(index=False))


if __name__ == '__main__':
    main()
