from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

from run_equity_gap_walkforward_experiment import (
    BASE_EXTENDED,
    GEOMETRY_EXTENDED,
    build_candidates,
    generate_market_with_shocks,
    portfolio_metrics,
)


@dataclass(frozen=True)
class OptionPolicy:
    name: str
    dte_sessions: int
    target_delta: float
    option_allocation: float
    iv_markup: float = 1.0
    half_spread_bps: float = 150.0


OPTION_POLICIES = (
    OptionPolicy('call_14d_delta35', 14, 0.35, 1.0, 1.0, 180.0),
    OptionPolicy('call_30d_delta40', 30, 0.40, 1.0, 1.0, 140.0),
    OptionPolicy('call_45d_delta50', 45, 0.50, 1.0, 1.0, 110.0),
)


def bs_call(spot: float, strike: float, tau: float, sigma: float, rate: float = 0.04) -> float:
    if tau <= 0:
        return max(spot - strike, 0.0)
    sigma = max(float(sigma), 1e-6)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)
    return spot * norm.cdf(d1) - strike * math.exp(-rate * tau) * norm.cdf(d2)


def strike_for_delta(spot: float, tau: float, sigma: float, delta: float, rate: float = 0.04) -> float:
    d1 = norm.ppf(np.clip(delta, 0.02, 0.98))
    return spot / math.exp(d1 * sigma * math.sqrt(tau) - (rate + 0.5 * sigma * sigma) * tau)


def estimate_iv(row: pd.Series, policy: OptionPolicy) -> float:
    # Annualize 30-minute realized volatility (13 regular-session bars/day, 252 sessions/year).
    rv = max(float(row.get('realized_vol_13', 0.01)), 1e-5) * math.sqrt(13 * 252)
    symbol_floor = {'AAPL': 0.22, 'MSFT': 0.20, 'TSLA': 0.38}.get(str(row.symbol), 0.25)
    event_premium = 0.12 if float(row.get('shock_volume_flag', 0.0)) > 0 else 0.0
    return max(symbol_floor, rv) * policy.iv_markup + event_premium


def option_return(row: pd.Series, policy: OptionPolicy) -> dict:
    entry_spot = float(row.entry_price)
    exit_spot = float(row.exit_price)
    entry_iv = estimate_iv(row, policy)
    tau0 = policy.dte_sessions / 252.0
    elapsed_sessions = max(float(row.bars_held) / 13.0, 1.0 / 13.0)
    tau1 = max((policy.dte_sessions - elapsed_sessions) / 252.0, 0.0)
    strike = strike_for_delta(entry_spot, tau0, entry_iv, policy.target_delta)
    entry_mid = bs_call(entry_spot, strike, tau0, entry_iv)
    # IV mean reverts, but expands when the underlying trade suffers adverse excursion.
    adverse = max(-float(row.mae), 0.0)
    favorable = max(float(row.mfe), 0.0)
    exit_iv = max(0.10, entry_iv * (0.96 + min(adverse * 8.0, 0.25) - min(favorable * 2.0, 0.08)))
    exit_mid = bs_call(exit_spot, strike, tau1, exit_iv)
    spread = policy.half_spread_bps / 10_000.0
    entry_fill = entry_mid * (1.0 + spread)
    exit_fill = max(exit_mid * (1.0 - spread), 0.0)
    ret = exit_fill / max(entry_fill, 1e-9) - 1.0
    return {
        'option_policy': policy.name,
        'option_return': max(ret, -1.0),
        'option_entry_premium': entry_fill,
        'option_exit_premium': exit_fill,
        'option_strike': strike,
        'entry_iv': entry_iv,
        'exit_iv': exit_iv,
        'dte_sessions': policy.dte_sessions,
        'target_delta': policy.target_delta,
    }


def make_pairwise_training(train: pd.DataFrame, features: list[str], max_pairs_per_time: int = 6) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict] = []
    labels: list[int] = []
    rng = np.random.default_rng(9107)
    for _, group in train.groupby('signal_time'):
        if len(group) < 2:
            continue
        indexes = list(group.index)
        pairs = [(indexes[i], indexes[j]) for i in range(len(indexes)) for j in range(i + 1, len(indexes))]
        rng.shuffle(pairs)
        for left_idx, right_idx in pairs[:max_pairs_per_time]:
            left = train.loc[left_idx]
            right = train.loc[right_idx]
            if abs(float(left.net_return) - float(right.net_return)) < 1e-8:
                continue
            rec = {f: float(left[f]) - float(right[f]) for f in features}
            rec['left_symbol'] = str(left.symbol)
            rec['right_symbol'] = str(right.symbol)
            rows.append(rec)
            labels.append(int(float(left.net_return) > float(right.net_return)))
            # Reverse pair makes symmetry explicit.
            rev = {f: -rec[f] for f in features}
            rev['left_symbol'] = str(right.symbol)
            rev['right_symbol'] = str(left.symbol)
            rows.append(rev)
            labels.append(1 - labels[-1])
    return pd.DataFrame(rows), np.asarray(labels, dtype=int)


def fit_pairwise_ranker(train: pd.DataFrame, features: list[str]) -> Pipeline:
    pair_x, pair_y = make_pairwise_training(train, features)
    transformer = ColumnTransformer([
        ('continuous', StandardScaler(), features),
        ('symbols', OneHotEncoder(handle_unknown='ignore'), ['left_symbol', 'right_symbol']),
    ])
    model = Pipeline([
        ('transform', transformer),
        ('ranker', LogisticRegression(C=0.35, max_iter=2000, class_weight='balanced')),
    ])
    model.fit(pair_x[features + ['left_symbol', 'right_symbol']], pair_y)
    return model


def rank_scores(model: Pipeline, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    # Score each candidate against a neutral zero-feature reference for consistent ordering.
    scoring = pd.DataFrame({f: frame[f].astype(float).to_numpy() for f in features})
    scoring['left_symbol'] = frame.symbol.astype(str).to_numpy()
    scoring['right_symbol'] = 'CASH'
    return model.predict_proba(scoring[features + ['left_symbol', 'right_symbol']])[:, 1]


def select_ranked(test: pd.DataFrame, min_rank_probability: float, top_k: int = 1) -> pd.DataFrame:
    ranked = test.sort_values(['signal_time', 'rank_probability'], ascending=[True, False])
    ranked = ranked.groupby('signal_time', as_index=False).head(top_k)
    ranked = ranked[ranked.rank_probability >= min_rank_probability].copy()
    accepted = []
    next_free = pd.Timestamp.min.tz_localize('UTC')
    for _, row in ranked.sort_values('signal_time').iterrows():
        if pd.Timestamp(row.signal_time) < next_free:
            continue
        accepted.append(row)
        next_free = pd.Timestamp(row.exit_time)
    return pd.DataFrame(accepted, columns=ranked.columns)


def option_portfolio_metrics(selected: pd.DataFrame, policy: OptionPolicy) -> tuple[float, float, pd.DataFrame]:
    rows = []
    equity = 1.0
    peak = 1.0
    drawdown = 0.0
    for _, row in selected.iterrows():
        result = option_return(row, policy)
        effective = policy.option_allocation * result['option_return']
        equity *= 1.0 + effective
        peak = max(peak, equity)
        drawdown = max(drawdown, (peak - equity) / peak)
        rows.append({**row.to_dict(), **result, 'effective_portfolio_return': effective, 'equity_after': equity})
    return equity - 1.0, drawdown, pd.DataFrame(rows)


def walk_forward_rank_options(candidates: pd.DataFrame, features: list[str], min_rank_probability: float = 0.57) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    times = np.sort(candidates.signal_time.unique())
    boundaries = [int(len(times) * x) for x in (0.40, 0.55, 0.70, 0.85, 1.00)]
    summaries = []
    stock_rows = []
    option_rows = []
    for fold in range(4):
        train_end = times[boundaries[fold]]
        test_end = times[boundaries[fold + 1] - 1]
        train = candidates[candidates.signal_time < train_end].copy()
        test = candidates[(candidates.signal_time >= train_end) & (candidates.signal_time <= test_end)].copy()
        if len(train) < 150 or len(test) < 25:
            continue
        ranker = fit_pairwise_ranker(train, features)
        test['rank_probability'] = rank_scores(ranker, test, features)
        selected = select_ranked(test, min_rank_probability)
        stock_ret, stock_dd = portfolio_metrics(selected)

        # Pairwise AUC on test pairs measures whether the model orders contemporaneous candidates correctly.
        pair_x, pair_y = make_pairwise_training(test, features)
        pair_auc = float('nan')
        if len(np.unique(pair_y)) == 2 and len(pair_y):
            pair_auc = roc_auc_score(pair_y, ranker.predict_proba(pair_x[features + ['left_symbol', 'right_symbol']])[:, 1])

        base = {
            'fold': fold + 1,
            'train_candidates': len(train),
            'test_candidates': len(test),
            'selected_trades': len(selected),
            'pairwise_auc': pair_auc,
            'stock_mean_return_bps': selected.net_return.mean() * 10000 if len(selected) else np.nan,
            'stock_portfolio_return': stock_ret,
            'stock_max_drawdown': stock_dd,
            'symbol_counts': json.dumps(selected.symbol.value_counts().to_dict()),
        }
        for policy in OPTION_POLICIES:
            opt_ret, opt_dd, details = option_portfolio_metrics(selected, policy)
            summaries.append({
                **base,
                'option_policy': policy.name,
                'option_portfolio_return': opt_ret,
                'option_max_drawdown': opt_dd,
                'option_mean_return': details.option_return.mean() if len(details) else np.nan,
                'option_win_rate': (details.option_return > 0).mean() if len(details) else np.nan,
                'option_total_losses': int((details.option_return <= -0.999).sum()) if len(details) else 0,
            })
            if len(details):
                details['fold'] = fold + 1
                option_rows.append(details)
        selected = selected.copy()
        selected['fold'] = fold + 1
        stock_rows.append(selected)
    return pd.DataFrame(summaries), pd.concat(stock_rows, ignore_index=True), pd.concat(option_rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', type=int, default=1300)
    parser.add_argument('--seed', type=int, default=9381)
    parser.add_argument('--output', default='outputs/equity_rank_options')
    parser.add_argument('--rank-threshold', type=float, default=0.57)
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    frames, shocks = generate_market_with_shocks(args.sessions, args.seed)
    candidates = build_candidates(frames, max_bars=26)
    candidates.to_csv(output / 'candidates_2_sessions.csv', index=False)
    shocks.to_csv(output / 'earnings_like_shocks.csv', index=False)

    results = []
    for feature_name, features in {'baseline': BASE_EXTENDED, 'geometry': GEOMETRY_EXTENDED}.items():
        summary, stocks, options = walk_forward_rank_options(candidates, features, args.rank_threshold)
        summary.insert(0, 'feature_set', feature_name)
        results.append(summary)
        stocks.to_csv(output / f'stock_selected_{feature_name}.csv', index=False)
        options.to_csv(output / f'option_selected_{feature_name}.csv', index=False)

    folds = pd.concat(results, ignore_index=True)
    aggregate = folds.groupby(['feature_set', 'option_policy'], as_index=False).agg(
        positive_stock_folds=('stock_portfolio_return', lambda s: int((s > 0).sum())),
        positive_option_folds=('option_portfolio_return', lambda s: int((s > 0).sum())),
        selected_trades=('selected_trades', 'sum'),
        mean_pairwise_auc=('pairwise_auc', 'mean'),
        mean_stock_return_bps=('stock_mean_return_bps', 'mean'),
        compounded_stock_return=('stock_portfolio_return', lambda s: float(np.prod(1.0 + s) - 1.0)),
        compounded_option_return=('option_portfolio_return', lambda s: float(np.prod(1.0 + s) - 1.0)),
        worst_option_fold=('option_portfolio_return', 'min'),
        max_option_drawdown=('option_max_drawdown', 'max'),
        mean_option_trade_return=('option_mean_return', 'mean'),
        mean_option_win_rate=('option_win_rate', 'mean'),
        total_option_total_losses=('option_total_losses', 'sum'),
    ).sort_values('compounded_option_return', ascending=False)

    folds.to_csv(output / 'fold_results.csv', index=False)
    aggregate.to_csv(output / 'summary.csv', index=False)
    payload = {
        'rank_threshold': args.rank_threshold,
        'best': aggregate.iloc[0].to_dict(),
        'all_results': aggregate.to_dict(orient='records'),
    }
    (output / 'results.json').write_text(json.dumps(payload, indent=2, default=str))
    print(aggregate.to_string(index=False))


if __name__ == '__main__':
    main()
