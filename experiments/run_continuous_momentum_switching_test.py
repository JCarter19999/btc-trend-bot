"""Tests whether switching stocks continuously whenever the daily momentum
leaderboard changes (rather than holding the current live fixed-10-day hold)
improves on `simple_trend`. Joey's framing: "if day 0 Apple is strongest but
day 5 Tesla is, is it worth switching continuously." Prior project findings
(ATR-stop removal, hold-duration monotonicity) suggest reacting faster hurts
for this exact strategy -- this tests the specific mechanism directly rather
than relying on analogy.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_equity_real_data_walkforward import (
    add_features, build_candidates, load_config, load_csv_dir, _select_fold_winners, FEATURES,
)
from btc_trend_bot.portfolio_sim import SizingMode, simulate_single_position

LIVE_CFG_PATH = Path("/home/joey/equity_v2_4/config/simple_trend_exit_regime_strategy.yaml")
OUT_DIR = ROOT / "outputs" / "continuous_momentum_switching_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_NULL_SEEDS = 1000


def load_all():
    cfg = load_config(LIVE_CFG_PATH)
    all_symbols = tuple(dict.fromkeys((*cfg.symbols, cfg.benchmark)))
    frames = load_csv_dir(all_symbols, ROOT / "data" / "real")
    return cfg, frames


def build_daily_leaderboard(cfg, frames) -> pd.DataFrame:
    """Daily #1-ranked symbol by relative_strength_20 among symbols with
    market_above_ema50>=1 (falling back to the full pool if none qualify),
    same convention as _select_fold_winners' simple_trend branch. Computed
    for EVERY day both symbols have valid features -- not restricted to the
    build_candidates' additional signal-quality mask, since a genuine daily
    leaderboard should reflect momentum ranking on every day, not just
    already-filtered candidate days."""
    bench = frames[cfg.benchmark]
    feats = {}
    for sym in cfg.symbols:
        f = add_features(frames[sym], bench)
        feats[sym] = f

    all_dates = sorted(set.union(*(set(f.index) for f in feats.values())))
    rows = []
    for d in all_dates:
        candidates = []
        for sym, f in feats.items():
            if d not in f.index:
                continue
            row = f.loc[d]
            rs = row.get("relative_strength_20")
            above = row.get("market_above_ema50")
            if pd.isna(rs):
                continue
            candidates.append((sym, float(rs), above))
        if not candidates:
            continue
        qualified = [(s, rs) for s, rs, above in candidates if pd.notna(above) and above >= 1]
        pool = qualified if qualified else [(s, rs) for s, rs, _ in candidates]
        pool.sort(key=lambda x: x[1], reverse=True)
        rows.append({"date": d, "leader": pool[0][0], "leader_rel_strength_20": pool[0][1],
                     "n_qualified": len(qualified), "n_pool": len(pool)})
    board = pd.DataFrame(rows).set_index("date").sort_index()
    board.to_csv(OUT_DIR / "daily_momentum_leaderboard.csv")
    return board


def reconstruct_baseline(cfg, frames):
    """Reproduces the validated 188-trade full-history fixed-10-day-hold
    result exactly, per this session's established method: bypass
    walk_forward()'s 756-bar warmup by calling _select_fold_winners directly
    against the FULL candidate set (selection='simple_trend' ignores
    'train' entirely), then run through simulate_single_position (the
    single-position-correct simulator, NOT simulate_capital which allows
    overlap)."""
    candidates = build_candidates(frames, cfg.benchmark, cfg)
    rng = np.random.default_rng(0)
    winners = _select_fold_winners(candidates, candidates, cfg, 0, rng, False, "simple_trend")
    # EQUITY_EXIT_REGIME_SIMPLE_TREND.md's validated characterization used
    # fixed-notional $2,500/trade (no compounding), NOT the live deployment
    # config's position_fraction=0.25 fixed-fractional sizing -- these are
    # two different sizing conventions for two different purposes (research
    # characterization vs. live capital allocation). Match the doc being
    # reproduced.
    sizing = SizingMode("fixed_notional_2500", "fixed_notional", cfg.initial_capital)
    path, stats = simulate_single_position(winners, cfg, sizing)
    return winners, path, stats


def continuous_switching_sim(cfg, frames, board: pd.DataFrame, cost_bps_each_side: float, switch_dates: pd.Index | None = None):
    """Holds the current position as long as it stays the #1-ranked symbol.
    On any day the #1 changes, exits at that day's close and opens a new
    position at the NEXT day's open (same next-bar-entry, no-lookahead
    convention as simulate_trade -- signal known at close of day t, entered
    at open of t+1). If switch_dates is given, forces switches on exactly
    those dates instead of following the real leaderboard (used for the
    random-switching control, holding switch COUNT fixed).
    """
    prices = {sym: frames[sym] for sym in cfg.symbols}
    dates = board.index

    if switch_dates is None:
        leader = board["leader"]
        change_mask = leader != leader.shift(1)
        change_mask.iloc[0] = True
        switch_on = set(dates[change_mask])
    else:
        switch_on = set(switch_dates)
        leader = board["leader"]

    trades = []
    current_symbol = None
    entry_date = None
    entry_price = None

    def _close_price(sym, d):
        f = prices[sym]
        return float(f.loc[d, "close"]) if d in f.index else None

    def _open_price(sym, d):
        f = prices[sym]
        return float(f.loc[d, "open"]) if d in f.index else None

    ordered_dates = list(dates)
    for idx, d in enumerate(ordered_dates):
        target_symbol = leader.loc[d]
        must_switch = d in switch_on
        if current_symbol is None:
            # open first position at the NEXT day's open
            if idx + 1 < len(ordered_dates):
                nd = ordered_dates[idx + 1]
                op = _open_price(target_symbol, nd)
                if op is not None:
                    current_symbol, entry_date, entry_price = target_symbol, nd, op
            continue
        if must_switch and (switch_dates is not None or target_symbol != current_symbol):
            cp = _close_price(current_symbol, d)
            if cp is not None and entry_price is not None:
                exit_fill = cp * (1 - cost_bps_each_side / 10000)
                entry_fill = entry_price * (1 + cost_bps_each_side / 10000)
                net_return = exit_fill / entry_fill - 1
                trades.append({"signal_time": d, "symbol": current_symbol, "entry_time": entry_date,
                               "exit_time": d, "net_return": net_return})
            new_target = target_symbol if switch_dates is None else leader.loc[d]
            if idx + 1 < len(ordered_dates):
                nd = ordered_dates[idx + 1]
                op = _open_price(new_target, nd)
                if op is not None:
                    current_symbol, entry_date, entry_price = new_target, nd, op
                else:
                    current_symbol = None

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, {"trades": 0}
    # Fixed-notional $2,500/trade, no compounding -- same convention as the
    # baseline reconstruction, for a fair apples-to-apples comparison
    # (matches EQUITY_EXIT_REGIME_SIMPLE_TREND.md's sizing, not the live
    # deployment config's separate position_fraction convention).
    equity = cfg.initial_capital
    peak = equity
    rows = []
    for _, t in trades_df.iterrows():
        notional = cfg.initial_capital
        pnl = notional * t.net_return
        equity = max(0.0, equity + pnl)
        peak = max(peak, equity)
        rows.append({**t.to_dict(), "equity": equity, "drawdown": 1 - equity / peak})
    path = pd.DataFrame(rows)
    stats = {
        "trades": len(path), "win_rate": float((path.net_return > 0).mean()),
        "mean_return": float(path.net_return.mean()), "expectancy_bps": float(path.net_return.mean() * 10000),
        "total_return": float(equity / cfg.initial_capital - 1), "max_drawdown": float(path.drawdown.max()),
        "ending_equity": float(equity),
    }
    return path, stats


def main():
    cfg, frames = load_all()
    print("=== Building daily momentum leaderboard ===")
    board = build_daily_leaderboard(cfg, frames)
    print(f"Leaderboard: {len(board)} days, {board.index.min()} to {board.index.max()}")
    n_switches_real = int((board["leader"] != board["leader"].shift(1)).sum())
    print(f"Real leader changes over full history: {n_switches_real}")

    print("\n=== Reconstructing validated fixed-10-day-hold baseline ===")
    winners, base_path, base_stats = reconstruct_baseline(cfg, frames)
    taken = base_path[base_path.trade_taken] if len(base_path) else base_path
    base_win_rate = float((taken.trade_pnl > 0).mean())
    base_pf = float(taken.loc[taken.trade_pnl > 0, "trade_pnl"].sum() / abs(taken.loc[taken.trade_pnl < 0, "trade_pnl"].sum()))
    print(f"Baseline: {len(taken)} trades, win_rate={base_win_rate:.3f}, PF={base_pf:.3f}, "
          f"total_return={base_stats['total_return']*100:.1f}%, max_dd={base_stats['max_drawdown']*100:.1f}%")
    print("EXPECTED: 188 trades, 58.0% win rate, PF 2.04, +457.4% total return, 22.1% max_dd")

    print("\n=== Continuous rank-driven switching, 2bps cost ===")
    cont_path_2bp, cont_stats_2bp = continuous_switching_sim(cfg, frames, board, cost_bps_each_side=2.0)
    print(json.dumps(cont_stats_2bp, indent=2))

    print("\n=== Continuous rank-driven switching, 5bps cost stress ===")
    cont_path_5bp, cont_stats_5bp = continuous_switching_sim(cfg, frames, board, cost_bps_each_side=5.0)
    print(json.dumps(cont_stats_5bp, indent=2))

    real_switch_dates = list(board.index[board["leader"] != board["leader"].shift(1)])
    n_switch = len(real_switch_dates)

    print(f"\n=== Random-switching control ({n_switch} switches, {N_NULL_SEEDS} seeds), 2bps cost ===")
    null_returns = []
    for seed in range(N_NULL_SEEDS):
        rng = np.random.default_rng(seed)
        random_dates = pd.Index(rng.choice(board.index[1:], size=min(n_switch, len(board) - 1), replace=False)).sort_values()
        _, stats = continuous_switching_sim(cfg, frames, board, cost_bps_each_side=2.0, switch_dates=random_dates)
        null_returns.append(stats.get("total_return", np.nan))
    null_returns = np.array([x for x in null_returns if np.isfinite(x)])
    real_return = cont_stats_2bp.get("total_return", np.nan)
    pct = float((null_returns < real_return).mean() * 100) if len(null_returns) else float("nan")
    print(f"Real total_return={real_return*100:.2f}%  null_mean={null_returns.mean()*100:.2f}% "
          f"null_std={null_returns.std()*100:.2f}%  percentile={pct:.1f}")

    print("\n=== Out-of-sample split (first half vs second half) ===")
    mid_date = board.index[len(board) // 2]
    board_first = board[board.index < mid_date]
    board_second = board[board.index >= mid_date]
    _, stats_first = continuous_switching_sim(cfg, frames, board_first, cost_bps_each_side=2.0)
    _, stats_second = continuous_switching_sim(cfg, frames, board_second, cost_bps_each_side=2.0)
    print(f"First half  ({board_first.index.min()} to {board_first.index.max()}): {json.dumps(stats_first)}")
    print(f"Second half ({board_second.index.min()} to {board_second.index.max()}): {json.dumps(stats_second)}")

    with open(OUT_DIR / "summary.json", "w") as fh:
        json.dump({
            "baseline_fixed_10day_hold": {"trades": int(len(taken)), "win_rate": base_win_rate,
                                          "profit_factor": base_pf, **{k: v for k, v in base_stats.items()}},
            "continuous_switching_2bp": cont_stats_2bp,
            "continuous_switching_5bp": cont_stats_5bp,
            "n_leader_changes_full_history": n_switches_real,
            "random_switching_control": {"real_total_return": float(real_return), "null_mean": float(null_returns.mean()),
                                         "null_std": float(null_returns.std()), "percentile": pct, "n_seeds": len(null_returns)},
            "oos_split": {"first_half": stats_first, "second_half": stats_second},
        }, fh, indent=2, default=str)
    cont_path_2bp.to_csv(OUT_DIR / "continuous_switching_trades_2bp.csv", index=False)
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
