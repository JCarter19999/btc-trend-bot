"""Multi-position portfolio study: does widening the universe to the S&P 100
and allowing up to 5 concurrent positions (each capped at 50% of capital,
so 20% each when fully deployed) increase trades/year and net return while
KEEPING the validated 10-day hold -- rather than shortening hold duration,
which the exit-strategy study already showed fights this strategy's actual
edge (medium-horizon drift capture, not a fast signal: 5/10/15/20-day holds
scored 106.6/243.3/272.3/452.7 bps/trade, monotonically *better* the longer
you hold).

Baseline being beaten: the live-deployed simple_trend strategy, 4 symbols,
single position, 188 trades over 2018-2026, 243.3 bps/trade expectancy,
22.1% max drawdown (EQUITY_EXIT_REGIME_SIMPLE_TREND.md).

Sizing rule (per Joey, clarified 2026-07-23): NOT a fixed 20% per slot.
Each new position gets min(50% of capital, currently-available cash
fraction) -- so 1-2 strong signals on a quiet day get sized up to 50% each,
while a fully-loaded 5-position book settles to ~20% each. Fixed-notional
per position (fraction x initial_capital, decided at entry, never
re-based on compounding) -- consistent with this project's established
practice of decoupling position sizing from compounding equity so selector
quality doesn't get confounded with path-dependent sizing (see
portfolio_sim.py's docstring).

Known caveat, stated plainly: the S&P 100 universe is TODAY's membership
projected back to 2018 -- survivorship bias (no delisted/removed names),
and cross-sectional correlation among large-caps means 5 "independent"
slots aren't fully independent bets. Not solved here; flagged so the
result isn't oversold.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_equity_real_data_walkforward import FEATURES, add_features, load_config, load_csv_dir  # noqa: E402
from download_sp100_universe import SP100_SYMBOLS, BENCHMARK  # noqa: E402

MAX_POSITIONS = 5
MAX_POSITION_FRACTION = 0.5
HOLD_BARS = 10  # matches the deployed strategy exactly -- not re-tuned
SLIPPAGE_BPS_EACH_SIDE = 2.0
INITIAL_CAPITAL = 2500.0
N_RANDOM_SEEDS = 50
MIN_ALLOC = 0.01  # skip an entry if less than 1% of capital would be allocated


def _mask(row: pd.Series) -> bool:
    return bool(
        row.relative_volume > 0.6 and row.atr_pct > 0
        and abs(row.ema_spread_atr) < 8 and row.return_20 > -0.25
    )


@dataclass
class Slot:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    notional_fraction: float
    notional_dollars: float
    exit_date: pd.Timestamp


def build_feature_frames(symbols: tuple[str, ...], data_dir: Path) -> dict[str, pd.DataFrame]:
    frames = load_csv_dir(symbols, data_dir)
    bench = frames[BENCHMARK]
    out = {}
    for symbol in symbols:
        if symbol == BENCHMARK:
            continue
        f = add_features(frames[symbol], bench)
        out[symbol] = f
    return out


def precompute(feature_frames: dict[str, pd.DataFrame], trading_dates: pd.DatetimeIndex) -> dict:
    """One pass over all symbols x dates -- shared by every simulation run
    (simple_trend + all 50 random-control seeds), since the eligible-
    candidate set and prices don't depend on ranking/seed, only the fill
    order does. Doing this once instead of per-run is what makes 51 runs
    over ~2000 days x 101 symbols tractable."""
    open_px: dict[str, dict[pd.Timestamp, float]] = {}
    close_px: dict[str, dict[pd.Timestamp, float]] = {}
    daily_candidates: dict[pd.Timestamp, list[tuple[str, float]]] = {d: [] for d in trading_dates}

    for symbol, f in feature_frames.items():
        open_px[symbol] = f["open"].to_dict()
        close_px[symbol] = f["close"].to_dict()
        sub = f[FEATURES + ["open", "close"]].dropna(subset=FEATURES)
        eligible = sub[sub.apply(_mask, axis=1) & (sub.get("market_above_ema50", 1) >= 1)]
        for date, rel_strength in eligible["relative_strength_20"].items():
            if date in daily_candidates:
                daily_candidates[date].append((symbol, float(rel_strength)))

    return {"open_px": open_px, "close_px": close_px, "daily_candidates": daily_candidates}


def simulate(pre: dict, trading_dates: pd.DatetimeIndex, *, rank_mode: str, seed: int | None = None) -> dict:
    """rank_mode: 'simple_trend' (rank by relative_strength_20 desc) or
    'random' (random order among eligible each day, given `seed`)."""
    rng = np.random.default_rng(seed) if rank_mode == "random" else None
    open_px, close_px, daily_candidates = pre["open_px"], pre["close_px"], pre["daily_candidates"]
    cash_dollars = INITIAL_CAPITAL
    committed_fraction = 0.0
    open_slots: list[Slot] = []
    completed = []
    equity_rows = []

    n = len(trading_dates)
    for i in range(60, n - 1):  # need i+1 for entry fill; warmup for feature stability
        today = trading_dates[i]

        # 1) process exits due today
        still_open = []
        for slot in open_slots:
            if slot.exit_date == today:
                px = close_px[slot.symbol].get(today)
                if px is None:
                    still_open.append(slot)  # data gap -- hold one more day rather than guess a price
                    continue
                exit_price = px * (1 - SLIPPAGE_BPS_EACH_SIDE / 10000)
                trade_return = exit_price / slot.entry_price - 1
                pnl = slot.notional_dollars * trade_return
                cash_dollars += slot.notional_dollars + pnl
                committed_fraction -= slot.notional_fraction
                completed.append({
                    "symbol": slot.symbol, "entry_date": slot.entry_date, "exit_date": today,
                    "entry_price": slot.entry_price, "exit_price": exit_price,
                    "notional_fraction": slot.notional_fraction, "notional_dollars": slot.notional_dollars,
                    "trade_return": trade_return, "pnl": pnl,
                })
            else:
                still_open.append(slot)
        open_slots = still_open
        committed_fraction = max(0.0, committed_fraction)

        # 2) fill available slots with new entries (signal today, fill at tomorrow's open)
        held_symbols = {s.symbol for s in open_slots}
        available_slots = MAX_POSITIONS - len(open_slots)
        if available_slots > 0:
            candidates = [c for c in daily_candidates.get(today, []) if c[0] not in held_symbols]
            if candidates:
                if rank_mode == "simple_trend":
                    candidates = sorted(candidates, key=lambda c: c[1], reverse=True)
                else:
                    candidates = list(candidates)
                    rng.shuffle(candidates)

                tomorrow = trading_dates[i + 1]
                for symbol, _ in candidates[:available_slots]:
                    available_cash_fraction = 1.0 - committed_fraction
                    fraction = min(MAX_POSITION_FRACTION, available_cash_fraction)
                    if fraction < MIN_ALLOC:
                        break
                    entry_px = open_px[symbol].get(tomorrow)
                    if entry_px is None:
                        continue
                    entry_price = entry_px * (1 + SLIPPAGE_BPS_EACH_SIDE / 10000)
                    notional_dollars = fraction * INITIAL_CAPITAL
                    exit_idx = min(i + 1 + HOLD_BARS - 1, n - 1)
                    open_slots.append(Slot(
                        symbol=symbol, entry_date=tomorrow, entry_price=entry_price,
                        notional_fraction=fraction, notional_dollars=notional_dollars,
                        exit_date=trading_dates[exit_idx],
                    ))
                    cash_dollars -= notional_dollars
                    committed_fraction += fraction

        # 3) mark-to-market today's portfolio value
        mtm = 0.0
        for slot in open_slots:
            px = close_px[slot.symbol].get(today)
            mtm += slot.notional_dollars * (px / slot.entry_price) if px is not None else slot.notional_dollars
        equity_rows.append({"date": today, "equity": cash_dollars + mtm, "n_open": len(open_slots)})

    equity_path = pd.DataFrame(equity_rows).set_index("date")
    trades = pd.DataFrame(completed)
    return summarize(equity_path, trades)


def summarize(equity_path: pd.DataFrame, trades: pd.DataFrame) -> dict:
    if trades.empty or equity_path.empty:
        return {"trades": 0}
    years = (equity_path.index[-1] - equity_path.index[0]).days / 365.25
    daily_returns = equity_path["equity"].pct_change().dropna()
    sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else None
    running_max = equity_path["equity"].cummax()
    drawdown = 1 - equity_path["equity"] / running_max
    return {
        "trades": int(len(trades)),
        "trades_per_year": float(len(trades) / years) if years > 0 else None,
        "win_rate": float((trades.trade_return > 0).mean()),
        "mean_trade_return_bps": float(trades.trade_return.mean() * 10000),
        "median_trade_return_bps": float(trades.trade_return.median() * 10000),
        "final_equity": float(equity_path["equity"].iloc[-1]),
        "net_return": float(equity_path["equity"].iloc[-1] / INITIAL_CAPITAL - 1),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.max()),
        "avg_positions_open": float(equity_path["n_open"].mean()),
        "years": years,
    }


def main() -> None:
    data_dir = ROOT / "data" / "real_sp100"
    universe = tuple(dict.fromkeys(SP100_SYMBOLS))
    print(f"Building features for {len(universe)} symbols...")
    feature_frames = build_feature_frames((*universe, BENCHMARK), data_dir)
    trading_dates = feature_frames[universe[0]].index
    # Use SPY-length calendar as master -- reload benchmark frame's own index for robustness
    bench_frame = load_csv_dir((BENCHMARK,), data_dir)[BENCHMARK]
    trading_dates = bench_frame.index

    print("Precomputing eligible-candidate set (shared across all runs)...")
    pre = precompute(feature_frames, trading_dates)

    print("Running simple_trend multi-position portfolio (S&P 100, max 5 positions, 50% cap)...")
    st_result = simulate(pre, trading_dates, rank_mode="simple_trend")
    print(json.dumps(st_result, indent=2, default=str))

    print(f"\nRunning {N_RANDOM_SEEDS}-seed random-selection control (same capital rules)...")
    random_results = [simulate(pre, trading_dates, rank_mode="random", seed=s) for s in range(N_RANDOM_SEEDS)]
    random_bps = [r["mean_trade_return_bps"] for r in random_results if r.get("trades", 0) > 0]
    pct = float((np.array(random_bps) < st_result["mean_trade_return_bps"]).mean() * 100) if random_bps else None

    print(f"\nsimple_trend mean trade return: {st_result['mean_trade_return_bps']:.1f} bps")
    print(f"random control: mean={np.mean(random_bps):.1f} bps, std={np.std(random_bps):.1f} bps")
    print(f"simple_trend percentile vs random: {pct}")

    out = ROOT / "outputs" / "equity_multi_position_sp100_study"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({
        "simple_trend": st_result,
        "random_mean_bps": float(np.mean(random_bps)) if random_bps else None,
        "random_std_bps": float(np.std(random_bps)) if random_bps else None,
        "simple_trend_percentile_vs_random": pct,
        "baseline_4stock_single_position": {
            "trades": 188, "trades_per_year": 188 / 8.5, "mean_trade_return_bps": 243.3, "max_drawdown": 0.221,
        },
    }, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
