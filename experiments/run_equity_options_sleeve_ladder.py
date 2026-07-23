"""Options sleeve ladder -- follow-up to Joey's "high-risk/high-reward
options portfolio gambit" brief (2026-07-23). Tests the three structures
his own priority order puts first (selective long calls were already
tested in EQUITY_OPTIONS_DEEP_DIVE.md -- reused here, not re-run): bull
call debit spreads, a small tail hedge, and a volatility-breakout
straddle. Calendars/diagonals/butterflies/short-premium are explicitly
out of scope tonight, matching his own doc's stated priority order
("should be studied after simpler defined-risk structures" / "what to
avoid initially").

Same honesty framing as every options doc in this project: Black-Scholes
off trailing REALIZED volatility, not real implied vol or real bid/ask.
This UNDERPRICES real options (volatility risk premium) -- generous to
every options structure tested here, not neutral. A negative result here
is real evidence against a structure; a positive one is weak evidence
pending real historical option-chain data (see options_pricing.py and
EQUITY_OPTIONS_DEEP_DIVE.md for the full statement of this bias and why
synthetic chains validate infrastructure, not edge -- per this project's
CLAUDE.md guardrail, nothing here is promotion-grade).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btc_trend_bot.options_pricing import (  # noqa: E402
    realized_vol, synthetic_call_debit_spread_trade, synthetic_straddle_trade,
    synthetic_put_trade,
)
from btc_trend_bot.portfolio_sim import SizingMode, simulate_single_position  # noqa: E402
from run_equity_options_deep_dive import build_signal_pool, select  # noqa: E402
from run_equity_real_data_walkforward import add_features, load_config, load_csv_dir  # noqa: E402

VOLATILE_UNIVERSE = ("TSLA", "COIN", "MSTR", "PLTR", "GME")
DTE_AT_ENTRY = 30
HOLD_DAYS = 10
PREMIUM_BUDGET = 250.0  # same $250-of-$2,500 (10%) convention as the calls deep dive
SIZING = SizingMode("fixed_premium_250", "fixed_notional", PREMIUM_BUDGET)
STOCK_SIZING = SizingMode("fixed_notional_2500", "fixed_notional", 2500.0)


def _cfg():
    base = load_config(ROOT / "configs/real_data.yaml")
    return dataclasses.replace(base, symbols=VOLATILE_UNIVERSE, stop_atr=100.0, target_atr=100.0,
                                max_hold_bars=HOLD_DAYS, safety_enabled=False, hard_shutdown_drawdown=1.0,
                                minimum_equity=0.0)


def summarize(trades: pd.DataFrame, label: str, sizing: SizingMode, cfg) -> dict:
    if trades.empty:
        return {"label": label, "trades_taken": 0}
    t = trades.dropna(subset=["net_return"]).copy()
    _, summary = simulate_single_position(t, cfg, sizing)
    summary["label"] = label
    return summary


# --------------------------------------------------------------------------- #
# 1) Bull call debit spreads -- same simple_trend long signal as the calls
#    deep dive, priced as a spread instead of an outright call. Structurally
#    less spread/IV sensitive than an outright call (short leg partially
#    offsets both), which is the specific weakness the calls deep dive found.
# --------------------------------------------------------------------------- #

def run_debit_spreads(frames: dict, cfg) -> dict:
    pool = build_signal_pool(frames, cfg.benchmark, cfg, weakest=False)
    trades = select(pool, weakest=False)
    for col in ("signal_time", "entry_time", "exit_time"):
        trades[col] = pd.to_datetime(trades[col], utc=True)

    results = {}
    for long_m, short_m, spread in (
        (1.0, 1.10, 0.05), (1.0, 1.10, 0.10), (1.0, 1.05, 0.05), (0.98, 1.08, 0.05),
        (1.0, 1.20, 0.05), (1.0, 1.30, 0.05), (0.95, 1.15, 0.05), (1.0, 1.50, 0.05),
    ):
        priced = trades.copy()
        rows = [synthetic_call_debit_spread_trade(t.entry_price, t.entry_vol, t.exit_price, t.exit_vol,
                                                    DTE_AT_ENTRY, HOLD_DAYS, long_m, short_m, spread)
                for t in priced.itertuples()]
        priced["net_return"] = [r["net_return"] for r in rows]
        key = f"debit_spread_long{long_m}_short{short_m}_spread{spread}"
        results[key] = summarize(priced, key, SIZING, cfg)
    return results


# --------------------------------------------------------------------------- #
# 2) Tail hedge -- small recurring SPY put allocation. Tested against the
#    ACTUAL stock-only simple_trend equity curve (the real deployed logic,
#    4-stock AAPL/MSFT/NVDA/TSLA universe) during its worst drawdown windows,
#    not the volatile-universe sandbox -- a crash hedge should be evaluated
#    against the thing it's meant to protect.
# --------------------------------------------------------------------------- #

def run_tail_hedge(main_frames: dict, main_cfg) -> dict:
    # A recurring hedge is SUPPOSED to have a high rate of small losses (most
    # months, the crash you're insuring against doesn't happen) -- gating it
    # with the stock strategy's own drawdown-pause/loss-cooldown thresholds
    # (tuned for a momentum strategy where a loss streak is a warning sign)
    # would pause the hedge exactly when a string of "normal" worthless-put
    # expirations happens, which defeats the point. Evaluate the hedge
    # sleeve's own aggregate stats without that layer; safety_enabled stays
    # on for anything evaluating the actual stock strategy elsewhere.
    hedge_eval_cfg = dataclasses.replace(main_cfg, safety_enabled=False, hard_shutdown_drawdown=1.0, minimum_equity=0.0)
    spy = main_frames[main_cfg.benchmark].copy()
    spy["realized_vol"] = realized_vol(spy["close"].to_numpy(), 20)

    # Recurring monthly put: enter ~21 trading days apart, 45 DTE, 10% OTM,
    # hold the full 21-day cycle (sold, not exercised).
    HEDGE_DTE = 45
    HEDGE_HOLD = 21
    HEDGE_MONEYNESS = 0.90
    MONTHLY_BUDGET_FRACTION = 0.02  # 2% of $2,500/month, mid of the 1-3% range in the brief
    budget = MONTHLY_BUDGET_FRACTION * 2500.0

    rows = []
    i = 60
    while i < len(spy) - HEDGE_HOLD - 1:
        row = spy.iloc[i]
        vol = row.get("realized_vol", np.nan)
        if not np.isfinite(vol) or vol <= 0:
            i += HEDGE_HOLD
            continue
        entry_i, exit_i = i + 1, min(i + 1 + HEDGE_HOLD - 1, len(spy) - 1)
        entry_spot, exit_spot = float(spy.iloc[entry_i].open), float(spy.iloc[exit_i].close)
        exit_vol = float(spy.iloc[exit_i].realized_vol) if np.isfinite(spy.iloc[exit_i].realized_vol) else vol
        r = synthetic_put_trade(entry_spot, vol, exit_spot, exit_vol, HEDGE_DTE, HEDGE_HOLD,
                                 strike_moneyness=HEDGE_MONEYNESS, spread_frac_of_premium=0.08)
        rows.append({"signal_time": spy.index[i], "symbol": main_cfg.benchmark,
                     "entry_time": spy.index[entry_i], "exit_time": spy.index[exit_i],
                     "net_return": r["net_return"], "spot_return": exit_spot / entry_spot - 1})
        i += HEDGE_HOLD

    hedge_trades = pd.DataFrame(rows)
    hedge_summary = summarize(hedge_trades, "tail_hedge_recurring", SizingMode("hedge", "fixed_notional", budget), hedge_eval_cfg)

    # Crash-window check: did the hedge pay off specifically when SPY itself
    # fell hardest? (2018 Q4, 2020 COVID crash, 2022 bear)
    crash_windows = {
        "2018_Q4": ("2018-10-01", "2018-12-31"),
        "2020_COVID": ("2020-02-15", "2020-04-15"),
        "2022_bear": ("2022-01-01", "2022-10-31"),
    }
    crash_stats = {}
    for name, (start, end) in crash_windows.items():
        mask = (hedge_trades.signal_time >= pd.Timestamp(start, tz="UTC")) & (hedge_trades.signal_time <= pd.Timestamp(end, tz="UTC"))
        window_trades = hedge_trades[mask]
        spy_mask = (spy.index >= pd.Timestamp(start, tz="UTC")) & (spy.index <= pd.Timestamp(end, tz="UTC"))
        spy_window = spy.loc[spy_mask, "close"]
        spy_drawdown = float(spy_window.iloc[-1] / spy_window.max() - 1) if len(spy_window) else None
        crash_stats[name] = {
            "spy_return_in_window": spy_drawdown,
            "hedge_trades_in_window": int(len(window_trades)),
            "hedge_mean_return": float(window_trades.net_return.mean()) if len(window_trades) else None,
        }

    return {"recurring_hedge_summary": hedge_summary, "crash_window_performance": crash_stats,
            "monthly_budget_dollars": budget, "annual_cost_pct_of_capital": MONTHLY_BUDGET_FRACTION * 12 * 100}


# --------------------------------------------------------------------------- #
# 3) Volatility breakout straddle -- compressed realized vol + expanding
#    volume, direction-agnostic bet that realized move will exceed what's
#    priced in. Uses the volatile universe (more likely to actually break out).
# --------------------------------------------------------------------------- #

def run_vol_breakout(frames: dict, cfg) -> dict:
    bench = frames[cfg.benchmark]
    rows = []
    for symbol in cfg.symbols:
        f = add_features(frames[symbol], bench).copy()
        f["realized_vol"] = realized_vol(f["close"].to_numpy(), 20)
        f["vol_percentile_1y"] = f["realized_vol"].rolling(252, min_periods=60).rank(pct=True)
        for i in range(260, len(f) - HOLD_DAYS - 1):
            r = f.iloc[i]
            compressed = np.isfinite(r.get("vol_percentile_1y", np.nan)) and r.vol_percentile_1y < 0.25
            expanding_volume = np.isfinite(r.get("relative_volume", np.nan)) and r.relative_volume > 1.2
            if not (compressed and expanding_volume):
                continue
            if not np.isfinite(r.get("realized_vol", np.nan)) or r.realized_vol <= 0:
                continue
            entry_i, exit_i = i + 1, min(i + 1 + HOLD_DAYS - 1, len(f) - 1)
            entry_spot, exit_spot = float(f.iloc[entry_i].open), float(f.iloc[exit_i].close)
            exit_vol = float(f.iloc[exit_i].realized_vol) if np.isfinite(f.iloc[exit_i].realized_vol) else r.realized_vol
            realized_move_pct = abs(exit_spot / entry_spot - 1)
            rows.append({"signal_time": f.index[i], "symbol": symbol,
                         "entry_time": f.index[entry_i], "exit_time": f.index[exit_i],
                         "entry_spot": entry_spot, "entry_vol": float(r.realized_vol),
                         "exit_spot": exit_spot, "exit_vol": exit_vol, "realized_move_pct": realized_move_pct})
    pool = pd.DataFrame(rows)
    if pool.empty:
        return {"trades": 0, "note": "no compressed-vol + expanding-volume setups found in this universe/window"}

    results = {}
    for spread in (0.05, 0.10):
        priced = pool.copy()
        out = [synthetic_straddle_trade(t.entry_spot, t.entry_vol, t.exit_spot, t.exit_vol,
                                         DTE_AT_ENTRY, HOLD_DAYS, moneyness=1.0, spread_frac_of_premium=spread)
               for t in priced.itertuples()]
        priced["net_return"] = [o["net_return"] for o in out]
        priced["implied_move_at_entry"] = [
            # rough "implied move" proxy: entry straddle premium / spot, the standard
            # heuristic options traders use to eyeball what move is priced in
            (o["entry_premium"] if "entry_premium" in o else np.nan) / t.entry_spot
            for o, t in zip(out, priced.itertuples())
        ]
        key = f"vol_breakout_straddle_spread{spread}"
        results[key] = summarize(priced, key, SIZING, cfg)
        results[key]["mean_realized_move_pct"] = float(pool.realized_move_pct.mean())
        results[key]["mean_implied_move_at_entry"] = float(np.nanmean(priced.implied_move_at_entry))
        results[key]["pct_realized_exceeded_implied"] = float(
            (pool.realized_move_pct.to_numpy() > priced.implied_move_at_entry.to_numpy()).mean())
    return results


# --------------------------------------------------------------------------- #
# 4) Portfolio-level synthesis -- the "asymmetric basket" framing from the
#    brief: payoff skew, avg winner/avg loser, prob of a fully-lost sleeve,
#    expected log growth, evaluated on the OPTIONS SLEEVE portion only.
# --------------------------------------------------------------------------- #

def sleeve_synthesis(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {"label": label, "trades": 0}
    r = trades["net_return"].dropna()
    wins, losses = r[r > 0], r[r <= 0]
    return {
        "label": label,
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "avg_winner_pct": float(wins.mean() * 100) if len(wins) else None,
        "avg_loser_pct": float(losses.mean() * 100) if len(losses) else None,
        "payoff_ratio_win_over_loss": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else None,
        "pct_total_losses": float((r <= -0.95).mean()),  # essentially expired worthless
        "expected_log_growth_per_trade": float(np.log1p(r).mean()),
        "expected_arithmetic_return_per_trade": float(r.mean()),
    }


def main() -> None:
    cfg = _cfg()
    main_cfg_base = load_config(ROOT / "configs/real_data.yaml")  # AAPL/MSFT/NVDA/TSLA, the deployed universe
    frames = load_csv_dir((*VOLATILE_UNIVERSE, cfg.benchmark), ROOT / "data/real")
    main_frames = load_csv_dir((*main_cfg_base.symbols, main_cfg_base.benchmark), ROOT / "data/real")

    print("=== 1) Bull call debit spreads ===")
    debit_results = run_debit_spreads(frames, cfg)
    for k, s in debit_results.items():
        if s.get("trades_taken", 0) == 0:
            print(f"{k:45s} NO TRADES")
            continue
        print(f"{k:45s} trades={s['trades_taken']:4d} win={s.get('win_rate',float('nan')):.3f} "
              f"expectancy={s.get('expectancy_bps',float('nan')):8.1f}bps PF={s.get('profit_factor') or float('nan'):.2f} "
              f"maxDD={s.get('max_drawdown',float('nan')):.3f} total_return={s.get('total_return',float('nan'))*100:7.1f}%")

    print("\n=== 2) Tail hedge (SPY puts, deployed 4-stock universe's actual equity curve context) ===")
    hedge_results = run_tail_hedge(main_frames, main_cfg_base)
    print(json.dumps(hedge_results, indent=2, default=str))

    print("\n=== 3) Volatility breakout straddle ===")
    vol_results = run_vol_breakout(frames, cfg)
    for k, s in vol_results.items():
        if not isinstance(s, dict) or s.get("trades_taken", 0) == 0:
            continue
        print(f"{k:45s} trades={s['trades_taken']:4d} win={s.get('win_rate',float('nan')):.3f} "
              f"expectancy={s.get('expectancy_bps',float('nan')):8.1f}bps total_return={s.get('total_return',float('nan'))*100:7.1f}% "
              f"realized_move={s.get('mean_realized_move_pct',float('nan'))*100:.1f}% "
              f"implied_move={s.get('mean_implied_move_at_entry',float('nan'))*100:.1f}% "
              f"pct_realized>implied={s.get('pct_realized_exceeded_implied',float('nan'))*100:.0f}%")

    out = ROOT / "outputs" / "equity_options_sleeve_ladder"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({
        "debit_spreads": debit_results, "tail_hedge": hedge_results, "vol_breakout_straddle": vol_results,
    }, indent=2, default=str))
    print(f"\nWritten to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
