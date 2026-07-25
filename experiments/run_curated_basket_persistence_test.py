"""Curated 8-symbol basket, 5 concurrent $500-notional slots, persistence-
gated swaps (X trading days a challenger must sustain top-5 rank before
displacing a held name). Tests whether a hand-picked basket + a dwell-time
filter recovers what the full-S&P-100 5-way study lost to signal dilution,
and what the continuous single-position switching test lost to whipsaw.

No fixed 10-day hold here -- positions ride until displaced by the
persistence rule (or never, if never displaced). This is a genuinely
different exit mechanic from simple_trend's live time_exit, by design:
the whole point is testing whether persistence-gated rank-driven exits do
better than a blind timer.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from run_equity_real_data_walkforward import add_features, load_csv_dir

SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "AVGO"]
BENCHMARK = "SPY"
SLOT_NOTIONAL = 500.0
N_SLOTS = 5
SLIPPAGE_BPS = {"baseline": 2.0, "stress": 5.0}


def load_symbol(sym: str) -> pd.DataFrame:
    for d in ("data/real", "data/real_sp100"):
        p = ROOT / d / f"{sym}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
            return df
    raise FileNotFoundError(sym)


def build_ranks() -> pd.DataFrame:
    bench = load_symbol(BENCHMARK)
    feats = {}
    for sym in SYMBOLS:
        f = add_features(load_symbol(sym), bench)
        feats[sym] = f[["relative_strength_20", "market_above_ema50", "close", "open"]]
    idx = sorted(set.intersection(*(set(f.dropna().index) for f in feats.values())))
    idx = pd.DatetimeIndex(idx)
    rs = pd.DataFrame({s: feats[s]["relative_strength_20"].reindex(idx) for s in SYMBOLS})
    above = pd.DataFrame({s: feats[s]["market_above_ema50"].reindex(idx) for s in SYMBOLS})
    close = pd.DataFrame({s: feats[s]["close"].reindex(idx) for s in SYMBOLS})
    openp = pd.DataFrame({s: feats[s]["open"].reindex(idx) for s in SYMBOLS})

    ranks = pd.DataFrame(index=idx, columns=SYMBOLS, dtype=float)
    for d in idx:
        pool = [s for s in SYMBOLS if above.loc[d, s] >= 1]
        if not pool:
            pool = SYMBOLS
        ordered = rs.loc[d, pool].sort_values(ascending=False)
        for rank, s in enumerate(ordered.index, start=1):
            ranks.loc[d, s] = rank
        for s in SYMBOLS:
            if s not in pool:
                ranks.loc[d, s] = len(SYMBOLS) + 1  # worse than any real rank
    return ranks, close, openp, idx


def simulate(ranks: pd.DataFrame, close: pd.DataFrame, openp: pd.DataFrame, idx: pd.DatetimeIndex,
             persistence_days: int, slippage_bps: float, seed: int | None = None,
             random_selection: bool = False) -> dict:
    rng = np.random.default_rng(seed if seed is not None else 0)
    held: dict[str, dict] = {}  # symbol -> {entry_price, entry_date}
    dwell = {s: 0 for s in SYMBOLS}
    trades = []
    equity_curve = []
    realized_pnl = 0.0

    for i, d in enumerate(idx):
        today_ranks = ranks.loc[d]
        if random_selection:
            top5 = list(rng.choice(SYMBOLS, size=N_SLOTS, replace=False))
        else:
            top5 = list(today_ranks.sort_values().index[:N_SLOTS])

        for s in SYMBOLS:
            if s not in held and s in top5:
                dwell[s] += 1
            elif s not in held:
                dwell[s] = 0

        confirmed = [s for s in SYMBOLS if s not in held and s in top5 and dwell[s] > persistence_days]
        confirmed.sort(key=lambda s: today_ranks[s])  # strongest first
        removal_candidates = [s for s in held if s not in top5]
        removal_candidates.sort(key=lambda s: -today_ranks[s])  # weakest (highest rank number) first

        if i == 0:
            # Initial fill bypasses the persistence requirement -- no dwell history yet.
            for s in top5[:N_SLOTS]:
                if i + 1 < len(idx):
                    entry_px = float(openp.iloc[i + 1][s]) * (1 + slippage_bps / 10000)
                    held[s] = {"entry_price": entry_px, "entry_date": idx[i + 1]}
        else:
            n_empty = N_SLOTS - len(held)
            for s in confirmed:
                if s in held:
                    continue
                if n_empty > 0:
                    if i + 1 < len(idx):
                        entry_px = float(openp.iloc[i + 1][s]) * (1 + slippage_bps / 10000)
                        held[s] = {"entry_price": entry_px, "entry_date": idx[i + 1]}
                        n_empty -= 1
                elif removal_candidates:
                    victim = removal_candidates.pop(0)
                    if i + 1 < len(idx):
                        exit_px = float(openp.iloc[i + 1][victim]) * (1 - slippage_bps / 10000)
                        ret = exit_px / held[victim]["entry_price"] - 1
                        pnl = SLOT_NOTIONAL * ret
                        realized_pnl += pnl
                        trades.append({"symbol": victim, "entry_date": held[victim]["entry_date"],
                                       "exit_date": idx[i + 1], "entry_price": held[victim]["entry_price"],
                                       "exit_price": exit_px, "net_return": ret, "pnl": pnl})
                        del held[victim]
                        entry_px = float(openp.iloc[i + 1][s]) * (1 + slippage_bps / 10000)
                        held[s] = {"entry_price": entry_px, "entry_date": idx[i + 1]}

        mtm = realized_pnl
        for s, pos in held.items():
            cur_px = float(close.loc[d, s])
            mtm += SLOT_NOTIONAL * (cur_px / pos["entry_price"] - 1)
        equity_curve.append({"date": d, "equity": 2500.0 + mtm, "n_held": len(held)})

    # close any remaining open positions at the final bar
    final_date = idx[-1]
    for s, pos in list(held.items()):
        exit_px = float(close.loc[final_date, s]) * (1 - slippage_bps / 10000)
        ret = exit_px / pos["entry_price"] - 1
        pnl = SLOT_NOTIONAL * ret
        trades.append({"symbol": s, "entry_date": pos["entry_date"], "exit_date": final_date,
                       "entry_price": pos["entry_price"], "exit_price": exit_px, "net_return": ret, "pnl": pnl})

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(equity_curve)
    ending_equity = 2500.0 + trades_df["pnl"].sum() if len(trades_df) else 2500.0
    max_dd = float((1 - curve_df["equity"] / curve_df["equity"].cummax()).max()) if len(curve_df) else 0.0
    return {
        "n_trades": len(trades_df),
        "win_rate": float((trades_df["net_return"] > 0).mean()) if len(trades_df) else float("nan"),
        "mean_return": float(trades_df["net_return"].mean()) if len(trades_df) else float("nan"),
        "total_return": ending_equity / 2500.0 - 1,
        "ending_equity": ending_equity,
        "max_drawdown": max_dd,
        "trades_df": trades_df,
        "curve_df": curve_df,
    }


def main():
    print("Building ranks/features for 8-symbol curated basket...")
    ranks, close, openp, idx = build_ranks()
    print(f"Date range: {idx[0].date()} to {idx[-1].date()}, {len(idx)} trading days")

    out_dir = ROOT / "outputs" / "curated_basket_persistence_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, X in [("X0_no_persistence", 0), ("X3", 3), ("X5", 5), ("X10", 10)]:
        r = simulate(ranks, close, openp, idx, persistence_days=X, slippage_bps=SLIPPAGE_BPS["baseline"])
        results[label] = r
        print(f"{label}: trades={r['n_trades']} win_rate={r['win_rate']*100:.1f}% "
              f"mean_ret={r['mean_return']*100:.3f}% total_return={r['total_return']*100:+.1f}% "
              f"max_dd={r['max_drawdown']*100:.1f}%")
        r["trades_df"].to_csv(out_dir / f"trades_{label}.csv", index=False)

    # Cost stress on the best-looking variant found (fill in after seeing baseline results)
    print("\n=== Cost stress (5bps) ===")
    stress_results = {}
    for label, X in [("X0_no_persistence", 0), ("X3", 3), ("X5", 5), ("X10", 10)]:
        r = simulate(ranks, close, openp, idx, persistence_days=X, slippage_bps=SLIPPAGE_BPS["stress"])
        stress_results[label] = r
        print(f"{label} (5bps): total_return={r['total_return']*100:+.1f}% max_dd={r['max_drawdown']*100:.1f}%")

    # Random-selection control (5 random symbols each day, no persistence) -- 20 seeds
    print("\n=== Random-selection control (20 seeds, no persistence) ===")
    random_totals = []
    for seed in range(20):
        r = simulate(ranks, close, openp, idx, persistence_days=0, slippage_bps=SLIPPAGE_BPS["baseline"],
                     seed=seed, random_selection=True)
        random_totals.append(r["total_return"])
    random_totals = np.array(random_totals)
    print(f"Random control: mean={random_totals.mean()*100:.1f}% std={random_totals.std()*100:.1f}%")

    # Out-of-sample split on the best rank-driven variant (use X5 as a reasonable representative;
    # will redo on whichever variant looks best once baseline numbers are in)
    mid = len(idx) // 2
    print("\n=== OOS split (first half / second half), all variants ===")
    oos_results = {}
    for label, X in [("X0_no_persistence", 0), ("X3", 3), ("X5", 5), ("X10", 10)]:
        first_idx = idx[:mid]
        second_idx = idx[mid:]
        r1 = simulate(ranks.loc[first_idx], close, openp, idx=first_idx, persistence_days=X,
                     slippage_bps=SLIPPAGE_BPS["baseline"])
        r2 = simulate(ranks.loc[second_idx], close, openp, idx=second_idx, persistence_days=X,
                     slippage_bps=SLIPPAGE_BPS["baseline"])
        oos_results[label] = (r1["total_return"], r2["total_return"])
        print(f"{label}: first_half={r1['total_return']*100:+.1f}% second_half={r2['total_return']*100:+.1f}%")

    summary = {
        "date_range": [str(idx[0].date()), str(idx[-1].date())],
        "n_days": len(idx),
        "baseline_2bps": {k: {kk: vv for kk, vv in v.items() if kk not in ("trades_df", "curve_df")}
                          for k, v in results.items()},
        "stress_5bps": {k: {kk: vv for kk, vv in v.items() if kk not in ("trades_df", "curve_df")}
                        for k, v in stress_results.items()},
        "random_control": {"mean_total_return": float(random_totals.mean()), "std": float(random_totals.std()),
                           "seeds": random_totals.tolist()},
        "oos_split": {k: {"first_half": v[0], "second_half": v[1]} for k, v in oos_results.items()},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
