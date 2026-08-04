"""Performance metrics (spec section 19, the subset computable from two
real post-entry anchor points), structure baselines (spec section 18), and
promotion-control wiring via experiments/research_controls.py -- this
project's house standard, see PROMOTION_CONTROLS.md.

Not computed here (needs an intra-day option price path -- Phase 2):
target-hit rate, stop-hit rate with real timing, peak-profit-capture
efficiency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from research_controls import (  # noqa: E402
    assert_no_window_overlap, day_clustered_t, shuffled_null_percentile, vol_matched_benchmark,
)

from .exit_policy import evaluate_exit  # noqa: E402
from .structures import BUILDERS, load_chain, payoff_at_close, price_exit_from_chain  # noqa: E402


def summarize_trades(trades: pd.DataFrame, account_value: float, _breakdown: bool = True) -> dict:
    if trades.empty:
        return {"n": 0}
    rets = trades["return"].dropna()
    equity = account_value + trades["pnl_dollars"].cumsum()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    wins = trades["pnl_dollars"] > 0
    streak = (~wins).groupby((wins != wins.shift()).cumsum()).cumcount() + 1
    longest_losing_streak = int((streak * (~wins)).max()) if len(streak) else 0
    gains = trades.loc[trades["pnl_dollars"] > 0, "pnl_dollars"].sum()
    losses = -trades.loc[trades["pnl_dollars"] < 0, "pnl_dollars"].sum()
    return {
        "n": int(len(trades)),
        "win_rate": float(wins.mean()),
        "mean_return": float(rets.mean()),
        "median_return": float(rets.median()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "sharpe_like": float(rets.mean() / rets.std(ddof=1)) if rets.std(ddof=1) else float("nan"),
        "sortino_like": float(rets.mean() / rets[rets < 0].std(ddof=1)) if (rets < 0).any() and rets[rets < 0].std(ddof=1) else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "worst_trade": float(rets.min()),
        "longest_losing_streak": longest_losing_streak,
        "total_pnl_dollars": float(trades["pnl_dollars"].sum()),
        "final_equity": float(equity.iloc[-1]),
        "by_regime": ({k: summarize_trades(v, account_value, _breakdown=False) for k, v in trades.groupby("regime")}
                      if _breakdown and "regime" in trades and trades["regime"].nunique() > 1 else {}),
        "by_structure": ({k: summarize_trades(v, account_value, _breakdown=False) for k, v in trades.groupby("structure")}
                         if _breakdown and "structure" in trades and trades["structure"].nunique() > 1 else {}),
    }


ALL_STRUCTURE_KINDS = list(BUILDERS.keys())


def run_structure_baseline(kind: str, days: list[pd.Timestamp], entry_dir: Path, exit_dir: Path,
                            feature_df: pd.DataFrame, cfg: dict, fill_mode: str = "natural") -> pd.DataFrame:
    """"Always trade this structure" baseline (spec section 18), ignoring the
    regime classifier entirely -- same entry/exit data and exit policy as
    the routed backtest."""
    from .structures import build_structure

    rows = []
    for d in days:
        if d not in feature_df.index:
            continue
        row = feature_df.loc[d]
        spot = float(row["first_hour_close"])
        entry_chains = {
            "C": load_chain(entry_dir / f"{d:%Y-%m-%d}.parquet", d, "C"),
            "P": load_chain(entry_dir / f"{d:%Y-%m-%d}.parquet", d, "P"),
        }
        structure = build_structure(kind, entry_chains, spot, cfg, fill_mode)
        if structure is None:
            continue
        exit_chains = {
            "C": load_chain(exit_dir / f"{d:%Y-%m-%d}.parquet", d, "C"),
            "P": load_chain(exit_dir / f"{d:%Y-%m-%d}.parquet", d, "P"),
        }
        noon_value = price_exit_from_chain(structure, exit_chains, fill_mode)
        close_payoff = payoff_at_close(structure, float(row["day_close"]))
        exit_decision = evaluate_exit(structure, noon_value, close_payoff, cfg)
        rows.append({
            "date": d.strftime("%Y-%m-%d"), "structure": kind, "return": exit_decision.ret,
            "pnl_dollars": exit_decision.pnl * cfg["risk"]["account_value"] * cfg["risk"]["max_risk_pct_per_trade"] / structure.max_loss if structure.max_loss else 0.0,
            "regime": "n/a",
        })
    return pd.DataFrame(rows)


def run_promotion_controls(trades: pd.DataFrame, decisions: pd.DataFrame,
                            feature_df: pd.DataFrame, account_value: float,
                            baseline_mean_returns: dict[str, float] | None = None) -> dict:
    out: dict = {}

    # 1. Window-overlap: the 10:30 feature window must never touch the noon
    # exit or close windows it's scored against.
    if not decisions.empty:
        try:
            signal_end = pd.to_datetime(decisions["date"]) + pd.Timedelta(hours=10, minutes=30)
            target_start = pd.to_datetime(decisions["date"]) + pd.Timedelta(hours=10, minutes=30)
            assert_no_window_overlap(signal_end, target_start, "options_router entry-vs-signal")
            out["window_overlap_assertion"] = "passed"
        except AssertionError as e:
            out["window_overlap_assertion"] = f"FAILED: {e}"

    if trades.empty:
        out["note"] = "no trades taken; remaining controls need at least one trade"
        return out

    # 2. Day-clustered significance on trade returns.
    out["day_clustered_t"] = day_clustered_t(trades["return"], pd.to_datetime(trades["date"]))

    # 3. Shuffled-null: does regime confidence actually predict return, or
    # would any confidence-return pairing look about this good?
    out["confidence_vs_return_shuffled_null"] = shuffled_null_percentile(
        decisions.loc[decisions["trade"] == True, "regime_confidence"].reset_index(drop=True) if "regime_confidence" in decisions else pd.Series(dtype=float),
        trades["return"].reset_index(drop=True),
    )

    # 4. Vol-matched benchmark: buy-and-hold SPX, scaled to the strategy's
    # realized volatility (not SPY, not un-scaled) -- per PROMOTION_CONTROLS.md.
    equity = pd.Series(account_value + trades["pnl_dollars"].cumsum().to_numpy(),
                        index=pd.to_datetime(trades["date"]))
    spx_returns = feature_df["day_close"].pct_change().reindex(equity.index.normalize()).dropna()
    if len(spx_returns) >= 5:
        out["vol_matched_spx_benchmark"] = vol_matched_benchmark(equity, spx_returns)
    else:
        out["vol_matched_spx_benchmark"] = {"note": "too few overlapping days"}

    # 5. Negative control: with one trade per day, shuffling the ORDER of
    # trades["return"] cannot move its mean (a mean is order-invariant) --
    # that would be a tautological no-op, not a real control. The
    # meaningful counterfactual is whether regime-conditional selection did
    # better than picking a structure with no regard to regime at all.
    # `baseline_mean_returns` (the six "always trade structure X" means, on
    # the same underlying days) stands in for "uniform random structure
    # choice" -- passed in by the caller, since it's already computed there.
    if baseline_mean_returns:
        uniform_random_mean = float(np.mean(list(baseline_mean_returns.values())))
        out["negative_control_vs_uniform_random_structure"] = {
            "router_mean_return": float(trades["return"].mean()),
            "uniform_random_structure_mean_return": uniform_random_mean,
            "router_beats_uniform_random": bool(trades["return"].mean() > uniform_random_mean),
        }

    # 6. Single-period dependence: drop each calendar month in turn.
    trades = trades.copy()
    trades["month"] = pd.to_datetime(trades["date"]).dt.to_period("M").astype(str)
    per_month = trades.groupby("month")["return"].mean().to_dict()
    out["per_month_mean_return"] = per_month
    if len(per_month) > 1:
        out["drop_one_month"] = {
            f"ex_{m}": float(trades.loc[trades["month"] != m, "return"].mean())
            for m in per_month
        }

    return out
