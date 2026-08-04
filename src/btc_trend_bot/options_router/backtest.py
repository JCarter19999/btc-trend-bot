"""Per-day event loop: features -> regime -> persistence -> router ->
structures -> risk -> exit policy -> one decision record per day, matching
spec section 25's per-decision output schema.

Only trading days present in BOTH data/opra_spx/ (entry) and
data/opra_spx_exit/ (exit) are backtested -- both are real cached OPRA
snapshots, no new Databento spend for Phase 1.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from .exit_policy import evaluate_exit
from .features import FIRST_HOUR_BARS, build_first_hour_features, fetch_confirmation_data
from .regimes import classify_regime, regime_persistence
from .risk import CONTRACT_MULTIPLIER, build_risk_manager
from .router import route
from .structures import load_chain, payoff_at_close, price_exit_from_chain

REPO_ROOT = Path(__file__).resolve().parents[3]


def available_days(entry_dir: Path, exit_dir: Path) -> list[pd.Timestamp]:
    entry_days = {Path(f).stem for f in glob.glob(str(entry_dir / "*.parquet"))}
    exit_days = {Path(f).stem for f in glob.glob(str(exit_dir / "*.parquet"))}
    common = sorted(entry_days & exit_days)
    return [pd.to_datetime(d) for d in common]


def run_backtest(cfg: dict, root: Path = REPO_ROOT, start: str | None = None,
                  end: str | None = None, fill_mode: str = "natural") -> tuple[pd.DataFrame, pd.DataFrame]:
    entry_dir = root / cfg["data"]["entry_dir"]
    exit_dir = root / cfg["data"]["exit_dir"]
    underlying = cfg["data"]["underlying"]

    days = available_days(entry_dir, exit_dir)
    if start:
        days = [d for d in days if d >= pd.Timestamp(start)]
    if end:
        days = [d for d in days if d <= pd.Timestamp(end)]

    bars = fetch_confirmation_data(cfg["data"]["confirmation_symbols"], underlying=underlying)
    feature_df = build_first_hour_features(
        bars, underlying=underlying,
        trailing_window_days=cfg["data"].get("trailing_window_days", 5),
    ).set_index("date")
    spx_bars = bars[underlying]

    risk = build_risk_manager(cfg)

    decisions: list[dict] = []
    trades: list[dict] = []

    for d in days:
        risk.roll_day(d)
        if d not in feature_df.index:
            decisions.append({"date": d.strftime("%Y-%m-%d"), "trade": False,
                               "reason": "no first-hour feature row (insufficient free-data bars)"})
            continue
        row = feature_df.loc[d].to_dict()
        spot = row["first_hour_close"]

        label, confidence, diagnostics = classify_regime(row, cfg)
        day_bars = spx_bars[spx_bars["date"] == d]
        persistence = regime_persistence(day_bars, label, cfg)

        entry_chains = {
            "C": load_chain(entry_dir / f"{d:%Y-%m-%d}.parquet", d, "C"),
            "P": load_chain(entry_dir / f"{d:%Y-%m-%d}.parquet", d, "P"),
        }

        routing = route(label, confidence, entry_chains, spot, cfg, fill_mode)
        can_trade, risk_reason = risk.can_trade()

        record = {
            "date": d.strftime("%Y-%m-%d"),
            "underlying": underlying,
            "initial_regime": label,
            "regime_confidence": confidence,
            "persistence_60m_matches": persistence.get(60, {}).get("matches_base"),
            "persistence_90m_matches": persistence.get(90, {}).get("matches_base"),
            "selected_structure": routing.selected_kind,
            "trade": False,
            "reasoning": routing.reason,
            "data_quality_warnings": None,
        }

        if not routing.trade:
            decisions.append(record)
            continue
        if not can_trade:
            record["reasoning"] = f"router selected {routing.selected_kind} but risk gate blocked: {risk_reason}"
            decisions.append(record)
            continue

        structure = routing.selected_structure
        contracts = risk.contracts_for(structure.max_loss)
        if contracts <= 0:
            record["reasoning"] = f"{routing.selected_kind} max loss too large for risk budget (0 contracts sized)"
            decisions.append(record)
            continue

        exit_chains = {
            "C": load_chain(exit_dir / f"{d:%Y-%m-%d}.parquet", d, "C"),
            "P": load_chain(exit_dir / f"{d:%Y-%m-%d}.parquet", d, "P"),
        }
        noon_value = price_exit_from_chain(structure, exit_chains, fill_mode)
        close_payoff = payoff_at_close(structure, row["day_close"])
        exit_decision = evaluate_exit(structure, noon_value, close_payoff, cfg)

        pnl_dollars = exit_decision.pnl * contracts * CONTRACT_MULTIPLIER
        risk.open_position = True
        risk.record_trade(pnl_dollars)

        record.update({
            "trade": True,
            "selected_strikes": ",".join(f"{leg.side}:{leg.cp}{leg.strike:g}" for leg in structure.legs),
            "entry_debit_or_credit": structure.debit,
            "maximum_profit": structure.max_profit,
            "maximum_loss": structure.max_loss,
            "contracts": contracts,
            "hard_stop": cfg["exit_policy"]["hard_stop_pct"],
            "profit_activation_threshold": cfg["exit_policy"]["profit_target_pct"],
            "latest_exit_time": "12:00 ET" if exit_decision.exit_at == "noon" else "close (cash-settled)",
            "exit_reason": exit_decision.exit_reason,
            "exit_return": exit_decision.ret,
            "pnl_dollars": pnl_dollars,
            "data_quality_warnings": exit_decision.data_quality_warning,
        })
        decisions.append(record)
        trades.append({
            "date": d.strftime("%Y-%m-%d"),
            "regime": label,
            "structure": routing.selected_kind,
            "exit_reason": exit_decision.exit_reason,
            "return": exit_decision.ret,
            "pnl_dollars": pnl_dollars,
            "contracts": contracts,
        })

    return pd.DataFrame(decisions), pd.DataFrame(trades)
