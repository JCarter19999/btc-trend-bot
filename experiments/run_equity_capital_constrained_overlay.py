from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SCHWAB_STOCK_COMMISSION = 0.0
SCHWAB_OPTION_CONTRACT_FEE = 0.65
SCHWAB_FRACTIONAL_MINIMUM = 1.0
OPTION_MULTIPLIER = 100


@dataclass(frozen=True)
class CapitalConfig:
    initial_capital: float
    monthly_deposit: float
    option_budget_fraction: float = 0.30
    mode: str = "cash_safe"  # cash_safe or additive_margin


def option_round_trip_pnl(entry_fill: float, realized_return: float, contracts: int) -> float:
    """Dollar P&L after Schwab's per-contract opening and closing fees.

    The synthetic realized return is measured on premium before broker fees.
    """
    if contracts <= 0:
        return 0.0
    gross_premium = float(entry_fill) * OPTION_MULTIPLIER * contracts
    gross_pnl = gross_premium * float(realized_return)
    fees = 2.0 * SCHWAB_OPTION_CONTRACT_FEE * contracts
    return gross_pnl - fees


def affordable_contracts(equity: float, entry_fill: float, budget_fraction: float) -> int:
    if not np.isfinite(entry_fill) or entry_fill <= 0 or equity <= 0:
        return 0
    budget = equity * budget_fraction
    all_in_open_cost = entry_fill * OPTION_MULTIPLIER + SCHWAB_OPTION_CONTRACT_FEE
    return max(0, int(np.floor(budget / all_in_open_cost)))


def _monthly_deposits_between(previous: pd.Timestamp | None, current: pd.Timestamp, amount: float) -> int:
    if previous is None or amount <= 0:
        return 0
    prev_month = previous.tz_convert(None).to_period("M")
    curr_month = current.tz_convert(None).to_period("M")
    return max(0, int(curr_month.ordinal - prev_month.ordinal))


def simulate_capital_path(trades: pd.DataFrame, config: CapitalConfig) -> tuple[pd.DataFrame, dict]:
    if config.mode not in {"cash_safe", "additive_margin"}:
        raise ValueError(f"Unsupported mode: {config.mode}")

    ordered = trades.sort_values("signal_time").copy()
    ordered["signal_time"] = pd.to_datetime(ordered["signal_time"], utc=True)

    equity = float(config.initial_capital)
    total_deposits = float(config.initial_capital)
    previous_time: pd.Timestamp | None = None
    rows: list[dict] = []
    option_qualified = 0
    option_affordable = 0
    total_contracts = 0
    first_option_time = None
    peak = equity
    max_drawdown = 0.0

    for _, trade in ordered.iterrows():
        signal_time = pd.Timestamp(trade.signal_time)
        months = _monthly_deposits_between(previous_time, signal_time, config.monthly_deposit)
        deposit = months * config.monthly_deposit
        equity += deposit
        total_deposits += deposit
        previous_time = signal_time

        starting_equity = equity
        qualified = bool(trade.get("option_overlay_eligible", False))
        if qualified:
            option_qualified += 1

        contracts = affordable_contracts(
            starting_equity,
            float(trade.get("entry_fill", np.nan)),
            config.option_budget_fraction,
        ) if qualified else 0
        if contracts > 0:
            option_affordable += 1
            total_contracts += contracts
            if first_option_time is None:
                first_option_time = signal_time

        option_open_cost = 0.0
        if contracts:
            option_open_cost = contracts * (
                float(trade.entry_fill) * OPTION_MULTIPLIER + SCHWAB_OPTION_CONTRACT_FEE
            )

        if config.mode == "cash_safe":
            # Preserve both legs without borrowing. The exact whole-contract premium is
            # reserved first; all remaining cash becomes fractional stock exposure.
            stock_notional = max(0.0, starting_equity - option_open_cost)
        else:
            # Preserve 100% stock notional and fund the option premium as additional
            # gross exposure. This requires margin or equivalent external buying power.
            stock_notional = starting_equity

        if 0 < stock_notional < SCHWAB_FRACTIONAL_MINIMUM:
            stock_notional = 0.0

        stock_pnl = stock_notional * float(trade.net_return)
        option_pnl = option_round_trip_pnl(
            float(trade.get("entry_fill", np.nan)),
            float(trade.get("realized_option_return", 0.0)),
            contracts,
        ) if contracts else 0.0

        equity = max(0.0, starting_equity + stock_pnl + option_pnl)
        peak = max(peak, equity)
        drawdown = 1.0 - equity / peak if peak > 0 else 1.0
        max_drawdown = max(max_drawdown, drawdown)

        rows.append({
            "signal_time": signal_time,
            "symbol": trade.symbol,
            "deposit": deposit,
            "starting_equity": starting_equity,
            "stock_notional": stock_notional,
            "stock_return": float(trade.net_return),
            "stock_pnl": stock_pnl,
            "option_qualified": qualified,
            "option_affordable": contracts > 0,
            "option_contracts": contracts,
            "option_entry_fill": float(trade.get("entry_fill", np.nan)),
            "option_open_cost": option_open_cost,
            "option_return": float(trade.get("realized_option_return", np.nan)),
            "option_pnl_after_fees": option_pnl,
            "ending_equity": equity,
            "drawdown": drawdown,
        })

        if equity <= 0:
            break

    path = pd.DataFrame(rows)
    net_profit = equity - total_deposits
    summary = {
        **asdict(config),
        "ending_equity": equity,
        "total_deposits": total_deposits,
        "net_profit_after_deposits": net_profit,
        "return_on_contributed_capital": net_profit / total_deposits if total_deposits else np.nan,
        "max_drawdown": max_drawdown,
        "stock_trades": len(path),
        "option_qualified_signals": option_qualified,
        "option_affordable_signals": option_affordable,
        "option_affordability_rate": option_affordable / option_qualified if option_qualified else 0.0,
        "total_option_contracts": total_contracts,
        "first_option_trade": first_option_time.isoformat() if first_option_time is not None else None,
        "ending_below_contributions": equity < total_deposits,
        "ruined": equity <= 0,
    }
    return path, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", required=True)
    parser.add_argument("--output", default="outputs/equity_capital_constrained_overlay")
    parser.add_argument("--monthly-deposit", type=float, default=50.0)
    parser.add_argument("--option-budget-fraction", type=float, default=0.30)
    parser.add_argument(
        "--initial-capitals",
        default="50,100,250,500,1000,2500,5000,10000",
    )
    args = parser.parse_args()

    trades = pd.read_csv(args.trades)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    capitals = [float(value.strip()) for value in args.initial_capitals.split(",") if value.strip()]
    summaries = []
    for mode in ("cash_safe", "additive_margin"):
        for initial in capitals:
            config = CapitalConfig(
                initial_capital=initial,
                monthly_deposit=args.monthly_deposit,
                option_budget_fraction=args.option_budget_fraction,
                mode=mode,
            )
            path, summary = simulate_capital_path(trades, config)
            summaries.append(summary)
            path.to_csv(output / f"path_{mode}_{int(initial)}.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output / "capital_summary.csv", index=False)
    metadata = {
        "schwab_fee_assumptions": {
            "online_listed_stock_commission": SCHWAB_STOCK_COMMISSION,
            "option_fee_per_contract_per_transaction": SCHWAB_OPTION_CONTRACT_FEE,
            "fractional_share_minimum": SCHWAB_FRACTIONAL_MINIMUM,
            "option_contract_multiplier": OPTION_MULTIPLIER,
            "regulatory_exchange_fees": "not separately modeled; variable and generally small",
        },
        "configuration": vars(args),
        "results": summaries,
    }
    (output / "results.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
