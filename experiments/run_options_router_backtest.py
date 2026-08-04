"""CPU-only options strategy router backtest (Phase 1).

Runs the regime classifier + router + structure payoff engine over the
trading days already cached in data/opra_spx/ (10:30 entry) and
data/opra_spx_exit/ (noon exit) -- no new Databento spend. Also runs the
six "always trade this structure" baselines from spec section 18 and the
promotion-controls suite from research_controls.py.

See OPTIONS_ROUTER_PHASE1.md for what is and isn't in scope for Phase 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from btc_trend_bot.options_router.backtest import available_days, run_backtest  # noqa: E402
from btc_trend_bot.options_router.features import build_first_hour_features, fetch_confirmation_data  # noqa: E402
from btc_trend_bot.options_router.reporting import (  # noqa: E402
    ALL_STRUCTURE_KINDS, run_promotion_controls, run_structure_baseline, summarize_trades,
)

OUT_DIR = ROOT / "outputs" / "options_router"


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for section in ["data", "regimes", "strategy_map", "structures", "risk", "exit_policy", "no_trade_filters"]:
        if section not in cfg:
            raise ValueError(f"config missing required section: {section}")
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "options_router.yaml"))
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--fill-mode", default="natural", choices=["natural", "midpoint", "worse_25pct"])
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    decisions, trades = run_backtest(cfg, root=ROOT, start=args.start, end=args.end, fill_mode=args.fill_mode)
    decisions.to_csv(OUT_DIR / "decisions.csv", index=False)
    trades.to_csv(OUT_DIR / "trades.csv", index=False)

    account_value = cfg["risk"]["account_value"]
    router_summary = summarize_trades(trades, account_value)

    entry_dir = ROOT / cfg["data"]["entry_dir"]
    exit_dir = ROOT / cfg["data"]["exit_dir"]
    days = available_days(entry_dir, exit_dir)
    if args.start:
        days = [d for d in days if str(d.date()) >= args.start]
    if args.end:
        days = [d for d in days if str(d.date()) <= args.end]

    bars = fetch_confirmation_data(cfg["data"]["confirmation_symbols"], underlying=cfg["data"]["underlying"])
    feature_df = build_first_hour_features(
        bars, underlying=cfg["data"]["underlying"],
        trailing_window_days=cfg["data"].get("trailing_window_days", 5),
    ).set_index("date")

    baselines = {}
    for kind in ALL_STRUCTURE_KINDS:
        bl_trades = run_structure_baseline(kind, days, entry_dir, exit_dir, feature_df, cfg, args.fill_mode)
        baselines[kind] = summarize_trades(bl_trades, account_value)

    baseline_mean_returns = {k: v["mean_return"] for k, v in baselines.items() if v.get("n", 0) > 0}
    promotion = run_promotion_controls(trades, decisions, feature_df, account_value, baseline_mean_returns)

    summary = {
        "n_days": len(days),
        "n_trades_routed": int(len(trades)),
        "router": router_summary,
        "baselines": baselines,
        "promotion_controls": promotion,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
