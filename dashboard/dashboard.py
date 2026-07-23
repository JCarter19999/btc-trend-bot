from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st
import yaml


st.set_page_config(
    page_title="Trading Bots Operations",
    page_icon="📈",
    layout="wide",
)

EQUITY_LEDGER_PATH = Path("/home/joey/equity_v2_4/runtime/equity_yfinance_paper.sqlite3")
EQUITY_STRATEGY_CONFIG_PATH = Path("/home/joey/equity_v2_4/config/schwab_paper_strategy.yaml")

# Fixed categorical hues (dataviz skill default order, first 5 slots) so a
# symbol keeps the same color everywhere in this tab regardless of sort order
# or which symbols happen to have trades yet -- color follows the entity, not
# its rank. Any symbol outside this set (a future universe change) falls back
# to muted gray rather than silently reassigning another symbol's color.
EQUITY_SYMBOL_COLORS = {
    "AAPL": "#2a78d6",   # blue
    "MSFT": "#eb6834",   # orange
    "NVDA": "#1baf7a",   # aqua
    "TSLA": "#eda100",   # yellow
    "AMZN": "#e87ba4",   # magenta
}
EQUITY_STATUS_GOOD = "#0ca30c"
EQUITY_STATUS_CRITICAL = "#d03b3b"
EQUITY_TEXT_SECONDARY = "#52514e"

# Two-tier safety-layer BTC candidate (see btc-trend-bot/TWO_TIER_SAFETY_LAYER.md)
# -- a separate local paper account from both the frozen production dry-run
# and the legacy paper.py account, written by
# btc-trend-bot/experiments/run_two_tier_safety_paper_step.py.
TWO_TIER_STATE_PATH = Path("/home/joey/btc-trend-bot/paper/two_tier_safety_state.json")
TWO_TIER_TRADES_PATH = Path("/home/joey/btc-trend-bot/paper/two_tier_safety_trades.csv")
TWO_TIER_RUNS_LOG_PATH = Path("/home/joey/btc-trend-bot/paper/two_tier_safety_runs.jsonl")
TWO_TIER_PRODUCTION_CONFIG_PATH = Path("/home/joey/btc-trend-bot/config/settings_production.yaml")
# Source of truth for the values below is CHOSEN_SAFETY in
# btc-trend-bot/experiments/run_two_tier_safety_paper_step.py; this file is
# the OOS-validation output those values were derived from and frozen from
# (see TWO_TIER_SAFETY_LAYER_RESULTS.md). Read here rather than importing the
# script directly, since that module pulls in the full btc_trend_bot package
# (ccxt, etc.) which this dashboard's venv does not have installed.
TWO_TIER_CHOSEN_SAFETY_PATH = Path(
    "/home/joey/btc-trend-bot/outputs/two_tier_safety_oos_validation/chosen_safety_config.json")
TWO_TIER_SAFETY_REASON_COLORS = {
    "": "#0ca30c",               # good -- trading normally
    "drawdown_pause": "#fab219",  # warning
    "loss_cooldown": "#ec835a",   # serious
    "hard_shutdown": "#d03b3b",   # critical
}


def get_secret(name: str) -> str:
    try:
        return str(st.secrets["supabase"][name]).strip()
    except Exception as exc:
        raise RuntimeError(
            "Missing .streamlit/secrets.toml. Copy secrets.toml.example to "
            ".streamlit/secrets.toml and add your Supabase URL and secret key."
        ) from exc


@st.cache_data(ttl=60, show_spinner=False)
def fetch_runs(limit: int = 5000) -> pd.DataFrame:
    url = get_secret("url").rstrip("/")
    key = get_secret("secret_key")

    endpoint = f"{url}/rest/v1/bot_runs"
    params = {
        "select": (
            "run_id,received_at,server_name,mode,strategy_version,"
            "started_at,finished_at,duration_seconds,exit_code,success,"
            "bar_timestamp,close_price,target_position,side,order_size,"
            "reason,stderr,"
            "portfolio_timestamp,base_currency,quote_currency,"
            "base_available,base_total,quote_available,quote_total,"
            "mark_price,base_value_quote,portfolio_value_quote,"
            "cash_percent,btc_percent,portfolio_error"
        ),
        "order": "started_at.desc",
        "limit": str(limit),
    }
    headers = {
        "apikey": key,
        "Accept": "application/json",
    }

    response = requests.get(
        endpoint,
        params=params,
        headers=headers,
        timeout=20,
    )
    response.raise_for_status()

    rows: list[dict[str, Any]] = response.json()
    frame = pd.DataFrame(rows)

    if frame.empty:
        return frame

    for column in (
        "received_at", "started_at", "finished_at", "bar_timestamp",
        "portfolio_timestamp",
    ):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")

    for column in (
        "duration_seconds",
        "close_price",
        "target_position",
        "order_size",
        "exit_code",
        "base_available",
        "base_total",
        "quote_available",
        "quote_total",
        "mark_price",
        "base_value_quote",
        "portfolio_value_quote",
        "cash_percent",
        "btc_percent",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "success" in frame.columns:
        frame["success"] = frame["success"].fillna(False).astype(bool)

    return frame.sort_values("started_at")


def age_text(timestamp: pd.Timestamp | None) -> str:
    if timestamp is None or pd.isna(timestamp):
        return "Unknown"

    now = pd.Timestamp.now(tz="UTC")
    seconds = max(0, int((now - timestamp).total_seconds()))

    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h ago"

    days = seconds / 86400
    return f"{days:.1f}d ago"


def health_status(latest: pd.Series) -> tuple[str, str]:
    finished = latest.get("finished_at")
    success = bool(latest.get("success", False))

    if not success:
        return "FAILED", "Latest execution returned an error."

    if pd.isna(finished):
        return "UNKNOWN", "No completion timestamp was reported."

    age_hours = (
        pd.Timestamp.now(tz="UTC") - finished
    ).total_seconds() / 3600

    if age_hours > 5:
        return "STALE", "No successful report has arrived in over five hours."

    return "HEALTHY", "Latest run succeeded and reporting is current."


def metric_value(value: Any, fallback: str = "—") -> str:
    if value is None or pd.isna(value):
        return fallback
    return str(value)


def render_btc_tab() -> None:
    st.title("₿ BTC Bot Operations")
    st.caption("Supabase-backed monitoring for the scheduled Hetzner deployment.")

    with st.sidebar:
        st.header("BTC bot controls")
        window = st.selectbox(
            "Time window",
            ["24 hours", "7 days", "30 days", "All available"],
            index=1,
        )

        if st.button("Refresh data", use_container_width=True):
            fetch_runs.clear()
            st.rerun()

        st.caption("Data is cached for 60 seconds.")

    try:
        all_runs = fetch_runs()
    except Exception as exc:
        st.error(f"Could not load Supabase data: {exc}")
        return

    if all_runs.empty:
        st.warning("No bot runs have been uploaded yet.")
        return

    cutoffs = {
        "24 hours": pd.Timedelta(hours=24),
        "7 days": pd.Timedelta(days=7),
        "30 days": pd.Timedelta(days=30),
    }

    if window == "All available":
        runs = all_runs.copy()
    else:
        cutoff = pd.Timestamp.now(tz="UTC") - cutoffs[window]
        runs = all_runs[all_runs["started_at"] >= cutoff].copy()

    if runs.empty:
        st.warning(f"No records are available for the selected window: {window}.")
        return

    latest = all_runs.sort_values("started_at").iloc[-1]
    status, status_help = health_status(latest)

    success_rate = 100 * runs["success"].mean() if len(runs) else 0
    latest_close = latest.get("close_price")
    latest_target = latest.get("target_position")
    latest_bar = latest.get("bar_timestamp")

    metric_columns = st.columns(6)

    metric_columns[0].metric(
        "Bot status",
        status,
        help=status_help,
        border=True,
    )
    metric_columns[1].metric(
        "Last report",
        age_text(latest.get("finished_at")),
        border=True,
    )
    metric_columns[2].metric(
        "Latest BTC close",
        f"${latest_close:,.2f}" if pd.notna(latest_close) else "—",
        border=True,
    )
    metric_columns[3].metric(
        "Target BTC exposure",
        f"{latest_target * 100:.0f}%" if pd.notna(latest_target) else "—",
        border=True,
    )
    metric_columns[4].metric(
        "Runs in window",
        f"{len(runs):,}",
        border=True,
    )
    metric_columns[5].metric(
        "Success rate",
        f"{success_rate:.1f}%",
        border=True,
    )

    st.caption(
        "Latest completed strategy bar: "
        + (
            latest_bar.strftime("%Y-%m-%d %H:%M UTC")
            if pd.notna(latest_bar)
            else "Unknown"
        )
        + f" · Mode: {metric_value(latest.get('mode'))}"
        + f" · Version: {metric_value(latest.get('strategy_version'))}"
    )

    st.subheader("Portfolio snapshot")

    portfolio_rows = all_runs.dropna(
        subset=["portfolio_timestamp", "portfolio_value_quote"]
    ).sort_values("portfolio_timestamp")

    if portfolio_rows.empty:
        st.info(
            "Portfolio telemetry has not arrived yet. It will appear after the "
            "updated Hetzner service completes its first run."
        )
    else:
        portfolio = portfolio_rows.iloc[-1]
        base_currency = metric_value(portfolio.get("base_currency"), "BTC")
        quote_currency = metric_value(portfolio.get("quote_currency"), "USD")

        portfolio_value = portfolio.get("portfolio_value_quote")
        quote_total = portfolio.get("quote_total")
        quote_available = portfolio.get("quote_available")
        base_total = portfolio.get("base_total")
        base_value = portfolio.get("base_value_quote")
        cash_percent = portfolio.get("cash_percent")
        btc_percent = portfolio.get("btc_percent")
        mark_price = portfolio.get("mark_price")

        portfolio_metrics = st.columns(5)
        portfolio_metrics[0].metric(
            f"Total portfolio ({quote_currency})",
            f"${portfolio_value:,.2f}" if pd.notna(portfolio_value) else "—",
            border=True,
        )
        portfolio_metrics[1].metric(
            f"Cash available ({quote_currency})",
            f"${quote_available:,.2f}" if pd.notna(quote_available) else "—",
            border=True,
        )
        portfolio_metrics[2].metric(
            f"Cash total ({quote_currency})",
            f"${quote_total:,.2f}" if pd.notna(quote_total) else "—",
            border=True,
        )
        portfolio_metrics[3].metric(
            f"{base_currency} holdings",
            f"{base_total:,.8f}" if pd.notna(base_total) else "—",
            border=True,
        )
        portfolio_metrics[4].metric(
            f"{base_currency} value",
            f"${base_value:,.2f}" if pd.notna(base_value) else "—",
            border=True,
        )

        portfolio_left, portfolio_right = st.columns([1, 2])

        with portfolio_left:
            allocation = pd.DataFrame(
                {
                    "asset": [f"Cash ({quote_currency})", base_currency],
                    "value": [quote_total, base_value],
                }
            ).dropna()

            allocation_chart = (
                alt.Chart(allocation)
                .mark_arc(innerRadius=65)
                .encode(
                    theta=alt.Theta("value:Q"),
                    color=alt.Color("asset:N", title="Allocation"),
                    tooltip=[
                        alt.Tooltip("asset:N", title="Asset"),
                        alt.Tooltip("value:Q", title="Value", format="$,.2f"),
                    ],
                )
                .properties(height=280, title="Current allocation")
            )
            st.altair_chart(allocation_chart, use_container_width=True)

            if pd.notna(cash_percent) and pd.notna(btc_percent):
                st.caption(
                    f"{cash_percent:.1f}% cash · {btc_percent:.1f}% {base_currency}"
                )

        with portfolio_right:
            history = (
                portfolio_rows.dropna(
                    subset=["portfolio_timestamp", "portfolio_value_quote"]
                )
                .sort_values("portfolio_timestamp")
                .drop_duplicates(
                    subset=["portfolio_timestamp"],
                    keep="last",
                )
            )

            portfolio_history_chart = (
                alt.Chart(history)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        "portfolio_timestamp:T",
                        title="Snapshot time",
                    ),
                    y=alt.Y(
                        "portfolio_value_quote:Q",
                        title=f"Portfolio value ({quote_currency})",
                        scale=alt.Scale(zero=False),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "portfolio_timestamp:T",
                            title="Snapshot",
                            format="%Y-%m-%d %H:%M UTC",
                        ),
                        alt.Tooltip(
                            "portfolio_value_quote:Q",
                            title="Portfolio value",
                            format="$,.2f",
                        ),
                        alt.Tooltip(
                            "cash_percent:Q",
                            title="Cash %",
                            format=".1f",
                        ),
                        alt.Tooltip(
                            "btc_percent:Q",
                            title=f"{base_currency} %",
                            format=".1f",
                        ),
                    ],
                )
                .properties(height=280, title="Portfolio value history")
                .interactive()
            )
            st.altair_chart(portfolio_history_chart, use_container_width=True)

        snapshot_time = portfolio.get("portfolio_timestamp")
        snapshot_age = age_text(snapshot_time) if pd.notna(snapshot_time) else "unknown"
        st.caption(
            f"Portfolio snapshot: {snapshot_age}"
            + (
                f" · Valuation price: ${mark_price:,.2f}"
                if pd.notna(mark_price)
                else ""
            )
            + " · Allocation covers the bot's configured base and quote assets."
        )

        portfolio_error = portfolio.get("portfolio_error")
        if isinstance(portfolio_error, str) and portfolio_error.strip():
            st.warning(f"Latest portfolio snapshot warning: {portfolio_error}")

    price_source = (
        runs.dropna(subset=["bar_timestamp", "close_price"])
        .sort_values(["bar_timestamp", "started_at"])
        .drop_duplicates(subset=["bar_timestamp"], keep="first")
    )

    allocation_source = (
        runs.dropna(subset=["bar_timestamp", "target_position"])
        .sort_values(["bar_timestamp", "started_at"])
        .drop_duplicates(subset=["bar_timestamp"], keep="first")
    )

    left, right = st.columns(2)

    with left:
        st.subheader("BTC close history")
        if price_source.empty:
            st.info("No price observations are available.")
        else:
            price_chart = (
                alt.Chart(price_source)
                .mark_line(point=True)
                .encode(
                    x=alt.X("bar_timestamp:T", title="Completed 4-hour bar"),
                    y=alt.Y(
                        "close_price:Q",
                        title="BTC close (USD)",
                        scale=alt.Scale(zero=False),
                    ),
                    tooltip=[
                        alt.Tooltip("bar_timestamp:T", title="Bar", format="%Y-%m-%d %H:%M UTC"),
                        alt.Tooltip("close_price:Q", title="Close", format="$,.2f"),
                    ],
                )
                .properties(height=320)
                .interactive()
            )
            st.altair_chart(price_chart, use_container_width=True)

    with right:
        st.subheader("Target BTC exposure")
        if allocation_source.empty:
            st.info("No target-position observations are available.")
        else:
            allocation_chart = (
                alt.Chart(allocation_source.assign(
                    target_percent=allocation_source["target_position"] * 100
                ))
                .mark_line(point=True, interpolate="step-after")
                .encode(
                    x=alt.X("bar_timestamp:T", title="Completed 4-hour bar"),
                    y=alt.Y(
                        "target_percent:Q",
                        title="Target exposure (%)",
                        scale=alt.Scale(domain=[0, 100]),
                    ),
                    tooltip=[
                        alt.Tooltip("bar_timestamp:T", title="Bar", format="%Y-%m-%d %H:%M UTC"),
                        alt.Tooltip("target_percent:Q", title="Target", format=".1f"),
                    ],
                )
                .properties(height=320)
                .interactive()
            )
            st.altair_chart(allocation_chart, use_container_width=True)

    st.subheader("Execution health")

    daily = runs.copy()
    daily["utc_day"] = daily["started_at"].dt.floor("D")
    daily_summary = (
        daily.groupby("utc_day", as_index=False)
        .agg(
            successful_runs=("success", "sum"),
            total_runs=("success", "size"),
            average_runtime=("duration_seconds", "mean"),
        )
    )
    daily_summary["failed_runs"] = (
        daily_summary["total_runs"] - daily_summary["successful_runs"]
    )

    health_long = daily_summary.melt(
        id_vars=["utc_day"],
        value_vars=["successful_runs", "failed_runs"],
        var_name="result",
        value_name="runs",
    )

    health_chart = (
        alt.Chart(health_long)
        .mark_bar()
        .encode(
            x=alt.X("utc_day:T", title="UTC day"),
            y=alt.Y("runs:Q", title="Executions"),
            color=alt.Color("result:N", title="Result"),
            tooltip=[
                alt.Tooltip("utc_day:T", title="Day", format="%Y-%m-%d"),
                alt.Tooltip("result:N", title="Result"),
                alt.Tooltip("runs:Q", title="Runs"),
            ],
        )
        .properties(height=260)
    )

    st.altair_chart(health_chart, use_container_width=True)

    decision_rows = runs.sort_values("started_at", ascending=False).copy()
    decision_rows["started_at"] = decision_rows["started_at"].dt.strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    decision_rows["bar_timestamp"] = decision_rows["bar_timestamp"].dt.strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    decision_rows["target_position"] = (
        decision_rows["target_position"] * 100
    ).round(2)
    decision_rows["close_price"] = decision_rows["close_price"].round(2)
    decision_rows["order_size"] = decision_rows["order_size"].round(8)
    decision_rows["duration_seconds"] = decision_rows["duration_seconds"].round(3)
    if "portfolio_value_quote" in decision_rows.columns:
        decision_rows["portfolio_value_quote"] = decision_rows["portfolio_value_quote"].round(2)
    if "cash_percent" in decision_rows.columns:
        decision_rows["cash_percent"] = decision_rows["cash_percent"].round(1)

    display_columns = [
        "started_at",
        "bar_timestamp",
        "close_price",
        "target_position",
        "side",
        "order_size",
        "reason",
        "success",
        "duration_seconds",
        "portfolio_value_quote",
        "cash_percent",
        "strategy_version",
    ]

    st.subheader("Recent decisions")
    st.dataframe(
        decision_rows[display_columns].head(100),
        use_container_width=True,
        hide_index=True,
        column_config={
            "started_at": "Run time",
            "bar_timestamp": "Strategy bar",
            "close_price": st.column_config.NumberColumn("BTC close", format="$%.2f"),
            "target_position": st.column_config.NumberColumn("Target BTC %", format="%.2f%%"),
            "side": "Decision",
            "order_size": st.column_config.NumberColumn("Order size BTC", format="%.8f"),
            "reason": "Reason",
            "success": "Success",
            "duration_seconds": st.column_config.NumberColumn("Runtime (s)", format="%.3f"),
            "portfolio_value_quote": st.column_config.NumberColumn("Portfolio value", format="$%.2f"),
            "cash_percent": st.column_config.NumberColumn("Cash %", format="%.1f%%"),
            "strategy_version": "Git version",
        },
    )

    failed = runs[~runs["success"]]
    if not failed.empty:
        with st.expander(f"Errors in selected window ({len(failed)})", expanded=True):
            st.dataframe(
                failed[
                    [
                        "started_at",
                        "exit_code",
                        "reason",
                        "stderr",
                        "strategy_version",
                    ]
                ].sort_values("started_at", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Export")
    csv_bytes = runs.sort_values("started_at", ascending=False).to_csv(
        index=False
    ).encode("utf-8")
    json_bytes = json.dumps(
        runs.sort_values("started_at", ascending=False)
        .assign(
            started_at=lambda df: df["started_at"].astype(str),
            finished_at=lambda df: df["finished_at"].astype(str),
            received_at=lambda df: df["received_at"].astype(str),
            bar_timestamp=lambda df: df["bar_timestamp"].astype(str),
            portfolio_timestamp=lambda df: df["portfolio_timestamp"].astype(str),
        )
        .to_dict(orient="records"),
        indent=2,
        default=str,
    ).encode("utf-8")

    download_left, download_right = st.columns(2)
    download_left.download_button(
        "Download selected runs as CSV",
        data=csv_bytes,
        file_name="btc_bot_runs.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_right.download_button(
        "Download selected runs as JSON",
        data=json_bytes,
        file_name="btc_bot_runs.json",
        mime="application/json",
        use_container_width=True,
    )


# --------------------------------------------------------------------------- #
# Equity paper trading (yfinance live-data stand-in) -- reads the local SQLite
# ledger written by equity_v2_4/experiments/run_equity_paper_step.py directly.
# No live-order code exists in that pipeline; every fill here is simulated.
# --------------------------------------------------------------------------- #

def _load_equity_strategy_config() -> dict:
    if not EQUITY_STRATEGY_CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(EQUITY_STRATEGY_CONFIG_PATH.read_text()) or {}


@st.cache_data(ttl=30, show_spinner=False)
def fetch_equity_ledger(db_path: str) -> dict[str, Any]:
    empty = {"completed_trades": pd.DataFrame(), "open_position": None,
             "runs": pd.DataFrame(), "kv": {}}
    if not Path(db_path).exists():
        return empty
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        trades = pd.read_sql_query("SELECT * FROM completed_trades ORDER BY signal_time", conn)
        runs = pd.read_sql_query("SELECT * FROM runs ORDER BY id DESC LIMIT 50", conn)
        open_row = conn.execute("SELECT * FROM open_position WHERE id=1").fetchone()
        kv_rows = conn.execute("SELECT key, value FROM kv_state").fetchall()
    finally:
        conn.close()

    for column in ("signal_time", "entry_time", "exit_time", "created_at"):
        if column in trades.columns:
            trades[column] = pd.to_datetime(trades[column], utc=True, errors="coerce")
    if "created_at" in runs.columns:
        runs["created_at"] = pd.to_datetime(runs["created_at"], utc=True, errors="coerce")

    return {
        "completed_trades": trades,
        "open_position": dict(open_row) if open_row else None,
        "runs": runs,
        "kv": {row["key"]: row["value"] for row in kv_rows},
    }


def _simulate_capital_path(trades: pd.DataFrame, strategy_cfg: dict) -> pd.DataFrame:
    """Mirrors run_equity_real_data_walkforward.simulate_capital's sizing and safety
    math exactly (position_fraction notional, drawdown pause, loss cooldown, hard
    shutdown) so the displayed equity curve matches what the live paper step itself
    used to gate new entries -- see that function for the authoritative version this
    must stay consistent with."""
    safety = strategy_cfg.get("safety", {})
    initial_capital = float(strategy_cfg.get("initial_capital", 2500.0))
    position_fraction = float(strategy_cfg.get("position_fraction", 0.25))
    safety_enabled = bool(safety.get("enabled", True))
    drawdown_pause = float(safety.get("drawdown_pause", 0.15))
    hard_shutdown_drawdown = float(safety.get("hard_shutdown_drawdown", 0.35))
    consecutive_loss_limit = int(safety.get("consecutive_loss_limit", 4))
    cooldown_trades = int(safety.get("cooldown_trades", 8))
    minimum_equity = float(safety.get("minimum_equity", 25.0))

    equity = initial_capital
    peak = equity
    cooldown = 0
    dd_cooldown = 0
    losses = 0
    hard = False
    rows: list[dict[str, Any]] = []
    for _, t in trades.sort_values("signal_time").iterrows():
        dd = 1 - equity / peak
        reason = ""
        if hard or equity < minimum_equity or dd >= hard_shutdown_drawdown:
            hard = True
            reason = "hard_shutdown"
        elif safety_enabled and cooldown > 0:
            cooldown -= 1
            reason = "loss_cooldown"
        elif safety_enabled and (dd_cooldown > 0 or dd >= drawdown_pause):
            if dd_cooldown == 0:
                dd_cooldown = cooldown_trades
            dd_cooldown -= 1
            reason = "drawdown_pause"
            if dd_cooldown == 0:
                peak = equity
        if reason:
            rows.append({"signal_time": t.signal_time, "symbol": t.symbol, "trade_taken": False,
                         "ending_equity": equity, "drawdown": dd})
            continue
        start = equity
        notional = start * position_fraction
        pnl = notional * float(t.net_return)
        equity = max(0.0, start + pnl)
        peak = max(peak, equity)
        dd = 1 - equity / peak
        if pnl < 0:
            losses += 1
            if safety_enabled and losses >= consecutive_loss_limit:
                cooldown = cooldown_trades
                losses = 0
        else:
            losses = 0
        rows.append({"signal_time": t.signal_time, "symbol": t.symbol, "trade_taken": True,
                     "ending_equity": equity, "drawdown": dd, "trade_pnl": pnl})
    return pd.DataFrame(rows)


def render_equity_tab() -> None:
    st.title("📈 Equity Paper Trading — yfinance live data")
    st.caption(
        "Phase 2 shadow deployment (equity_v2_4). Simulated fills only, replayed "
        "locally against real market data -- no live orders are ever placed."
    )

    if st.button("Refresh equity data", key="equity_refresh"):
        fetch_equity_ledger.clear()
        st.rerun()

    if not EQUITY_LEDGER_PATH.exists():
        st.info(f"No paper-trading ledger found yet at {EQUITY_LEDGER_PATH} -- "
                "the scheduled step hasn't run.")
        return

    strategy_cfg = _load_equity_strategy_config()
    data = fetch_equity_ledger(str(EQUITY_LEDGER_PATH))
    trades: pd.DataFrame = data["completed_trades"]
    open_position: dict | None = data["open_position"]
    runs: pd.DataFrame = data["runs"]
    kv: dict = data["kv"]

    halted = kv.get("halted") == "true"
    halt_reason = kv.get("halt_reason", "")
    last_processed = kv.get("last_processed_bar_date") or "—"

    path = _simulate_capital_path(trades, strategy_cfg) if len(trades) else pd.DataFrame()
    initial_capital = float(strategy_cfg.get("initial_capital", 2500.0))
    current_equity = float(path.iloc[-1].ending_equity) if len(path) else initial_capital
    total_return = current_equity / initial_capital - 1
    max_drawdown = float(path["drawdown"].max()) if len(path) else 0.0

    metric_columns = st.columns(6)
    metric_columns[0].metric(
        "Bot status", "HALTED" if halted else "RUNNING",
        help=halt_reason or None, border=True,
    )
    metric_columns[1].metric("Last processed bar", str(last_processed), border=True)
    metric_columns[2].metric("Simulated equity", f"${current_equity:,.2f}", border=True)
    metric_columns[3].metric("Total return", f"{total_return * 100:+.2f}%", border=True)
    metric_columns[4].metric("Max drawdown", f"{max_drawdown * 100:.1f}%", border=True)
    metric_columns[5].metric("Completed trades", f"{len(trades):,}", border=True)

    st.subheader("Open position")
    if open_position is None:
        st.info("Flat -- no open position.")
    else:
        pos_cols = st.columns(5)
        pos_cols[0].metric("Symbol", open_position.get("symbol") or "—")
        pos_cols[1].metric("Status", open_position.get("status") or "—")
        entry_price = open_position.get("entry_price")
        stop_price = open_position.get("stop_price")
        target_price = open_position.get("target_price")
        pos_cols[2].metric("Entry price", f"${entry_price:,.2f}" if entry_price else "pending fill")
        pos_cols[3].metric("Stop", f"${stop_price:,.2f}" if stop_price else "—")
        pos_cols[4].metric("Target", f"${target_price:,.2f}" if target_price else "—")
        predicted_return = open_position.get("predicted_return")
        if predicted_return is not None:
            st.caption(f"Predicted return at signal time: {predicted_return * 100:+.2f}%")

    st.subheader("Simulated equity curve")
    if path.empty:
        st.info("No completed trades yet -- the equity curve will appear after the first exit.")
    else:
        equity_chart = (
            alt.Chart(path)
            .mark_line(point=True)
            .encode(
                x=alt.X("signal_time:T", title="Signal time"),
                y=alt.Y("ending_equity:Q", title="Simulated equity ($)", scale=alt.Scale(zero=False)),
                color=alt.condition(alt.datum.trade_taken, alt.value("#2ca02c"), alt.value("#d62728")),
                tooltip=[
                    alt.Tooltip("signal_time:T", title="Signal time", format="%Y-%m-%d"),
                    alt.Tooltip("symbol:N", title="Symbol"),
                    alt.Tooltip("ending_equity:Q", title="Equity", format="$,.2f"),
                    alt.Tooltip("trade_taken:N", title="Trade taken"),
                ],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(equity_chart, use_container_width=True)

    st.subheader("Per-symbol trade activity")
    if trades.empty:
        st.info("No completed trades yet -- per-symbol breakdown will appear after the first exit.")
    else:
        per_symbol = trades.groupby("symbol").agg(
            trade_count=("net_return", "size"),
            avg_return=("net_return", "mean"),
        ).reset_index()
        pnl_by_symbol = (
            path[path["trade_taken"]].groupby("symbol")["trade_pnl"].sum().reset_index()
            if len(path) else pd.DataFrame(columns=["symbol", "trade_pnl"])
        )
        per_symbol = per_symbol.merge(pnl_by_symbol, on="symbol", how="left")
        per_symbol["trade_pnl"] = per_symbol["trade_pnl"].fillna(0.0)

        # Fixed left-to-right symbol order (shared by both charts below) so a
        # symbol sits in the same spot in each -- easier to cross-reference
        # count against P&L than if each chart sorted independently.
        symbol_order = [s for s in EQUITY_SYMBOL_COLORS if s in per_symbol["symbol"].values]
        symbol_order += sorted(s for s in per_symbol["symbol"] if s not in EQUITY_SYMBOL_COLORS)
        color_domain = list(EQUITY_SYMBOL_COLORS.keys())
        color_range = list(EQUITY_SYMBOL_COLORS.values())

        count_left, pnl_right = st.columns(2)

        with count_left:
            count_base = alt.Chart(per_symbol).encode(
                x=alt.X("symbol:N", title=None, sort=symbol_order),
                y=alt.Y("trade_count:Q", title="Trades"),
            )
            count_bars = count_base.mark_bar(cornerRadiusEnd=4, size=32).encode(
                color=alt.Color("symbol:N", scale=alt.Scale(domain=color_domain, range=color_range),
                                 legend=None, sort=symbol_order),
                tooltip=[
                    alt.Tooltip("symbol:N", title="Symbol"),
                    alt.Tooltip("trade_count:Q", title="Trades"),
                    alt.Tooltip("avg_return:Q", title="Avg return", format=".2%"),
                ],
            )
            count_labels = count_base.mark_text(dy=-8, color=EQUITY_TEXT_SECONDARY).encode(
                text=alt.Text("trade_count:Q", format="d"))
            st.altair_chart(
                (count_bars + count_labels).properties(height=280, title="Trades per symbol"),
                use_container_width=True,
            )

        with pnl_right:
            pnl_base = alt.Chart(per_symbol).encode(
                x=alt.X("symbol:N", title=None, sort=symbol_order),
                y=alt.Y("trade_pnl:Q", title="Simulated P&L ($)"),
            )
            pnl_bars = pnl_base.mark_bar(cornerRadiusEnd=4, size=32).encode(
                color=alt.condition(alt.datum.trade_pnl >= 0, alt.value(EQUITY_STATUS_GOOD),
                                     alt.value(EQUITY_STATUS_CRITICAL)),
                tooltip=[
                    alt.Tooltip("symbol:N", title="Symbol"),
                    alt.Tooltip("trade_pnl:Q", title="Simulated P&L", format="$,.2f"),
                    alt.Tooltip("trade_count:Q", title="Trades"),
                ],
            )
            pnl_labels_pos = (
                pnl_base.transform_filter(alt.datum.trade_pnl >= 0)
                .mark_text(dy=-8, color=EQUITY_TEXT_SECONDARY)
                .encode(text=alt.Text("trade_pnl:Q", format="$,.0f"))
            )
            pnl_labels_neg = (
                pnl_base.transform_filter(alt.datum.trade_pnl < 0)
                .mark_text(dy=12, color=EQUITY_TEXT_SECONDARY)
                .encode(text=alt.Text("trade_pnl:Q", format="$,.0f"))
            )
            st.altair_chart(
                (pnl_bars + pnl_labels_pos + pnl_labels_neg).properties(
                    height=280, title="Simulated P&L per symbol"),
                use_container_width=True,
            )
        st.caption(
            "P&L uses the same position-sized simulation as the equity curve above "
            "(position_fraction of equity per trade, compounded) -- not raw % return."
        )

    st.subheader("Predicted vs. actual return")
    if trades.empty:
        st.info("No completed trades yet -- this compares the Ridge model's predicted_return at "
                 "signal time against each trade's realized net_return once it appears.")
    else:
        predicted_domain = [s for s in EQUITY_SYMBOL_COLORS if s in trades["symbol"].values]
        predicted_domain += sorted(s for s in trades["symbol"] if s not in EQUITY_SYMBOL_COLORS)
        axis_min = float(min(trades["predicted_return"].min(), trades["net_return"].min()))
        axis_max = float(max(trades["predicted_return"].max(), trades["net_return"].max()))
        pad = (axis_max - axis_min) * 0.1 or 0.01
        diagonal = pd.DataFrame({"x": [axis_min - pad, axis_max + pad], "y": [axis_min - pad, axis_max + pad]})

        calibration_line = (
            alt.Chart(diagonal)
            .mark_line(strokeDash=[4, 4], color=EQUITY_TEXT_SECONDARY)
            .encode(x=alt.X("x:Q", scale=alt.Scale(domain=[axis_min - pad, axis_max + pad])),
                    y=alt.Y("y:Q", scale=alt.Scale(domain=[axis_min - pad, axis_max + pad])))
        )
        scatter = (
            alt.Chart(trades)
            .mark_circle(size=90, opacity=0.8)
            .encode(
                x=alt.X("predicted_return:Q", title="Predicted return (at signal time)",
                        axis=alt.Axis(format="%"), scale=alt.Scale(domain=[axis_min - pad, axis_max + pad])),
                y=alt.Y("net_return:Q", title="Actual return (realized)",
                        axis=alt.Axis(format="%"), scale=alt.Scale(domain=[axis_min - pad, axis_max + pad])),
                color=alt.Color("symbol:N", title="Symbol",
                                 scale=alt.Scale(domain=list(EQUITY_SYMBOL_COLORS.keys()),
                                                  range=list(EQUITY_SYMBOL_COLORS.values()))),
                tooltip=[
                    alt.Tooltip("symbol:N", title="Symbol"),
                    alt.Tooltip("signal_time:T", title="Signal time", format="%Y-%m-%d"),
                    alt.Tooltip("predicted_return:Q", title="Predicted", format=".2%"),
                    alt.Tooltip("net_return:Q", title="Actual", format=".2%"),
                    alt.Tooltip("exit_reason:N", title="Exit reason"),
                ],
            )
        )
        st.altair_chart((calibration_line + scatter).properties(height=340).interactive(),
                         use_container_width=True)
        st.caption(
            "Dashed line is perfect calibration (actual = predicted). Above the line: the trade did "
            "better than the model expected; below: worse. Scales differ naturally -- predicted_return "
            "is the raw Ridge forecast, actual net_return is clipped by the ATR stop/target/time exit, "
            "so realized swings are often larger in either direction than the prediction that triggered them."
        )

    st.subheader("Completed trades")
    if trades.empty:
        st.info("No completed trades yet.")
    else:
        display = trades.sort_values("signal_time", ascending=False).copy()
        for column in ("signal_time", "entry_time", "exit_time"):
            display[column] = display[column].dt.strftime("%Y-%m-%d %H:%M UTC")
        display["net_return"] = (display["net_return"] * 100).round(2)
        st.dataframe(
            display[["signal_time", "symbol", "entry_time", "entry_price", "exit_time",
                     "exit_price", "net_return", "exit_reason", "bars_held", "predicted_return"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "net_return": st.column_config.NumberColumn("Net return", format="%.2f%%"),
                "entry_price": st.column_config.NumberColumn("Entry", format="$%.2f"),
                "exit_price": st.column_config.NumberColumn("Exit", format="$%.2f"),
                "predicted_return": st.column_config.NumberColumn("Predicted return", format="%.4f"),
            },
        )

    st.subheader("Recent run log")
    if runs.empty:
        st.info("No run history yet.")
    else:
        runs_display = runs.copy()
        runs_display["created_at"] = runs_display["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        st.dataframe(
            runs_display[["created_at", "as_of_date", "status", "message"]],
            use_container_width=True,
            hide_index=True,
        )



# --------------------------------------------------------------------------- #
# Two-tier safety-layer BTC candidate -- reads the local state/trades/run-log
# files directly, same pattern as the equity tab reads its SQLite ledger.
# --------------------------------------------------------------------------- #

def _load_two_tier_production_cfg() -> dict:
    if not TWO_TIER_PRODUCTION_CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(TWO_TIER_PRODUCTION_CONFIG_PATH.read_text()) or {}


@st.cache_data(ttl=30, show_spinner=False)
def fetch_two_tier_safety_data(state_path: str, trades_path: str, runs_path: str) -> dict[str, Any]:
    state = None
    if Path(state_path).exists():
        state = json.loads(Path(state_path).read_text())

    trades = pd.DataFrame()
    if Path(trades_path).exists():
        trades = pd.read_csv(trades_path)
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)

    runs = pd.DataFrame()
    if Path(runs_path).exists():
        rows = [json.loads(line) for line in Path(runs_path).read_text().splitlines() if line.strip()]
        runs = pd.DataFrame(rows)
        if len(runs):
            runs["timestamp"] = pd.to_datetime(runs["timestamp"], utc=True)
            runs["created_at"] = pd.to_datetime(runs["created_at"], utc=True)

    return {"state": state, "trades": trades, "runs": runs}


TWO_TIER_SCHEDULE_HOURS_UTC = [0, 4, 8, 12, 16, 20]
TWO_TIER_SCHEDULE_MINUTE_UTC = 12  # deploy/two-tier-safety-paper.timer: *-*-* 00,04,08,12,16,20:12:00 UTC


def _next_two_tier_run_time(now: pd.Timestamp) -> pd.Timestamp:
    day_start = now.normalize()
    candidates = [
        day_start + pd.Timedelta(days=offset, hours=h, minutes=TWO_TIER_SCHEDULE_MINUTE_UTC)
        for offset in (0, 1)
        for h in TWO_TIER_SCHEDULE_HOURS_UTC
    ]
    return min(c for c in candidates if c > now)


def _format_future_delta(delta: pd.Timedelta) -> str:
    total_seconds = max(0, int(delta.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def render_two_tier_safety_tab() -> None:
    st.title("⚖️ Two-Tier Safety Layer -- BTC candidate")
    st.caption(
        "Research candidate (see TWO_TIER_SAFETY_LAYER.md in btc-trend-bot). Same frozen "
        "production trend signal, wrapped in a soft drawdown-pause + loss-cooldown + hard-shutdown "
        "risk layer. Pure local simulation against public market data -- no exchange keys, no live orders."
    )

    if st.button("Refresh data", key="two_tier_refresh"):
        fetch_two_tier_safety_data.clear()
        st.rerun()

    if TWO_TIER_CHOSEN_SAFETY_PATH.exists():
        safety_cfg = json.loads(TWO_TIER_CHOSEN_SAFETY_PATH.read_text())
        with st.expander("Safety configuration in effect (validated out-of-sample -- see TWO_TIER_SAFETY_LAYER_RESULTS.md)"):
            cfg_cols = st.columns(5)
            cfg_cols[0].metric("Drawdown pause", f"{safety_cfg['drawdown_pause'] * 100:.0f}%")
            cfg_cols[1].metric("Hard shutdown", f"{safety_cfg['hard_shutdown_drawdown'] * 100:.0f}%")
            cfg_cols[2].metric("Min equity floor", f"{safety_cfg['minimum_equity_fraction'] * 100:.0f}%")
            cfg_cols[3].metric("Loss-streak limit", f"{safety_cfg['consecutive_loss_limit_bars']} bars")
            cfg_cols[4].metric("Cooldown length", f"{safety_cfg['cooldown_bars']} bars")

    if not TWO_TIER_STATE_PATH.exists():
        st.info(f"No paper account yet at {TWO_TIER_STATE_PATH} -- the scheduled step hasn't run.")
        return

    cfg = _load_two_tier_production_cfg()
    initial_cash = float(cfg.get("paper", {}).get("initial_cash", 500.0))
    data = fetch_two_tier_safety_data(str(TWO_TIER_STATE_PATH), str(TWO_TIER_TRADES_PATH), str(TWO_TIER_RUNS_LOG_PATH))
    state: dict | None = data["state"]
    trades: pd.DataFrame = data["trades"]
    runs: pd.DataFrame = data["runs"]

    hard_halted = bool(state.get("hard_halted")) if state else False
    loss_cooldown = int(state.get("loss_cooldown_remaining", 0)) if state else 0
    dd_cooldown = int(state.get("dd_cooldown_remaining", 0)) if state else 0
    if hard_halted:
        safety_state = "HARD SHUTDOWN"
    elif loss_cooldown > 0:
        safety_state = f"loss cooldown ({loss_cooldown} bars left)"
    elif dd_cooldown > 0:
        safety_state = f"drawdown pause ({dd_cooldown} bars left)"
    else:
        safety_state = "trading normally"

    current_equity = float(runs["equity"].iloc[-1]) if len(runs) else initial_cash
    total_return = current_equity / initial_cash - 1
    max_drawdown = float(runs["drawdown"].min()) if len(runs) else 0.0
    last_bar = state.get("last_bar_timestamp") if state else None

    # --- Portfolio value, on top ------------------------------------------ #
    st.subheader("Portfolio value")
    portfolio_cols = st.columns(4)
    portfolio_cols[0].metric("Paper equity", f"${current_equity:,.2f}", border=True)
    portfolio_cols[1].metric("Total return", f"{total_return * 100:+.2f}%", border=True)
    portfolio_cols[2].metric("Max drawdown", f"{max_drawdown * 100:.1f}%", border=True)
    portfolio_cols[3].metric("Bot status", "HALTED" if hard_halted else "RUNNING",
                              help=state.get("hard_halt_reason") if state else None, border=True)
    st.caption(f"Safety state: **{safety_state}**")

    # --- Decisions, right after ------------------------------------------- #
    st.subheader("Decisions")
    if runs.empty:
        st.info("No run history yet.")
    else:
        runs_display = runs.sort_values("timestamp", ascending=False).copy()
        runs_display["timestamp"] = runs_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
        runs_display["safety_reason"] = runs_display["safety_reason"].replace("", "normal")
        st.dataframe(
            runs_display[["timestamp", "close", "raw_target", "applied_target", "safety_reason",
                          "side", "equity", "drawdown"]].head(100),
            use_container_width=True, hide_index=True,
            column_config={
                "close": st.column_config.NumberColumn("BTC close", format="$%.2f"),
                "raw_target": st.column_config.NumberColumn("Raw signal", format="%.2f"),
                "applied_target": st.column_config.NumberColumn("Applied", format="%.2f"),
                "equity": st.column_config.NumberColumn("Equity", format="$%.2f"),
                "drawdown": st.column_config.NumberColumn("Drawdown", format="%.2%"),
            },
        )

    # --- Last processed bar / next scheduled run timer --------------------- #
    now_utc = pd.Timestamp.now(tz="UTC")
    next_run = _next_two_tier_run_time(now_utc)
    timer_cols = st.columns(2)
    timer_cols[0].metric("Last processed bar", str(last_bar or "—"), border=True)
    timer_cols[1].metric("Next scheduled run", f"{next_run.strftime('%H:%M UTC')} (in {_format_future_delta(next_run - now_utc)})",
                          border=True)
    st.caption(
        "Scheduled every 4h at :12 past (deploy/two-tier-safety-paper.timer), 7 minutes after "
        "the real production step. \"Next scheduled run\" is computed from that fixed schedule, "
        "not read from systemd, so it reflects when the page was loaded/refreshed."
    )

    st.subheader("Current allocation")
    if state is None:
        st.info("No state yet.")
    else:
        cash = float(state.get("cash", 0.0))
        btc = float(state.get("btc", 0.0))
        alloc_cols = st.columns(3)
        alloc_cols[0].metric("Cash", f"${max(cash, 0.0):,.2f}", border=True)
        alloc_cols[1].metric("BTC held", f"{btc:.8f}", border=True)
        if len(runs):
            alloc_cols[2].metric("Applied exposure", f"{float(runs['applied_target'].iloc[-1]) * 100:.0f}%", border=True)

    st.subheader("Equity curve")
    if runs.empty:
        st.info("No run history yet -- the equity curve will appear after the first scheduled step.")
    else:
        equity_chart = (
            alt.Chart(runs)
            .mark_line(point=True)
            .encode(
                x=alt.X("timestamp:T", title="Bar"),
                y=alt.Y("equity:Q", title="Paper equity ($)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "safety_reason:N", title="State",
                    scale=alt.Scale(domain=list(TWO_TIER_SAFETY_REASON_COLORS.keys()),
                                     range=list(TWO_TIER_SAFETY_REASON_COLORS.values())),
                    legend=alt.Legend(labelExpr="datum.label === '' ? 'normal' : datum.label"),
                ),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Bar", format="%Y-%m-%d %H:%M UTC"),
                    alt.Tooltip("equity:Q", title="Equity", format="$,.2f"),
                    alt.Tooltip("drawdown:Q", title="Drawdown", format=".2%"),
                    alt.Tooltip("safety_reason:N", title="State"),
                ],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(equity_chart, use_container_width=True)

    st.subheader("Exposure vs. raw signal")
    if runs.empty:
        st.info("No run history yet.")
    else:
        exposure_long = runs.melt(
            id_vars=["timestamp"], value_vars=["raw_target", "applied_target"],
            var_name="series", value_name="exposure",
        )
        exposure_long["exposure"] = exposure_long["exposure"] * 100
        exposure_chart = (
            alt.Chart(exposure_long)
            .mark_line(point=True, interpolate="step-after")
            .encode(
                x=alt.X("timestamp:T", title="Bar"),
                y=alt.Y("exposure:Q", title="Exposure (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("series:N", title="Series",
                                 scale=alt.Scale(domain=["raw_target", "applied_target"],
                                                  range=["#898781", "#2a78d6"])),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Bar", format="%Y-%m-%d %H:%M UTC"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("exposure:Q", title="Exposure", format=".0f"),
                ],
            )
            .properties(height=260)
            .interactive()
        )
        st.altair_chart(exposure_chart, use_container_width=True)
        st.caption(
            "raw_target is what the frozen strategy signal wants; applied_target is what the "
            "safety layer actually allowed (0% whenever a pause or shutdown is active)."
        )

    st.subheader("Signal's implied return vs. actual return")
    if len(runs) < 2:
        st.info("Needs at least two processed bars to compute a bar-over-bar return -- this strategy "
                 "has no predicted-return concept like the equity model, so this is a proxy: what the "
                 "raw signal's exposure would have earned on the realized market move, vs. what the "
                 "account actually earned once the safety layer's overrides and costs are included.")
    else:
        runs_sorted = runs.sort_values("timestamp").reset_index(drop=True)
        bar_return = runs_sorted["close"].pct_change()
        proxy = pd.DataFrame({
            "timestamp": runs_sorted["timestamp"],
            "expected_return": runs_sorted["raw_target"].shift(1) * bar_return,
            "actual_return": runs_sorted["equity"].pct_change(),
            "safety_reason": runs_sorted["safety_reason"].shift(1).fillna(""),
        }).dropna(subset=["expected_return", "actual_return"])

        if proxy.empty:
            st.info("Not enough history yet to compute this.")
        else:
            axis_min = float(min(proxy["expected_return"].min(), proxy["actual_return"].min()))
            axis_max = float(max(proxy["expected_return"].max(), proxy["actual_return"].max()))
            pad = (axis_max - axis_min) * 0.1 or 0.01
            domain = [axis_min - pad, axis_max + pad]
            diagonal = pd.DataFrame({"x": domain, "y": domain})

            calibration_line = (
                alt.Chart(diagonal)
                .mark_line(strokeDash=[4, 4], color=EQUITY_TEXT_SECONDARY)
                .encode(x=alt.X("x:Q", scale=alt.Scale(domain=domain)),
                        y=alt.Y("y:Q", scale=alt.Scale(domain=domain)))
            )
            scatter = (
                alt.Chart(proxy)
                .mark_circle(size=90, opacity=0.8)
                .encode(
                    x=alt.X("expected_return:Q", title="Raw signal's implied return",
                            axis=alt.Axis(format="%"), scale=alt.Scale(domain=domain)),
                    y=alt.Y("actual_return:Q", title="Actual account return",
                            axis=alt.Axis(format="%"), scale=alt.Scale(domain=domain)),
                    color=alt.Color(
                        "safety_reason:N", title="State",
                        scale=alt.Scale(domain=list(TWO_TIER_SAFETY_REASON_COLORS.keys()),
                                         range=list(TWO_TIER_SAFETY_REASON_COLORS.values())),
                        legend=alt.Legend(labelExpr="datum.label === '' ? 'normal' : datum.label"),
                    ),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Bar", format="%Y-%m-%d %H:%M UTC"),
                        alt.Tooltip("expected_return:Q", title="Raw signal implied", format=".2%"),
                        alt.Tooltip("actual_return:Q", title="Actual", format=".2%"),
                        alt.Tooltip("safety_reason:N", title="State"),
                    ],
                )
            )
            st.altair_chart((calibration_line + scatter).properties(height=340).interactive(),
                             use_container_width=True)
            st.caption(
                "Not a model calibration chart -- this strategy doesn't predict a return magnitude. "
                "Dashed line is where the account earned exactly what raw exposure implied (normal bars "
                "should sit near it, off by transaction cost). Points that fall short of the line during "
                "\"drawdown_pause\"/\"loss_cooldown\"/\"hard_shutdown\" bars are the safety layer forgoing "
                "a raw gain the signal would have captured; points that land at 0% while raw was sharply "
                "negative are the safety layer avoiding a raw loss."
            )

    st.subheader("Time in each safety state")
    if runs.empty:
        st.info("No run history yet.")
    else:
        reason_counts = runs["safety_reason"].fillna("").value_counts().reset_index()
        reason_counts.columns = ["safety_reason", "bars"]
        reason_counts["label"] = reason_counts["safety_reason"].replace("", "normal")
        state_chart = (
            alt.Chart(reason_counts)
            .mark_bar(cornerRadiusEnd=4, size=32)
            .encode(
                x=alt.X("label:N", title=None,
                        sort=["normal", "drawdown_pause", "loss_cooldown", "hard_shutdown"]),
                y=alt.Y("bars:Q", title="Bars"),
                color=alt.Color("safety_reason:N",
                                 scale=alt.Scale(domain=list(TWO_TIER_SAFETY_REASON_COLORS.keys()),
                                                  range=list(TWO_TIER_SAFETY_REASON_COLORS.values())),
                                 legend=None),
                tooltip=[alt.Tooltip("label:N", title="State"), alt.Tooltip("bars:Q", title="Bars")],
            )
            .properties(height=240, title="Bars per safety state (since this account started)")
        )
        st.altair_chart(state_chart, use_container_width=True)

    st.subheader("Trades")
    if trades.empty:
        st.info("No trades yet.")
    else:
        display = trades.sort_values("timestamp", ascending=False).copy()
        display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
        st.dataframe(
            display[["timestamp", "side", "btc_delta", "fill_price", "fee", "target_position",
                     "safety_reason", "cash_after", "btc_after"]],
            use_container_width=True, hide_index=True,
            column_config={
                "btc_delta": st.column_config.NumberColumn("BTC delta", format="%.8f"),
                "fill_price": st.column_config.NumberColumn("Fill price", format="$%.2f"),
                "fee": st.column_config.NumberColumn("Fee", format="$%.4f"),
                "target_position": st.column_config.NumberColumn("Target", format="%.2f"),
                "cash_after": st.column_config.NumberColumn("Cash after", format="$%.2f"),
                "btc_after": st.column_config.NumberColumn("BTC after", format="%.8f"),
            },
        )


tab_btc, tab_equity, tab_two_tier = st.tabs(
    ["₿ BTC Bot", "📈 Equity Paper (yfinance)", "⚖️ Two-Tier Safety (BTC)"])
with tab_btc:
    render_btc_tab()
with tab_equity:
    render_equity_tab()
with tab_two_tier:
    render_two_tier_safety_tab()
