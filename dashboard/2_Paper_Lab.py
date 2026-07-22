from __future__ import annotations

from datetime import timedelta
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="BTC 5-Minute Paper Lab",
    page_icon="🧪",
    layout="wide",
)


def get_secret(name: str) -> str:
    try:
        return str(st.secrets["supabase"][name]).strip()
    except Exception as exc:
        raise RuntimeError(
            "Missing Supabase settings. The existing dashboard secrets must contain "
            "[supabase] url and secret_key."
        ) from exc


@st.cache_data(ttl=60, show_spinner=False)
def fetch_table(
    table: str,
    *,
    select: str,
    since: pd.Timestamp,
    order: str,
    max_rows: int = 50_000,
) -> pd.DataFrame:
    url = get_secret("url").rstrip("/")
    key = get_secret("secret_key")
    endpoint = f"{url}/rest/v1/{table}"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }
    params = {
        "select": select,
        "bar_timestamp": f"gte.{since.isoformat()}",
        "order": order,
    }

    rows: list[dict[str, Any]] = []
    page_size = 1000
    for start in range(0, max_rows, page_size):
        page_headers = {
            **headers,
            "Range": f"{start}-{start + page_size - 1}",
        }
        response = requests.get(
            endpoint,
            params=params,
            headers=page_headers,
            timeout=20,
        )
        response.raise_for_status()
        batch: list[dict[str, Any]] = response.json()
        rows.extend(batch)
        if len(batch) < page_size:
            break

    frame = pd.DataFrame(rows)
    if not frame.empty and "bar_timestamp" in frame:
        frame["bar_timestamp"] = pd.to_datetime(
            frame["bar_timestamp"],
            utc=True,
            errors="coerce",
        )
    return frame


def normalize_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    numeric_columns = [
        "signal_close",
        "mark_price",
        "bid",
        "ask",
        "target_position",
        "cash",
        "btc",
        "equity",
        "gross_equity",
        "cost_drag",
        "return_pct",
        "gross_return_pct",
        "drawdown",
        "peak_equity",
        "total_fees",
        "total_spread",
        "total_slippage",
        "total_turnover",
        "trade_count",
        "bars_processed",
        "streak_direction",
        "streak_length",
        "run_return_bps",
        "relative_volume",
        "body_fraction",
        "regime_spread_bps",
        "momentum_bps",
    ]
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["bar_timestamp", "strategy_id"])


def normalize_trades(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    numeric_columns = [
        "btc_delta",
        "mark_price",
        "reference_price",
        "fill_price",
        "gross_notional",
        "fee",
        "spread_cost",
        "slippage_cost",
        "cash_after",
        "btc_after",
        "equity_after",
    ]
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("bar_timestamp", ascending=False)


def percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.2%}"


st.title("🧪 BTC 5-Minute Paper Lab")
st.caption(
    "Independent simulated portfolios. No orders are submitted to Coinbase. "
    "Net equity includes estimated fees, spread, and slippage; gross equity does not."
)

with st.sidebar:
    st.header("Paper-lab controls")
    window = st.selectbox(
        "History window",
        ["24 hours", "7 days", "14 days", "30 days", "90 days"],
        index=2,
    )
    if st.button("Refresh paper data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Supabase snapshots are sampled every 15 minutes and on every trade.")

window_days = {
    "24 hours": 1,
    "7 days": 7,
    "14 days": 14,
    "30 days": 30,
    "90 days": 90,
}
cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days[window])

snapshot_select = (
    "strategy_id,bar_timestamp,quote_timestamp,signal_close,mark_price,bid,ask,"
    "signal,reason,target_position,cash,btc,equity,gross_equity,cost_drag,"
    "return_pct,gross_return_pct,drawdown,peak_equity,total_fees,total_spread,"
    "total_slippage,total_turnover,trade_count,bars_processed,streak_direction,"
    "streak_length,run_return_bps,relative_volume,body_fraction,broader_trend_up,"
    "regime_spread_bps,momentum_bps"
)
trade_select = (
    "strategy_id,bar_timestamp,quote_timestamp,side,btc_delta,mark_price,"
    "reference_price,fill_price,gross_notional,fee,spread_cost,slippage_cost,"
    "cash_after,btc_after,equity_after"
)

try:
    snapshots = normalize_snapshots(
        fetch_table(
            "paper_portfolio_snapshots",
            select=snapshot_select,
            since=cutoff,
            order="bar_timestamp.asc",
        )
    )
    trades = normalize_trades(
        fetch_table(
            "paper_trades",
            select=trade_select,
            since=cutoff,
            order="bar_timestamp.desc",
            max_rows=10_000,
        )
    )
except Exception as exc:
    st.error(f"Unable to load paper-lab data: {exc}")
    st.stop()

if snapshots.empty:
    st.info("No paper snapshots are available yet. Run btc-paper-5m.service once.")
    st.stop()

all_strategies = sorted(snapshots["strategy_id"].dropna().unique().tolist())
selected = st.sidebar.multiselect(
    "Strategies",
    all_strategies,
    default=all_strategies,
)
if not selected:
    st.warning("Select at least one strategy.")
    st.stop()

filtered = snapshots[snapshots["strategy_id"].isin(selected)].copy()
latest = (
    filtered.sort_values("bar_timestamp")
    .groupby("strategy_id", as_index=False)
    .tail(1)
    .sort_values("strategy_id")
)

st.subheader("Latest simulated portfolios")
card_columns = st.columns(min(5, max(1, len(latest))))
for column, (_, row) in zip(card_columns, latest.iterrows()):
    with column:
        st.metric(
            str(row["strategy_id"]),
            f"${row['equity']:,.2f}",
            percent(row["return_pct"]),
        )
        st.caption(
            f"Trades {int(row['trade_count'])} · "
            f"Drawdown {percent(row['drawdown'])} · "
            f"Costs ${row['cost_drag']:,.2f}"
        )

st.subheader("Net portfolio value")
net_chart = (
    alt.Chart(filtered)
    .mark_line()
    .encode(
        x=alt.X("bar_timestamp:T", title="UTC time"),
        y=alt.Y("equity:Q", title="Simulated portfolio value (USD)", scale=alt.Scale(zero=False)),
        color=alt.Color("strategy_id:N", title="Strategy"),
        tooltip=[
            alt.Tooltip("bar_timestamp:T", title="Time"),
            alt.Tooltip("strategy_id:N", title="Strategy"),
            alt.Tooltip("equity:Q", title="Net equity", format=",.2f"),
            alt.Tooltip("return_pct:Q", title="Return", format=".2%"),
            alt.Tooltip("trade_count:Q", title="Trades", format=".0f"),
        ],
    )
    .properties(height=380)
    .interactive()
)
st.altair_chart(net_chart, use_container_width=True)

st.subheader("Transaction-cost damage")
cost_chart = (
    alt.Chart(filtered)
    .mark_line()
    .encode(
        x=alt.X("bar_timestamp:T", title="UTC time"),
        y=alt.Y("cost_drag:Q", title="Gross equity minus net equity (USD)"),
        color=alt.Color("strategy_id:N", title="Strategy"),
        tooltip=[
            alt.Tooltip("bar_timestamp:T", title="Time"),
            alt.Tooltip("strategy_id:N", title="Strategy"),
            alt.Tooltip("cost_drag:Q", title="Cost drag", format=",.2f"),
            alt.Tooltip("total_fees:Q", title="Fees", format=",.2f"),
            alt.Tooltip("total_spread:Q", title="Spread", format=",.2f"),
            alt.Tooltip("total_slippage:Q", title="Slippage", format=",.2f"),
        ],
    )
    .properties(height=300)
    .interactive()
)
st.altair_chart(cost_chart, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Drawdown")
    drawdown_chart = (
        alt.Chart(filtered)
        .mark_line()
        .encode(
            x=alt.X("bar_timestamp:T", title="UTC time"),
            y=alt.Y("drawdown:Q", title="Drawdown", axis=alt.Axis(format="%")),
            color=alt.Color("strategy_id:N", title="Strategy"),
        )
        .properties(height=280)
        .interactive()
    )
    st.altair_chart(drawdown_chart, use_container_width=True)

with right:
    st.subheader("Target BTC exposure")
    exposure_chart = (
        alt.Chart(filtered)
        .mark_line(interpolate="step-after")
        .encode(
            x=alt.X("bar_timestamp:T", title="UTC time"),
            y=alt.Y(
                "target_position:Q",
                title="Target exposure",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color("strategy_id:N", title="Strategy"),
        )
        .properties(height=280)
        .interactive()
    )
    st.altair_chart(exposure_chart, use_container_width=True)

st.subheader("Strategy comparison")
comparison = latest[
    [
        "strategy_id",
        "bar_timestamp",
        "equity",
        "gross_equity",
        "return_pct",
        "gross_return_pct",
        "drawdown",
        "trade_count",
        "total_fees",
        "total_spread",
        "total_slippage",
        "total_turnover",
        "cost_drag",
        "target_position",
        "regime_spread_bps",
        "momentum_bps",
        "signal",
        "reason",
    ]
].copy()
for column in ("return_pct", "gross_return_pct", "drawdown", "target_position"):
    comparison[column] = comparison[column] * 100.0
st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True,
    column_config={
        "return_pct": st.column_config.NumberColumn(format="%.2f%%"),
        "gross_return_pct": st.column_config.NumberColumn(format="%.2f%%"),
        "drawdown": st.column_config.NumberColumn(format="%.2f%%"),
        "target_position": st.column_config.NumberColumn(format="%.0f%%"),
        "equity": st.column_config.NumberColumn(format="$%.2f"),
        "gross_equity": st.column_config.NumberColumn(format="$%.2f"),
        "total_fees": st.column_config.NumberColumn(format="$%.2f"),
        "total_spread": st.column_config.NumberColumn(format="$%.2f"),
        "total_slippage": st.column_config.NumberColumn(format="$%.2f"),
        "cost_drag": st.column_config.NumberColumn(format="$%.2f"),
        "regime_spread_bps": st.column_config.NumberColumn(format="%.1f bps"),
        "momentum_bps": st.column_config.NumberColumn(format="%.1f bps"),
    },
)

st.subheader("Recent simulated trades")
selected_trades = trades[trades["strategy_id"].isin(selected)] if not trades.empty else trades
if selected_trades.empty:
    st.caption("No simulated trades in the selected window.")
else:
    st.dataframe(selected_trades.head(200), use_container_width=True, hide_index=True)

st.download_button(
    "Download selected snapshots CSV",
    filtered.to_csv(index=False).encode("utf-8"),
    file_name="btc_5m_paper_snapshots.csv",
    mime="text/csv",
)
