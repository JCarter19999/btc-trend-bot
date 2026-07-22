import json
import streamlit as st
from btc_trend_bot.v1.config import load_v1_config
from btc_trend_bot.v1.storage import connect, get_meta

st.set_page_config(page_title="BTC v1 Operations", layout="wide")
st.title("BTC/USDT v1 Operations")
cfg = load_v1_config()
st.caption(cfg["system"]["strategy_version"])
conn = connect(cfg["storage"]["sqlite_path"])
cols = st.columns(4)
cols[0].metric("Starting equity", f"${cfg['system']['starting_equity_usdt']:.2f}")
cols[1].metric("Max allocation", f"{100 * cfg['risk']['max_allocation_fraction']:.0f}%")
cols[2].metric("Safety margin", f"{cfg['model']['safety_margin_bps']:.1f} bps")
cols[3].metric("New matured", get_meta(conn, "newly_matured_since_training", 0))
st.subheader("Recent scheduler checks")
rows = conn.execute("SELECT created_at,status,details_json FROM scheduler_runs ORDER BY id DESC LIMIT 25").fetchall()
st.dataframe([{"created_at": row[0], "status": row[1], **json.loads(row[2])} for row in rows], use_container_width=True)
st.subheader("Promotion audit")
rows = conn.execute("SELECT created_at,old_model_id,new_model_id,approved_by,reason FROM promotion_events ORDER BY id DESC LIMIT 25").fetchall()
st.dataframe(rows, use_container_width=True)
