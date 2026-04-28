import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from user_config import USERNAMES

IST = ZoneInfo("Asia/Kolkata")

DEFAULT_USERS = USERNAMES
BASE_DIR = Path(st.sidebar.text_input("Strategy TS base dir", r"/mnt/Quant_Research/Risk_dashboard_inputs/strategy_ts"))
user = st.sidebar.selectbox("User", DEFAULT_USERS, index=0)

st.sidebar.header("Refresh")
do_auto = st.sidebar.checkbox("Auto-refresh", value=True)
every_s = st.sidebar.number_input("Every (seconds)", min_value=1, value=30, step=1)
if do_auto:
    st_autorefresh(interval=int(every_s * 1000), key="strat_tag_autorefresh")

day = st.sidebar.date_input("Date", value=datetime.now(IST).date())
yyyymmdd = day.strftime("%Y%m%d")

# locate dataset folder
read_path = BASE_DIR / f"date={yyyymmdd}" / f"user={user}"
if not read_path.exists():
    st.error(f"No strategy_ts parquet folder found:\n{read_path}")
    st.stop()

# ── Safe parquet loader: skips corrupt / zero-byte / partial files ──────────
def safe_read_parquet_dir(folder: Path) -> tuple[pd.DataFrame, list[str]]:
    files = sorted(folder.glob("*.parquet"))
    if not files:
        return pd.DataFrame(), []

    frames, skipped = [], []
    for f in files:
        if f.stat().st_size < 12:
            skipped.append(f.name + "  [empty/tiny]")
            continue
        try:
            with open(f, "rb") as fh:
                header = fh.read(4)
                fh.seek(-4, 2)
                footer = fh.read(4)
            if header != b"PAR1" or footer != b"PAR1":
                skipped.append(f.name + "  [bad magic bytes — likely partial write]")
                continue
        except Exception as e:
            skipped.append(f.name + f"  [unreadable: {e}]")
            continue
        try:
            frames.append(pd.read_parquet(f, engine="pyarrow"))
        except Exception as e:
            skipped.append(f.name + f"  [pyarrow error: {e}]")

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, skipped


df, skipped_files = safe_read_parquet_dir(read_path)

if skipped_files:
    with st.expander(f"⚠️ {len(skipped_files)} corrupt/partial file(s) skipped", expanded=False):
        for s in skipped_files:
            st.code(s)

if df.empty:
    st.info("No intraday points yet.")
    st.stop()

# normalize
df["ts"] = pd.to_datetime(df["ts_utc"], utc=True).dt.tz_convert("Asia/Kolkata")
for c in ["DayPnL", "SimDayPnL", "NetPnL", "SimCarryPnL", "allocated_margin"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df["tag"] = df["tag"].astype(str).fillna("__NO_TAG__")

st.title("📉 Strategy plots — Live DayPnL vs Mock SimDayPnL")
st.caption(f"{user} • {yyyymmdd} • points={len(df):,} • tags={df['tag'].nunique():,}")

# controls
all_tags = sorted(df["tag"].unique().tolist())
mode = st.sidebar.radio("View mode", ["Select one tag", "Show many tags"], index=0)

if mode == "Select one tag":
    tag = st.selectbox("Strategy tag", all_tags, index=0)
    sub = df[df["tag"] == tag].sort_values("ts")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["ts"], y=sub["DayPnL"], mode="lines", name="Live DayPnL"))
    fig.add_trace(go.Scatter(x=sub["ts"], y=sub["SimDayPnL"], mode="lines", name="Mock SimDayPnL"))
    fig.add_trace(go.Scatter(x=sub["ts"], y=(sub["DayPnL"] - sub["SimDayPnL"]), mode="lines", name="Live - Mock"))

    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), title=f"{tag}")
    st.plotly_chart(fig, use_container_width=True)

else:
    # Showing ALL tags can be heavy; default to top N by last |NetPnL|
    st.info("Showing many tags can be heavy if you have lots of strategies. Use the Top-N slider.")

    # compute last point per tag
    last_by_tag = (
        df.sort_values("ts")
          .groupby("tag", as_index=False)
          .tail(1)
    )
    last_by_tag["abs_net"] = last_by_tag.get("NetPnL", 0.0).abs()
    last_by_tag = last_by_tag.sort_values("abs_net", ascending=False)

    top_n = st.slider("Top N strategies (by |last NetPnL|)", min_value=5, max_value=min(100, len(last_by_tag)), value=min(20, len(last_by_tag)))
    tags = last_by_tag["tag"].head(top_n).tolist()

    for tag in tags:
        sub = df[df["tag"] == tag].sort_values("ts")
        with st.expander(tag, expanded=False):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sub["ts"], y=sub["DayPnL"], mode="lines", name="Live DayPnL"))
            fig.add_trace(go.Scatter(x=sub["ts"], y=sub["SimDayPnL"], mode="lines", name="Mock SimDayPnL"))
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

cols_to_use = ["tag","DayPnL","SimDayPnL","NetPnL","allocated_margin",'ts']
with st.expander("Raw tail", expanded=False):
    st.dataframe(df[cols_to_use].sort_values("ts").tail(200), use_container_width=True)