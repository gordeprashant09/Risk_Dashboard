import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
from user_config import USERNAMES

IST = ZoneInfo("Asia/Kolkata")

# Force white background so chart numbers/axis labels are visible
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: white !important; }
    [data-testid="stHeader"] { background-color: white !important; }
    .stPlotlyChart { background-color: white !important; }
    div[class*="block-container"] { background-color: white !important; color: #111111 !important; }
    p, span, label { color: #111111 !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar
# -------------------------
# BASE_DIR = Path(st.sidebar.text_input("Parquet base dir", r"Y:\Risk_dashboard_inputs"))
BASE_DIR = Path(st.sidebar.text_input("Parquet base dir", r"/mnt/Quant_Research/Risk_dashboard_inputs"))
user = st.sidebar.selectbox("User", USERNAMES + ["__ALL__"], index=0)

index_choice = st.sidebar.selectbox(
    "Index to overlay",
    ["None", "NIFTY", "BANKNIFTY", "SENSEX"],   # ✅ disable option
    index=1,  # default NIFTY (change to 0 if you want default None)
    help="Uses parquet columns: nifty_s / bn_s / sx_s. Choose None to disable overlay.",
)

IDX_COL_MAP = {
    "NIFTY": "nifty_s",
    "BANKNIFTY": "bn_s",
    "SENSEX": "sx_s",
}
idx_col = IDX_COL_MAP.get(index_choice)  # None if disabled

today = datetime.now(IST).strftime("%Y%m%d")

date_dir = BASE_DIR / f"date={today}"
user_dir1 = date_dir / f"user={user}"   # hive-style partition
user_dir2 = date_dir / user            # plain folder name

if user_dir1.exists():
    read_path = user_dir1
elif user_dir2.exists():
    read_path = user_dir2
else:
    st.error(
        f"No parquet folder found for {user} on {today}.\n"
        f"Tried:\n- {user_dir1}\n- {user_dir2}"
    )
    st.stop()

# ── Safe parquet loader: skips corrupt / zero-byte / partial files ──────────
def safe_read_parquet_dir(folder: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Read every *.parquet file individually; skip any that raise an error.
    Returns (combined_df, list_of_skipped_filenames).
    """
    files = sorted(folder.glob("*.parquet"))
    if not files:
        return pd.DataFrame(), []

    frames, skipped = [], []
    for f in files:
        # Guard 1: zero-byte or suspiciously tiny file
        if f.stat().st_size < 12:
            skipped.append(f.name + "  [empty/tiny]")
            continue
        # Guard 2: check magic bytes (PAR1) without loading the whole file
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
        # Guard 3: actually parse it
        try:
            frames.append(pd.read_parquet(f, engine="pyarrow"))
        except Exception as e:
            skipped.append(f.name + f"  [pyarrow error: {e}]")

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, skipped


df, skipped_files = safe_read_parquet_dir(read_path)

if skipped_files:
    with st.expander(f"⚠️ {len(skipped_files)} corrupt/partial file(s) skipped — click to see", expanded=False):
        for s in skipped_files:
            st.code(s)

if df.empty:
    st.info("No intraday parquet points yet for today.")
    st.stop()

# -------------------------
# Cleanup / time
# -------------------------
sort_col = "ts_utc" if "ts_utc" in df.columns else ("ts" if "ts" in df.columns else None)
if sort_col is None:
    st.error("Expected a timestamp column 'ts_utc' (preferred) or 'ts' in parquet.")
    st.stop()

df = df.sort_values(sort_col)

# Build tz-aware IST timestamp column
if sort_col == "ts_utc":
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce").dt.tz_convert(IST)
else:
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    if getattr(df["ts"].dt, "tz", None) is None:
        df["ts"] = df["ts"].dt.tz_localize(IST)

df = df.dropna(subset=["ts"])

# tz-naive IST for cleaner hover labels (no +05:30)
ts_plot = df["ts"].dt.tz_convert(IST).dt.tz_localize(None)

# -------------------------
# Series
# -------------------------
pnl = pd.to_numeric(df.get("pnl"), errors="coerce").ffill().fillna(0.0)
margin = pd.to_numeric(df.get("margin_total"), errors="coerce").ffill().fillna(0.0)

# Index series (optional)
has_idx = False
idx_series = None
if idx_col is not None and idx_col in df.columns:
    idx_series = pd.to_numeric(df[idx_col], errors="coerce").ffill()
    has_idx = np.isfinite(idx_series.to_numpy()).any()
elif idx_col is not None:
    st.sidebar.warning(f"Column '{idx_col}' not found. Overlay will be skipped.")

peak = pnl.cummax()
dd = pnl - peak

# -------------------------
# Plot
# -------------------------
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.72, 0.28],
    specs=[[{"secondary_y": True}], [{}]],
)

# Row 1: PnL (y) — solid blue line like old dashboard
fig.add_trace(
    go.Scatter(
        x=ts_plot, y=pnl, mode="lines", name="PnL (₹)",
        line=dict(color="#1f77b4", width=1.5)
    ),
    row=1, col=1, secondary_y=False
)

# Row 1: Margin (y2) — dashed light blue
fig.add_trace(
    go.Scatter(
        x=ts_plot, y=margin, mode="lines", name="Margin (₹)",
        line=dict(dash="dash", color="#aec7e8", width=1.5)
    ),
    row=1, col=1, secondary_y=True
)

# Row 1: Index overlay — solid green
if has_idx:
    fig.add_trace(
        go.Scatter(
            x=ts_plot, y=idx_series, mode="lines", name=f"{index_choice} Spot",
            line=dict(dash="solid", width=1, color="#2ca02c")
        ),
        row=1, col=1, secondary_y=False
    )
    fig.data[-1].update(yaxis="y4")

# Row 2: Drawdown — pink fill like old dashboard
fig.add_trace(
    go.Scatter(
        x=ts_plot, y=dd, mode="lines", name="Drawdown (₹)",
        fill="tozeroy",
        line=dict(color="rgba(255,100,100,0.8)", width=1),
        fillcolor="rgba(255,150,150,0.4)"
    ),
    row=2, col=1
)

fig.update_yaxes(title_text="PnL (₹)", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="Margin (₹)", row=1, col=1, secondary_y=True)
fig.update_yaxes(title_text="Drawdown (₹)", row=2, col=1)

layout_updates = dict(
    height=650,
    margin=dict(l=10, r=10, t=40, b=10),
    title=f"{user} • Intraday PnL vs Margin" + (f" (+ {index_choice})" if has_idx else ""),
    # White background like old dashboard
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color="#111111", size=12),
    legend=dict(
        bgcolor="white",
        bordercolor="#cccccc",
        borderwidth=1,
        font=dict(color="#111111")
    ),
    xaxis=dict(
        gridcolor="#e0e0e0",
        linecolor="#aaaaaa",
        tickfont=dict(color="#111111"),
        title=dict(font=dict(color="#111111")),
        zerolinecolor="#aaaaaa",
    ),
    yaxis=dict(
        gridcolor="#e0e0e0",
        linecolor="#aaaaaa",
        tickfont=dict(color="#111111"),
        title=dict(font=dict(color="#111111")),
        zerolinecolor="#aaaaaa",
    ),
    xaxis2=dict(
        gridcolor="#e0e0e0",
        linecolor="#aaaaaa",
        tickfont=dict(color="#111111"),
        title=dict(font=dict(color="#111111")),
    ),
    yaxis3=dict(
        gridcolor="#e0e0e0",
        linecolor="#aaaaaa",
        tickfont=dict(color="#111111"),
        title=dict(font=dict(color="#111111")),
        zerolinecolor="#aaaaaa",
    ),
)

# Margin axis tweak
layout_updates["yaxis2"] = dict(
    title="Margin (₹)",
    overlaying="y",
    side="right",
    position=0.97,
)

# Add yaxis4 only if index exists
if has_idx:
    layout_updates["yaxis4"] = dict(
        title=f"{index_choice} Spot",
        overlaying="y",
        side="right",
        position=1.0,
        showgrid=False,
        anchor="x",
    )

fig.update_layout(**layout_updates)

st.plotly_chart(fig, config={"responsive": True}, use_container_width=True)

# -------------------------
# Debug / sanity checks
# -------------------------
with st.expander("Debug: axis mapping"):
    st.write("Row1 -> y/y2, Row2 -> y3, Index overlay -> y4 (if enabled)")
    # st.write("Layout y-axes present:", [k for k in fig.layout.keys() if k.startswith("yaxis")])

    trace_info = []
    for i, tr in enumerate(fig.data):
        trace_info.append({
            "i": i,
            "name": tr.name,
            "xaxis": getattr(tr, "xaxis", "x"),
            "yaxis": getattr(tr, "yaxis", "y"),
        })
    st.dataframe(pd.DataFrame(trace_info), use_container_width=True)

with st.expander("Raw tail"):
    st.dataframe(df.tail(200), use_container_width=True)

