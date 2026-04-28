#strategy_pnl.py
import os
import json
from datetime import datetime, time
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st
import redis
from streamlit_autorefresh import st_autorefresh
from user_config import USERNAMES

# 👇 Import your performance dashboard component
from components.performance_dashboard import render_performance_dashboard, PerfConfig


# Auto-refresh (2s)
st_autorefresh(interval=3000, key="strategy_pnl_refresh")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PWD = os.getenv("REDIS_PASSWORD", "") or None

IST = ZoneInfo("Asia/Kolkata")

EOD_SAVE_TIME = time(15, 35)  # 15:35 IST
# STRATEGY_EOD_DIR = os.getenv("STRATEGY_PNL_EOD_DIR", r"Y:\daily_task\daily_strategy_pnl_snaps")
STRATEGY_EOD_DIR = os.getenv("STRATEGY_PNL_EOD_DIR", r"/mnt/Quant_Research/daily_task/daily_strategy_pnl_snaps")
EOD_FLAG_TTL_SEC = 10 * 24 * 3600  # keep flags for 10 days

USERS = USERNAMES + ["__ALL__"]
ACCOUNTS = USERNAMES

# filter out positional trades
PAT_POSITIONAL = r":\d{1,2}[A-Za-z]{3}\d{2}$"


def redis_client() -> redis.Redis:
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PWD,
        decode_responses=True,
        socket_timeout=2.0,
        socket_connect_timeout=2.0,
    )


def _safe_sum(df: pd.DataFrame, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return 0.0
    s = pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum()
    try:
        return float(s)
    except Exception:
        return 0.0


def _coerce_numeric_keep_text(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns while keeping tag/times/account as text."""
    if df is None or df.empty:
        return df
    df = df.copy()
    for c in df.columns:
        if c not in ("tag", "MinTime", "MaxTime", "account"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _filter_positional(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "tag" not in df.columns:
        return df
    return df[~df["tag"].astype(str).str.contains(PAT_POSITIONAL, case=False, regex=True, na=False)].copy()


def _load_one_snapshot_df(r: redis.Redis, acct: str) -> tuple[dict | None, pd.DataFrame]:
    """Load one account snapshot -> (obj, cleaned_df). If missing/bad, returns (None, empty_df)."""
    raw = r.get(f"risk:strategy_pnl:latest:{acct}")
    if not raw:
        return None, pd.DataFrame()

    try:
        obj = json.loads(raw)
    except Exception:
        return None, pd.DataFrame()

    rows = obj.get("rows") or []
    df = pd.DataFrame(rows)
    if df.empty:
        return obj, df

    df["account"] = acct
    df = _filter_positional(df)
    df = _coerce_numeric_keep_text(df)
    return obj, df


def _load_combined_snapshots_df(r: redis.Redis) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads all accounts (excluding __ALL__) and concatenates.
    Returns:
      - combined_df
      - status_df columns: account, username, as_of, status
    """
    keys = [f"risk:strategy_pnl:latest:{acct}" for acct in ACCOUNTS]
    pipe = r.pipeline()
    for k in keys:
        pipe.get(k)
    raws = pipe.execute()

    dfs: list[pd.DataFrame] = []
    status_rows: list[dict] = []

    for acct, raw in zip(ACCOUNTS, raws):
        if not raw:
            status_rows.append({"account": acct, "username": "", "as_of": "", "status": "MISSING"})
            continue

        try:
            obj = json.loads(raw)
        except Exception:
            status_rows.append({"account": acct, "username": "", "as_of": "", "status": "BAD_JSON"})
            continue

        username = obj.get("username", "") or obj.get("user", "") or acct
        as_of = obj.get("as_of", "")

        rows = obj.get("rows") or []
        df_i = pd.DataFrame(rows)

        if df_i.empty:
            status_rows.append({"account": acct, "username": username, "as_of": as_of, "status": "EMPTY"})
            continue

        df_i["account"] = acct
        df_i = _filter_positional(df_i)
        df_i = _coerce_numeric_keep_text(df_i)

        if df_i.empty:
            status_rows.append({"account": acct, "username": username, "as_of": as_of, "status": "FILTERED_EMPTY"})
            continue

        dfs.append(df_i)
        status_rows.append({"account": acct, "username": username, "as_of": as_of, "status": "OK"})

    combined = pd.concat(dfs, ignore_index=True, sort=False) if dfs else pd.DataFrame()
    status_df = pd.DataFrame(status_rows)
    return combined, status_df


def try_save_strategy_table_eod(
    r: redis.Redis,
    user: str,
    df_out: pd.DataFrame,
    cols_out: list[str] | None = None,
) -> tuple[bool, str]:
    """
    Saves once per day after 15:35 IST.
    Uses Redis SET NX as a lock/flag so auto-refresh won't resave repeatedly.
    Returns: (saved_now, filepath_or_reason)
    """
    now = datetime.now(IST)
    if now.time() < EOD_SAVE_TIME:
        return False, "Not EOD yet"

    yyyymmdd = now.strftime("%Y%m%d")
    flag_key = f"risk:strategy_pnl:eod_saved:{user}:{yyyymmdd}"

    did_set = r.set(flag_key, now.isoformat(timespec="seconds"), nx=True, ex=EOD_FLAG_TTL_SEC)
    if not did_set:
        return False, "Already saved"

    os.makedirs(STRATEGY_EOD_DIR, exist_ok=True)
    safe_user = str(user).replace("/", "_").replace("\\", "_").replace(":", "_")
    path = os.path.join(STRATEGY_EOD_DIR, f"strategy_table_{safe_user}_{yyyymmdd}.csv")

    out = df_out.copy()
    if cols_out:
        cols_out = [c for c in cols_out if c in out.columns]
        if cols_out:
            out = out[cols_out].copy()

    for c in ["MinTime", "MaxTime"]:
        if c in out.columns:
            out[c] = out[c].astype(str)

    out.to_csv(path, index=False)
    return True, path

def try_save_strategy_table_eod_all_accounts(
    r: redis.Redis,
    cols_out: list[str] | None = None,
) -> list[dict]:
    """
    Saves EOD strategy tables for ALL ACCOUNTS (DMA16/DMA19/...) once per day.
    Uses per-user NX flags inside try_save_strategy_table_eod, so it's safe to call every refresh.

    Returns list of dicts: [{"account":..., "saved":..., "info":...}, ...]
    """
    results: list[dict] = []

    for acct in ACCOUNTS:
        obj_i, df_i = _load_one_snapshot_df(r, acct)
        if df_i is None or df_i.empty:
            results.append({"account": acct, "saved": False, "info": "Snapshot empty/missing"})
            continue

        saved_now, info = try_save_strategy_table_eod(r, acct, df_i, cols_out=cols_out)
        results.append({"account": acct, "saved": bool(saved_now), "info": info})

    return results

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("Strategy PnL")

user = st.sidebar.selectbox("User", USERS, index=0, key="selected_user")
tag_filter = st.sidebar.text_input("Filter tag (contains)", value="", key="strategy_tag_filter").strip()
only_negative = st.sidebar.checkbox("Only negative NetPnL", value=False, key="strategy_only_negative")

# ----------------------------
# Load Data
# ----------------------------
r = redis_client()
status_df = None

if user == "__ALL__":
    df, status_df = _load_combined_snapshots_df(r)

    st.title("Strategy Level PnL")
    st.caption("User: __ALL__ (showing per-account slippage metrics)")

    if status_df is not None and not status_df.empty:
        with st.expander("Snapshot status (per account)", expanded=False):
            st.dataframe(status_df, use_container_width=True)

    if df.empty:
        st.warning("No strategy PnL snapshots found for any account (or all are empty/filtered).")
        st.stop()

else:
    obj, df = _load_one_snapshot_df(r, user)
    if obj is None or df.empty:
        st.warning(
            f"No strategy PnL snapshot found for {user}. "
            f"Ensure strategy_pnl_worker.py is running and writing risk:strategy_pnl:latest:{user}."
        )
        st.stop()

    st.title("Strategy Level PnL")
    st.caption(f"User: {obj.get('username')} | As of: {obj.get('as_of')}")

# ----------------------------
# Apply Filters (these affect table + slippage metrics)
# ----------------------------
if df.empty:
    st.info("No rows in snapshot.")
    st.stop()

if tag_filter and "tag" in df.columns:
    df = df[df["tag"].astype(str).str.contains(tag_filter, case=False, na=False)]

if only_negative and "NetPnL" in df.columns:
    df = df[df["NetPnL"] < 0]

if df.empty:
    st.info("No rows after filters.")
    st.stop()

# Sort
if "NetPnL" in df.columns:
    df = df.sort_values("NetPnL", ascending=True)

# ----------------------------
# Metrics
# ----------------------------
if user == "__ALL__":
    st.subheader("Slippage metrics by account")

    # map account -> username/as_of from status_df
    user_map = {}
    asof_map = {}
    if status_df is not None and not status_df.empty:
        for _, row in status_df.iterrows():
            user_map[str(row.get("account", ""))] = str(row.get("username", "")) or str(row.get("account", ""))
            asof_map[str(row.get("account", ""))] = str(row.get("as_of", ""))

    # build summary table too (handy for copy/paste)
    summary_rows = []

    for acct in ACCOUNTS:
        sub = df[df.get("account", "").astype(str) == acct] if "account" in df.columns else pd.DataFrame()
        if sub.empty:
            continue

        day = _safe_sum(sub, "DayPnL")
        sim = _safe_sum(sub, "SimDayPnL")
        slip = day - sim
        margin = _safe_sum(sub, "allocated_margin")
        slip_pct = (slip / abs(day) * 100.0) if abs(day) > 1e-9 else 0.0
        slip_margin_pct = (slip / margin * 100.0) if margin else 0.0
        net = _safe_sum(sub, "NetPnL")

        username = user_map.get(acct, acct)
        as_of = asof_map.get(acct, "")

        st.markdown(f"#### {acct} — {username}" + (f"  \n_As of: {as_of}_" if as_of else ""))
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("NetPnL", f"{net:,.0f}")
        c2.metric("DayPnL", f"{day:,.0f}")
        c3.metric("SimDayPnL", f"{sim:,.0f}")
        c4.metric("Slippage", f"{slip:,.0f}")
        c5.metric("Slip %", f"{slip_pct:.3f}")
        c6.metric("Slip wrt Margin %", f"{slip_margin_pct:.3f}")

        summary_rows.append(
            {
                "account": acct,
                "username": username,
                "as_of": as_of,
                "NetPnL": net,
                "DayPnL": day,
                "SimDayPnL": sim,
                "Slippage": slip,
                "Slip %": slip_pct,
                "Margin": margin,
                "Slip wrt Margin %": slip_margin_pct,
                "Rows": len(sub),
            }
        )

    if summary_rows:
        with st.expander("Per-account slippage summary table", expanded=False):
            s = pd.DataFrame(summary_rows)
            # sort by Slippage (worst first)
            if "Slippage" in s.columns:
                s = s.sort_values("Slippage", ascending=True)
            st.dataframe(s, use_container_width=True)

else:
    # Single-user totals (original behavior)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total NetPnL", f"{_safe_sum(df, 'NetPnL'):,.0f}")
    c2.metric("Total DayPnL", f"{_safe_sum(df, 'DayPnL'):,.0f}")
    c3.metric("Total CarryPnL", f"{_safe_sum(df, 'CarryPnL'):,.0f}")
    c4.metric("Total Alloc Margin", f"{_safe_sum(df, 'allocated_margin'):,.0f}")

    total_slip = _safe_sum(df, "DayPnL") - _safe_sum(df, "SimDayPnL")
    den = _safe_sum(df, "DayPnL")
    slip_percent = (total_slip / abs(den) * 100.0) if abs(den) > 1e-9 else 0.0
    total_margin = _safe_sum(df, "allocated_margin")
    slip_margin_pct = (total_slip / total_margin * 100.0) if total_margin else 0.0

    st.subheader("slip summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total slippage", f"{total_slip:,.0f}")
    c2.metric("slip_percent %", f"{slip_percent:.3f}")
    c3.metric("slippage wrt margin", f"{slip_margin_pct:.3f} %")

    st.caption(
        """
Total Slippage = Total Day PnL - Total Sim Day PnL,
Slip % = (Total Slip / abs(Total Day PnL)) * 100,
Slippage w.r.t Margin = (Total Slip / Total Margin) * 100
"""
    )

# ----------------------------
# Strategy Table
# ----------------------------
st.subheader("Strategy table")

# add a column which represents the sim and live difference
if "DayPnL" in df.columns and "SimDayPnL" in df.columns:
    df["slip"] = pd.to_numeric(df["DayPnL"], errors="coerce").fillna(0.0) - pd.to_numeric(
        df["SimDayPnL"], errors="coerce"
    ).fillna(0.0)
else:
    df["slip"] = 0.0

wanted = [
    "account",  # will show for __ALL__
    "tag",
    "slip",
    "CarryPnL",
    "DayPnL",
    "Expenses",
    "NetPnL",
    "allocated_margin",
    "net_pnl/margin (%)",
    "MinPnL",
    "MinTime",
    "MaxPnL",
    "MaxTime",
    "SimCarryPnL",
    "SimDayPnL",
    "Slippage (%)",
]
cols = [c for c in wanted if c in df.columns]
st.dataframe(round(df[cols], 2) if cols else df, use_container_width=True, height=400)

# ---- EOD one-time save (after 15:35 IST) ----
# Save for ALL accounts (DMA16/DMA19/DMA09/DMA20) once per day.
# NOTE: This saves each account's own snapshot table (not the UI-filtered view).
eod_results = try_save_strategy_table_eod_all_accounts(r, cols_out=cols if cols else None)

# Show only when something actually saved now (avoid spam on refresh)
saved_now_rows = [x for x in eod_results if x.get("saved")]
if saved_now_rows:
    with st.expander("EOD saves (saved now)", expanded=True):
        st.dataframe(pd.DataFrame(saved_now_rows), use_container_width=True)
        st.success(f"EOD saved for {len(saved_now_rows)} account(s).")

# ============================================================
# Performance Dashboard
# - show only for single user (NOT for __ALL__)
# ============================================================
if user != "__ALL__":
    st.divider()
    st.header("Performance Dashboard")
    cfg = PerfConfig()
    render_performance_dashboard(cfg, show_login=False, debug=True)


st.markdown("---")
st.subheader("Strategy payoff (expiry-style)")

pay_raw = r.get(f"risk:strategy_payoff:latest:{user}")
if not pay_raw:
    st.info("No payoff published yet. Ensure strategy_pnl_worker is publishing risk:strategy_payoff:latest:<user>.")
else:
    try:
        pay = json.loads(pay_raw)
    except Exception:
        st.warning("Payoff payload is not valid JSON.")
        pay = {}

    pdata = (pay.get("data") or {})
    if not pdata:
        st.info("Payoff data is empty (no open positions or no spot available).")
    else:
        tags = sorted(pdata.keys())

        # ---- Persist selection across refresh (per-user keys) ----
        tag_key = f"payoff_tag_sel::{user}"
        ul_key  = f"payoff_ul_sel::{user}"

        prev_tag = st.session_state.get(tag_key)
        tag_index = tags.index(prev_tag) if (prev_tag in tags) else 0
        sel_tag = st.selectbox("Tag", tags, index=tag_index, key=tag_key)

        uls = sorted((pdata.get(sel_tag) or {}).keys())
        if not uls:
            st.info("No underlying payoff curves for this tag.")
        else:
            prev_ul = st.session_state.get(ul_key)
            ul_index = uls.index(prev_ul) if (prev_ul in uls) else 0
            sel_ul = st.selectbox("Underlying", uls, index=ul_index, key=ul_key)

            curve = pdata.get(sel_tag, {}).get(sel_ul, {})
            S = curve.get("S", [])
            pnl = curve.get("pnl", [])
            S0 = curve.get("S0", None)

            if not S or not pnl:
                st.info("Curve is empty.")
            else:
                # ---- Plot payoff with 0-line + green fill above 0 ----
                df_curve = pd.DataFrame({"S": S, "PnL": pnl}).sort_values("S")
                st.caption(f"S0: {S0} | points: {len(df_curve)}")

                x = df_curve["S"].to_numpy(dtype=float)
                y = df_curve["PnL"].to_numpy(dtype=float)

                fig = go.Figure()

                # Green filled area where PnL > 0
                y_pos = np.maximum(y, 0.0)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([y_pos, np.zeros_like(y_pos)[::-1]]),
                    fill="toself",
                    fillcolor="rgba(34,197,94,0.25)",  # green with alpha
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Profit area",
                ))

                # Payoff line
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color="black", width=2),
                    name="Payoff",
                    hovertemplate="Spot: %{x}<br>P&L: %{y}<extra></extra>",
                ))

                # Solid zero line
                fig.add_hline(y=0, line_width=2, line_dash="solid", line_color="gray")

                # Optional: show S0 marker if available
                if S0 is not None:
                    try:
                        s0f = float(S0)
                        if np.isfinite(s0f):
                            fig.add_vline(x=s0f, line_width=1, line_dash="dash", line_color="orange")
                    except Exception:
                        pass

                fig.update_layout(
                    height=380,
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="Spot",
                    yaxis_title="PnL (₹)",
                    showlegend=False,
                )

                st.plotly_chart(fig, use_container_width=True)


