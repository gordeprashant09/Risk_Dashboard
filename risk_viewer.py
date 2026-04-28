# risk_viewer.py
"""
Streamlit viewer (lightweight):
- Reads precomputed risk snapshots from Redis
- No heavy compute (no IV/greeks/scenarios)

Reads:
- risk:outputs:latest:{username}  (JSON)
Optionally reads:
- margin:outputs:latest:{username} (already embedded in snapshot by worker)
"""

from __future__ import annotations

import os
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone, time

import numpy as np
import pandas as pd
import redis
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

# -------------------------
# CONFIG
# -------------------------

from user_config import USERNAMES   # auto-loaded from users.yaml

COMBINED_USER = "__ALL__"

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))
REDIS_PWD  = os.getenv("REDIS_PASSWORD", "") or None

IST = timezone(timedelta(hours=5, minutes=30))


# -------------------------
# HELPERS
# -------------------------

def _finite_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def _sum_scenario_dicts(dict_list):
    """
    dict_list: list[dict] where each dict is {shock_label: pnl_value}
    returns: aggregated dict {shock_label: sum(pnl_value)}
    """
    out = defaultdict(float)
    for d in dict_list:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            fv = _finite_float(v)
            if fv is None:
                continue
            out[str(k)] += fv
    return dict(out)

def _sort_shock_keys(keys, kind: str):
    """
    kind in {"spot","vol","time"}
    spot keys like "-10.0%", "1.0%"
    vol  keys like "+1%", "-2%"   (old)
    time keys like "T+1D", "T+3D"
    """
    keys = [str(k) for k in keys]

    if kind == "spot":
        def f(k):
            m = re.search(r"([-+]?\d+(\.\d+)?)\s*%", k)
            return float(m.group(1)) if m else 1e9
        return sorted(keys, key=f)

    if kind == "vol":
        def f(k):
            m = re.search(r"([-+]?\d+)\s*%", k)
            return int(m.group(1)) if m else 10**9
        return sorted(keys, key=f)

    # time
    def f(k):
        m = re.search(r"T\+(\d+)\s*D", k)
        return int(m.group(1)) if m else 10**9
    return sorted(keys, key=f)

def _parse_time_key(k: str) -> int | None:
    m = re.search(r"T\+(\d+)\s*D", str(k))
    return int(m.group(1)) if m else None

def _parse_vol_key(k: str) -> int | None:
    # worker publishes V+1 / V-2 etc.
    m = re.search(r"V\s*([+-]?\d+)", str(k))
    return int(m.group(1)) if m else None

def _nearest_int(x: int, choices: list[int]) -> int:
    if not choices:
        return x
    return min(choices, key=lambda c: abs(int(c) - int(x)))

def aggregate_scenarios_all_underlyings(scn: dict) -> dict:
    """
    scn format:
      scn[ul]["spot"] = {shock: value}
      scn[ul]["vol"]  = {shock: value}
      scn[ul]["time"] = {shock: value}
    returns:
      {"spot": {...}, "vol": {...}, "time": {...}}
    """
    spot_list = []
    vol_list  = []
    time_list = []

    for ul, packs in (scn or {}).items():
        if not isinstance(packs, dict):
            continue
        spot_list.append(packs.get("spot", {}) or {})
        vol_list.append(packs.get("vol", {}) or {})
        time_list.append(packs.get("time", {}) or {})  # ✅ fixed syntax bug

    return {
        "spot": _sum_scenario_dicts(spot_list),
        "vol":  _sum_scenario_dicts(vol_list),
        "time": _sum_scenario_dicts(time_list),
    }

def aggregate_combo_all_underlyings(scn: dict) -> dict:
    """
    TRUE combined cube aggregation:

    Input per UL:
      scn[ul]["combo"][T+XD][V+Y][spot_key] = pnl_rupees

    Output (aggregated across ULs):
      combo_agg[T+XD][V+Y][spot_key] = sum_pnl_rupees
    """
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for ul, packs in (scn or {}).items():
        if not isinstance(packs, dict):
            continue
        combo = packs.get("combo") or {}
        if not isinstance(combo, dict):
            continue

        for tkey, vold in combo.items():
            if not isinstance(vold, dict):
                continue
            for vkey, spotd in vold.items():
                if not isinstance(spotd, dict):
                    continue
                for skey, val in spotd.items():
                    fv = _finite_float(val)
                    if fv is None:
                        continue
                    out[str(tkey)][str(vkey)][str(skey)] += fv

    # convert nested defaultdicts -> dict
    return {t: {v: dict(sd) for v, sd in vd.items()} for t, vd in out.items()}

def records_to_df_payoff(x):
    """Accept list[dict] or dict and return DataFrame safely."""
    if x is None:
        return pd.DataFrame()
    if isinstance(x, list):
        return pd.DataFrame(x)
    if isinstance(x, dict):
        try:
            return pd.DataFrame(x)
        except Exception:
            return pd.DataFrame([x])
    return pd.DataFrame()

def redis_client(host, port, db, pwd):
    return redis.Redis(
        host=host, port=port, db=db, password=pwd,
        decode_responses=True,
        socket_timeout=2.0, socket_connect_timeout=2.0
    )

def read_snapshot(r: redis.Redis, username: str) -> dict | None:
    raw = r.get(f"risk:outputs:latest:{username}")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None

def records_to_df(obj: dict) -> pd.DataFrame:
    """
    Your worker packs tables using df_to_records(...).
    This function expects dict like:
      {"columns":[...], "data":[...]}.
    """
    if not obj:
        return pd.DataFrame()
    cols = obj.get("columns") or []
    data = obj.get("data") or []
    if not data:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(data)
    if cols and all(c in df.columns for c in cols):
        df = df[cols]
    return df

# -------------------------
# UI
# -------------------------

st.set_page_config(page_title="Risk Viewer (Fast)", layout="wide")
st.title("📈 Risk Viewer (Fast)")
st.caption("All calculations run in risk_worker.py. Viewer only reads Redis snapshots.")


# Page switch
view_mode = st.sidebar.radio("View", ["Account", "Combined (all accounts)"], index=0)
if view_mode == "Account":
    username = st.sidebar.selectbox("Username", USERNAMES, index=0)
else:
    username = COMBINED_USER
    st.sidebar.caption(f"Combined key: {COMBINED_USER}")

    
st.sidebar.header("Redis")
host = st.sidebar.text_input("Host", value=REDIS_HOST)
port = st.sidebar.number_input("Port", value=REDIS_PORT, step=1)
dbn  = st.sidebar.number_input("DB", value=REDIS_DB, step=1)
pwd  = st.sidebar.text_input("Password", value="" if REDIS_PWD is None else str(REDIS_PWD), type="password")

st.sidebar.header("Refresh")
do_auto = st.sidebar.checkbox("Auto-refresh", value=True)
every_s = st.sidebar.number_input("Every (seconds)", min_value=1, value=3, step=1)
if do_auto:
    st_autorefresh(interval=int(every_s * 1000), key="risk_viewer_autorefresh")


# connect
try:
    r = redis_client(host, int(port), int(dbn), pwd or None)
    r.ping()
except Exception as e:
    st.error(f"Redis connection failed: {e}")
    st.stop()

snap = read_snapshot(r, username)
if not snap:
    st.warning(f"No snapshot found for {username}. Is the worker running and publishing this key?")
    st.stop()

as_of = snap.get("as_of", "")
kpis = snap.get("kpis", {}) or {}
cfg  = snap.get("config", {}) or {}
mm   = snap.get("pnl_minmax", {}) or {}
margin = snap.get("margin", {}) or {}

# Header
title_name = "Combined Portfolio (All Accounts)" if view_mode != "Account" else username
st.subheader(f"{title_name} — Snapshot")
st.caption(f"as_of: {as_of} · version: {snap.get('version','')}")

if view_mode != "Account":
    users_included = snap.get("users_included") or []
    if users_included:
        st.caption(f"Users included: {', '.join(map(str, users_included))}")

# KPIs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Carry P&L (₹)", f"{kpis.get('carry_pnl',0):,.0f}")
c2.metric("Day P&L (₹)",   f"{kpis.get('day_pnl',0):,.0f}")
c3.metric("Expenses (₹)",  f"{kpis.get('expenses',0):,.0f}")
c4.metric("Net P&L (₹)",   f"{kpis.get('net_pnl',0):,.0f}")
c5.metric("Open legs",     f"{kpis.get('legs_open',0):,}")

st.caption(f"rf={cfg.get('rf')} · q_div={cfg.get('q_div')} · cutoff={cfg.get('cutoff')} · spot_mode={cfg.get('spot_mode')}")

# -------------------------
# TRUE Combined Shock Scenarios (TOP of dashboard)
# -------------------------

scn = snap.get("scenarios") or {}
# print(scn)
st.subheader("Shock Scenarios — Spot shocks (TRUE combined with Vol points + Time days)")

if not scn:
    st.info("No scenarios found in snapshot.")
else:
    combo_agg = aggregate_combo_all_underlyings(scn)

    if not combo_agg:
        st.warning("No TRUE combined scenarios found (missing scenarios[ul]['combo']). Enable combo publishing in risk_worker.py.")
        st.caption("You can still view the old separate spot/vol/time scenarios inside the Portfolio Greeks tab.")
    else:
        # available axes
        avail_times = sorted([t for t in (_parse_time_key(k) for k in combo_agg.keys()) if t is not None])
        # default to 0 if present else smallest
        default_time = 0 if 0 in avail_times else (avail_times[0] if avail_times else 0)

        cI1, cI2, cI3 = st.columns([1.2, 1.2, 2.6])
        with cI1:
            req_time = st.number_input(
                "Time shock (days)",
                value=int(st.session_state.get("combo_time_days", default_time)),
                step=1,
                min_value=0,
                key="combo_time_days",
            )

        # snap time to available
        sel_time = _nearest_int(int(req_time), avail_times) if avail_times else int(req_time)
        tkey = f"T+{int(sel_time)}D"

        avail_vols_for_time = []
        if tkey in combo_agg:
            avail_vols_for_time = sorted([v for v in (_parse_vol_key(k) for k in combo_agg[tkey].keys()) if v is not None])

        default_vol = 0 if 0 in avail_vols_for_time else (avail_vols_for_time[0] if avail_vols_for_time else 0)

        with cI2:
            req_vol = st.number_input(
                "Vol shock (points)",
                value=int(st.session_state.get("combo_vol_pts", default_vol)),
                step=1,
                key="combo_vol_pts",
                help="1 vol point = +0.01 absolute IV",
            )

        sel_vol = _nearest_int(int(req_vol), avail_vols_for_time) if avail_vols_for_time else int(req_vol)
        vkey = f"V{int(sel_vol):+d}"

        with cI3:
            st.caption(
                "Table shows PnL under joint repricing: "
                "S -> S*(1+spot%), IV -> IV + (vol_pts/100), T -> T - (days/365)."
            )

        # warn if snapped
        if avail_times and int(req_time) != int(sel_time):
            st.info(f"Time shock snapped to nearest available: requested {int(req_time)}D → using {int(sel_time)}D")
        if avail_vols_for_time and int(req_vol) != int(sel_vol):
            st.info(f"Vol shock snapped to nearest available: requested {int(req_vol)} → using {int(sel_vol)}")

        if tkey not in combo_agg:
            st.warning(f"{tkey} not available. Available: {', '.join(sorted(combo_agg.keys()))}")
        elif vkey not in combo_agg[tkey]:
            st.warning(f"{vkey} not available under {tkey}. Available: {', '.join(sorted(combo_agg[tkey].keys()))}")
        else:
            spot_map = combo_agg[tkey][vkey] or {}
            cols = _sort_shock_keys(list(spot_map.keys()), "spot")

            df = pd.DataFrame([[spot_map.get(k, 0.0) for k in cols]], columns=cols, index=["ALL"])
            st.markdown(f"**Spot shocks (₹ Lacs)** — slice: `{tkey}` & `{vkey}`")
            st.dataframe((df / 1e5).round(2), use_container_width=True)
st.divider()
# Min/Max PnL
with st.expander("📌 P&L Min/Max (today)", expanded=False):

    def _as_float(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    def _as_ts(s):
        s = (s or "").strip()
        if not s:
            return pd.NaT
        try:
            ts = pd.to_datetime(s, errors="coerce")
            if pd.isna(ts):
                return pd.NaT
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.tz_convert("Asia/Kolkata")
            return ts
        except Exception:
            return pd.NaT

    rows = []
    for key, label in [("carry", "Carry"), ("day", "Day"), ("net", "Net")]:
        rows.append({
            "Metric": label,
            "Min P&L": _as_float(mm.get(f"{key}_min")),
            "Min Time": _as_ts(mm.get(f"{key}_min_at")),
            "Max P&L": _as_float(mm.get(f"{key}_max")),
            "Max Time": _as_ts(mm.get(f"{key}_max_at")),
        })

    mm_df = pd.DataFrame(rows)

    styler = (
        mm_df.style
        .format({
            "Min P&L": lambda x: "—" if pd.isna(x) else f"{x:,.0f}",
            "Max P&L": lambda x: "—" if pd.isna(x) else f"{x:,.0f}",
            "Min Time": lambda x: "—" if pd.isna(x) else x.strftime("%H:%M:%S"),
            "Max Time": lambda x: "—" if pd.isna(x) else x.strftime("%H:%M:%S"),
        })
        .set_properties(subset=["Min Time", "Max Time"], **{"color": "#6b7280"})
        .set_properties(subset=["Min P&L", "Max P&L"], **{"font-weight": "700"})
    )

    st.dataframe(styler, width="stretch", hide_index=True)
    st.caption(
        f"created: {mm.get('created_at','—')} · updated: {mm.get('updated_at','—')} · last_at: {mm.get('last_at','—')}"
    )

# Margin
with st.expander("🧾 Margin (from margin worker)", expanded=True):
    if not margin or not margin.get("has_output", False):
        st.info("No margin output found yet (or margin worker not running).")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("SPAN", f"₹{float(margin.get('span',0)/1e7):,.3f} Cr")
        m2.metric("Exposure", f"₹{float(margin.get('exposure',0)/1e7):,.3f} Cr")
        m3.metric("Total", f"₹{float(margin.get('total',0)/1e7):,.3f} Cr")
        m4.metric("Computed at", margin.get("computed_at",""))

        mm2 = margin.get("minmax") or {}
        if mm2:
            st.caption("Margin min/max hash (if produced by margin worker):")
            st.json(mm2, expanded=False)

# Tables / Tabs
tab_labels = ["P&L by UL+Expiry", "P&L by UL", "Portfolio Greeks", "Top legs"]
if view_mode == "Account":
    tab_labels.append("payoff")

tabs = st.tabs(tab_labels)

ul_exp = records_to_df(((snap.get("tables") or {}).get("ul_exp_pnl") or {}))
ul_ul  = records_to_df(((snap.get("tables") or {}).get("ul_pnl") or {}))
pf     = records_to_df(((snap.get("tables") or {}).get("pf_greeks") or {}))

with tabs[0]:
    st.subheader("Carry + Day — P&L by underlying + expiry (incl. expenses)")
    st.dataframe(ul_exp.round(0), width="stretch")

with tabs[1]:
    st.subheader("Carry + Day — P&L by underlying (incl. expenses)")
    st.dataframe(ul_ul.round(0), width="stretch")

with tabs[2]:
    st.subheader("Portfolio Greeks (qty-weighted) by UL + expiry")

    # Number format 
    try:
        cols_to_format = ["Vega","Θ","volga","charm","Notional Δ","gross_delta"]
        pf[cols_to_format] = pf[cols_to_format]/1e5
    except :
        st.info("error while formatting numbers")
    st.dataframe(pf.round(2), use_container_width=True)
    st.caption(f"values in these columns are in lacs :{cols_to_format}")

    scn = snap.get("scenarios") or {}
    if not scn:
        st.caption("No scenarios in snapshot (enable scenario publishing in worker).")
    else:
        # If combo exists, allow browsing per UL for the SAME (time, vol) chosen at top
        req_time = int(st.session_state.get("combo_time_days", 0))
        req_vol  = int(st.session_state.get("combo_vol_pts", 0))
        tkey = f"T+{req_time}D"
        vkey = f"V{req_vol:+d}"

        for ul, packs in scn.items():
            with st.expander(f"{ul} • Scenarios", expanded=False):
                # show TRUE combo slice (if present)
                combo = (packs or {}).get("combo") or {}
                if isinstance(combo, dict) and combo:
                    # 1) TRUE Combined spot shocks (existing)
                    if (tkey in combo) and (vkey in (combo.get(tkey) or {})):
                        spot_map = (combo[tkey][vkey] or {})
                        cols = _sort_shock_keys(list(spot_map.keys()), "spot")
                        dfc = pd.DataFrame([[spot_map.get(k, 0.0) for k in cols]], columns=cols, index=[ul])
                        st.markdown(f"**TRUE Combined spot shocks (₹ Lacs)** — `{tkey}` & `{vkey}`")
                        st.dataframe((dfc / 1e5).round(2), use_container_width=True)
                    else:
                        st.caption("TRUE combo cube present, but this (time, vol) slice not available for this UL.")

                    # 2) NEW: Spot × Vol grid at T+0D
                    t0 = "T+0D"
                    if t0 in combo and isinstance(combo.get(t0), dict) and combo[t0]:
                        vol_keys = list(combo[t0].keys())  # e.g. ["V-2","V+0","V+1",...]
                        # sort vol keys numerically
                        def _vol_num(vk: str):
                            try:
                                return int(str(vk).replace("V", ""))
                            except Exception:
                                return 0
                        vol_keys = sorted(vol_keys, key=_vol_num)

                        # spot columns: take from first vol slice that has data
                        first_spot_map = None
                        for vk in vol_keys:
                            m = combo[t0].get(vk) or {}
                            if isinstance(m, dict) and m:
                                first_spot_map = m
                                break

                        if first_spot_map:
                            spot_cols = _sort_shock_keys(list(first_spot_map.keys()), "spot")

                            rows = []
                            idx = []
                            for vk in vol_keys:
                                m = combo[t0].get(vk) or {}
                                if not isinstance(m, dict):
                                    continue
                                rows.append([m.get(sc, 0.0) for sc in spot_cols])
                                idx.append(vk)

                            if rows:
                                df_sv = pd.DataFrame(rows, index=idx, columns=spot_cols)
                                st.markdown(f"**TRUE Spot × Vol grid (₹ Lacs)** — `{t0}`")
                                st.dataframe((df_sv / 1e5).round(4), use_container_width=True)
                            else:
                                st.caption("TRUE combo cube has T+0D but no vol slices with spot maps.")
                        else:
                            st.caption("TRUE combo cube has T+0D but spot maps are empty.")
                    else:
                        st.caption("TRUE combo cube present, but `T+0D` slice is missing (TIME_DAYS_COMBO must include 0).")

                else:
                    st.caption("No TRUE combo cube for this UL (missing `scenarios[ul]['combo']`).")

                # keep old separate scenario views (optional, useful for debugging)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Spot-only shocks (old)**")
                    # df = pd.DataFrame([combo['T+0D']["V+0"]],index=[ul])
                    df = pd.DataFrame([packs.get("spot", {})], index=[ul])
                    st.dataframe((df / 1e5).round(2), use_container_width=True)
                with c2:
                    st.markdown("**Vol-only shocks (old)**")
                    df = pd.DataFrame([packs.get("vol", {})], index=[ul])
                    st.dataframe((df / 1e5).round(2), use_container_width=True)
                with c3:
                    st.markdown("**Time-only shocks (old)**")
                    df = pd.DataFrame([packs.get("time", {})], index=[ul])
                    st.dataframe((df / 1e5).round(2), use_container_width=True)

        st.caption("Top table uses TRUE joint reprices from `combo`. The old spot/vol/time tables are single-axis shocks for reference.")

with tabs[3]:
    top = snap.get("top_legs") or {}
    if not top:
        st.info("No top legs available for this snapshot.")
    else:
        sub_tabs = st.tabs(["Top |Δ|", "Top |Vega|", "Top |Γ|"])
        with sub_tabs[0]:
            st.dataframe(records_to_df(top.get("by_abs_delta") or {}).round(6), width="stretch")
        with sub_tabs[1]:
            st.dataframe(records_to_df(top.get("by_abs_vega") or {}).round(6), width="stretch")
        with sub_tabs[2]:
            st.dataframe(records_to_df(top.get("by_abs_gamma") or {}).round(6), width="stretch")

# Payoff tab only for Account view
if view_mode == "Account":
    with tabs[4]:
        import plotly.graph_objects as go

        now_dt = datetime.now(IST)
        st.subheader(f"Payoff curves — {username} (Expiry payoff)")

        # ---- Controls (ONLY grid range) ----
        cA, _ = st.columns([1.2, 3.0])
        with cA:
            grid_pct_choice = st.selectbox("Grid range", ["±3%", "±5%", "±8%", "±10%"], index=1)
            grid_pct = {"±3%": 0.03, "±5%": 0.05, "±8%": 0.08, "±10%": 0.10}[grid_pct_choice]

        pack = snap.get("payoff_pack") or {}
        expiry_map = (pack.get("expiry") or {}) if isinstance(pack, dict) else {}

        if not expiry_map:
            st.info("No payoff data in snapshot for this user (missing payoff_pack/expiry).")
            st.stop()

        keys = sorted(expiry_map.keys())
        sel = st.selectbox("Select UL|Expiry", keys, index=0, key=f"pay_sel_{username}")

        dfe = records_to_df_payoff(expiry_map.get(sel) or [])
        if not dfe.empty:
            dfe["S"] = pd.to_numeric(dfe.get("S"), errors="coerce")
            dfe["pnl_entry"] = pd.to_numeric(dfe.get("pnl_entry"), errors="coerce")
            dfe = dfe.dropna(subset=["S", "pnl_entry"]).sort_values("S")
        else:
            st.info("No curve data for this selection.")
            st.stop()

        # ---- S0 (for clipping) ----
        # Use midpoint as S0 for clipping if you don't have an explicit S0 in payoff_pack
        S0 = float(dfe["S"].iloc[len(dfe)//2]) if len(dfe) else np.nan

        # ---- Clip to chosen grid ----
        if np.isfinite(S0) and S0 > 0:
            lo_s, hi_s = S0 * (1.0 - grid_pct), S0 * (1.0 + grid_pct)
            dfe = dfe[(dfe["S"] >= lo_s) & (dfe["S"] <= hi_s)].copy()

        if dfe.empty:
            st.info("No curve data after clipping.")
            st.stop()

        # ---- Plot with 0-line + green fill above 0 ----
        x = dfe["S"].to_numpy(dtype=float)
        y = dfe["pnl_entry"].to_numpy(dtype=float)

        y_pos = np.maximum(y, 0.0)  # fill only positive area

        fig = go.Figure()

        # Green fill (only where payoff > 0)
        fig.add_trace(go.Scatter(
            x=x, y=y_pos,
            mode="none",
            fill="tozeroy",
            fillcolor="rgba(0, 180, 0, 0.25)",
            name="PnL > 0"
        ))

        # Payoff curve
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            name="Payoff",
            line=dict(width=2)
        ))

        # Solid 0 line
        fig.add_trace(go.Scatter(
            x=[float(np.min(x)), float(np.max(x))],
            y=[0.0, 0.0],
            mode="lines",
            name="0",
            line=dict(color="black", width=2)
        ))

        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Spot (S)",
            yaxis_title="PnL (₹)",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Curve data", expanded=False):
            show = dfe[["S", "pnl_entry"]].copy()
            show["S"] = show["S"].round(2)
            show["pnl_entry"] = show["pnl_entry"].round(0)
            st.dataframe(show, use_container_width=True)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 14px; color: gray;'>"
    "© Subhkam Ventures. All rights reserved."
    "</div>",
    unsafe_allow_html=True,
)
