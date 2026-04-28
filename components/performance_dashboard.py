# components/performance_dashboard.py
"""
Performance Dashboard module (importable)

Usage (inside any Streamlit page):
    from components.performance_dashboard import render_performance_dashboard, PerfConfig
    cfg = PerfConfig()  # or override paths/margins
    render_performance_dashboard(cfg, show_login=False)

Patch notes vs your old page script:
- ✅ NO pip-install inside Streamlit
- ✅ NO st.set_page_config here (caller page owns that)
- ✅ Controls moved to an in-page expander (no sidebar clutter)
- ✅ Safer file parsing + graceful “missing yfinance / missing files” handling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import os
import re
import bisect
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
import streamlit as st

# --- add near other imports at top ---
from datetime import datetime  # already present
import os  # already present
import pandas as pd  # already present
import streamlit as st  # already present


# -------------------------
# Debug Renderer
# -------------------------
def _render_nifty_debug(cfg: PerfConfig) -> None:
    """Shows what Nifty file is being read + raw vs parsed date ranges."""
    with st.expander("DEBUG: Nifty data being loaded", expanded=False):
        path = cfg.input_file_nifty
        st.write("Nifty file path:", path)
        exists = bool(path) and os.path.exists(path)
        st.write("Exists:", exists)

        if not exists:
            st.warning("Nifty file does not exist at this path.")
            return

        try:
            st.write("Last modified (local):", datetime.fromtimestamp(os.path.getmtime(path)))
            st.write("File size (MB):", round(os.path.getsize(path) / (1024 * 1024), 2))
        except Exception as e:
            st.warning(f"Could not read file stats: {e}")

        # 1) RAW file check
        try:
            raw = pd.read_excel(path)
            st.write("RAW shape:", raw.shape)
            st.write("RAW columns:", list(raw.columns))

            raw_dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
            st.write("RAW min date:", raw_dates.min())
            st.write("RAW max date:", raw_dates.max())
            st.dataframe(raw.tail(20), use_container_width=True)
        except Exception as e:
            st.error(f"RAW read_excel failed: {e}")
            return

        # 2) PARSED (what dashboard uses)
        try:
            parsed = load_strat_data(path, "nifty")
            st.write("PARSED rows:", 0 if parsed is None else len(parsed))

            if parsed is None or parsed.empty:
                st.warning("Parsed Nifty dataframe is empty (after filtering).")
                return

            if "Date" in parsed.columns:
                st.write("PARSED min date:", parsed["Date"].min())
                st.write("PARSED max date:", parsed["Date"].max())

            show_cols = [c for c in ["Date", "Daily Return"] if c in parsed.columns]
            st.dataframe(parsed[show_cols].tail(20) if show_cols else parsed.tail(20), use_container_width=True)
        except Exception as e:
            st.error(f"PARSED load_strat_data failed: {e}")
            return



# Optional dependency
try:
    import yfinance as yf  # type: ignore
    _HAS_YF = True
except Exception:
    yf = None
    _HAS_YF = False


# -------------------------
# Config
# -------------------------
@dataclass
class PerfConfig:
    # Files
    input_file_qi: str = r"/mnt/Quant_Research/daily_task/daily_trade_data/PNL_Reports_QI/daily_pnl_summary.csv"
    input_file_nifty: str = r"/mnt/Quant_Research/daily_task/nifty_daily_pnl.csv.xlsx"
    input_file_vb: str = r"/mnt/Quant_Research/daily_task/daily_emkay_trade_data/PNL_Reports_DMATCS08/daily_pnl_summary.csv"

    file_16: str = r"/mnt/Quant_Research/daily_task/daily_emkay_trade_data/PNL_Reports_DMATCS16/daily_pnl_summary.csv"
    file_19: str = r"/mnt/Quant_Research/daily_task/daily_emkay_trade_data/PNL_Reports_DMATCS19/daily_pnl_summary.csv"
    file_09: str = r"/mnt/Quant_Research/daily_task/daily_emkay_trade_data/PNL_Reports_DMATCS09/daily_pnl_summary.csv"
    file_fut01: str = r"/mnt/Quant_Research/daily_task/daily_emkay_trade_data/PNL_Reports_Futures_5RPCG02/daily_pnl_summary.csv"

    # Margins
    margin_qi: float = 250_000_000
    margin_fut01: float = 250_000_000
    trading_days: int = 252

    # Cache TTL for metrics
    cache_ttl_sec: int = 3600


# -------------------------
# Utilities
# -------------------------
def clean_numeric(x) -> float:
    if pd.isna(x) or str(x).strip() == "":
        return 0.0
    try:
        s = str(x).replace(",", "").replace("%", "").strip()
        return float(s)
    except Exception:
        return 0.0


# def smart_parse_date(date_str):
#     """Handles mixed formats: DD-MM-YYYY and YYYY-MM-DD in the same file."""
#     if pd.isna(date_str):
#         return pd.NaT
#     s = str(date_str).strip()

#     # YYYY-MM-DD
#     if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
#         return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")

#     # DD-MM-YYYY
#     if re.match(r"^\d{2}-\d{2}-\d{4}$", s):
#         return pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")

#     return pd.to_datetime(s, dayfirst=True, errors="coerce")

def smart_parse_date(x):
    """Robust mixed date parser, handles datetime objects + ambiguous strings."""
    if pd.isna(x) or str(x).strip() == "":
        return pd.NaT

    # If it's already a datetime-like value (common when reading Excel), keep it
    if isinstance(x, (pd.Timestamp, datetime, np.datetime64)):
        return pd.to_datetime(x, errors="coerce")

    s = str(x).strip()

    # Common explicit formats first (fast + deterministic)
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y"):
        dt = pd.to_datetime(s, format=fmt, errors="coerce")
        if not pd.isna(dt):
            return dt

    # Ambiguous cases: try both dayfirst True/False and pick the plausible one (not in far future)
    dt_a = pd.to_datetime(s, dayfirst=True, errors="coerce")
    dt_b = pd.to_datetime(s, dayfirst=False, errors="coerce")

    if pd.isna(dt_a):
        return dt_b
    if pd.isna(dt_b):
        return dt_a

    # Heuristic: prefer the one that is not in the future (relative to "today")
    today = pd.Timestamp.today().normalize()
    a_future = dt_a.normalize() > today
    b_future = dt_b.normalize() > today

    if a_future and not b_future:
        return dt_b
    if b_future and not a_future:
        return dt_a

    # If both plausible, keep dayfirst=True behavior (your original intent)
    return dt_a

def _ensure_parent_dir(path: str) -> None:
    try:
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
    except Exception:
        pass


def get_vb_margin(dt: datetime) -> float:
    return 100_000_000 if dt < datetime(2025, 11, 1) else 250_000_000


def get_09_margin(dt) -> float:
    d = dt.date() if isinstance(dt, (datetime, pd.Timestamp)) else dt
    if d >= datetime(2026, 2, 16).date():
        return 75_000_000
    elif d >= datetime(2026, 2, 10).date():
        return 50_000_000
    else:
        return 30_000_000


_RAW_SCHEDULE = {
    "2025-09-10": 10000000, "2025-10-10": 20000000, "2025-10-13": 30000000, "2025-10-14": 30000000,
    "2025-10-15": 40000000, "2025-10-16": 50000000, "2025-10-17": 50000000, "2025-10-20": 50000000,
    "2025-10-21": 50000000, "2025-10-23": 50000000, "2025-10-24": 50000000, "2025-10-27": 50000000,
    "2025-10-28": 50000000, "2025-10-29": 50000000, "2025-10-30": 50000000, "2025-10-31": 50000000,
    "2025-11-03": 50000000, "2025-11-04": 50000000, "2025-11-06": 50000000, "2025-11-07": 50000000,
    "2025-11-10": 50000000, "2025-11-11": 50000000, "2025-11-12": 50000000, "2025-11-13": 50000000,
    "2025-11-14": 50000000, "2025-11-17": 50000000, "2025-11-18": 50000000, "2025-11-19": 60000000,
    "2025-11-20": 60000000, "2025-11-21": 60000000, "2025-11-24": 60000000, "2025-11-25": 60000000,
    "2025-11-26": 60000000, "2025-11-27": 60000000, "2025-11-28": 70000000, "2025-12-01": 70000000,
    "2025-12-02": 70000000, "2025-12-03": 70000000, "2025-12-04": 70000000, "2025-12-05": 70000000,
    "2025-12-08": 70000000, "2025-12-09": 80000000, "2025-12-10": 90000000, "2025-12-11": 100000000,
    "2025-12-12": 100000000, "2025-12-15": 120000000, "2025-12-16": 120000000, "2025-12-17": 140000000,
    "2025-12-18": 140000000, "2025-12-19": 140000000, "2025-12-22": 160000000, "2025-12-23": 160000000,
    "2025-12-24": 160000000, "2025-12-26": 160000000, "2025-12-29": 180000000, "2025-12-30": 180000000,
    "2025-12-31": 200000000, "2026-01-01": 200000000, "2026-01-02": 200000000, "2026-01-05": 220000000,
    "2026-01-06": 220000000, "2026-01-07": 240000000, "2026-01-08": 240000000, "2026-01-09": 240000000,
    "2026-01-12": 240000000, "2026-01-13": 240000000, "2026-01-14": 240000000, "2026-01-15": 240000000,
    "2026-01-16": 240000000, "2026-01-19": 240000000, "2026-01-20": 240000000, "2026-01-21": 240000000,
    "2026-01-22": 240000000, "2026-01-23": 240000000,
    "2026-02-16": 260000000, "2026-02-18": 280000000
}
_MS_1619 = {datetime.strptime(k, "%Y-%m-%d").date(): v for k, v in _RAW_SCHEDULE.items()}
_SORTED_SCHED_DATES = sorted(_MS_1619.keys())


def get_16_19_margin(dt) -> float:
    d_key = dt.date() if isinstance(dt, (datetime, pd.Timestamp)) else dt
    if d_key == datetime(2025, 10, 9).date():
        return 10_000_000
    if d_key in _MS_1619:
        return _MS_1619[d_key]
    if d_key < _SORTED_SCHED_DATES[0]:
        return _MS_1619[_SORTED_SCHED_DATES[0]]
    if d_key > _SORTED_SCHED_DATES[-1]:
        return _MS_1619[_SORTED_SCHED_DATES[-1]]
    idx = bisect.bisect_left(_SORTED_SCHED_DATES, d_key)
    return _MS_1619[_SORTED_SCHED_DATES[idx - 1]]


def refresh_nifty_data(output_xlsx: str) -> Tuple[bool, str]:
    if not _HAS_YF:
        return False, "yfinance not installed. Install once: pip install yfinance"

    try:
        raw = yf.download("^NSEI", start="2024-01-01", progress=False)
        if raw is None or raw.empty:
            return False, "No data returned by yfinance."

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw[["Close"]].copy()
        df["Daily Return"] = df["Close"].pct_change().fillna(0.0)

        df.reset_index(inplace=True)
        # Keep compatibility with your old 'iloc[:,6]' expectation:
        # add 4 dummy columns so Daily Return becomes col index 6.
        for i in range(1, 5):
            df.insert(i + 1, f"dummy_{i}", 0)

        _ensure_parent_dir(output_xlsx)
        df.to_excel(output_xlsx, index=False)
        return True, "Nifty updated successfully."
    except Exception as e:
        return False, f"Nifty refresh error: {e}"


def load_strat_data(files, mode: str) -> pd.DataFrame:
    """
    Reads daily_pnl_summary file(s) and returns a cleaned df with columns:
      Date, Net PNL, ...
    For nifty, expects an xlsx with return in either 'Daily Return' or column 6.
    """

    def process_raw(f_path: str) -> pd.DataFrame:
        if not f_path or not os.path.exists(f_path):
            return pd.DataFrame()
        tdf = pd.read_excel(f_path) if f_path.lower().endswith(".xlsx") else pd.read_csv(f_path)
        if tdf.empty:
            return tdf

        # rename first col -> Date
        tdf = tdf.copy()
        tdf.rename(columns={tdf.columns[0]: "Date"}, inplace=True)

        tdf["Date"] = tdf["Date"].apply(smart_parse_date)
        tdf["Date"] = tdf["Date"].dt.tz_localize(None, nonexistent="NaT", ambiguous="NaT")
        tdf.dropna(subset=["Date"], inplace=True)

        if "Net PNL" in tdf.columns:
            tdf["Net PNL"] = tdf["Net PNL"].apply(clean_numeric)

        return tdf

    try:
        if mode == "1619":
            f16, f19 = files
            df16, df19 = process_raw(f16), process_raw(f19)
            if df16.empty:
                return pd.DataFrame()

            mask16 = df16["Net PNL"].abs() > 1.0
            if not mask16.any():
                return pd.DataFrame()

            anchor_date = df16.loc[mask16.idxmax(), "Date"]
            df16 = df16[df16["Date"] >= anchor_date].copy()
            df19 = df19[df19["Date"] >= anchor_date].copy()

            res = pd.merge(df16, df19, on="Date", how="left").fillna(0)
            # merged net pnl
            res["Net PNL"] = res.get("Net PNL_x", 0) + res.get("Net PNL_y", 0)
            return res.sort_values("Date")

        # single file
        tdf = process_raw(files)
        if tdf.empty:
            return tdf

        if mode == "nifty":
            # old logic used tdf.iloc[:,6] (Daily Return). Make robust:
            if "Daily Return" in tdf.columns:
                ret_series = tdf["Daily Return"].apply(clean_numeric)
            else:
                # fallback to column 6 if present
                if tdf.shape[1] >= 7:
                    ret_series = tdf.iloc[:, 6].apply(clean_numeric)
                else:
                    return pd.DataFrame()

            mask = ret_series.abs() > 0.000001
            return tdf.loc[mask.idxmax():].copy() if mask.any() else pd.DataFrame()

        # non-nifty
        mask = tdf["Net PNL"].abs() > 1.0
        return tdf.loc[mask.idxmax():].copy() if mask.any() else pd.DataFrame()

    except Exception:
        return pd.DataFrame()


def get_quarter_start(dt: datetime) -> datetime:
    q_month = ((dt.month - 1) // 3) * 3 + 1
    return datetime(dt.year, q_month, 1)


def get_metrics(df: pd.DataFrame, mode: str, cfg: PerfConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if df is None or df.empty:
        return {}, {"first_date": "No Data", "abs_pnl": "0", "raw_pnl": 0}

    df = df.copy()
    print(df.tail())
    if mode == "nifty":
        # return column:
        if "Daily Return" in df.columns:
            df["ret"] = df["Daily Return"].apply(clean_numeric)
        elif df.shape[1] >= 7:
            df["ret"] = df.iloc[:, 6].apply(clean_numeric)
        else:
            return {}, {"first_date": "No Data", "abs_pnl": "0", "raw_pnl": 0}

    elif mode == "vb":
        df["ret"] = df.apply(lambda r: r["Net PNL"] / get_vb_margin(r["Date"]), axis=1)

    elif mode == "1619":
        df["ret"] = df.apply(lambda r: r["Net PNL"] / get_16_19_margin(r["Date"]), axis=1)

    elif mode == "09":
        df["ret"] = df.apply(lambda r: r["Net PNL"] / get_09_margin(r["Date"]), axis=1)

    elif mode == "fut01":
        df["ret"] = df["Net PNL"] / cfg.margin_fut01

    elif mode == "total":
        df["total_margin"] = df.apply(
            lambda r: get_16_19_margin(r["Date"]) + get_09_margin(r["Date"]) + cfg.margin_fut01,
            axis=1,
        )
        df["ret"] = df["Net PNL"] / df["total_margin"]

    else:
        df["ret"] = df["Net PNL"] / cfg.margin_qi

    latest = df["Date"].max()
    mtd_df = df[(df["Date"].dt.month == latest.month) & (df["Date"].dt.year == latest.year)]
    qtd_start = get_quarter_start(latest)
    qtd_df = df[df["Date"] >= qtd_start]
    fy_start = datetime(latest.year if latest.month >= 4 else latest.year - 1, 4, 1)
    ytd_df = df[df["Date"] >= fy_start]

    periodic = {
        "ftd": float(df["ret"].iloc[-1]),
        "mtd": float(mtd_df["ret"].sum()) if not mtd_df.empty else 0.0,
        "qtd": float(qtd_df["ret"].sum()) if not qtd_df.empty else 0.0,
        "ytd": float(ytd_df["ret"].sum()) if not ytd_df.empty else 0.0,
        "ftd_d": latest.strftime("%d/%m/%y"),
        "mtd_d": f"{mtd_df['Date'].min().strftime('%d/%m/%y')}<br>{latest.strftime('%d/%m/%y')}" if not mtd_df.empty else "-",
        "qtd_d": f"{qtd_df['Date'].min().strftime('%d/%m/%y')}<br>{latest.strftime('%d/%m/%y')}" if not qtd_df.empty else "-",
        "ytd_d": f"{ytd_df['Date'].min().strftime('%d/%m/%y')}<br>{latest.strftime('%d/%m/%y')}" if not ytd_df.empty else "-",
    }

    df["NAV"] = 100 * (1 + df["ret"]).cumprod()
    ann_ret = (df["NAV"].iloc[-1] / 100) ** (cfg.trading_days / max(1, len(df))) - 1
    vol = float(df["ret"].std()) * np.sqrt(cfg.trading_days)
    max_dd = float((df["NAV"] / df["NAV"].cummax() - 1).min())

    raw_pnl = float(df["Net PNL"].sum()) if ("Net PNL" in df.columns and mode != "nifty") else 0.0

    agg = {
        "ann_ret": f"{ann_ret:.1%}",
        "win_rate": f"{(df['ret'] > 0).mean():.0%}",
        "max_dd": f"{max_dd:.1%}",
        "sharpe": f"{ann_ret/vol:.2f}" if vol > 0 else "0.00",
        "calmar": f"{ann_ret/abs(max_dd):.2f}" if max_dd != 0 else "0.00",
        "vol": f"{vol:.1%}",
        "first_date": df["Date"].min().strftime("%d-%m-%Y"),
        "last_date": latest.strftime("%d-%m-%Y"),
        "row_count": str(len(df)),
        "abs_pnl": f"{raw_pnl:,.0f}" if mode != "nifty" else "n/a",
        "raw_pnl": raw_pnl,
    }
    return periodic, agg


@st.cache_data(ttl=3600)
def _get_all_cached(cfg_dict: dict) -> dict:
    cfg = PerfConfig(**cfg_dict)

    data = {}

    df_1619 = load_strat_data([cfg.file_16, cfg.file_19], "1619")
    df_09 = load_strat_data(cfg.file_09, "09")
    data["1619_p"], data["1619"] = get_metrics(df_1619, "1619", cfg)
    data["09_p"], data["09"] = get_metrics(df_09, "09", cfg)

    df_fut01 = load_strat_data(cfg.file_fut01, "fut01")
    data["fut01_p"], data["fut01"] = get_metrics(df_fut01, "fut01", cfg)

    # total
    strat_dfs = [("1619", df_1619), ("09", df_09), ("fut01", df_fut01)]
    non_empty = [(k, df) for k, df in strat_dfs if df is not None and not df.empty]

    if len(non_empty) >= 2:
        merged = non_empty[0][1][["Date", "Net PNL"]].rename(columns={"Net PNL": f"Net PNL_{non_empty[0][0]}"})
        for k, df in non_empty[1:]:
            merged = pd.merge(
                merged,
                df[["Date", "Net PNL"]].rename(columns={"Net PNL": f"Net PNL_{k}"}),
                on="Date",
                how="outer",
            )
        merged.fillna(0, inplace=True)
        pnl_cols = [c for c in merged.columns if c.startswith("Net PNL_")]
        merged["Net PNL"] = merged[pnl_cols].sum(axis=1)
        data["total_p"], data["total"] = get_metrics(merged.sort_values("Date"), "total", cfg)
    elif len(non_empty) == 1:
        data["total_p"], data["total"] = get_metrics(non_empty[0][1], "total", cfg)
    else:
        data["total_p"], data["total"] = ({}, {"first_date": "No Data", "abs_pnl": "0", "raw_pnl": 0})

    # peers
    for k, f, m in [
        ("vb", cfg.input_file_vb, "vb"),
        ("qi", cfg.input_file_qi, "qi"),
        ("nifty", cfg.input_file_nifty, "nifty"),
    ]:
        df = load_strat_data(f, m)
        data[f"{k}_p"], data[k] = get_metrics(df, m, cfg)

    return data


def _css() -> str:
    return """<style>
        .scroll-wrapper { overflow-x: auto; width: 100%; -webkit-overflow-scrolling: touch; }
        .main-table { border-collapse: collapse; margin-bottom: 30px; table-layout: auto; min-width: 2400px; }
        .master-header, .sub-header, .row-label, .data-cell { text-align: center; font-size: 14px; padding: 6px 8px; font-weight: bold; color: black; border: none; white-space: nowrap; }
        .master-divider { border-right: 2.5px solid black !important; }
        .master-header { background-color: #eee; border-bottom: 2px solid black; }
        .sub-header { border-bottom: 1px solid #ccc; min-width: 70px; }
        .row-label { background-color: #f2f2f2; text-align: left; min-width: 140px; position: sticky; left: 0; z-index: 1; border-right: 2.5px solid black !important; }
        .total-row { background-color: #fffde7; border-top: 2px solid black; border-bottom: 1px solid black; }
        .meta-text { color: #777; font-weight: normal; font-size: 13px; line-height: 1.4; white-space: pre-wrap; }
        .pos { color: green !important; } .neg { color: red !important; }
    </style>"""


def _render_periodic_table(d: dict) -> None:
    st.subheader("Periodic Performance")
    heads = ["Strategy Performance", "Benchmark (Nifty)", "Peer (QiCap)", "Peer (VB)", "vs Bench (bps)", "vs Qi (bps)", "vs VB (bps)"]
    period_cols = ["FTD", "MTD", "QTD", "YTD"]
    num_periods = len(period_cols)

    html = '<div class="scroll-wrapper"><table class="main-table"><thead><tr><th class="master-header master-divider"></th>'
    for h in heads:
        html += f'<th colspan="{num_periods}" class="master-header master-divider">{h}</th>'
    html += '</tr><tr><th class="sub-header master-divider">Name</th>'
    for _ in range(7):
        for i, s in enumerate(period_cols):
            html += f'<th class="sub-header {"master-divider" if i==num_periods-1 else ""}">{s}</th>'
    html += "</tr></thead><tbody>"

    period_keys = ["ftd", "mtd", "qtd", "ytd"]
    strat_rows = [("DMATCS16+19", "1619_p"), ("DMATCS09", "09_p"), ("FUT01", "fut01_p")]
    num_strats = len(strat_rows)

    for idx_s, (name, k_p) in enumerate(strat_rows):
        html += f'<tr><td class="row-label">{name}</td>'

        for i, p in enumerate(period_keys):
            v = float(d.get(k_p, {}).get(p, 0.0) or 0.0)
            html += f'<td class="data-cell {"master-divider" if i==num_periods-1 else ""}">{v:.2%}</td>'

        if idx_s == 0:
            for peer in ["nifty_p", "qi_p", "vb_p"]:
                for i, p in enumerate(period_keys):
                    v = float(d.get(peer, {}).get(p, 0.0) or 0.0)
                    html += f'<td rowspan="{num_strats}" class="data-cell {"master-divider" if i==num_periods-1 else ""}">{v:.2%}</td>'

        for peer in ["nifty_p", "qi_p", "vb_p"]:
            for i, p in enumerate(period_keys):
                diff = (float(d.get(k_p, {}).get(p, 0.0) or 0.0) - float(d.get(peer, {}).get(p, 0.0) or 0.0)) * 10000
                html += f'<td class="data-cell {"pos" if diff>=0 else "neg"} {"master-divider" if i==num_periods-1 else ""}">{diff:+.0f}</td>'

        html += "</tr>"

    # Total row
    html += '<tr class="total-row"><td class="row-label">TOTAL</td>'
    for i, p in enumerate(period_keys):
        v = float(d.get("total_p", {}).get(p, 0.0) or 0.0)
        html += f'<td class="data-cell {"master-divider" if i==num_periods-1 else ""}">{v:.2%}</td>'
    for peer in ["nifty_p", "qi_p", "vb_p"]:
        for i, p in enumerate(period_keys):
            v = float(d.get(peer, {}).get(p, 0.0) or 0.0)
            html += f'<td class="data-cell {"master-divider" if i==num_periods-1 else ""}">{v:.2%}</td>'
    for peer in ["nifty_p", "qi_p", "vb_p"]:
        for i, p in enumerate(period_keys):
            diff = (float(d.get("total_p", {}).get(p, 0.0) or 0.0) - float(d.get(peer, {}).get(p, 0.0) or 0.0)) * 10000
            html += f'<td class="data-cell {"pos" if diff>=0 else "neg"} {"master-divider" if i==num_periods-1 else ""}">{diff:+.0f}</td>'
    html += "</tr>"

    # Dates footer row
    date_keys = ["ftd_d", "mtd_d", "qtd_d", "ytd_d"]
    html += '<tr><td class="row-label meta-text">Dates</td>'
    for i, p in enumerate(date_keys):
        triple_date = (
            f'{d.get("1619_p", {}).get(p, "")}<br><br>'
            f'{d.get("09_p", {}).get(p, "")}<br><br>'
            f'{d.get("fut01_p", {}).get(p, "")}'
        )
        html += f'<td class="data-cell meta-text {"master-divider" if i==num_periods-1 else ""}">{triple_date}</td>'

    for pk in ["nifty_p", "qi_p", "vb_p"]:
        for i, p in enumerate(date_keys):
            html += f'<td class="data-cell meta-text {"master-divider" if i==num_periods-1 else ""}">{d.get(pk, {}).get(p, "")}</td>'

    for _ in range(num_periods * 3):
        html += '<td class="data-cell meta-text"></td>'
    html += "</tr></tbody></table></div>"

    st.markdown(html, unsafe_allow_html=True)


def _render_aggregate_table(d: dict) -> None:
    st.subheader("Aggregate Performance")
    m_keys = ["ann_ret", "win_rate", "max_dd", "sharpe", "calmar", "vol"]

    html2 = '<div class="scroll-wrapper"><table class="main-table"><thead><tr><th class="master-header master-divider"></th>'
    for h in ["Strategy Performance", "Benchmark (Nifty)", "Peer (QiCap)", "Peer (VB)"]:
        html2 += f'<th colspan="6" class="master-header master-divider">{h}</th>'
    html2 += '</tr><tr><th class="sub-header master-divider">Name</th>'
    for _ in range(4):
        for i, s in enumerate(["Ann", "Win", "MDD", "Shrp", "Calm", "Vol"]):
            html2 += f'<th class="sub-header {"master-divider" if i==5 else ""}">{s}</th>'
    html2 += "</tr></thead><tbody>"

    agg_rows = [("DMATCS16+19", "1619"), ("DMATCS09", "09"), ("FUT01", "fut01")]
    num_agg = len(agg_rows)

    for idx_s, (name, k_code) in enumerate(agg_rows):
        html2 += f'<tr><td class="row-label">{name}</td>'
        for i, m in enumerate(m_keys):
            html2 += f'<td class="data-cell {"master-divider" if i==5 else ""}">{d.get(k_code, {}).get(m, "")}</td>'

        if idx_s == 0:
            for peer in ["nifty", "qi", "vb"]:
                for i, m in enumerate(m_keys):
                    html2 += f'<td rowspan="{num_agg}" class="data-cell {"master-divider" if i==5 else ""}">{d.get(peer, {}).get(m, "")}</td>'

        html2 += "</tr>"

    # Total row
    html2 += '<tr class="total-row"><td class="row-label">TOTAL</td>'
    for i, m in enumerate(m_keys):
        html2 += f'<td class="data-cell {"master-divider" if i==5 else ""}">{d.get("total", {}).get(m, "")}</td>'
    for peer in ["nifty", "qi", "vb"]:
        for i, m in enumerate(m_keys):
            html2 += f'<td class="data-cell {"master-divider" if i==5 else ""}">{d.get(peer, {}).get(m, "")}</td>'
    html2 += "</tr>"

    # Metadata
    for lbl, m in [("net_pnl INR", "abs_pnl"), ("First Date", "first_date"), ("Last Date", "last_date"), ("Days analyzed", "row_count")]:
        html2 += f'<tr><td class="row-label">{lbl}</td>'

        color1 = "pos" if float(d.get("1619", {}).get("raw_pnl", 0) or 0) >= 0 else "neg"
        color2 = "pos" if float(d.get("09", {}).get("raw_pnl", 0) or 0) >= 0 else "neg"
        color3 = "pos" if float(d.get("fut01", {}).get("raw_pnl", 0) or 0) >= 0 else "neg"

        val_str = (
            f'<span class="{color1 if lbl=="net_pnl INR" else ""}">{d.get("1619", {}).get(m, "")}</span><br>'
            f'<span class="{color2 if lbl=="net_pnl INR" else ""}">{d.get("09", {}).get(m, "")}</span><br>'
            f'<span class="{color3 if lbl=="net_pnl INR" else ""}">{d.get("fut01", {}).get(m, "")}</span>'
        )
        html2 += f'<td colspan="6" class="data-cell master-divider meta-text">{val_str}</td>'

        for pk in ["nifty", "qi", "vb"]:
            p_color = "pos" if float(d.get(pk, {}).get("raw_pnl", 0) or 0) >= 0 else "neg"
            p_val = f'<span class="{p_color if lbl=="net_pnl INR" else ""}">{d.get(pk, {}).get(m, "")}</span>'
            html2 += f'<td colspan="6" class="data-cell master-divider meta-text">{p_val}</td>'

        html2 += "</tr>"

    st.markdown(html2 + "</tbody></table></div>", unsafe_allow_html=True)


# -------------------------
# Public Renderer
# -------------------------
def render_performance_dashboard(
    cfg: PerfConfig | None = None,
    show_login: bool = False,
    debug: bool = False,   # ✅ add this
) -> None:
    """
    Renders the whole dashboard in-place (below whatever is on the page).

    - No sidebar usage.
    - Controls in an in-page expander.
    - Optional login gate (show_login=True), but default is False.
    - Optional debug expander (debug=True).
    """
    cfg = cfg or PerfConfig()

    if show_login:
        auth_key = "perf_authenticated"
        if auth_key not in st.session_state:
            st.subheader("Login")
            u = st.text_input("User", key="perf_user")
            p = st.text_input("Pass", type="password", key="perf_pass")
            if st.button("Login", key="perf_login_btn"):
                if u == "abc" and p == "new":
                    st.session_state[auth_key] = True
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            return

    with st.expander("Performance controls", expanded=False):
        c1, c2 = st.columns(2)

        with c1:
            if st.button("🔄 Refresh All", key="perf_refresh_all"):
                _get_all_cached.clear()
                st.rerun()

        with c2:
            if st.button("📉 Refresh Nifty", key="perf_refresh_nifty"):
                ok, msg = refresh_nifty_data(cfg.input_file_nifty)
                if ok:
                    st.success(msg)
                    _get_all_cached.clear()
                    st.rerun()
                else:
                    st.warning(msg)

        if not _HAS_YF:
            st.caption("Note: yfinance not installed. Nifty refresh button will not work until you install it once.")

    # ✅ show debug expander only if asked
    if debug:
        _render_nifty_debug(cfg)

    st.markdown(_css(), unsafe_allow_html=True)

    d = _get_all_cached(cfg.__dict__)

    have_any = any(isinstance(v, dict) and len(v) > 0 for k, v in d.items())
    if not have_any:
        st.warning("No performance data available (files missing or empty). Check input paths in PerfConfig.")
        return

    _render_periodic_table(d)
    _render_aggregate_table(d)
