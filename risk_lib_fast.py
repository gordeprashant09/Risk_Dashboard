# risk_lib_fast.py
"""
Fast risk library (per-contract IV + greeks vectorized; scenarios disabled)

Designed for:
- risk_worker.py: compute snapshots for ALL users in a single-process loop
- risk_viewer.py: only reads + displays results from Redis (no heavy compute)

Key changes vs your monolithic Streamlit script:
- Vectorized per-contract IV (bisection) + greeks (arrays)  ✅
- Scenarios repricing loop REMOVED (returns empty scenarios) ✅
- No Streamlit dependencies in library                         ✅
"""

from __future__ import annotations

import re
import math
import calendar
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# TIME / HOLIDAYS
# =============================================================================

IST = timezone(timedelta(hours=5, minutes=30))

SECONDS_IN_DAY = 24 * 3600
SECONDS_IN_YEAR = 252 * SECONDS_IN_DAY  # trading-day seconds

@lru_cache(maxsize=1)
def last_5yr_holidays() -> tuple[date, ...]:
    holidays_dict = {
        2026: [(26,1),(3,3),(26,3),(31,3),(3,4),(14,4),(1,5),(28,5),(26,6),(14,9),(2,10),(20,10),(24,11),(25,12)],
        2025: [(26,2),(14,3),(31,3),(10,4),(14,4),(18,4),(1,5),(15,8),(27,8),(2,10),(21,10),(22,10),(5,11),(25,12)],
        2024: [(22,1),(26,1),(8,3),(25,3),(29,3),(11,4),(17,4),(1,5),(17,6),(15,8),(2,10),(1,11),(15,11),(20,11),(25,12)],
        2023: [(26,1),(7,3),(30,3),(4,4),(7,4),(14,4),(1,5),(28,6),(15,8),(19,9),(2,10),(24,10),(14,11),(27,11),(25,12)],
        2022: [(26,1),(1,3),(18,3),(14,4),(15,4),(3,5),(9,8),(15,8),(31,8),(5,10),(24,10),(26,10),(8,11)],
        2021: [(26,1),(11,3),(29,3),(2,4),(14,4),(21,4),(13,5),(21,7),(19,8),(10,9),(15,10),(4,11),(5,11),(19,11)],
        2020: [(26,1),(21,2),(6,4),(10,4),(14,4),(7,5),(31,7),(12,8),(15,8),(29,8),(2,10),(25,10),(29,10),(14,11),(30,11),(25,12)],
    }
    out: list[date] = []
    for year, dms in holidays_dict.items():
        for d, m in dms:
            out.append(date(year, m, d))
    return tuple(out)

DEFAULT_HOLIDAY_LIST = set(last_5yr_holidays())

def as_aware_ist(dt_like) -> datetime:
    if isinstance(dt_like, pd.Timestamp):
        if dt_like.tzinfo is None:
            return dt_like.tz_localize("Asia/Kolkata").to_pydatetime()
        return dt_like.tz_convert("Asia/Kolkata").to_pydatetime()
    if isinstance(dt_like, datetime):
        if dt_like.tzinfo is None:
            return dt_like.replace(tzinfo=IST)
        return dt_like.astimezone(IST)
    if isinstance(dt_like, date):
        return datetime(dt_like.year, dt_like.month, dt_like.day, 0, 0, tzinfo=IST)
    raise TypeError(f"Unsupported datetime-like: {type(dt_like)}")

def is_trading_day(d: date, holiday_list: set[date]) -> bool:
    return (d.weekday() < 5) and (d not in holiday_list)

def get_prev_valid_date(now_dt: datetime, holidays: set[date]) -> date:
    now_dt = as_aware_ist(now_dt)
    prev_date = now_dt.date() - timedelta(days=1)
    while prev_date.weekday() >= 5 or prev_date in holidays:
        prev_date -= timedelta(days=1)
    return prev_date

def T_years_intraday(
    expiry_date,
    now_dt: datetime,
    cutoff: time = time(15, 30),
    holiday_list: set[date] | None = None,
    floor_after_cutoff: bool = True
) -> float:
    if expiry_date is None:
        return float("nan")

    holiday_list = holiday_list or set()
    now_dt = as_aware_ist(now_dt)

    today = now_dt.date()
    expiry_date = expiry_date.date() if isinstance(expiry_date, datetime) else expiry_date

    seconds = 0.0

    # expiry today
    if expiry_date == today:
        if not is_trading_day(today, holiday_list):
            return 1e-8 if floor_after_cutoff else 0.0
        expiry_dt_ = as_aware_ist(datetime.combine(today, cutoff))
        seconds = (expiry_dt_ - now_dt).total_seconds()
        if seconds <= 0:
            return 1e-8 if floor_after_cutoff else 0.0
        return seconds / SECONDS_IN_YEAR

    # future expiry
    if is_trading_day(today, holiday_list):
        today_cutoff_dt = as_aware_ist(datetime.combine(today, cutoff))
        if now_dt < today_cutoff_dt:
            seconds += (today_cutoff_dt - now_dt).total_seconds()

    d = today + timedelta(days=1)
    while d < expiry_date:
        if is_trading_day(d, holiday_list):
            seconds += SECONDS_IN_DAY
        d += timedelta(days=1)

    if is_trading_day(expiry_date, holiday_list):
        seconds += (
            datetime.combine(expiry_date, cutoff) - datetime.combine(expiry_date, time(0, 0))
        ).total_seconds()

    if seconds <= 0:
        return 1e-8 if floor_after_cutoff else 0.0
    return seconds / SECONDS_IN_YEAR


# =============================================================================
# SYMBOL PARSING
# =============================================================================

MONTH3_TO_INT = {m.upper(): i for i, m in enumerate(calendar.month_abbr) if m}

_SPOT      = re.compile(r"^[A-Z]+$")
_FUT_MON   = re.compile(r"^(?P<ul>[A-Z]+)(?P<yy>\d{2})(?P<mon>[A-Z]{3})FUT$")
_OPT_MON   = re.compile(r"^(?P<ul>[A-Z]+)(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<strike>\d+)(?P<opt>CE|PE)$")
_OPT_WK = re.compile(r"^(?P<ul>[A-Z]+)(?P<yy>\d{2})(?P<mon1>[A-Z0-9])(?P<dd>\d{2})(?P<strike>\d+)(?P<opt>CE|PE)$")

_OPT_SPACED = re.compile(r"^(?P<ul>[A-Z]+)\s+(?P<dd>\d{2})(?P<mon>[A-Z]{3})(?P<yyyy>\d{4})\s+(?P<opt>CE|PE)\s+(?P<strike>\d+)$")
_FUT_SPACED = re.compile(r"^(?P<ul>[A-Z]+)\s+(?P<dd>\d{2})(?P<mon>[A-Z]{3})(?P<yyyy>\d{4})\s+FUT$")

DEFAULT_WEEKLY_MONTH_MAP = {str(i): i for i in range(1, 10)}
DEFAULT_WEEKLY_MONTH_MAP.update({"O": 10, "N": 11, "D": 12})

DEFAULT_MONTHLY_EXPIRY_WEEKDAY = {
    "NIFTY":calendar.TUESDAY, "BANKNIFTY":calendar.TUESDAY, "FINNIFTY":calendar.TUESDAY,
    "MIDCPNIFTY":calendar.TUESDAY, "SENSEX":calendar.THURSDAY, "BANKEX":calendar.FRIDAY,
}

@dataclass(frozen=True)
class Canonical:
    ul: str
    kind: str           # 'SPOT' | 'FUT' | 'OPT'
    expiry: date | None
    opt: str | None
    strike: int | None

def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    cal_ = calendar.monthcalendar(year, month)
    days = [wk[weekday] for wk in cal_ if wk[weekday] != 0]
    expday = date(year, month, days[-1])
    while (expday in DEFAULT_HOLIDAY_LIST) or (expday.weekday() >= 5):
        expday = expday - timedelta(days=1)
    return expday

@lru_cache(maxsize=50000)
def parse_symbol(sym: str) -> Canonical | None:
    if not sym:
        return None
    s = str(sym).strip().upper()

    m = _OPT_SPACED.fullmatch(s)
    if m:
        ul, dd, mon3, yyyy = m.group("ul"), int(m.group("dd")), m.group("mon"), int(m.group("yyyy"))
        mon = MONTH3_TO_INT.get(mon3)
        if not mon:
            return None
        try:
            exp = date(yyyy, mon, dd)
        except ValueError:
            return None
        return Canonical(ul, "OPT", exp, m.group("opt"), int(m.group("strike")))

    m = _FUT_SPACED.fullmatch(s)
    if m:
        ul, dd, mon3, yyyy = m.group("ul"), int(m.group("dd")), m.group("mon"), int(m.group("yyyy"))
        mon = MONTH3_TO_INT.get(mon3)
        if not mon:
            return None
        try:
            exp = date(yyyy, mon, dd)
        except ValueError:
            return None
        return Canonical(ul, "FUT", exp, None, None)

    s_nospace = s.replace(" ", "")
    if _SPOT.fullmatch(s_nospace):
        return Canonical(s_nospace, "SPOT", None, None, None)

    m = _FUT_MON.fullmatch(s_nospace)
    if m:
        ul, yy, mon3 = m.group("ul"), int(m.group("yy")), m.group("mon")
        mon = MONTH3_TO_INT.get(mon3)
        if not mon:
            return None
        wd = DEFAULT_MONTHLY_EXPIRY_WEEKDAY.get(ul, calendar.THURSDAY)
        exp = _last_weekday_of_month(2000 + yy, mon, wd)
        return Canonical(ul, "FUT", exp, None, None)

    m = _OPT_MON.fullmatch(s_nospace)
    if m:
        ul, yy, mon3 = m.group("ul"), int(m.group("yy")), m.group("mon")
        mon = MONTH3_TO_INT.get(mon3)
        if not mon:
            return None
        wd = DEFAULT_MONTHLY_EXPIRY_WEEKDAY.get(ul, calendar.THURSDAY)
        exp = _last_weekday_of_month(2000 + yy, mon, wd)
        return Canonical(ul, "OPT", exp, m.group("opt"), int(m.group("strike")))

    m = _OPT_WK.fullmatch(s_nospace)
    if m:
        ul, yy, mon1, dd = m.group("ul"), int(m.group("yy")), m.group("mon1"), int(m.group("dd"))
        mon = DEFAULT_WEEKLY_MONTH_MAP.get(mon1)
        exp = None
        if mon:
            try:
                exp = date(2000 + yy, mon, dd)
            except ValueError:
                exp = None
        return Canonical(ul, "OPT", exp, m.group("opt"), int(m.group("strike")))

    return None

def to_symbol_tradebook_spaced(c: Canonical | None) -> str | None:
    if c is None:
        return None
    if c.kind == "SPOT":
        return c.ul
    if not isinstance(c.expiry, date):
        return None
    mon3 = calendar.month_abbr[c.expiry.month].upper()
    if c.kind == "FUT":
        return f"{c.ul} {c.expiry.day:02d}{mon3}{c.expiry.year} FUT"
    if c.kind == "OPT" and c.opt and c.strike is not None:
        return f"{c.ul} {c.expiry.day:02d}{mon3}{c.expiry.year} {c.opt} {c.strike}"
    return None

def ltp_to_tradebook_spaced_df(ltp_df: pd.DataFrame, symbol_col="symbol", out_col="TradebookLike") -> pd.DataFrame:
    out = ltp_df.copy()
    out[out_col] = out[symbol_col].astype(str).map(lambda s: to_symbol_tradebook_spaced(parse_symbol(s)))
    return out

def build_spot_map(ltp_df: pd.DataFrame) -> dict[str, float]:
    spot_map: dict[str, float] = {}
    if ltp_df is None or ltp_df.empty or "symbol" not in ltp_df.columns or "ltp" not in ltp_df.columns:
        return spot_map

    sym = ltp_df["symbol"].astype(str)
    spot_rows = ltp_df[sym.str.fullmatch(r"[A-Z]+", case=False, na=False)].copy()
    for _, r in spot_rows.iterrows():
        try:
            ul = str(r["symbol"]).upper()
            spot_map[ul] = float(r["ltp"])
        except Exception:
            continue

    fut_rows = ltp_df[sym.str.contains("FUT", case=False, na=False)].copy()
    if not fut_rows.empty:
        fut_rows["_canon"] = fut_rows["symbol"].astype(str).map(parse_symbol)
        fut_rows = fut_rows[fut_rows["_canon"].notna()].copy()
        if not fut_rows.empty:
            fut_rows["ul"] = fut_rows["_canon"].map(lambda c: c.ul)
            fut_rows["expiry"] = fut_rows["_canon"].map(lambda c: c.expiry)
            fut_rows = fut_rows[pd.notna(fut_rows["expiry"])].copy()
            fut_rows["ltp"] = pd.to_numeric(fut_rows["ltp"], errors="coerce")
            fut_rows = fut_rows.dropna(subset=["ltp", "expiry"]).sort_values(["ul", "expiry"])
            for ul, grp in fut_rows.groupby("ul", sort=False):
                if ul.upper() not in spot_map:
                    try:
                        spot_map[ul.upper()] = float(grp.iloc[0]["ltp"])
                    except Exception:
                        pass
    return spot_map

def build_synth_spot_by_expiry_from_atm_cp(ltp_df: pd.DataFrame) -> tuple[dict[tuple[str, date], float], pd.DataFrame]:
    """
    Synthetic spot via ATM put-call parity (NO discounting):
        S_syn = K_atm + C_atm - P_atm
    """
    if ltp_df is None or ltp_df.empty or "symbol" not in ltp_df.columns or "ltp" not in ltp_df.columns:
        return {}, pd.DataFrame()

    df = ltp_df.copy()
    df["_canon"] = df["symbol"].astype(str).map(parse_symbol)
    df = df[df["_canon"].notna()].copy()
    if df.empty:
        return {}, pd.DataFrame()

    df["ul"] = df["_canon"].map(lambda c: str(c.ul).upper())
    df["kind"] = df["_canon"].map(lambda c: c.kind)
    df["expiry"] = df["_canon"].map(lambda c: c.expiry)
    df["opt"] = df["_canon"].map(lambda c: c.opt)
    df["strike"] = df["_canon"].map(lambda c: c.strike)
    df["ltp"] = pd.to_numeric(df["ltp"], errors="coerce")

    opt = df[(df["kind"] == "OPT") & df["expiry"].notna() & df["strike"].notna()].copy()
    opt = opt.dropna(subset=["ul", "expiry", "opt", "strike", "ltp"])
    if opt.empty:
        return {}, pd.DataFrame()

    piv = opt.pivot_table(
        index=["ul", "expiry", "strike"],
        columns="opt",
        values="ltp",
        aggfunc="last"
    ).reset_index()

    if "CE" not in piv.columns or "PE" not in piv.columns:
        return {}, pd.DataFrame()

    piv = piv.dropna(subset=["CE", "PE"]).copy()
    if piv.empty:
        return {}, pd.DataFrame()

    piv["abs_cp"] = (piv["CE"] - piv["PE"]).abs()

    out_map: dict[tuple[str, date], float] = {}
    dbg_rows: list[dict] = []

    for (ul, exp), grp in piv.groupby(["ul", "expiry"], sort=False):
        pick = grp.sort_values("abs_cp").iloc[0]
        K = float(pick["strike"])
        C = float(pick["CE"])
        P = float(pick["PE"])
        S_syn = K + (C - P)
        out_map[(str(ul).upper(), exp)] = float(S_syn)
        dbg_rows.append({"ul": str(ul).upper(), "expiry": exp, "K_atm": K, "C_atm": C, "P_atm": P, "S_syn": float(S_syn)})

    return out_map, pd.DataFrame(dbg_rows)

def build_synth_spot_by_expiry_from_atm_cp_updated(
    ltp_df: pd.DataFrame,
    spot_map: dict[str, float],
    pct_window: float = 0.02,      # ±2% window around spot
    min_pairs: int = 3,            # need at least this many strikes to use median
    clamp_pct: float = 0.05        # if synth deviates >5%, fall back to spot
) -> tuple[dict[tuple[str, date], float], pd.DataFrame]:
    print("using median type synthetic")

    if ltp_df is None or ltp_df.empty or "symbol" not in ltp_df.columns or "ltp" not in ltp_df.columns:
        return {}, pd.DataFrame()

    df = ltp_df.copy()
    df["_canon"] = df["symbol"].astype(str).map(parse_symbol)
    df = df[df["_canon"].notna()].copy()
    if df.empty:
        return {}, pd.DataFrame()

    df["ul"]     = df["_canon"].map(lambda c: str(c.ul).upper())
    df["kind"]   = df["_canon"].map(lambda c: c.kind)
    df["expiry"] = df["_canon"].map(lambda c: c.expiry)
    df["opt"]    = df["_canon"].map(lambda c: str(c.opt).upper() if c.opt is not None else None)
    df["strike"] = pd.to_numeric(df["_canon"].map(lambda c: c.strike), errors="coerce")
    df["ltp"]    = pd.to_numeric(df["ltp"], errors="coerce")

    opt = df[(df["kind"] == "OPT") & df["expiry"].notna() & df["strike"].notna()].copy()
    opt = opt.dropna(subset=["ul", "expiry", "opt", "strike", "ltp"])
    opt = opt[opt["opt"].isin(["CE", "PE"])].copy()
    if opt.empty:
        return {}, pd.DataFrame()

    piv = (
        opt.pivot_table(index=["ul","expiry","strike"], columns="opt", values="ltp", aggfunc="last")
           .reset_index()
    )
    if "CE" not in piv.columns or "PE" not in piv.columns:
        return {}, pd.DataFrame()

    piv = piv.dropna(subset=["CE","PE"]).copy()

    out_map = {}
    dbg_rows = []

    for (ul, exp), grp in piv.groupby(["ul","expiry"], sort=False):
        S_ref = float(spot_map.get(str(ul).upper(), np.nan))
        if not np.isfinite(S_ref) or S_ref <= 0:
            continue

        lo, hi = S_ref * (1 - pct_window), S_ref * (1 + pct_window)
        w = grp[(grp["strike"] >= lo) & (grp["strike"] <= hi)].copy()

        # if window too sparse, fall back to nearest strikes
        if len(w) < min_pairs:
            w = grp.assign(dist=(grp["strike"] - S_ref).abs()).sort_values("dist").head(max(min_pairs, 5)).copy()

        w["F_i"] = w["strike"] + (w["CE"] - w["PE"])
        F_med = float(np.nanmedian(w["F_i"].to_numpy()))

        # sanity clamp
        if not np.isfinite(F_med) or abs(F_med / S_ref - 1.0) > clamp_pct:
            S_syn = S_ref
            note = "FALLBACK_TO_SPOT"
        else:
            S_syn = F_med
            note = "MEDIAN_PARITY"

        out_map[(str(ul).upper(), exp)] = float(S_syn)
        dbg_rows.append({
            "ul": str(ul).upper(), "expiry": exp,
            "S_ref": S_ref, "S_syn": S_syn, "note": note,
            "n_strikes_used": int(len(w)),
            "F_min": float(np.nanmin(w["F_i"])), "F_max": float(np.nanmax(w["F_i"]))
        })

    return out_map, pd.DataFrame(dbg_rows)

# =============================================================================
# FIFO ROLL (unchanged logic from your script)
# =============================================================================

def filter_expired_positions(df: pd.DataFrame, sym_col: str, now_dt: datetime, cutoff: time) -> pd.DataFrame:
    parsed = df[sym_col].astype(str).map(parse_symbol)
    df2 = df.copy()
    df2["_expiry"] = parsed.map(lambda c: c.expiry if c else None)

    def alive(exp):
        if not isinstance(exp, date):
            return True
        exp_dt = datetime(exp.year, exp.month, exp.day, cutoff.hour, cutoff.minute, tzinfo=IST)
        return exp_dt > as_aware_ist(now_dt)

    return df2[df2["_expiry"].map(alive)].drop(columns=["_expiry"], errors="ignore")

def _prepare_prev_map(prev_df: pd.DataFrame, sym_col: str, sym_col_prev: str | None,
                      prev_qty_col: str | None, prev_avg_col: str | None) -> dict[str, tuple[float, float]]:
    if (
        prev_df is None or prev_df.empty or
        sym_col_prev is None or prev_qty_col is None or prev_avg_col is None or
        sym_col_prev not in prev_df.columns or prev_qty_col not in prev_df.columns or prev_avg_col not in prev_df.columns
    ):
        return {}

    prev = prev_df[[sym_col_prev, prev_qty_col, prev_avg_col]].copy()
    prev.columns = [sym_col, "qty", "avg"]
    prev["qty"] = pd.to_numeric(prev["qty"], errors="coerce").fillna(0.0)
    prev["avg"] = pd.to_numeric(prev["avg"], errors="coerce")
    prev_agg = prev.groupby(sym_col, as_index=False).agg({"qty": "sum", "avg": "last"})

    out: dict[str, tuple[float, float]] = {}
    for _, r in prev_agg.iterrows():
        s = str(r[sym_col])
        q = float(r["qty"])
        a = float(r["avg"]) if pd.notna(r["avg"]) else float("nan")
        out[s] = (q, a)
    return out

def _prepare_trades_df(today_df: pd.DataFrame, sym_col: str, qty_col_today: str | None, avg_col_today: str | None) -> pd.DataFrame:
    if (
        today_df is None or today_df.empty or
        sym_col not in today_df.columns or
        qty_col_today is None or avg_col_today is None or
        qty_col_today not in today_df.columns or avg_col_today not in today_df.columns
    ):
        return pd.DataFrame(columns=[sym_col, "t_qty", "t_px"])

    cols = [sym_col, qty_col_today, avg_col_today]
    time_col = None
    for candidate in ["OrderGeneratedDateTime", "LastUpdateDateTime", "ExchangeTransactTime"]:
        if candidate in today_df.columns:
            cols.append(candidate)
            time_col = candidate
            break

    trades = today_df[cols].copy()
    if time_col:
        trades = trades.sort_values(time_col).drop(columns=[time_col])

    trades.columns = [sym_col, "t_qty", "t_px"]
    trades["t_qty"] = pd.to_numeric(trades["t_qty"], errors="coerce").fillna(0.0)
    trades["t_px"]  = pd.to_numeric(trades["t_px"],  errors="coerce").fillna(0.0)
    return trades

def _fifo_simulate_symbol(prev_qty: float, prev_avg: float, trades_df: pd.DataFrame) -> dict[str, float]:
    lots: list[dict[str, float | str]] = []
    realized = 0.0

    pq = float(prev_qty or 0.0)
    try:
        pa = float(prev_avg)
    except Exception:
        pa = float("nan")

    if abs(pq) > 1e-8 and math.isfinite(pa):
        lots.append({"qty": pq, "px": pa, "src": "prev"})

    if trades_df is not None and not trades_df.empty:
        for _, tr in trades_df.iterrows():
            q = float(tr["t_qty"])
            px = float(tr["t_px"])
            if abs(q) < 1e-8:
                continue

            trade_left = q
            if not lots:
                lots.append({"qty": trade_left, "px": px, "src": "today"})
                continue

            i = 0
            while i < len(lots) and abs(trade_left) > 1e-8:
                lot = lots[i]
                lot_qty = float(lot["qty"])
                if lot_qty * trade_left > 0:
                    i += 1
                    continue

                closable = min(abs(lot_qty), abs(trade_left))
                closed_qty = math.copysign(closable, lot_qty)
                realized += (px - float(lot["px"])) * closed_qty

                lot["qty"] = lot_qty - closed_qty
                trade_left = trade_left + closed_qty

                if abs(float(lot["qty"])) < 1e-8:
                    lots.pop(i)
                else:
                    i += 1

            if abs(trade_left) > 1e-8:
                lots.append({"qty": trade_left, "px": px, "src": "today"})

    net_qty_end = sum(float(l["qty"]) for l in lots)
    if abs(net_qty_end) > 1e-8:
        carry_notional = sum(float(l["qty"]) * float(l["px"]) for l in lots)
        carry_avg_end = carry_notional / net_qty_end
    else:
        carry_avg_end = 0.0

    q_overnight = sum(float(l["qty"]) for l in lots if l.get("src") == "prev")
    if abs(q_overnight) > 1e-8:
        prev_notional = sum(float(l["qty"]) * float(l["px"]) for l in lots if l.get("src") == "prev")
        prev_close = prev_notional / q_overnight
    else:
        prev_close = float("nan")

    q_today = sum(float(l["qty"]) for l in lots if l.get("src") == "today")
    if abs(q_today) > 1e-8:
        today_notional = sum(float(l["qty"]) * float(l["px"]) for l in lots if l.get("src") == "today")
        today_avg = today_notional / q_today
    else:
        today_avg = float("nan")

    return {
        "NetQty_end": net_qty_end,
        "CarryAvg_end": carry_avg_end,
        "RealizedPnL_day": realized,
        "Q_overnight": q_overnight,
        "PrevClose": prev_close,
        "Q_today": q_today,
        "TodayBuyAvg": today_avg,
    }

def roll_positions_weighted_average(
    prev_df: pd.DataFrame,
    today_df: pd.DataFrame,
    sym_col: str,
    sym_col_prev: str,
    qty_col_today: str,
    avg_col_today: str,
    prev_qty_col: str,
    prev_avg_col: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prev_map = _prepare_prev_map(prev_df, sym_col, sym_col_prev, prev_qty_col, prev_avg_col)
    trades_all = _prepare_trades_df(today_df, sym_col, qty_col_today, avg_col_today)

    syms_prev  = set(prev_map.keys())
    syms_today = set(trades_all[sym_col].dropna().unique()) if not trades_all.empty else set()
    all_syms   = sorted(syms_prev | syms_today)

    rolled_rows: list[dict] = []
    realized_rows: list[dict] = []
    mtm_rows: list[dict] = []

    for s in all_syms:
        prev_qty, prev_avg = prev_map.get(s, (0.0, float("nan")))
        trades_sym = trades_all[trades_all[sym_col] == s][["t_qty", "t_px"]].copy() if not trades_all.empty else pd.DataFrame(columns=["t_qty", "t_px"])
        res = _fifo_simulate_symbol(prev_qty, prev_avg, trades_sym)

        realized_rows.append({sym_col: s, "NetQty_end": res["NetQty_end"], "CarryAvg_end": res["CarryAvg_end"], "RealizedPnL_day": res["RealizedPnL_day"]})

        if abs(res["NetQty_end"]) > 1e-8:
            rolled_rows.append({sym_col: s, "NetQty": res["NetQty_end"], "CarryAvg": res["CarryAvg_end"], "RealizedPnL": res["RealizedPnL_day"]})

        mtm_rows.append({sym_col: s, "Q_overnight": res["Q_overnight"], "PrevClose": res["PrevClose"], "Q_today": res["Q_today"], "TodayBuyAvg": res["TodayBuyAvg"]})

    rolled = pd.DataFrame(rolled_rows, columns=[sym_col, "NetQty", "CarryAvg", "RealizedPnL"])
    realized_all = pd.DataFrame(realized_rows, columns=[sym_col, "NetQty_end", "CarryAvg_end", "RealizedPnL_day"])
    mtm_components = pd.DataFrame(mtm_rows, columns=[sym_col, "Q_overnight", "PrevClose", "Q_today", "TodayBuyAvg"])

    if not rolled.empty:
        rolled["ul"] = rolled[sym_col].astype(str).str.split().str[0]
    else:
        rolled["ul"] = []

    if not realized_all.empty:
        realized_all["ul"] = realized_all[sym_col].astype(str).str.split().str[0]
    else:
        realized_all["ul"] = []

    if not mtm_components.empty:
        for c in ["Q_overnight", "Q_today"]:
            if c in mtm_components.columns:
                mtm_components[c] = pd.to_numeric(mtm_components[c], errors="coerce").fillna(0.0)

    return rolled, realized_all, mtm_components


# =============================================================================
# P&L HELPERS (day + carry + expenses)
# =============================================================================

def compute_day_pnl_from_trades(tb_today: pd.DataFrame, ltp_df: pd.DataFrame, sym_col: str, qty_col: str, avg_col: str) -> pd.DataFrame:
    if tb_today is None or tb_today.empty:
        return pd.DataFrame(columns=[sym_col, "ul", "expiry", "DayPnL"])

    ltp_conv = ltp_to_tradebook_spaced_df(ltp_df, symbol_col="symbol", out_col="TradebookLike")
    ltp_agg = (
        ltp_conv.dropna(subset=["TradebookLike"])
                .drop_duplicates(subset=["TradebookLike"], keep="last")
                .rename(columns={"ltp": "LTP"})[["TradebookLike", "LTP"]]
    )

    df = tb_today[[sym_col, qty_col, avg_col]].copy()
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0)
    df[avg_col] = pd.to_numeric(df[avg_col], errors="coerce")
    df["TradebookLike"] = df[sym_col].astype(str)

    df = df.merge(ltp_agg, on="TradebookLike", how="left")
    df["LTP"] = pd.to_numeric(df["LTP"], errors="coerce")
    df["DayPnL_contrib"] = df[qty_col] * (df["LTP"] - df[avg_col])

    parsed = df[sym_col].astype(str).map(parse_symbol)
    df["ul"] = parsed.map(lambda c: c.ul if c else None)
    df["expiry"] = parsed.map(lambda c: c.expiry if c else None)
    df["ul"] = df["ul"].fillna(df[sym_col].astype(str).str.split().str[0])

    out = (
        df.groupby([sym_col, "ul", "expiry"], as_index=False)["DayPnL_contrib"]
          .sum(min_count=1)
          .rename(columns={"DayPnL_contrib": "DayPnL"})
    )
    return out

def compute_carry_pnl_from_prev(prev_df: pd.DataFrame, ltp_df: pd.DataFrame, sym_col_prev: str,
                               qty_col_prev: str, avg_col_prev: str, tradebook_like_col: str = "PrevSymTradebookLike") -> pd.DataFrame:
    if prev_df is None or prev_df.empty:
        return pd.DataFrame(columns=["SymbolPrev", "ul", "expiry", "CarryPnL"])

    df = prev_df.copy()
    if tradebook_like_col in df.columns:
        df["TradebookLike"] = df[tradebook_like_col].astype(str)
    else:
        df["TradebookLike"] = df[sym_col_prev].astype(str).map(lambda s: to_symbol_tradebook_spaced(parse_symbol(s)))

    df["qty_prev"] = pd.to_numeric(df[qty_col_prev], errors="coerce").fillna(0.0)
    df["prev_px"] = pd.to_numeric(df[avg_col_prev], errors="coerce")

    ltp_conv = ltp_to_tradebook_spaced_df(ltp_df, symbol_col="symbol", out_col="TradebookLike")
    ltp_agg = (
        ltp_conv.dropna(subset=["TradebookLike"])
                .drop_duplicates(subset=["TradebookLike"], keep="last")
                .rename(columns={"ltp": "LTP"})[["TradebookLike", "LTP"]]
    )
    df = df.merge(ltp_agg, on="TradebookLike", how="left")
    df["LTP"] = pd.to_numeric(df["LTP"], errors="coerce")

    df["CarryPnL"] = df["qty_prev"] * (df["LTP"] - df["prev_px"])

    parsed = df[sym_col_prev].astype(str).map(parse_symbol)
    df["ul"] = parsed.map(lambda c: c.ul if c else None)
    df["expiry"] = parsed.map(lambda c: c.expiry if c else None)
    df["ul"] = df["ul"].fillna(df[sym_col_prev].astype(str).str.split().str[0])

    out = (
        df.groupby(["TradebookLike", "ul", "expiry"], as_index=False)["CarryPnL"]
          .sum(min_count=1)
          .rename(columns={"TradebookLike": "SymbolPrev"})
    )
    return out

def compute_trading_expenses(tb_today: pd.DataFrame, sym_col: str, qty_col: str, avg_col: str, cost_per_cr: float = 10000.0) -> pd.DataFrame:
    if tb_today is None or tb_today.empty:
        return pd.DataFrame(columns=["ul", "expiry", "PremiumTraded", "Expenses"])

    df = tb_today[[sym_col, qty_col, avg_col]].copy()
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0)
    df[avg_col] = pd.to_numeric(df[avg_col], errors="coerce")

    parsed = df[sym_col].astype(str).map(parse_symbol)
    df["ul"] = parsed.map(lambda c: c.ul if c else None)
    df["kind"] = parsed.map(lambda c: c.kind if c else None)
    df["expiry"] = parsed.map(lambda c: c.expiry if c else None)

    df = df[df["kind"] == "OPT"].copy()
    df = df.dropna(subset=["ul"])
    df["PremiumTraded"] = df[qty_col].abs() * df[avg_col]
    df["PremiumTraded"] = pd.to_numeric(df["PremiumTraded"], errors="coerce").fillna(0.0)

    out = df.groupby(["ul", "expiry"], as_index=False)["PremiumTraded"].sum()
    out["Expenses"] = cost_per_cr * (out["PremiumTraded"] / 1e7)
    return out


# =============================================================================
# VECTORIZED BS / IV
# =============================================================================

MIN_IV = 0.03
MAX_IV = 3.0

_SQRT2 = np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)

def _erf_approx_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * ax)
    tau = t * np.exp(
        -ax*ax - 1.26551223
        + 1.00002368*t
        + 0.37409196*t**2
        + 0.09678418*t**3
        - 0.18628806*t**4
        + 0.27886807*t**5
        - 1.13520398*t**6
        + 1.48851587*t**7
        - 0.82215223*t**8
        + 0.17087277*t**9
    )
    return sign * (1.0 - tau)

def _norm_cdf_vec(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf_approx_vec(x / _SQRT2))

def _norm_pdf_vec(x: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)

def _bs_price_vec(S, K, T, r, q, sigma, cp):
    S = np.asarray(S, float); K = np.asarray(K, float); T = np.asarray(T, float)
    sigma = np.asarray(sigma, float); cp = np.asarray(cp, float)
    sqrtT = np.sqrt(T)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma*sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    Nd1 = _norm_cdf_vec(cp * d1)
    Nd2 = _norm_cdf_vec(cp * d2)
    return cp * (S * disc_q * Nd1 - K * disc_r * Nd2)

def implied_vol_vec(price, S, K, T, r, q, cp, tol=1e-6, max_iter=60):
    price = np.asarray(price, float)
    S     = np.asarray(S, float)
    K     = np.asarray(K, float)
    T     = np.asarray(T, float)
    cp    = np.asarray(cp, float)

    out = np.full(price.shape, np.nan, dtype=float)

    ok = np.isfinite(price) & np.isfinite(S) & np.isfinite(K) & np.isfinite(T) & np.isfinite(cp)
    ok &= (price > 0) & (S > 0) & (K > 0) & (T > 0) & ((cp == 1) | (cp == -1))
    if not np.any(ok):
        return out

    idx = np.where(ok)[0]
    p = price[idx]; s = S[idx]; k = K[idx]; t = T[idx]; c = cp[idx]

    intrinsic = np.maximum(c * (s - k), 0.0)
    upper = np.where(c == 1.0, s * np.exp(-q * t), k * np.exp(-r * t))
    ok2 = (p >= intrinsic - 1e-6) & (p <= upper + 1e-6)
    if not np.any(ok2):
        return out

    idx2 = idx[ok2]
    p2 = price[idx2]; s2 = S[idx2]; k2 = K[idx2]; t2 = T[idx2]; c2 = cp[idx2]

    lo = np.full(p2.shape, MIN_IV, dtype=float)
    hi = np.full(p2.shape, MAX_IV, dtype=float)

    flo = _bs_price_vec(s2, k2, t2, r, q, lo, c2) - p2
    fhi = _bs_price_vec(s2, k2, t2, r, q, hi, c2) - p2

    need = (flo * fhi > 0)
    for _ in range(4):
        if not np.any(need):
            break
        hi[need] = np.minimum(hi[need] * 1.5, 5.0)
        fhi[need] = _bs_price_vec(s2[need], k2[need], t2[need], r, q, hi[need], c2[need]) - p2[need]
        need = (flo * fhi > 0) & (hi < 5.0)

    valid = (flo * fhi <= 0)
    if not np.any(valid):
        return out

    lo_v = lo.copy(); hi_v = hi.copy()
    flo_v = flo.copy(); fhi_v = fhi.copy()

    for _ in range(max_iter):
        mid = 0.5 * (lo_v + hi_v)
        fmid = _bs_price_vec(s2, k2, t2, r, q, mid, c2) - p2
        done = np.abs(fmid) < tol
        if np.all(done | ~valid):
            break

        left = (flo_v * fmid <= 0)
        hi_v = np.where(left, mid, hi_v)
        fhi_v = np.where(left, fmid, fhi_v)
        lo_v = np.where(~left, mid, lo_v)
        flo_v = np.where(~left, fmid, flo_v)

    iv = 0.5 * (lo_v + hi_v)
    iv = np.where(valid, iv, np.nan)
    out[idx2] = iv
    return out

def bs_greeks_vec(S, K, T, r, q, sigma, cp):
    S = np.asarray(S, float); K = np.asarray(K, float); T = np.asarray(T, float)
    sigma = np.asarray(sigma, float); cp = np.asarray(cp, float)
    sqrtT = np.sqrt(T)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma*sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    pdf1 = _norm_pdf_vec(d1)
    Nd1  = _norm_cdf_vec(cp * d1)
    Nd2  = _norm_cdf_vec(cp * d2)

    price = cp * (S * disc_q * Nd1 - K * disc_r * Nd2)
    delta = cp * disc_q * Nd1
    gamma = disc_q * pdf1 / (S * sigma * sqrtT)
    vega  = S * disc_q * pdf1 * sqrtT
    theta = (-S * disc_q * pdf1 * sigma / (2 * sqrtT)
             + cp * q * S * disc_q * Nd1
             - cp * r * K * disc_r * Nd2)
    rho   = cp * K * T * disc_r * Nd2

    volga = vega * d1 * d2 / sigma
    vanna = -disc_q * pdf1 * d2 / sigma

    dd1_dT = (2.0*T*(r - q) - d2*sigma*sqrtT) / (2.0*T*sigma*sqrtT)
    dDelta_dT = (-cp*q*disc_q*Nd1) + (disc_q*pdf1*dd1_dT)
    charm = -dDelta_dT

    return {
        "price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho,
        "vanna": vanna, "volga": volga, "charm": charm,
    }


# =============================================================================
# LTP CONTEXT (compute once per tick, reuse for all users)
# =============================================================================

def prepare_ltp_context(ltp_df: pd.DataFrame) -> dict[str, Any]:
    """
    Precompute:
    - ltp_agg: TradebookLike -> LTP
    - spot_map: UL -> spot
    - parity_map: (UL, expiry) -> synthetic spot
    - parity_dbg: debug df
    """
    ltp_df = ltp_df.copy()
    ltp_df["ltp"] = pd.to_numeric(ltp_df["ltp"], errors="coerce")

    ltp_conv = ltp_to_tradebook_spaced_df(ltp_df, symbol_col="symbol", out_col="TradebookLike")
    ltp_agg = (
        ltp_conv.dropna(subset=["TradebookLike"])
                .drop_duplicates(subset=["TradebookLike"], keep="last")
                .rename(columns={"ltp": "LTP"})[["TradebookLike", "LTP"]]
    )

    spot_map = build_spot_map(ltp_df)
    parity_map, parity_dbg = build_synth_spot_by_expiry_from_atm_cp_updated(ltp_df,spot_map)

    return {
        "ltp_raw": ltp_df,
        "ltp_agg": ltp_agg,
        "spot_map": spot_map,
        "parity_map": parity_map,
        "parity_dbg": parity_dbg,
    }


# =============================================================================
# ENGINE (FAST): vectorized IV+greeks, scenarios disabled
# =============================================================================

def run_engine_fast_from_ctx(
    tb_df: pd.DataFrame,
    ltp_ctx: dict[str, Any],
    tb_symbol_col: str,
    qty_col: str,
    avg_col: str,
    rf: float,
    q_div: float,
    now_dt: datetime,
    cutoff: time,
    spot_mode: str = "Synthetic (K_atm+(C_atm - P_atm))",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    """
    Returns:
      tbm (leg-level)
      pf_greeks (MultiIndex underlying, expiry)
      scenarios (empty dict by design)
      parity_dbg (debug df from ltp_ctx)
    """
    if tb_df is None or tb_df.empty:
        empty = pd.DataFrame()
        return empty, empty, {}, ltp_ctx.get("parity_dbg", pd.DataFrame())

    ltp_agg = ltp_ctx["ltp_agg"]
    spot_map: dict[str, float] = ltp_ctx["spot_map"]
    parity_map: dict[tuple[str, date], float] = ltp_ctx["parity_map"]
    parity_dbg: pd.DataFrame = ltp_ctx.get("parity_dbg", pd.DataFrame())

    tbm = tb_df.merge(ltp_agg, left_on=tb_symbol_col, right_on="TradebookLike", how="left")

    # Parse
    parsed = tbm[tb_symbol_col].astype(str).map(parse_symbol)
    tbm["ul"]     = parsed.map(lambda c: c.ul if c else None)
    tbm["kind"]   = parsed.map(lambda c: c.kind if c else None)
    tbm["expiry"] = parsed.map(lambda c: c.expiry if c else None)
    tbm["opt"]    = parsed.map(lambda c: c.opt if c else None)
    tbm["strike"] = parsed.map(lambda c: c.strike if c else None)
    tbm["cp"]     = tbm["opt"].map(lambda x: +1 if str(x).upper()=="CE" else (-1 if str(x).upper()=="PE" else np.nan))

    tbm["T"] = tbm["expiry"].map(lambda exp: T_years_intraday(exp, now_dt, cutoff, holiday_list=DEFAULT_HOLIDAY_LIST))

    tbm["LTP"]   = pd.to_numeric(tbm["LTP"], errors="coerce")
    tbm[avg_col] = pd.to_numeric(tbm[avg_col], errors="coerce")
    tbm[qty_col] = pd.to_numeric(tbm[qty_col], errors="coerce")

    # Spot + S_used (vectorized mapping, no .apply)
    ul_u = tbm["ul"].astype(str).str.upper()
    tbm["S_spot"] = ul_u.map(spot_map).astype(float)

    if spot_mode == "Spot":
        tbm["S_used"] = tbm["S_spot"]
    else:
        # key = (ul, expiry)
        keys = list(zip(ul_u.tolist(), tbm["expiry"].tolist()))
        s_par = pd.Series([parity_map.get(k, np.nan) for k in keys], index=tbm.index, dtype=float)
        tbm["S_used"] = s_par.where(np.isfinite(s_par), tbm["S_spot"])

    tbm["S"] = tbm["S_used"]

    # Vectorized IV + Greeks
    GREEK_KEYS = ["price","delta","gamma","vega","theta","rho","vanna","volga","charm"]
    for k in GREEK_KEYS:
        tbm[f"g_{k}"] = np.nan
    tbm["iv"] = np.nan

    opt_mask = (
        (tbm["kind"] == "OPT")
        & tbm["expiry"].notna()
        & np.isfinite(tbm["S_used"])
        & np.isfinite(pd.to_numeric(tbm["strike"], errors="coerce"))
        & np.isfinite(pd.to_numeric(tbm["T"], errors="coerce"))
        & np.isfinite(pd.to_numeric(tbm["LTP"], errors="coerce"))
        & (pd.to_numeric(tbm["T"], errors="coerce") > 1e-12)
        & np.isfinite(pd.to_numeric(tbm["cp"], errors="coerce"))
    )

    if opt_mask.any():
        idx = tbm.index[opt_mask]
        S = tbm.loc[idx, "S_used"].to_numpy(float)
        K = pd.to_numeric(tbm.loc[idx, "strike"], errors="coerce").to_numpy(float)
        T = pd.to_numeric(tbm.loc[idx, "T"], errors="coerce").to_numpy(float)
        P = pd.to_numeric(tbm.loc[idx, "LTP"], errors="coerce").to_numpy(float)
        cp = pd.to_numeric(tbm.loc[idx, "cp"], errors="coerce").to_numpy(float)

        iv = implied_vol_vec(P, S, K, T, rf, q_div, cp)
        tbm.loc[idx, "iv"] = iv

        good = np.isfinite(iv) & (iv > 0)
        if np.any(good):
            g = bs_greeks_vec(S[good], K[good], T[good], rf, q_div, iv[good], cp[good])
            good_idx = idx[good]
            for k in GREEK_KEYS:
                tbm.loc[good_idx, f"g_{k}"] = g[k]

    fs_mask = tbm["kind"].isin(["FUT", "SPOT"])
    if fs_mask.any():
        base_px = tbm.loc[fs_mask, "LTP"].where(np.isfinite(tbm.loc[fs_mask, "LTP"]), tbm.loc[fs_mask, "S_used"])
        tbm.loc[fs_mask, "g_price"] = base_px
        tbm.loc[fs_mask, "g_delta"] = 1.0
        tbm.loc[fs_mask, "g_gamma"] = 0.0
        tbm.loc[fs_mask, "g_vega"]  = 0.0
        tbm.loc[fs_mask, "g_theta"] = 0.0
        tbm.loc[fs_mask, "g_rho"]   = 0.0
        tbm.loc[fs_mask, "g_vanna"] = 0.0
        tbm.loc[fs_mask, "g_volga"] = 0.0
        tbm.loc[fs_mask, "g_charm"] = 0.0

    # MTM decomposition (if columns exist; otherwise fallback)
    has_decomposed_mtm = all(col in tbm.columns for col in ["Q_overnight","Q_today","PrevClose","TodayBuyAvg"])
    if has_decomposed_mtm:
        tbm["Q_overnight"] = pd.to_numeric(tbm["Q_overnight"], errors="coerce").fillna(0.0)
        tbm["Q_today"]     = pd.to_numeric(tbm["Q_today"], errors="coerce").fillna(0.0)
        tbm["PrevClose"]   = pd.to_numeric(tbm["PrevClose"], errors="coerce")
        tbm["TodayBuyAvg"] = pd.to_numeric(tbm["TodayBuyAvg"], errors="coerce")

        ltp = tbm["LTP"]
        mtm_overnight = (ltp - tbm["PrevClose"]) * tbm["Q_overnight"]
        mtm_today     = (ltp - tbm["TodayBuyAvg"]) * tbm["Q_today"]

        tbm["MTM_overnight"] = np.where(np.isfinite(mtm_overnight), mtm_overnight, 0.0)
        tbm["MTM_today"]     = np.where(np.isfinite(mtm_today),     mtm_today,     0.0)
        tbm["MTM"]           = tbm["MTM_overnight"] + tbm["MTM_today"]
    else:
        tbm["MTM"] = (tbm["LTP"] - tbm[avg_col]) * tbm[qty_col]
        tbm["MTM_overnight"] = np.nan
        tbm["MTM_today"] = np.nan

    # Position greeks
    tbm["pos_delta"] = tbm["g_delta"] * tbm[qty_col]
    tbm["pos_gamma"] = tbm["g_gamma"] * tbm[qty_col]
    tbm["pos_vega"]  = tbm["g_vega"]  * tbm[qty_col]
    tbm["pos_theta"] = tbm["g_theta"] * tbm[qty_col]
    tbm["pos_rho"]   = tbm["g_rho"]   * tbm[qty_col]
    tbm["pos_vanna"] = tbm["g_vanna"] * tbm[qty_col]
    tbm["pos_volga"] = tbm["g_volga"] * tbm[qty_col]
    tbm["pos_charm"] = tbm["g_charm"] * tbm[qty_col]

    tbm["notional_delta"] = tbm["pos_delta"] * tbm["S_used"]
    tbm["abs_gross_delta"] = tbm["pos_delta"].abs() * tbm["S_used"]

    cols = ["pos_delta","pos_gamma","pos_vega","pos_theta","pos_vanna","pos_volga","pos_charm","notional_delta","abs_gross_delta"]
    pf_by_ul_exp = (
        tbm.groupby(["ul","expiry"], dropna=False)[cols]
           .sum(min_count=1)
           .rename_axis(["underlying","expiry"])
    )
    if not pf_by_ul_exp.empty:
        total = pf_by_ul_exp.sum().to_frame().T
        total.index = pd.MultiIndex.from_tuples([("Total", None)], names=["underlying","expiry"])
        pf_by_ul_exp = pd.concat([pf_by_ul_exp, total], axis=0)

    nice = pf_by_ul_exp.rename(
        columns={
            "pos_delta":"Δ","pos_gamma":"Γ","pos_vega":"Vega","pos_theta":"Θ",
            "pos_vanna":"vanna","pos_volga":"volga","pos_charm":"charm",
            "notional_delta":"Notional Δ","abs_gross_delta":"gross_delta"
        }
    )

    scenarios: dict[str, dict[str, pd.DataFrame]] = {}  # disabled
    return tbm, nice, scenarios, parity_dbg


def run_engine_fast_from_ctx2(
    tb_df: pd.DataFrame,
    ltp_ctx: dict[str, Any],
    tb_symbol_col: str,
    qty_col: str,
    avg_col: str,
    rf: float,
    q_div: float,
    spot_shocks:list[float],
    vol_shocks:list[float],
    time_shocks:list[int],
    now_dt: datetime,
    cutoff: time,
    spot_mode: str = "Synthetic (K_atm+(C_atm - P_atm))",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    """
    Returns:
      tbm (leg-level)
      pf_greeks (MultiIndex underlying, expiry)
      scenarios (empty dict by design)
      parity_dbg (debug df from ltp_ctx)
    """
    if tb_df is None or tb_df.empty:
        empty = pd.DataFrame()
        return empty, empty, {}, ltp_ctx.get("parity_dbg", pd.DataFrame())

    ltp_agg = ltp_ctx["ltp_agg"]
    spot_map: dict[str, float] = ltp_ctx["spot_map"]
    parity_map: dict[tuple[str, date], float] = ltp_ctx["parity_map"]
    parity_dbg: pd.DataFrame = ltp_ctx.get("parity_dbg", pd.DataFrame())

    tbm = tb_df.merge(ltp_agg, left_on=tb_symbol_col, right_on="TradebookLike", how="left")

    # Parse
    parsed = tbm[tb_symbol_col].astype(str).map(parse_symbol)
    tbm["ul"]     = parsed.map(lambda c: c.ul if c else None)
    tbm["kind"]   = parsed.map(lambda c: c.kind if c else None)
    tbm["expiry"] = parsed.map(lambda c: c.expiry if c else None)
    tbm["opt"]    = parsed.map(lambda c: c.opt if c else None)
    tbm["strike"] = parsed.map(lambda c: c.strike if c else None)
    tbm["cp"]     = tbm["opt"].map(lambda x: +1 if str(x).upper()=="CE" else (-1 if str(x).upper()=="PE" else np.nan))

    tbm["T"] = tbm["expiry"].map(lambda exp: T_years_intraday(exp, now_dt, cutoff, holiday_list=DEFAULT_HOLIDAY_LIST))

    tbm["LTP"]   = pd.to_numeric(tbm["LTP"], errors="coerce")
    tbm[avg_col] = pd.to_numeric(tbm[avg_col], errors="coerce")
    tbm[qty_col] = pd.to_numeric(tbm[qty_col], errors="coerce")

    # Spot + S_used (vectorized mapping, no .apply)
    ul_u = tbm["ul"].astype(str).str.upper()
    tbm["S_spot"] = ul_u.map(spot_map).astype(float)

    if spot_mode == "Spot":
        tbm["S_used"] = tbm["S_spot"]
    else:
        # key = (ul, expiry)
        keys = list(zip(ul_u.tolist(), tbm["expiry"].tolist()))
        s_par = pd.Series([parity_map.get(k, np.nan) for k in keys], index=tbm.index, dtype=float)
        tbm["S_used"] = s_par.where(np.isfinite(s_par), tbm["S_spot"])

    tbm["S"] = tbm["S_used"]

    # Vectorized IV + Greeks
    GREEK_KEYS = ["price","delta","gamma","vega","theta","rho","vanna","volga","charm"]
    for k in GREEK_KEYS:
        tbm[f"g_{k}"] = np.nan
    tbm["iv"] = np.nan

    opt_mask = (
        (tbm["kind"] == "OPT")
        & tbm["expiry"].notna()
        & np.isfinite(tbm["S_used"])
        & np.isfinite(pd.to_numeric(tbm["strike"], errors="coerce"))
        & np.isfinite(pd.to_numeric(tbm["T"], errors="coerce"))
        & np.isfinite(pd.to_numeric(tbm["LTP"], errors="coerce"))
        & (pd.to_numeric(tbm["T"], errors="coerce") > 1e-12)
        & np.isfinite(pd.to_numeric(tbm["cp"], errors="coerce"))
    )

    if opt_mask.any():
        idx = tbm.index[opt_mask]
        S = tbm.loc[idx, "S_used"].to_numpy(float)
        K = pd.to_numeric(tbm.loc[idx, "strike"], errors="coerce").to_numpy(float)
        T = pd.to_numeric(tbm.loc[idx, "T"], errors="coerce").to_numpy(float)
        P = pd.to_numeric(tbm.loc[idx, "LTP"], errors="coerce").to_numpy(float)
        cp = pd.to_numeric(tbm.loc[idx, "cp"], errors="coerce").to_numpy(float)

        iv = implied_vol_vec(P, S, K, T, rf, q_div, cp)
        tbm.loc[idx, "iv"] = iv

        good = np.isfinite(iv) & (iv > 0)
        if np.any(good):
            g = bs_greeks_vec(S[good], K[good], T[good], rf, q_div, iv[good], cp[good])
            good_idx = idx[good]
            for k in GREEK_KEYS:
                tbm.loc[good_idx, f"g_{k}"] = g[k]

    fs_mask = tbm["kind"].isin(["FUT", "SPOT"])
    if fs_mask.any():
        base_px = tbm.loc[fs_mask, "LTP"].where(np.isfinite(tbm.loc[fs_mask, "LTP"]), tbm.loc[fs_mask, "S_used"])
        tbm.loc[fs_mask, "g_price"] = base_px
        tbm.loc[fs_mask, "g_delta"] = 1.0
        tbm.loc[fs_mask, "g_gamma"] = 0.0
        tbm.loc[fs_mask, "g_vega"]  = 0.0
        tbm.loc[fs_mask, "g_theta"] = 0.0
        tbm.loc[fs_mask, "g_rho"]   = 0.0
        tbm.loc[fs_mask, "g_vanna"] = 0.0
        tbm.loc[fs_mask, "g_volga"] = 0.0
        tbm.loc[fs_mask, "g_charm"] = 0.0

    # MTM decomposition (if columns exist; otherwise fallback)
    has_decomposed_mtm = all(col in tbm.columns for col in ["Q_overnight","Q_today","PrevClose","TodayBuyAvg"])
    if has_decomposed_mtm:
        tbm["Q_overnight"] = pd.to_numeric(tbm["Q_overnight"], errors="coerce").fillna(0.0)
        tbm["Q_today"]     = pd.to_numeric(tbm["Q_today"], errors="coerce").fillna(0.0)
        tbm["PrevClose"]   = pd.to_numeric(tbm["PrevClose"], errors="coerce")
        tbm["TodayBuyAvg"] = pd.to_numeric(tbm["TodayBuyAvg"], errors="coerce")

        ltp = tbm["LTP"]
        mtm_overnight = (ltp - tbm["PrevClose"]) * tbm["Q_overnight"]
        mtm_today     = (ltp - tbm["TodayBuyAvg"]) * tbm["Q_today"]

        tbm["MTM_overnight"] = np.where(np.isfinite(mtm_overnight), mtm_overnight, 0.0)
        tbm["MTM_today"]     = np.where(np.isfinite(mtm_today),     mtm_today,     0.0)
        tbm["MTM"]           = tbm["MTM_overnight"] + tbm["MTM_today"]
    else:
        tbm["MTM"] = (tbm["LTP"] - tbm[avg_col]) * tbm[qty_col]
        tbm["MTM_overnight"] = np.nan
        tbm["MTM_today"] = np.nan

    # Position greeks
    tbm["pos_delta"] = tbm["g_delta"] * tbm[qty_col]
    tbm["pos_gamma"] = tbm["g_gamma"] * tbm[qty_col]
    tbm["pos_vega"]  = tbm["g_vega"]  * tbm[qty_col]
    tbm["pos_theta"] = tbm["g_theta"] * tbm[qty_col]
    tbm["pos_rho"]   = tbm["g_rho"]   * tbm[qty_col]
    tbm["pos_vanna"] = tbm["g_vanna"] * tbm[qty_col]
    tbm["pos_volga"] = tbm["g_volga"] * tbm[qty_col]
    tbm["pos_charm"] = tbm["g_charm"] * tbm[qty_col]

    tbm["notional_delta"] = tbm["pos_delta"] * tbm["S_used"]
    tbm["abs_gross_delta"] = tbm["pos_delta"].abs() * tbm["S_used"]

    cols = ["pos_delta","pos_gamma","pos_vega","pos_theta","pos_vanna","pos_volga","pos_charm","notional_delta","abs_gross_delta"]
    pf_by_ul_exp = (
        tbm.groupby(["ul","expiry"], dropna=False)[cols]
           .sum(min_count=1)
           .rename_axis(["underlying","expiry"])
    )
    if not pf_by_ul_exp.empty:
        total = pf_by_ul_exp.sum().to_frame().T
        total.index = pd.MultiIndex.from_tuples([("Total", None)], names=["underlying","expiry"])
        pf_by_ul_exp = pd.concat([pf_by_ul_exp, total], axis=0)

    nice = pf_by_ul_exp.rename(
        columns={
            "pos_delta":"Δ","pos_gamma":"Γ","pos_vega":"Vega","pos_theta":"Θ",
            "pos_vanna":"vanna","pos_volga":"volga","pos_charm":"charm",
            "notional_delta":"Notional Δ","abs_gross_delta":"gross_delta"
        }
    )

    # scenarios: dict[str, dict[str, pd.DataFrame]] = {}  # disabled
    scenarios = build_scenarios_approx_by_ul(tbm=tbm,
                                             spot_shocks=spot_shocks,
                                             vol_shocks=vol_shocks,
                                             time_shocks=time_shocks
                                             )
    return tbm, nice, scenarios, parity_dbg


# =============================================================================
# MARGIN INPUTS (for your separate margin worker)
# =============================================================================

def enc_fut_key(ul: str, exp_i: int) -> str:
    return f"{str(ul).upper()}|{int(exp_i)}"

def enc_opt_key(ul: str, exp_i: int, strike: float, opt_type: str) -> str:
    return f"{str(ul).upper()}|{int(exp_i)}|{float(strike)}|{str(opt_type).upper()}"

def build_margin_inputs_from_tbm(tbm: pd.DataFrame, ltp_raw: pd.DataFrame, qty_col: str) -> dict:
    if tbm is None or tbm.empty:
        return {
            "positions_units": [],
            "underlying_price": {},
            "fut_price_by_underlying_expiry": {},
            "opt_ltp_by_contract_key": {},
            "prev_close_by_underlying": {},
        }

    df = tbm.copy()
    df["kind"] = df["kind"].astype(str).str.upper()
    df["ul"]   = df["ul"].astype(str).str.upper()
    df["expiry_int"] = df["expiry"].map(lambda d: int(d.strftime("%Y%m%d")) if isinstance(d, date) else None)

    pos = df[["ul","kind","expiry_int","opt","strike",qty_col]].copy()
    pos = pos.rename(columns={"ul":"underlying","expiry_int":"expiry","opt":"option_type",qty_col:"qty_units"})
    pos["qty_units"] = pd.to_numeric(pos["qty_units"], errors="coerce").fillna(0.0)
    pos["strike"] = pd.to_numeric(pos["strike"], errors="coerce")

    pos["option_type"] = pos["option_type"].astype(str).str.upper()
    pos.loc[pos["kind"] != "OPT", "option_type"] = None
    pos.loc[pos["kind"] == "OPT", "option_type"] = pos.loc[pos["kind"] == "OPT", "option_type"].replace({"NONE": None, "NAN": None})

    pos = pos[pos["kind"].isin(["FUT","OPT"]) & pos["expiry"].notna()].copy()
    grp_cols = ["underlying","kind","expiry","option_type","strike"]
    pos = pos.groupby(grp_cols, as_index=False)["qty_units"].sum()
    pos = pos[pos["qty_units"].abs() > 1e-12].copy()

    underlying_price = build_spot_map(ltp_raw)

    fut = df[df["kind"] == "FUT"].copy()
    fut = fut.dropna(subset=["ul","expiry_int","LTP"])
    fut["LTP"] = pd.to_numeric(fut["LTP"], errors="coerce")
    fut = fut.dropna(subset=["LTP"]).drop_duplicates(subset=["ul","expiry_int"], keep="last")
    fut_price_by_underlying_expiry = {enc_fut_key(r["ul"], int(r["expiry_int"])): float(r["LTP"]) for _, r in fut.iterrows()}

    opt = df[df["kind"] == "OPT"].copy()
    opt = opt.dropna(subset=["ul","expiry_int","opt","strike","LTP"])
    opt["LTP"] = pd.to_numeric(opt["LTP"], errors="coerce")
    opt["strike"] = pd.to_numeric(opt["strike"], errors="coerce")
    opt["opt"] = opt["opt"].astype(str).str.upper()
    opt = opt.dropna(subset=["LTP","strike"]).drop_duplicates(subset=["ul","expiry_int","opt","strike"], keep="last")
    opt_ltp_by_contract_key = {enc_opt_key(r["ul"], int(r["expiry_int"]), float(r["strike"]), str(r["opt"])): float(r["LTP"]) for _, r in opt.iterrows()}

    return {
        "positions_units": pos.to_dict(orient="records"),
        "underlying_price": underlying_price,
        "fut_price_by_underlying_expiry": fut_price_by_underlying_expiry,
        "opt_ltp_by_contract_key": opt_ltp_by_contract_key,
        "prev_close_by_underlying": {},
    }


# =============================================================================
# SMALL UTILS (serialization + payload-id)
# =============================================================================

def df_to_records(df: pd.DataFrame, max_rows: int | None = None) -> dict[str, Any]:
    if df is None:
        return {"columns": [], "data": []}
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).copy()
    return {"columns": list(df.columns), "data": df.to_dict(orient="records")}

def payload_id(payload: dict) -> str:
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode()).hexdigest()

# def compute_payoff_pack(
#     tbm: pd.DataFrame,
#     qty_col: str,
#     avg_col: str,
#     spot_col: str = "S_used",
#     # expiry payoff is computed wide; viewer slices for ±3/5/8/10
#     expiry_grid_pct_wide: float = 0.25,
#     expiry_n_points: int = 301,
#     # limit payload
#     max_groups: int = 12,
#     # which greeks to aggregate
#     delta_col: str = "g_delta",
#     gamma_col: str = "g_gamma",
#     vega_col: str = "g_vega",
#     theta_col: str = "g_theta",
# ) -> dict:
#     """
#     Returns:
#       {
#         "expiry": { "UL|YYYY-MM-DD": [{"S":..., "pnl_entry":...}, ...] },
#         "greeks_agg": {
#             "UL|YYYY-MM-DD": {"S0":..., "QΔ":..., "QΓ":..., "QVega":..., "QTheta":...}
#         },
#         "meta": {"expiry_grid_pct_wide":..., "expiry_n_points":...}
#       }

#     - "expiry": intrinsic-at-expiry PnL vs entry (avg_col), for OPT/FUT/SPOT.
#     - "greeks_agg": qty-weighted greek sums to generate fast approx MTM curves in viewer:
#         pnl ≈ QΔ*dS + 0.5*QΓ*dS^2 + QVega*dσ + QTheta*dt_years
#     """
#     if tbm is None or tbm.empty:
#         return {"expiry": {}, "greeks_agg": {}, "meta": {"expiry_grid_pct_wide": expiry_grid_pct_wide, "expiry_n_points": expiry_n_points}}

#     df = tbm.copy()
#     df["ul"] = df["ul"].astype(str).str.upper()
#     df["kind"] = df["kind"].astype(str).str.upper()
#     df["expiry_s"] = df["expiry"].astype(str).replace({"None": "", "NaT": ""})

#     # numeric
#     df[qty_col] = pd.to_numeric(df.get(qty_col), errors="coerce").fillna(0.0)
#     df[avg_col] = pd.to_numeric(df.get(avg_col), errors="coerce")
#     df[spot_col] = pd.to_numeric(df.get(spot_col), errors="coerce")

#     df["strike"] = pd.to_numeric(df.get("strike"), errors="coerce")
#     df["cp"] = pd.to_numeric(df.get("cp"), errors="coerce")  # +1/-1

#     for c in [delta_col, gamma_col, vega_col, theta_col]:
#         if c not in df.columns:
#             df[c] = 0.0
#         df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

#     # rank groups by abs notional to cap payload
#     df["_abs_notional"] = (df[qty_col].abs() * df[spot_col].abs()).fillna(0.0)
#     grp_rank = (
#         df.groupby(["ul", "expiry_s"], as_index=False)["_abs_notional"]
#           .sum()
#           .sort_values("_abs_notional", ascending=False)
#     )
#     groups = list(zip(grp_rank["ul"].tolist(), grp_rank["expiry_s"].tolist()))[:max_groups]

#     expiry_out: dict[str, list[dict]] = {}
#     greeks_out: dict[str, dict] = {}

#     for ul, exp_s in groups:
#         g = df[(df["ul"] == ul) & (df["expiry_s"] == exp_s)].copy()
#         if g.empty:
#             continue

#         # anchor spot for the group
#         S0 = float(g[spot_col].dropna().iloc[0]) if g[spot_col].notna().any() else np.nan
#         if not np.isfinite(S0) or S0 <= 0:
#             continue

#         key = f"{ul}|{exp_s}"

#         # -------------------------
#         # A) Expiry payoff (wide grid)
#         # -------------------------
#         lo = max(1.0, S0 * (1.0 - float(expiry_grid_pct_wide)))
#         hi = S0 * (1.0 + float(expiry_grid_pct_wide))
#         S_grid = np.linspace(lo, hi, int(expiry_n_points), dtype=float)

#         qty = g[qty_col].to_numpy(float)
#         entry = g[avg_col].to_numpy(float)

#         pnl_total = np.zeros_like(S_grid)

#         # FUT/SPOT: (S - entry)
#         fs = g["kind"].isin(["FUT", "SPOT"]).to_numpy(bool)
#         if np.any(fs):
#             pnl_total += (qty[fs][:, None] * (S_grid[None, :] - entry[fs][:, None])).sum(axis=0)

#         # OPT: intrinsic - entry
#         op = (g["kind"] == "OPT").to_numpy(bool)
#         if np.any(op):
#             K  = g["strike"].to_numpy(float)[op]
#             cp = g["cp"].to_numpy(float)[op]
#             q  = qty[op]
#             e  = entry[op]
#             intrinsic = np.maximum(cp[:, None] * (S_grid[None, :] - K[:, None]), 0.0)
#             pnl_total += (q[:, None] * (intrinsic - e[:, None])).sum(axis=0)

#         expiry_out[key] = [{"S": float(s), "pnl_entry": float(p)} for s, p in zip(S_grid, pnl_total)]

#         # -------------------------
#         # B) Greeks aggregates (for viewer fast curve)
#         # -------------------------
#         QD = float((g[qty_col] * g[delta_col]).sum())
#         QG = float((g[qty_col] * g[gamma_col]).sum())
#         QV = float((g[qty_col] * g[vega_col]).sum())   # vega is per +1.00 abs vol (your BS)
#         QT = float((g[qty_col] * g[theta_col]).sum())  # theta is per year (your BS)

#         greeks_out[key] = {"S0": float(S0), "QΔ": QD, "QΓ": QG, "QVega": QV, "QTheta": QT}

#     return {
#         "expiry": expiry_out,
#         "greeks_agg": greeks_out,
#         "meta": {"expiry_grid_pct_wide": float(expiry_grid_pct_wide), "expiry_n_points": int(expiry_n_points)},
#     }

def compute_payoff_pack(
    tbm: pd.DataFrame,
    qty_col: str,
    avg_col: str,
    spot_col: str = "S_used",
    expiry_grid_pct_wide: float = 0.25,
    expiry_n_points: int = 301,
    max_groups: int = 12,
    # fallback greeks (only used if pos_* not present)
    delta_col: str = "g_delta",
    gamma_col: str = "g_gamma",
    vega_col: str = "g_vega",
    theta_col: str = "g_theta",
) -> dict:
    """
    Returns:
      {
        "expiry": { "UL|YYYY-MM-DD": [{"S":..., "pnl_entry":...}, ...] },
        "greeks_agg": {
            "UL|YYYY-MM-DD": {"S0":..., "MTM0":..., "QΔ":..., "QΓ":..., "QVega":..., "QTheta":...}
        },
        "meta": {"expiry_grid_pct_wide":..., "expiry_n_points":...}
      }

    - expiry: intrinsic-at-expiry PnL vs entry (avg_col), for OPT/FUT/SPOT.
    - greeks_agg: uses SUM(pos_*) if available; else falls back to SUM(qty*g_*).
      Viewer can do:
        pnl(S) ≈ MTM0 + QΔ*dS + 0.5*QΓ*dS^2 + QVega*dσ + QTheta*dt_years
    """
    if tbm is None or tbm.empty:
        return {
            "expiry": {},
            "greeks_agg": {},
            "meta": {"expiry_grid_pct_wide": float(expiry_grid_pct_wide), "expiry_n_points": int(expiry_n_points)},
        }

    df = tbm.copy()

    # --- normalize key cols ---
    df["ul"] = df.get("ul", "").astype(str).str.upper()
    df["kind"] = df.get("kind", "").astype(str).str.upper()
    df["expiry_s"] = df.get("expiry", "").astype(str).replace({"None": "", "NaT": ""})

    # --- numeric coercions ---
    df[qty_col] = pd.to_numeric(df.get(qty_col), errors="coerce").fillna(0.0)
    df[avg_col] = pd.to_numeric(df.get(avg_col), errors="coerce")  # keep NaN for now; we nan_to_num later
    df[spot_col] = pd.to_numeric(df.get(spot_col), errors="coerce")

    df["strike"] = pd.to_numeric(df.get("strike"), errors="coerce")
    # cp: +1 call, -1 put. If missing, infer from opt ("CE"/"PE") if present.
    if "cp" in df.columns:
        df["cp"] = pd.to_numeric(df.get("cp"), errors="coerce")
    else:
        df["cp"] = np.nan

    if df["cp"].isna().any() and "opt" in df.columns:
        opt = df["opt"].astype(str).str.upper()
        df.loc[df["cp"].isna() & opt.str.contains("C"), "cp"] = 1.0
        df.loc[df["cp"].isna() & opt.str.contains("P"), "cp"] = -1.0

    # fallback greeks cols (only used if pos_* not present)
    for c in [delta_col, gamma_col, vega_col, theta_col]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # rank groups by abs notional to cap payload
    df["_abs_notional"] = (df[qty_col].abs() * df[spot_col].abs()).fillna(0.0)
    grp_rank = (
        df.groupby(["ul", "expiry_s"], as_index=False)["_abs_notional"]
          .sum()
          .sort_values("_abs_notional", ascending=False)
    )
    groups = list(zip(grp_rank["ul"].tolist(), grp_rank["expiry_s"].tolist()))[:max_groups]

    expiry_out: dict[str, list[dict]] = {}
    greeks_out: dict[str, dict] = {}

    have_pos_greeks = all(c in df.columns for c in ["pos_delta", "pos_gamma", "pos_vega", "pos_theta"])

    for ul, exp_s in groups:
        g = df[(df["ul"] == ul) & (df["expiry_s"] == exp_s)].copy()
        if g.empty:
            continue

        # anchor spot for the group (use median to avoid odd rows)
        spots = pd.to_numeric(g[spot_col], errors="coerce").dropna()
        S0 = float(spots.median()) if not spots.empty else np.nan
        if not np.isfinite(S0) or S0 <= 0:
            continue

        key = f"{ul}|{exp_s}"

        # -------------------------
        # A) Expiry payoff (wide grid)
        # -------------------------
        lo = max(1.0, S0 * (1.0 - float(expiry_grid_pct_wide)))
        hi = S0 * (1.0 + float(expiry_grid_pct_wide))
        S_grid = np.linspace(lo, hi, int(expiry_n_points), dtype=float)

        qty = g[qty_col].to_numpy(float)
        entry = g[avg_col].to_numpy(float)
        entry = np.nan_to_num(entry, nan=0.0)  # avoids NaN poisoning

        pnl_total = np.zeros_like(S_grid)

        # FUT/SPOT: q * (S - entry)
        fs = g["kind"].isin(["FUT", "SPOT"]).to_numpy(bool)
        if np.any(fs):
            pnl_total += (qty[fs][:, None] * (S_grid[None, :] - entry[fs][:, None])).sum(axis=0)

        # OPT: q * (intrinsic - entry)
        op = (g["kind"] == "OPT").to_numpy(bool)
        if np.any(op):
            K  = pd.to_numeric(g.loc[op, "strike"], errors="coerce").to_numpy(float)
            cp = pd.to_numeric(g.loc[op, "cp"], errors="coerce").to_numpy(float)
            q  = qty[op]
            e  = entry[op]

            # keep only valid option rows
            valid = np.isfinite(K) & np.isfinite(cp)
            if np.any(valid):
                K = K[valid]
                cp = cp[valid]
                q = q[valid]
                e = e[valid]

                intrinsic = np.maximum(cp[:, None] * (S_grid[None, :] - K[:, None]), 0.0)
                pnl_total += (q[:, None] * (intrinsic - e[:, None])).sum(axis=0)

        expiry_out[key] = [{"S": float(s), "pnl_entry": float(p)} for s, p in zip(S_grid, pnl_total)]

        # -------------------------
        # B) Greeks aggregates (for viewer fast curve)
        # -------------------------
        if have_pos_greeks:
            QD = float(pd.to_numeric(g["pos_delta"], errors="coerce").fillna(0.0).sum())
            QG = float(pd.to_numeric(g["pos_gamma"], errors="coerce").fillna(0.0).sum())
            QV = float(pd.to_numeric(g["pos_vega"],  errors="coerce").fillna(0.0).sum())
            QT = float(pd.to_numeric(g["pos_theta"], errors="coerce").fillna(0.0).sum())
        else:
            QD = float((g[qty_col] * g[delta_col]).sum())
            QG = float((g[qty_col] * g[gamma_col]).sum())
            QV = float((g[qty_col] * g[vega_col]).sum())
            QT = float((g[qty_col] * g[theta_col]).sum())

        MTM0 = 0.0
        if "MTM" in g.columns:
            MTM0 = float(pd.to_numeric(g["MTM"], errors="coerce").fillna(0.0).sum())

        greeks_out[key] = {"S0": float(S0), "MTM0": MTM0, "QΔ": QD, "QΓ": QG, "QVega": QV, "QTheta": QT}

    return {
        "expiry": expiry_out,
        "greeks_agg": greeks_out,
        "meta": {"expiry_grid_pct_wide": float(expiry_grid_pct_wide), "expiry_n_points": int(expiry_n_points)},
    }


def build_scenarios_approx_by_ul(
    tbm: pd.DataFrame,
    spot_shocks: list[float],
    vol_shocks: list[float],   # absolute vol (e.g. +0.01 for +1 vol point)
    time_shocks: list[int],    # days forward
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Approx scenario P&L using aggregated greeks:
      Spot:  pnl ≈ Δ*dS + 0.5*Γ*dS^2
      Vol:   pnl ≈ Vega*dσ + 0.5*Volga*dσ^2
      Time:  pnl ≈ Θ*dt_years   (Θ is calendar theta per year from BS)
    Returns old-style:
      scenarios[ul] = {"spot": df(1xN), "vol": df(1xN), "time": df(1xN)}
    """
    scenarios: dict[str, dict[str, pd.DataFrame]] = {}
    if tbm is None or tbm.empty or "ul" not in tbm.columns:
        return scenarios

    need_cols = ["pos_delta", "pos_gamma", "pos_vega", "pos_theta", "pos_volga", "S_used"]
    for c in need_cols:
        if c not in tbm.columns:
            tbm[c] = 0.0

    df = tbm.copy()
    df["ul"] = df["ul"].astype(str).str.upper()

    # aggregate greeks by UL
    agg = (
        df.groupby("ul", dropna=False)[["pos_delta", "pos_gamma", "pos_vega", "pos_theta", "pos_volga"]]
          .sum(min_count=1)
          .fillna(0.0)
    )

    # pick a single S0 per UL (median is stable intraday)
    S0 = (
        pd.to_numeric(df["S_used"], errors="coerce")
        .groupby(df["ul"])
        .median()
    )
    agg["S0"] = S0

    agg = agg[np.isfinite(agg["S0"]) & (agg["S0"] > 0)].copy()
    if agg.empty:
        return scenarios

    D = agg["pos_delta"].astype(float)
    G = agg["pos_gamma"].astype(float)
    V = agg["pos_vega"].astype(float)
    Th = agg["pos_theta"].astype(float)
    Vo = agg["pos_volga"].astype(float)
    S0v = agg["S0"].astype(float)

    # Spot scenarios
    spot_tbl = pd.DataFrame(index=agg.index)
    for pct in spot_shocks:
        dS = S0v * float(pct)
        spot_tbl[f"{pct*100:.1f}%"] = (D * dS) + (0.5 * G * (dS ** 2))

    # Vol scenarios (dv is absolute sigma change)
    vol_tbl = pd.DataFrame(index=agg.index)
    for dv in vol_shocks:
        dv = float(dv)
        vol_tbl[f"{int(round(dv*100)):+d}%"] = (V * dv) + (0.5 * Vo * (dv ** 2))

    # Time scenarios
    time_tbl = pd.DataFrame(index=agg.index)
    for days in time_shocks:
        dt_years = float(days) / 365.0
        time_tbl[f"T+{int(days)}D"] = Th * dt_years

    # pack per UL (old viewer expects per-UL dict of DataFrames)
    for ul in agg.index.tolist():
        scenarios[str(ul)] = {
            "spot": spot_tbl.loc[[ul]],
            "vol":  vol_tbl.loc[[ul]],
            "time": time_tbl.loc[[ul]],
        }

    return scenarios
