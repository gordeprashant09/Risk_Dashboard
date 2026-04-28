# strategy_pnl_worker.py
"""
Strategy-level PnL worker (isolated; does NOT modify risk_worker.py)

Inputs:
- Redis LTP hash (default: last_price)
- Tradebook CSV per user (must contain: TradingSymbol, LastTradedQuantity, OrderAverageTradedPrice, and a strategy tag col)
- Prev EOD positions CSV per user (optional)
    IMPORTANT: To compute tag-wise CarryPnL correctly, prev EOD file must contain a TAG column.
    If prev EOD is missing OR has no tag column => CarryPnL = 0 across all tags.
- Mock sqlite DBs per user: intraday + positional (table: orders)
    If positional mock db missing/empty => SimCarryPnL = 0 across all tags.
- Allocated margin CSV (NOW USER-SPECIFIC):
    Path pattern:
      Y:\Risk_dashboard_inputs\{dma}\multiplier\multiplier_{dma}_{YYYYMMDD}.csv
    Columns:
      ['Strategy Name', 'Multiplier', 'Total Margin']
    Map:
      tag -> Total Margin   (tag is your strategy identifier)

Outputs (Redis):
- risk:strategy_pnl:latest:{username}   (JSON)
- risk:strategy_pnl_minmax:{username}:{YYYYMMDD}  (HASH, tag-level net min/max + timestamps)

Output columns:
tag, CarryPnL, DayPnL, Expenses, NetPnL,
allocated_margin, net_pnl/margin (%),
MinPnL, MinTime, MaxPnL, MaxTime,
SimCarryPnL, SimDayPnL, Slippage (%), sim_live_diff %
"""

from __future__ import annotations

import os
import time as _time
import json
import math
import logging
import sqlite3
from datetime import datetime, time, date

import numpy as np
import pandas as pd
import redis
import warnings
warnings.filterwarnings("ignore")

from risk_lib_fast import (
    IST, DEFAULT_HOLIDAY_LIST,
    get_prev_valid_date,
    parse_symbol,
    to_symbol_tradebook_spaced,
    prepare_ltp_context,
)

# -------------------------
# CONFIG
# -------------------------
from user_config import USERDETAILS, USERNAMES, get_user_col, normalize_symbol, build_tradebook_path as _build_tb_path, build_mock_paths as _build_mock_paths

TRADEBOOK_PATH_TEMPLATE = os.getenv(
    "TRADEBOOK_PATH_TEMPLATE",
    # r"Y:\Risk_dashboard_inputs\{dma}\tradebook\tradebook_T611_{yyyy_mm_dd}.csv"
    "/mnt/Quant_Research/Risk_dashboard_inputs/{dma}/tradebook/tradebook_T611_{yyyy_mm_dd}.csv"
)

# NOTE: for correct tag-wise CarryPnL, your prev file must include a tag column
PREV_EOD_TEMPLATE = os.getenv(
    "PREV_EOD_TEMPLATE",
    # r"Y:\Risk_dashboard_inputs\{dma}\eod_files\net_positions_eod_{yymmdd}.csv"
    "/mnt/Quant_Research/Risk_dashboard_inputs/{dma}/eod_files/net_positions_eod_{yymmdd}.csv"
)

# Mock DB paths (your pattern)
MOCK_INTRADAY_DB_TEMPLATE = os.getenv(
    "MOCK_INTRADAY_DB_TEMPLATE",
    # r"Y:\Risk_dashboard_inputs\mock_{dma}\db\orders.sqlite"
    "/mnt/Quant_Research/Risk_dashboard_inputs/mock_{dma}/db/orders.sqlite"
)
# Default to a {dma}_P pattern (override with env if different)
MOCK_POSITIONAL_DB_TEMPLATE = os.getenv(
    "MOCK_POSITIONAL_DB_TEMPLATE",
    # r"Y:\Risk_dashboard_inputs\mock_{dma}_P\db\orders.sqlite"
    "/mnt/Quant_Research/Risk_dashboard_inputs/mock_{dma}_P/db/orders.sqlite"
)

# NEW: user-specific allocated margin file template
# Example: Y:\Risk_dashboard_inputs\DMA16\multiplier\multiplier_DMA16_20260220.csv
ALLOC_MARGIN_TEMPLATE = os.getenv(
    "ALLOC_MARGIN_TEMPLATE",
    # r"Y:\Risk_dashboard_inputs\{dma}\multiplier\multiplier_{dma}_{yyyymmdd}.csv"
    "/mnt/Quant_Research/Risk_dashboard_inputs/{dma}/multiplier/multiplier_{dma}_{yyyymmdd}.csv"
)

TB_SYMBOL_COL = os.getenv("TB_SYMBOL_COL", "TradingSymbol")
TB_QTY_COL    = os.getenv("TB_QTY_COL", "LastTradedQuantity")
TB_AVG_COL    = os.getenv("TB_AVG_COL", "OrderAverageTradedPrice")

# YOU SAID: tradebook has strategy tag in column "OrderUniqueIdentifier"
TB_TAG_COL    = os.getenv("TB_TAG_COL", "OrderUniqueIdentifier")

# Prev file columns
PREV_SYM_COL  = os.getenv("PREV_SYM_COL", "instrument")
PREV_QTY_COL  = os.getenv("PREV_QTY_COL", "net_open_qty")
PREV_AVG_COL  = os.getenv("PREV_AVG_COL", "bhav_today")
# NEW: prev tag column (must exist for tag-wise carry)
PREV_TAG_COL  = os.getenv("PREV_TAG_COL", "tag")

# Expenses model (same as your worker)
COST_PER_CR = float(os.getenv("COST_PER_CR", "10000.0"))  # per 1 crore premium traded (₹1e7)

CUTOFF_HOUR   = int(os.getenv("EXPIRY_CUTOFF_HOUR", "15"))
CUTOFF_MINUTE = int(os.getenv("EXPIRY_CUTOFF_MIN", "30"))
CUT = time(CUTOFF_HOUR, CUTOFF_MINUTE)

LOOP_SECONDS = float(os.getenv("STRAT_LOOP_SECONDS", "2.0"))

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))
REDIS_PWD  = os.getenv("REDIS_PASSWORD", "") or None
LTP_HASH_KEY = os.getenv("LTP_HASH_KEY", "last_price")

# ---- Strategy payoff publish config ----
STRAT_PAYOFF_GRID_PCT = float(os.getenv("STRAT_PAYOFF_GRID_PCT", "0.08"))
STRAT_PAYOFF_N_POINTS = int(os.getenv("STRAT_PAYOFF_N_POINTS", "201"))
STRAT_PAYOFF_TOP_N    = int(os.getenv("STRAT_PAYOFF_TOP_N", "20"))  # you asked default=20

#stop script config
STOP_AFTER_CUTOFF = os.getenv("STOP_AFTER_CUTOFF", "1") == "1"
STOP_TIME_HOUR    = int(os.getenv("STOP_TIME_HOUR", "15"))
STOP_TIME_MINUTE  = int(os.getenv("STOP_TIME_MINUTE", "30"))
STOP_TIME = time(STOP_TIME_HOUR, STOP_TIME_MINUTE)

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("strategy_pnl_worker")

# -------------------------
# strategy_pnl_storing helper
# -------------------------
import pathlib
from datetime import timezone

STRAT_TS_BASE_DIR = os.getenv(
    "STRAT_TS_BASE_DIR",
    # r"Y:\Risk_dashboard_inputs\strategy_ts"
    "/mnt/Quant_Research/Risk_dashboard_inputs/strategy_ts"
)
STRAT_TS_WRITE_EVERY_SECONDS = float(os.getenv("STRAT_TS_WRITE_EVERY_SECONDS", "10"))  # downsample per-tag writes

def write_strategy_ts_parquet(
    username: str,
    now_dt: datetime,
    out_df: pd.DataFrame,
):
    """
    Writes one parquet file per write event containing ALL tags for that user at that timestamp.

    Folder:
      {STRAT_TS_BASE_DIR}/date=YYYYMMDD/user=<username>/*.parquet

    Columns saved:
      ts_utc, ts_ist, tag, DayPnL, SimDayPnL, NetPnL, SimCarryPnL, allocated_margin
    """
    if out_df is None or out_df.empty:
        return

    day = now_dt.strftime("%Y%m%d")
    out_dir = pathlib.Path(STRAT_TS_BASE_DIR) / f"date={day}" / f"user={username}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_utc = now_dt.astimezone(timezone.utc).isoformat(timespec="seconds")
    ts_ist = now_dt.isoformat(timespec="seconds")

    keep = [c for c in ["tag", "DayPnL", "SimDayPnL", "NetPnL", "SimCarryPnL", "allocated_margin"] if c in out_df.columns]
    d = out_df[keep].copy()
    d["ts_utc"] = ts_utc
    d["ts_ist"] = ts_ist

    # unique file per write
    fname = f"ts_{now_dt.strftime('%H%M%S')}_{int(now_dt.timestamp())}.parquet"
    d.to_parquet(out_dir / fname, index=False, engine="pyarrow", compression="zstd")

#-------------------------------------------------------------------------------------
# -------------------------
# UTILS
# -------------------------
def _sf(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)

def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")

def redis_client() -> redis.Redis:
    return redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PWD,
        decode_responses=True, socket_timeout=2.0, socket_connect_timeout=2.0
    )

def fetch_ltp_df(r: redis.Redis) -> pd.DataFrame:
    data = r.hgetall(LTP_HASH_KEY) or {}
    if not data:
        return pd.DataFrame(columns=["symbol", "ltp"])
    df = pd.DataFrame(list(data.items()), columns=["symbol", "ltp"])
    df["ltp"] = pd.to_numeric(df["ltp"], errors="coerce")
    df = df.dropna(subset=["ltp"])
    return df

def read_csv_safe(path: str) -> pd.DataFrame | None:
    try:
        if not path or not os.path.exists(path):
            return None
        return pd.read_csv(path)
    except Exception:
        return None

def build_tradebook_path(user: str, now_dt: datetime) -> str:
    base_dir = "/mnt/Quant_Research/Risk_dashboard_inputs"
    return _build_tb_path(user, base_dir, str(now_dt.date()))

def build_prev_eod_path(user: str, prev_date: date) -> str:
    meta = USERDETAILS[user]
    return PREV_EOD_TEMPLATE.format(dma=meta["dma"], yymmdd=prev_date.strftime("%y%m%d"))

def build_mock_db_paths(user: str) -> tuple[str, str]:
    base_dir = "/mnt/Quant_Research/Risk_dashboard_inputs"
    return _build_mock_paths(user, base_dir)

def build_alloc_margin_path(user: str, now_dt: datetime) -> str:
    meta = USERDETAILS[user]
    return ALLOC_MARGIN_TEMPLATE.format(dma=meta["dma"], yyyymmdd=now_dt.strftime("%Y%m%d"))

def maybe_make_signed_qty(tb_raw: pd.DataFrame) -> pd.DataFrame:
    if tb_raw is None or tb_raw.empty:
        return tb_raw

    if "OrderSide" not in tb_raw.columns or TB_QTY_COL not in tb_raw.columns:
        tb_raw = tb_raw.copy()
        tb_raw["signed_qty"] = pd.to_numeric(tb_raw.get(TB_QTY_COL, 0.0), errors="coerce").fillna(0.0)
        return tb_raw

    side = tb_raw["OrderSide"].astype(str).str.upper()
    is_buy = side.isin(["BUY", "B"])
    qty_base = pd.to_numeric(tb_raw[TB_QTY_COL], errors="coerce").fillna(0.0)

    tb_raw = tb_raw.copy()
    tb_raw["signed_qty"] = qty_base.where(is_buy, -qty_base)
    return tb_raw

def load_alloc_margin_map(csv_path: str) -> dict[str, float]:
    """
    NEW user-specific multiplier file columns:
      ['Strategy Name', 'Multiplier', 'Total Margin']

    Map:
      tag -> Total Margin
    """
    mp: dict[str, float] = {}
    if not csv_path or not os.path.exists(csv_path):
        log.warning("Allocated margin file not found: %s", csv_path)
        return mp

    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    strat_c = cols.get("strategy name")
    marg_c = cols.get("total margin")

    if not strat_c or not marg_c:
        raise ValueError(
            f"Margin file must have columns 'Strategy Name' and 'Total Margin'. Found: {list(df.columns)}"
        )

    for _, row in df.iterrows():
        tag = str(row.get(strat_c, "")).strip()
        if not tag or tag.lower() in ("nan", "none"):
            continue
        mp[tag] = _sf(row.get(marg_c, 0.0), 0.0)

    return mp

def alloc_margin_for(tag: str, mp: dict[str, float]) -> float:
    return float(mp.get(str(tag), 0.0))


# -------------------------
# LIVE (tradebook + prev_eod) strategy PnL
# -------------------------
def compute_live_strategy_pnl(
    tb_raw: pd.DataFrame,
    prev_raw: pd.DataFrame | None,
    ltp_ctx: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
      out_df with columns: tag, CarryPnL, DayPnL, Expenses, NetPnL
      meta dict about carry availability
    """
    meta = {
        "carry_available": False,
        "prev_eod_missing": True,
        "prev_eod_has_tag": False,
    }

    if tb_raw is None or tb_raw.empty:
        return pd.DataFrame(columns=["tag", "CarryPnL", "DayPnL", "Expenses", "NetPnL"]), meta

    tb = maybe_make_signed_qty(tb_raw.copy())

    # ensure tag column exists on TB (your tag col = OrderUniqueIdentifier)
    if TB_TAG_COL in tb.columns:
        tb["tag"] = tb[TB_TAG_COL].astype(str).fillna("__NO_TAG__")
    elif "tag" in tb.columns:
        tb["tag"] = tb["tag"].astype(str).fillna("__NO_TAG__")
    else:
        tb["tag"] = "__NO_TAG__"

    # Canonical symbol for joining LTP
    tb["TradebookLike"] = tb[TB_SYMBOL_COL].astype(str).map(lambda s: to_symbol_tradebook_spaced(parse_symbol(s)))

    ltp_agg = ltp_ctx.get("ltp_agg")
    if ltp_agg is not None and not ltp_agg.empty:
        tb = tb.merge(ltp_agg[["TradebookLike", "LTP"]], on="TradebookLike", how="left")
    else:
        ltp_map = dict(zip(ltp_ctx["ltp_raw"]["symbol"].astype(str), ltp_ctx["ltp_raw"]["ltp"].astype(float)))
        tb["LTP"] = tb["TradebookLike"].astype(str).map(ltp_map)

    tb["LTP"] = pd.to_numeric(tb["LTP"], errors="coerce")
    tb["avg_px"] = pd.to_numeric(tb.get(TB_AVG_COL, np.nan), errors="coerce")
    tb["signed_qty"] = pd.to_numeric(tb["signed_qty"], errors="coerce").fillna(0.0)

    # DayPnL per trade row: (LTP - avg) * signed_qty
    tb["DayPnL_row"] = (tb["LTP"] - tb["avg_px"]) * tb["signed_qty"]
    tb["DayPnL_row"] = tb["DayPnL_row"].where(np.isfinite(tb["DayPnL_row"]), 0.0)

    # Expenses: premium traded = |qty| * avg_px ; expenses = premium/1e7 * COST_PER_CR
    tb["PremiumTraded_row"] = (tb["signed_qty"].abs() * tb["avg_px"]).where(np.isfinite(tb["avg_px"]), 0.0)
    tb["Expenses_row"] = tb["PremiumTraded_row"] / 1e7 * float(COST_PER_CR)
    tb["Expenses_row"] = tb["Expenses_row"].where(np.isfinite(tb["Expenses_row"]), 0.0)

    live_by_tag = (
        tb.groupby("tag", as_index=False)[["DayPnL_row", "Expenses_row"]]
          .sum()
          .rename(columns={"DayPnL_row": "DayPnL", "Expenses_row": "Expenses"})
    )

    # ---- CarryPnL ----
    # IMPORTANT: We DO NOT map prev symbols -> today's tags anymore.
    # We only compute carry if prev_raw exists AND has PREV_TAG_COL.
    if prev_raw is None or prev_raw.empty:
        meta["prev_eod_missing"] = True
        live_by_tag["CarryPnL"] = 0.0
        live_by_tag["NetPnL"] = live_by_tag["CarryPnL"] + live_by_tag["DayPnL"] - live_by_tag["Expenses"]
        return live_by_tag[["tag", "CarryPnL", "DayPnL", "Expenses", "NetPnL"]], meta

    meta["prev_eod_missing"] = False
    if PREV_TAG_COL not in prev_raw.columns:
        meta["prev_eod_has_tag"] = False
        live_by_tag["CarryPnL"] = 0.0
        live_by_tag["NetPnL"] = live_by_tag["CarryPnL"] + live_by_tag["DayPnL"] - live_by_tag["Expenses"]
        return live_by_tag[["tag", "CarryPnL", "DayPnL", "Expenses", "NetPnL"]], meta

    meta["prev_eod_has_tag"] = True
    meta["carry_available"] = True

    pr = prev_raw.copy()
    pr["tag"] = pr[PREV_TAG_COL].astype(str).fillna("__NO_TAG__")

    # canonical symbol
    pr["TradebookLike"] = pr[PREV_SYM_COL].astype(str).map(lambda s: to_symbol_tradebook_spaced(parse_symbol(s)))

    # LTP join
    if ltp_agg is not None and not ltp_agg.empty:
        pr = pr.merge(ltp_agg[["TradebookLike", "LTP"]], on="TradebookLike", how="left")
    else:
        ltp_map = dict(zip(ltp_ctx["ltp_raw"]["symbol"].astype(str), ltp_ctx["ltp_raw"]["ltp"].astype(float)))
        pr["LTP"] = pr["TradebookLike"].astype(str).map(ltp_map)

    pr["LTP"] = pd.to_numeric(pr["LTP"], errors="coerce")
    pr["prev_close"] = pd.to_numeric(pr.get(PREV_AVG_COL, np.nan), errors="coerce")
    pr["prev_qty"] = pd.to_numeric(pr.get(PREV_QTY_COL, 0.0), errors="coerce").fillna(0.0)

    # If prev_close missing, carry should be 0 for that row -> set prev_close = LTP
    pr["prev_close"] = pr["prev_close"].where(np.isfinite(pr["prev_close"]), pr["LTP"])

    pr["CarryPnL_row"] = (pr["LTP"] - pr["prev_close"]) * pr["prev_qty"]
    pr["CarryPnL_row"] = pr["CarryPnL_row"].where(np.isfinite(pr["CarryPnL_row"]), 0.0)

    carry_by_tag = (
        pr.groupby("tag", as_index=False)[["CarryPnL_row"]]
          .sum()
          .rename(columns={"CarryPnL_row": "CarryPnL"})
    )

    out = live_by_tag.merge(carry_by_tag, on="tag", how="outer").fillna(0.0)
    out["NetPnL"] = out["CarryPnL"] + out["DayPnL"] - out["Expenses"]
    return out[["tag", "CarryPnL", "DayPnL", "Expenses", "NetPnL"]], meta


# -------------------------
# MOCK (sqlite orders) strategy PnL
# -------------------------
# def _read_orders(db_path: str) -> pd.DataFrame:
#     if not db_path or not os.path.exists(db_path):
#         return pd.DataFrame()
#     try:
#         con = sqlite3.connect(db_path)
#         df = pd.read_sql_query("SELECT * FROM orders", con)
#         con.close()
#         return df
#     except Exception:
#         print(f"failed to read the intraday db :{db_path}")
#         return pd.DataFrame()

def _read_orders(db_path: str, retries: int = 4, sleep_s: float = 0.15) -> pd.DataFrame:
    """
    Robust reader for sqlite that may be mid-copy / locked.
    - Retries a few times with short sleep
    - Uses busy_timeout to handle transient locks
    Returns empty DF if still failing.
    """
    if not db_path or not os.path.exists(db_path):
        return pd.DataFrame()

    last_err = None
    for i in range(retries):
        try:
            # timeout helps when file is locked; busy_timeout helps inside sqlite too
            con = sqlite3.connect(db_path, timeout=1.0)
            con.execute("PRAGMA busy_timeout=1000;")  # ms
            df = pd.read_sql_query("SELECT * FROM orders", con)
            con.close()
            return df
        except Exception as e:
            last_err = e
            try:
                con.close()
            except Exception:
                pass
            _time.sleep(sleep_s)

    print(f"failed to read sqlite after {retries} retries: {db_path} | last_err={last_err}")
    return pd.DataFrame()


def compute_mock_strategy_pnl(
    intraday_db: str,
    positional_db: str,
    ltp_ctx: dict,
    prev_raw: pd.DataFrame | None,
    today: date,
    user: str | None
) -> pd.DataFrame:
    """
    Returns columns: tag, SimCarryPnL, SimDayPnL

    Rules:
    - If positional db missing/empty => SimCarryPnL = 0 across tags.
    - If prev_raw missing/empty => no prev_close map => carry effectively 0 (we enforce).
    """
    ltp_agg = ltp_ctx.get("ltp_agg")
    if ltp_agg is not None and not ltp_agg.empty:
        ltp_map = dict(zip(ltp_agg["TradebookLike"].astype(str), ltp_agg["LTP"].astype(float)))
    else:
        ltp_map = dict(zip(ltp_ctx["ltp_raw"]["symbol"].astype(str), ltp_ctx["ltp_raw"]["ltp"].astype(float)))

    # prev_close map
    prev_close_map = {}
    if prev_raw is not None and not prev_raw.empty and PREV_AVG_COL in prev_raw.columns:
        pr = prev_raw.copy()
        pr["TradebookLike"] = pr[PREV_SYM_COL].astype(str).map(lambda s: to_symbol_tradebook_spaced(parse_symbol(s)))
        pr["prev_close"] = pd.to_numeric(pr.get(PREV_AVG_COL, np.nan), errors="coerce")
        prev_close_map = pr.dropna(subset=["TradebookLike"]).set_index("TradebookLike")["prev_close"].to_dict()

    # ---- SimDayPnL (intraday) ----
    intr = _read_orders(intraday_db)
    # filter_sim_live
    # if user in ["DMA20", "DMA09"]:
    #     intr = filter_after_today_1035_ist_direct(intr, "timestamp")

    # Capture ALL unique tags from DB BEFORE date-filtering.
    # If the date-filter removes all rows (timezone mismatch, format issue, etc.)
    # we still need the tag universe so downstream merges produce non-empty output.
    all_tags_from_intr: list = []

    sim_day = pd.DataFrame(columns=["tag", "SimDayPnL"])
    if not intr.empty:
        intr = intr.copy()
        intr["tag"] = intr.get("tag", intr.get("strategy_id", "__NO_TAG__")).astype(str).fillna("__NO_TAG__")
        intr["symbol"] = intr.get("symbol", "").astype(str)

        # Save tag universe before filtering
        all_tags_from_intr = intr["tag"].unique().tolist()

        fq = intr.get("filled_quantity")
        q0 = intr.get("quantity")
        intr["qty"] = pd.to_numeric(fq if fq is not None else q0, errors="coerce").fillna(0.0)

        side = intr.get("side", "").astype(str).str.upper()
        intr["signed_qty"] = intr["qty"].where(side.isin(["BUY", "B"]), -intr["qty"])

        intr["px"] = pd.to_numeric(intr.get("average_price", intr.get("price", np.nan)), errors="coerce")
        intr["TradebookLike"] = intr["symbol"].map(lambda s: to_symbol_tradebook_spaced(parse_symbol(s)))

        intr["ts"] = pd.to_datetime(intr.get("timestamp", pd.NaT), errors="coerce")
        # Robust date comparison: strip timezone if present to avoid comparison errors
        try:
            ts_dates = intr["ts"].dt.tz_convert(None).dt.date if intr["ts"].dt.tz is not None else intr["ts"].dt.date
        except Exception:
            ts_dates = intr["ts"].dt.date
        intr = intr[ts_dates == today]

        intr["LTP"] = intr["TradebookLike"].astype(str).map(ltp_map)
        intr["LTP"] = pd.to_numeric(intr["LTP"], errors="coerce")

        intr["pnl_row"] = (intr["LTP"] - intr["px"]) * intr["signed_qty"]
        intr["pnl_row"] = intr["pnl_row"].where(np.isfinite(intr["pnl_row"]), 0.0)

        if not intr.empty:
            sim_day = (
                intr.groupby("tag", as_index=False)[["pnl_row"]]
                    .sum()
                    .rename(columns={"pnl_row": "SimDayPnL"})
            )

    # If date-filter wiped all rows but we know the tags, seed with 0.0 so
    # the tag universe flows through all downstream merges.
    if sim_day.empty and all_tags_from_intr:
        log.info("mock: sim_day empty after date-filter; seeding %d tags with SimDayPnL=0", len(all_tags_from_intr))
        sim_day = pd.DataFrame({"tag": all_tags_from_intr, "SimDayPnL": 0.0})

    # ---- SimCarryPnL (positional) ----
    if (not positional_db) or (not os.path.exists(positional_db)):
        out = sim_day.copy()
        if out.empty:
            # Genuinely no data at all
            return pd.DataFrame(columns=["tag", "SimCarryPnL", "SimDayPnL"])
        out["SimCarryPnL"] = 0.0
        return out[["tag", "SimCarryPnL", "SimDayPnL"]]

    pos = _read_orders(positional_db)
    if pos is None or pos.empty:
        out = sim_day.copy()
        if out.empty:
            return pd.DataFrame(columns=["tag", "SimCarryPnL", "SimDayPnL"])
        out["SimCarryPnL"] = 0.0
        return out[["tag", "SimCarryPnL", "SimDayPnL"]]

    pos = pos.copy()
    pos["tag"] = pos.get("tag", pos.get("strategy_id", "__NO_TAG__")).astype(str).fillna("__NO_TAG__")
    pos["symbol"] = pos.get("symbol", "").astype(str)

    fq = pos.get("filled_quantity")
    q0 = pos.get("quantity")
    pos["qty"] = pd.to_numeric(fq if fq is not None else q0, errors="coerce").fillna(0.0)

    side = pos.get("side", "").astype(str).str.upper()
    pos["signed_qty"] = pos["qty"].where(side.isin(["BUY", "B"]), -pos["qty"])

    pos["TradebookLike"] = pos["symbol"].map(lambda s: to_symbol_tradebook_spaced(parse_symbol(s)))
    pos["ts"] = pd.to_datetime(pos.get("timestamp", pd.NaT), errors="coerce")

    # keep trades strictly before today (overnight inventory)
    pos = pos[pos["ts"].dt.date < today]

    net = (
        pos.groupby(["tag", "TradebookLike"], as_index=False)[["signed_qty"]]
           .sum()
           .rename(columns={"signed_qty": "net_qty"})
    )

    net["LTP"] = net["TradebookLike"].astype(str).map(ltp_map)
    net["prev_close"] = net["TradebookLike"].astype(str).map(prev_close_map)

    net["LTP"] = pd.to_numeric(net["LTP"], errors="coerce")
    net["prev_close"] = pd.to_numeric(net["prev_close"], errors="coerce")

    # If prev_close missing => carry should be 0; enforce by filling NaN with LTP (diff=0)
    net["prev_close"] = net["prev_close"].where(np.isfinite(net["prev_close"]), net["LTP"])

    net["carry_row"] = (net["LTP"] - net["prev_close"]) * net["net_qty"]
    net["carry_row"] = net["carry_row"].where(np.isfinite(net["carry_row"]), 0.0)

    sim_carry = (
        net.groupby("tag", as_index=False)[["carry_row"]]
           .sum()
           .rename(columns={"carry_row": "SimCarryPnL"})
    )

    out = sim_carry.merge(sim_day, on="tag", how="outer").fillna(0.0)
    return out[["tag", "SimCarryPnL", "SimDayPnL"]]


# -------------------------
# TAG min/max tracker (NetPnL)
# -------------------------
def update_tag_minmax(
    r: redis.Redis,
    username: str,
    now_dt: datetime,
    tag: str,
    net_pnl: float,
) -> tuple[float, str, float, str]:
    """
    key = risk:strategy_pnl_minmax:{user}:{YYYYMMDD}
    fields per tag:
      {tag}::net_min, {tag}::net_min_at, {tag}::net_max, {tag}::net_max_at
    """
    dkey = f"risk:strategy_pnl_minmax:{username}:{now_dt.strftime('%Y%m%d')}"
    tkey = str(tag).replace("\n", " ").strip()

    net = _sf(net_pnl, 0.0)
    now_iso = _iso(now_dt)

    f_min = f"{tkey}::net_min"
    f_min_at = f"{tkey}::net_min_at"
    f_max = f"{tkey}::net_max"
    f_max_at = f"{tkey}::net_max_at"

    cur = r.hmget(dkey, [f_min, f_min_at, f_max, f_max_at])
    cur_min = _sf(cur[0], None) if cur[0] is not None else None
    cur_max = _sf(cur[2], None) if cur[2] is not None else None
    min_at = cur[1] or ""
    max_at = cur[3] or ""

    if (cur_min is None) or (net < cur_min):
        cur_min = net
        min_at = now_iso
        r.hset(dkey, f_min, str(cur_min))
        r.hset(dkey, f_min_at, min_at)

    if (cur_max is None) or (net > cur_max):
        cur_max = net
        max_at = now_iso
        r.hset(dkey, f_max, str(cur_max))
        r.hset(dkey, f_max_at, max_at)

    r.expire(dkey, 2 * 24 * 3600)
    return float(cur_min), min_at, float(cur_max), max_at


# -------------------------
# FILTER HELPERS (unchanged)
# -------------------------
def filter_after_today_1035_local(df, time_col="timestamp"):
    ts = pd.to_datetime(
        df[time_col],
        format="%d-%m-%Y %H:%M:%S.%f"
    )
    today = pd.Timestamp.now().normalize()
    cutoff = today + pd.Timedelta(hours=10, minutes=8)
    filtered_df = df[ts >= cutoff]
    if not filtered_df.empty:
        print("Minimum datetime after filtering:", ts[ts >= cutoff].min())
    else:
        print("No rows after filtering")
    return filtered_df

def filter_after_today_1035_ist_direct(df, time_col="exchange_timestamp"):
    ts = pd.to_datetime(df[time_col])
    today_ist = pd.Timestamp.now(tz="Asia/Kolkata").normalize()
    cutoff_ist = today_ist + pd.Timedelta(hours=10, minutes=8)
    filtered_df = df[ts >= cutoff_ist]
    if not filtered_df.empty:
        print("Minimum IST datetime after filtering:", ts[ts >= cutoff_ist].min())
    else:
        print("No rows after filtering")
    return filtered_df

# -------------------------
# strategy_level payoff helpers
# -------------------------
def _pick_time_col(df: pd.DataFrame) -> str | None:
    """Pick a reasonable timestamp column for ordering trades."""
    for c in ["ExchangeTransactTime", "ExchangeTransactTime", "OrderGeneratedDateTime", "LastUpdateDateTime", "ExchangeTransactTime"]:
        if c in df.columns:
            return c
    return None

def _to_dt_series(df: pd.DataFrame, col: str) -> pd.Series:
    # Your tradebook sometimes uses dd-mm-YYYY format
    s = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    return s

def _wac_roll_apply(q0: float, a0: float, q: float, px: float) -> tuple[float, float]:
    """
    Lightweight position roll per (tag, symbol) to compute remaining open avg.
    - same sign: weighted avg
    - opposite sign: reduce; if flip, new avg=trade px
    """
    if not np.isfinite(px) or abs(q) < 1e-12:
        return q0, a0
    if abs(q0) < 1e-12:
        return q, float(px)

    # same direction => weighted avg
    if q0 * q > 0:
        den = abs(q0) + abs(q)
        if den <= 1e-12:
            return q0 + q, a0
        a = (a0 * abs(q0) + float(px) * abs(q)) / den
        return q0 + q, float(a)

    # opposite direction => close/reduce/flip
    if abs(q) < abs(q0):
        # partial close; keep avg of remaining
        return q0 + q, a0
    if abs(q) == abs(q0):
        # flat
        return 0.0, 0.0

    # flip
    q_new = q0 + q
    return q_new, float(px)

def build_open_positions_by_tag(
    tb_raw: pd.DataFrame,
    prev_raw: pd.DataFrame | None,
    ltp_ctx: dict,
    now_dt: datetime,
) -> pd.DataFrame:
    """
    Produces current open inventory by (tag, TradebookLike):
      columns: tag, TradebookLike, qty, avg, ul, kind, strike, cp
    Includes prev_raw ONLY if it has PREV_TAG_COL (consistent with your carry rule).
    """
    if tb_raw is None or tb_raw.empty:
        return pd.DataFrame(columns=["tag","TradebookLike","qty","avg","ul","kind","strike","cp"])

    tb = maybe_make_signed_qty(tb_raw.copy())

    # tag column on TB
    if TB_TAG_COL in tb.columns:
        tb["tag"] = tb[TB_TAG_COL].astype(str).fillna("__NO_TAG__")
    elif "tag" in tb.columns:
        tb["tag"] = tb["tag"].astype(str).fillna("__NO_TAG__")
    else:
        tb["tag"] = "__NO_TAG__"

    tb["TradebookLike"] = tb[TB_SYMBOL_COL].astype(str).map(lambda s: to_symbol_tradebook_spaced(parse_symbol(s)))
    tb["signed_qty"] = pd.to_numeric(tb.get("signed_qty", 0.0), errors="coerce").fillna(0.0)
    tb["px"] = pd.to_numeric(tb.get(TB_AVG_COL, np.nan), errors="coerce")

    # Order by time if possible
    tcol = _pick_time_col(tb)
    if tcol:
        tb["_ts"] = _to_dt_series(tb, tcol)
        tb = tb.sort_values("_ts")

    # initial state from prev_raw if tag exists
    state_qty: dict[tuple[str,str], float] = {}
    state_avg: dict[tuple[str,str], float] = {}

    if prev_raw is not None and (not prev_raw.empty) and (PREV_TAG_COL in prev_raw.columns):
        pr = prev_raw.copy()
        pr["tag"] = pr[PREV_TAG_COL].astype(str).fillna("__NO_TAG__")
        pr["TradebookLike"] = pr[PREV_SYM_COL].astype(str).map(lambda s: to_symbol_tradebook_spaced(parse_symbol(s)))
        pr["qty0"] = pd.to_numeric(pr.get(PREV_QTY_COL, 0.0), errors="coerce").fillna(0.0)
        pr["avg0"] = pd.to_numeric(pr.get(PREV_AVG_COL, np.nan), errors="coerce")

        for _, r in pr.iterrows():
            tag = str(r["tag"])
            sym = r["TradebookLike"]
            if not sym or (not isinstance(sym, str)):
                continue
            q0 = float(r["qty0"])
            a0 = float(r["avg0"]) if np.isfinite(r["avg0"]) else 0.0
            if abs(q0) < 1e-12:
                continue
            key = (tag, sym)
            state_qty[key] = state_qty.get(key, 0.0) + q0
            # if multiple rows collide, just keep last avg (rare); better would be WAC, but OK
            state_avg[key] = a0

    # apply today's trades
    for _, r in tb.iterrows():
        tag = str(r["tag"])
        sym = r["TradebookLike"]
        if not sym or not isinstance(sym, str):
            continue
        q = float(r["signed_qty"])
        px = float(r["px"]) if np.isfinite(r["px"]) else np.nan
        if abs(q) < 1e-12 or (not np.isfinite(px)):
            continue
        key = (tag, sym)
        q0 = float(state_qty.get(key, 0.0))
        a0 = float(state_avg.get(key, 0.0))
        q1, a1 = _wac_roll_apply(q0, a0, q, px)
        state_qty[key] = q1
        state_avg[key] = a1

    rows = []
    for (tag, sym), q in state_qty.items():
        if abs(q) < 1e-12:
            continue
        a = float(state_avg.get((tag, sym), 0.0))
        c = parse_symbol(sym)
        if not c:
            continue
        ul = str(c.ul).upper()
        kind = str(c.kind).upper()
        strike = float(c.strike) if c.strike is not None else np.nan
        cp = +1 if (c.opt and str(c.opt).upper()=="CE") else (-1 if (c.opt and str(c.opt).upper()=="PE") else np.nan)

        rows.append({
            "tag": tag,
            "TradebookLike": sym,
            "qty": float(q),
            "avg": float(a),
            "ul": ul,
            "kind": kind,
            "strike": strike,
            "cp": cp,
        })

    return pd.DataFrame(rows)

def compute_strategy_payoff_pack(
    open_pos: pd.DataFrame,
    spot_map: dict[str, float],
    grid_pct: float = 0.08,
    n_points: int = 201,
    top_tags: list[str] | None = None,
) -> dict:
    """
    Returns JSON-able payoff pack:
      { "meta": {...}, "data": { tag: { ul: {"S0":..., "S":[...], "pnl":[...]} } } }
    """
    pack = {
        "meta": {"grid_pct": float(grid_pct), "n_points": int(n_points)},
        "data": {}
    }
    if open_pos is None or open_pos.empty:
        return pack

    df = open_pos.copy()
    df["tag"] = df["tag"].astype(str)
    df["ul"] = df["ul"].astype(str).str.upper()

    if top_tags is not None:
        df = df[df["tag"].isin(set(top_tags))].copy()
        if df.empty:
            return pack

    for tag, gtag in df.groupby("tag", sort=False):
        pack["data"].setdefault(tag, {})
        for ul, g in gtag.groupby("ul", sort=False):
            S0 = spot_map.get(ul)
            try:
                S0 = float(S0)
            except Exception:
                S0 = np.nan
            if not np.isfinite(S0) or S0 <= 0:
                continue

            lo, hi = S0 * (1.0 - grid_pct), S0 * (1.0 + grid_pct)
            S = np.linspace(lo, hi, int(n_points), dtype=float)
            pnl = np.zeros_like(S, dtype=float)

            for _, r in g.iterrows():
                qty = float(r.get("qty", 0.0) or 0.0)
                avg = float(r.get("avg", 0.0) or 0.0)
                kind = str(r.get("kind", "")).upper()

                if abs(qty) < 1e-12:
                    continue

                if kind == "OPT":
                    K = float(r.get("strike", np.nan))
                    cp = float(r.get("cp", np.nan))
                    if not np.isfinite(K) or not np.isfinite(cp):
                        continue
                    intrinsic = np.maximum(cp * (S - K), 0.0)
                    pnl += (intrinsic - avg) * qty
                elif kind in ("FUT", "SPOT"):
                    pnl += (S - avg) * qty

            pack["data"][tag][ul] = {
                "S0": float(S0),
                "S": S.tolist(),
                "pnl": pnl.tolist(),
            }

    return pack


def should_stop_for_day(now_dt: datetime) -> bool:
    """Stop the worker after STOP_TIME on the same day (IST-aware now_dt)."""
    if not STOP_AFTER_CUTOFF:
        return False
    try:
        return now_dt.time() >= STOP_TIME
    except Exception:
        return False

# -------------------------
# MAIN LOOP
# -------------------------
def main():
    r = redis_client()
    r.ping()
    log.info("strategy_pnl_worker started | users=%d | loop=%.2fs", len(USERDETAILS), LOOP_SECONDS)

    last_ts_write: dict[str, float] = {}  # username -> last write epoch seconds
    while True:
        tick_start = _time.time()
        now_dt = datetime.now(IST)
        
        # ---- STOP after cutoff (15:30 IST by default) ----
        if should_stop_for_day(now_dt):
            log.info(
                "Stopping strategy_pnl_worker: now=%s reached stop_time=%02d:%02d IST",
                now_dt.isoformat(timespec="seconds"),
                STOP_TIME.hour,
                STOP_TIME.minute,
            )
            return
        
        today = now_dt.date()
        prev_date = get_prev_valid_date(now_dt, DEFAULT_HOLIDAY_LIST)

        # LTP once per tick
        ltp_raw = fetch_ltp_df(r)
        if ltp_raw.empty:
            log.warning("LTP empty; skipping tick")
            _time.sleep(LOOP_SECONDS)
            continue
        ltp_ctx = prepare_ltp_context(ltp_raw)

        for username in USERDETAILS.keys():
            try:
                tb_path = build_tradebook_path(username, now_dt)
                tb_raw = read_csv_safe(tb_path)
                # Do NOT skip when tb_raw is None/empty — mock must still run.
                if tb_raw is not None and tb_raw.empty:
                    tb_raw = None  # normalise empty df -> None

                # Normalize per-user columns + compact symbols -> spaced format
                if tb_raw is not None:
                    sym_col = get_user_col(username, "col_symbol")
                    if sym_col in tb_raw.columns:
                        tb_raw[sym_col] = tb_raw[sym_col].astype(str).map(
                            lambda s: normalize_symbol(s, username)
                        )
                    tb_raw.rename(columns={
                        get_user_col(username, "col_qty"):    TB_QTY_COL,
                        get_user_col(username, "col_price"):  TB_AVG_COL,
                        get_user_col(username, "col_side"):   "OrderSide",
                        get_user_col(username, "col_symbol"): TB_SYMBOL_COL,
                        get_user_col(username, "col_tag"):    TB_TAG_COL,
                    }, inplace=True)

                # filter_sim_live
                # if username in ["DMA20", "DMA09"]:
                #     tb_raw = filter_after_today_1035_local(tb_raw, "ExchangeTransactTime")

                prev_path = build_prev_eod_path(username, prev_date)
                prev_raw = read_csv_safe(prev_path)
                if prev_raw is None or prev_raw.empty:
                    prev_raw = None  # treat as missing

                # live table (no best-effort mapping; carry only if prev_raw has PREV_TAG_COL)
                live, live_meta = compute_live_strategy_pnl(tb_raw, prev_raw, ltp_ctx)

                # mock table
                intr_db, pos_db = build_mock_db_paths(username)
                mock = compute_mock_strategy_pnl(intr_db, pos_db, ltp_ctx, prev_raw, today, user=username)

                # merge (guarantee tag exists)
                out = live.merge(mock, on="tag", how="outer").fillna(0.0)

                #payoff block
                # -------- Strategy payoff publish (top N tags) --------
                try:
                    spot_map = (ltp_ctx.get("spot_map") or {})
                    # choose top tags by abs(NetPnL)
                    top_n = int(STRAT_PAYOFF_TOP_N)
                    tags_top = (
                        out.assign(_abs=out["NetPnL"].abs())
                        .sort_values("_abs", ascending=False)
                        .head(top_n)["tag"]
                        .astype(str)
                        .tolist()
                    )

                    open_pos = build_open_positions_by_tag(tb_raw, prev_raw, ltp_ctx, now_dt)
                    payoff_pack = compute_strategy_payoff_pack(
                        open_pos=open_pos,
                        spot_map=spot_map,
                        grid_pct=STRAT_PAYOFF_GRID_PCT,
                        n_points=STRAT_PAYOFF_N_POINTS,
                        top_tags=tags_top,
                    )

                    payoff_payload = {
                        "username": username,
                        "as_of": _iso(now_dt),
                        "top_n": top_n,
                        "meta": payoff_pack.get("meta", {}),
                        "data": payoff_pack.get("data", {}),
                    }
                    r.set(
                        f"risk:strategy_payoff:latest:{username}",
                        json.dumps(payoff_payload, default=str, ensure_ascii=False),
                    )
                except Exception as e:
                    log.warning("strategy payoff publish failed user=%s: %s", username, e)

                # ---- USER-SPECIFIC ALLOCATED MARGIN ----
                alloc_path = build_alloc_margin_path(username, now_dt)
                alloc_mp = load_alloc_margin_map(alloc_path)
                # print(username,alloc_mp )
                out["allocated_margin"] = out["tag"].map(lambda t: alloc_margin_for(str(t), alloc_mp)).astype(float)

                # net_pnl/margin (%)
                out["net_pnl/margin (%)"] = np.where(
                    out["allocated_margin"] > 0,
                    out["NetPnL"] / out["allocated_margin"] * 100.0,
                    np.nan
                )

                # Slippage (%) = (DayPnL - SimDayPnL) / abs(SimDayPnL) * 100
                out["Slippage (%)"] = np.where(
                    np.abs(out["SimDayPnL"]) > 1e-9,
                    (out["DayPnL"] - out["SimDayPnL"]) / np.abs(out["SimDayPnL"]) * 100.0,
                    np.nan
                )

                # sim_live_diff % = (NetPnL - (SimCarryPnL+SimDayPnL)) / allocated_margin * 100
                out["sim_live_diff %"] = np.where(
                    out["allocated_margin"] > 0,
                    (out["NetPnL"] - (out["SimCarryPnL"] + out["SimDayPnL"])) / out["allocated_margin"] * 100.0,
                    np.nan
                )

                # Tag min/max (NetPnL)
                mins, mins_at, maxs, maxs_at = [], [], [], []
                for _, row in out.iterrows():
                    mn, mn_at, mx, mx_at = update_tag_minmax(
                        r, username, now_dt, str(row["tag"]), float(row["NetPnL"])
                    )
                    mins.append(mn)
                    mins_at.append(mn_at[11:19] if mn_at else "")  # HH:MM:SS
                    maxs.append(mx)
                    maxs_at.append(mx_at[11:19] if mx_at else "")

                out["MinPnL"] = mins
                out["MinTime"] = mins_at
                out["MaxPnL"] = maxs
                out["MaxTime"] = maxs_at

                # Order columns
                col_order = [
                    "tag",
                    "CarryPnL", "DayPnL", "Expenses", "NetPnL",
                    "allocated_margin", "net_pnl/margin (%)",
                    "MinPnL", "MinTime", "MaxPnL", "MaxTime",
                    "SimCarryPnL", "SimDayPnL",
                    "Slippage (%)", "sim_live_diff %"
                ]
                for c in col_order:
                    if c not in out.columns:
                        out[c] = np.nan
                out = out[col_order].sort_values("NetPnL", ascending=True)

                payload = {
                    "username": username,
                    "as_of": _iso(now_dt),
                    "alloc_margin_path": alloc_path,
                    "rows": out.to_dict(orient="records"),
                    "meta": {
                        **live_meta,
                        "has_mock_intraday_db": bool(intr_db and os.path.exists(intr_db)),
                        "has_mock_positional_db": bool(pos_db and os.path.exists(pos_db)),
                        "intraday_db_path": intr_db,
                        "positional_db_path": pos_db,
                        "tradebook_tag_col": TB_TAG_COL,
                        "prev_tag_col": PREV_TAG_COL,
                        "prev_path": prev_path,
                        "tradebook_path": tb_path,
                        "alloc_margin_cols_expected": ["Strategy Name", "Multiplier", "Total Margin"],
                    }
                }

                r.set(
                    f"risk:strategy_pnl:latest:{username}",
                    json.dumps(payload, default=str, ensure_ascii=False)
                )

                #storing strategywise pnl
                # ---- write per-tag timeseries locally (downsampled) ----
                # now_s = _time.time()
                # prev_s = last_ts_write.get(username, 0.0)
                # if (now_s - prev_s) >= STRAT_TS_WRITE_EVERY_SECONDS:
                #     try:
                #         write_strategy_ts_parquet(username, now_dt, out)
                #         last_ts_write[username] = now_s
                #     except Exception as e:
                #         log.warning("strategy_ts parquet write failed user=%s: %s", username, e)

            except Exception as e:
                log.error("user=%s failed: %s", username, e)

        elapsed = _time.time() - tick_start
        print(f"single loop time elapsed: {elapsed}")
        _time.sleep(max(0.05, LOOP_SECONDS - elapsed))


if __name__ == "__main__":
    main()