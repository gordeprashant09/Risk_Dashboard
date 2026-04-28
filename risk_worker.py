# risk_worker.py
"""
Single-process risk worker:
- Fetch LTPs from Redis hash (default: last_price)
- For each user:
  - read tradebook CSV
  - read prev EOD net positions CSV (optional)
  - build effective open book (FIFO roll)
  - run fast engine (vectorized IV+greeks)
  - compute P&L summary + min/max tracking in Redis
  - publish margin inputs (for separate margin worker) if enabled
  - publish risk snapshot for viewer

Redis keys written:
- risk:outputs:latest:{username}           (JSON)
- risk:pnl_minmax:{username}:{YYYYMMDD}    (HASH)
- margin:inputs:latest:{username}          (JSON)   (optional)

Reads (if available):
- margin:outputs:latest:{username}         (JSON)
- margin:outputs:minmax:{username}:{span_date} (HASH)
"""

from __future__ import annotations

import os
import time as _time
import json
import traceback
import logging
from datetime import datetime, time, date,timezone
import pathlib
import numpy as np
import pandas as pd
import redis
import math
from parquet_ts_store import DailyParquetStore
import warnings

warnings.filterwarnings("ignore")

from risk_lib_fast import (
    IST, DEFAULT_HOLIDAY_LIST,
    get_prev_valid_date, filter_expired_positions,
    roll_positions_weighted_average,
    prepare_ltp_context, run_engine_fast_from_ctx2,
    compute_day_pnl_from_trades, compute_carry_pnl_from_prev, compute_trading_expenses,
    build_margin_inputs_from_tbm, payload_id,
    df_to_records,
    compute_payoff_pack,
    to_symbol_tradebook_spaced,
    parse_symbol
)

# -------------------------
# Code snippet for combined user
# -------------------------
from dataclasses import dataclass, field

COMBINED_USER = "__ALL__"

def _to_float2(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)

def _merge_combo_sum(dst_combo: dict, src_combo: dict) -> dict:
    """
    combo format:
      combo[T+XD][V+Y][spot%] = pnl
    """
    if not isinstance(src_combo, dict) or not src_combo:
        return dst_combo
    if not isinstance(dst_combo, dict):
        dst_combo = {}

    for tkey, vold in src_combo.items():
        if not isinstance(vold, dict):
            continue
        dst_combo.setdefault(tkey, {})
        for vkey, spotd in vold.items():
            if not isinstance(spotd, dict):
                continue
            dst_combo[tkey].setdefault(vkey, {})
            for skey, val in spotd.items():
                dst_combo[tkey][vkey][skey] = _to_float2(dst_combo[tkey][vkey].get(skey, 0.0)) + _to_float2(val, 0.0)

    return dst_combo

def _merge_scenarios_sum(dst: dict, src: dict) -> dict:
    """
    dst/src format (viewer format):
      { ul: { "spot":{shock:val,...}, "vol":{...}, "time":{...}, "combo":{T:{V:{S:val}}} }, ... }
    """
    if not src:
        return dst
    for ul, packs in (src or {}).items():
        if ul not in dst:
            dst[ul] = {}

        for k, shock_map in (packs or {}).items():
            if k == "combo":
                dst[ul]["combo"] = _merge_combo_sum(dst[ul].get("combo", {}), shock_map or {})
                continue

            if k not in dst[ul]:
                dst[ul][k] = {}

            if not isinstance(shock_map, dict):
                continue

            for shock, val in shock_map.items():
                dst[ul][k][shock] = _to_float2(dst[ul][k].get(shock, 0.0)) + _to_float2(val, 0.0)

    return dst

def _add_df_sum(acc_df: pd.DataFrame | None, df: pd.DataFrame | None, index_cols: list[str]) -> pd.DataFrame | None:
    if df is None or df.empty:
        return acc_df
    d = df.copy()
    if "expiry" in d.columns:
        d["expiry"] = d["expiry"].astype(str).replace({"None": "", "NaT": ""})

    d = d.set_index(index_cols)
    d = d.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if acc_df is None or acc_df.empty:
        return d
    return acc_df.add(d, fill_value=0.0)

@dataclass
class CombinedAcc:
    users: list[str] = field(default_factory=list)
    kpis_sum: dict = field(default_factory=dict)
    margin_sum: dict = field(default_factory=dict)

    pf_sum: pd.DataFrame | None = None
    ul_exp_sum: pd.DataFrame | None = None
    ul_sum: pd.DataFrame | None = None

    scenarios_sum: dict = field(default_factory=dict)

    def add_user(
        self,
        username: str,
        snap: dict,
        pf: pd.DataFrame | None,
        ul_exp_pnl_df: pd.DataFrame | None,
        ul_pnl_df: pd.DataFrame | None,
        scenarios_json: dict | None,
    ):
        self.users.append(username)

        # KPIs: sum numeric
        k = (snap.get("kpis") or {})
        for key in ["carry_pnl", "day_pnl", "expenses", "net_pnl", "legs_open"]:
            self.kpis_sum[key] = _to_float2(self.kpis_sum.get(key, 0.0)) + _to_float2(k.get(key, 0.0))

        # Margin: sum numeric
        m = (snap.get("margin") or {})
        for key in ["span", "exposure", "total"]:
            self.margin_sum[key] = _to_float2(self.margin_sum.get(key, 0.0)) + _to_float2(m.get(key, 0.0))

        # PF: drop Total row then add
        if pf is not None and not pf.empty:
            pf2 = pf.copy()
            try:
                pf2 = pf2.drop(index=("Total", None), errors="ignore")
            except Exception:
                pass

            pf2 = pf2.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            if self.pf_sum is None:
                self.pf_sum = pf2
            else:
                self.pf_sum = self.pf_sum.add(pf2, fill_value=0.0)

        # Sum pnl tables
        self.ul_exp_sum = _add_df_sum(self.ul_exp_sum, ul_exp_pnl_df, ["ul", "expiry"])
        self.ul_sum     = _add_df_sum(self.ul_sum,     ul_pnl_df,     ["ul"])

        # Sum scenarios (spot/vol/time + combo cube)
        self.scenarios_sum = _merge_scenarios_sum(self.scenarios_sum, scenarios_json or {})

def finalize_pf_with_total(pf_sum: pd.DataFrame | None) -> pd.DataFrame:
    if pf_sum is None or pf_sum.empty:
        return pd.DataFrame()
    out = pf_sum.copy()
    total = out.sum().to_frame().T
    total.index = pd.MultiIndex.from_tuples([("Total", None)], names=["underlying", "expiry"])
    return pd.concat([out, total], axis=0)

#----------------------------------------------------------------------------------------------------------------------------------------

# -------------------------
# Base directory for parquet
# -------------------------
BASE_DIR = r"/mnt/Quant_Research/Risk_dashboard_inputs"

# -------------------------
# Local scenarios store (JSONL)
# -------------------------
SCENARIO_LOCAL_DIR = os.getenv(
     "SCENARIO_LOCAL_DIR",
     os.path.join(BASE_DIR, "scenario_logs"))

# -------------------------
# CONFIG
# -------------------------
# ---- Users loaded from users.yaml (edit that file to add/remove users) ----
from user_config import USERDETAILS, USERNAMES, get_user_col, normalize_symbol, build_tradebook_path as _build_tb_path

TRADEBOOK_PATH_TEMPLATE = os.getenv(
    "TRADEBOOK_PATH_TEMPLATE",
    "/mnt/Quant_Research/Risk_dashboard_inputs/{dma}/tradebook/tradebook_T611_{yyyy_mm_dd}.csv"
    # "Y:\\Risk_dashboard_inputs\\{dma}\\tradebook\\tradebook_T611_{yyyy_mm_dd}.csv"
)

PREV_EOD_TEMPLATE = os.getenv(
    "PREV_EOD_TEMPLATE",
    # "Y:\\Risk_dashboard_inputs\\{dma}\\eod_files\\net_positions_eod_{yymmdd}.csv"
    "/mnt/Quant_Research/Risk_dashboard_inputs/{dma}/eod_files/net_positions_eod_{yymmdd}.csv"
)

TB_SYMBOL_COL = os.getenv("TB_SYMBOL_COL", "TradingSymbol")
TB_QTY_COL    = os.getenv("TB_QTY_COL", "LastTradedQuantity")
TB_AVG_COL    = os.getenv("TB_AVG_COL", "OrderAverageTradedPrice")

PREV_SYM_COL  = os.getenv("PREV_SYM_COL", "instrument")
PREV_QTY_COL  = os.getenv("PREV_QTY_COL", "net_open_qty")
PREV_AVG_COL  = os.getenv("PREV_AVG_COL", "bhav_today")

CUTOFF_HOUR   = int(os.getenv("EXPIRY_CUTOFF_HOUR", "15"))
CUTOFF_MINUTE = int(os.getenv("EXPIRY_CUTOFF_MIN", "30"))
CUT = time(CUTOFF_HOUR, CUTOFF_MINUTE)

STOP_AFTER_CUTOFF = os.getenv("STOP_AFTER_CUTOFF", "1") == "1"   # enable/disable
STOP_TIME_HOUR    = int(os.getenv("STOP_TIME_HOUR", "15"))
STOP_TIME_MINUTE  = int(os.getenv("STOP_TIME_MINUTE", "30"))
STOP_TIME = time(STOP_TIME_HOUR, STOP_TIME_MINUTE)

RF   = float(os.getenv("RF", "0.065"))
QDIV = float(os.getenv("QDIV", "0.0"))

SPOT_MODE = os.getenv("SPOT_MODE", "Synthetic (K_atm+(C_atm - P_atm))")
PUBLISH_MARGIN = os.getenv("PUBLISH_MARGIN", "1") == "1"

# NEW: publish true combined scenarios cube
PUBLISH_COMBO_SCENARIOS = os.getenv("PUBLISH_COMBO_SCENARIOS", "1") == "1"

LOOP_SECONDS = float(os.getenv("RISK_LOOP_SECONDS", "2.0"))
TOP_LEGS_N   = int(os.getenv("TOP_LEGS_N", "50"))

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))
REDIS_PWD  = os.getenv("REDIS_PASSWORD", "") or None
LTP_HASH_KEY = os.getenv("LTP_HASH_KEY", "last_price")

# Shock scenarios (existing engine outputs)
SPOT_SHOCKS = [-.2,-.1, -.05, -.02, -.01, -0.005, 0.005, 0.01, 0.02, 0.05, 0.1]  # fractions
VOL_SHOCKS  = [-0.1,-.05, -.03,-.02, .01, 0.01, 0.02,0.03, 0.05,0.1,0.2]                           # abs IV shifts (engine)
TIME_SHOCKS = [1, 3, 5]                                                     # days (engine)

# NEW: grids for TRUE combined cube
# - spot uses your SPOT_SHOCKS (+ include 0)
# - vol uses points (1 pt = +0.01 abs IV) derived from VOL_SHOCKS (+ include -1/0/+1)
# - time uses days (+ include 0)
SPOT_SHOCKS_COMBO = list(dict.fromkeys([0.0] + SPOT_SHOCKS))
VOL_PTS_COMBO     = sorted(set([int(round(v * 100.0)) for v in (VOL_SHOCKS or [])] + [-1, 0, 1]))
TIME_DAYS_COMBO   = sorted(set([0] + (TIME_SHOCKS or [])))

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("risk_worker")

# json helper
def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        x = float(o)
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, (pd.Series,)):
        return o.tolist()
    if isinstance(o, (pd.DataFrame,)):
        return o.to_dict(orient="records")
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    return str(o)

# -------------------------
# HELPERS
# -------------------------
def redis_client() -> redis.Redis:
    return redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PWD,
        decode_responses=True,
        socket_timeout=2.0, socket_connect_timeout=2.0
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
    # Uses per-user tradebook_prefix from users.yaml
    base_dir = "/mnt/Quant_Research/Risk_dashboard_inputs"
    return _build_tb_path(user, base_dir, str(now_dt.date()))

def build_prev_eod_path(user: str, prev_date) -> str:
    meta = USERDETAILS[user]
    return PREV_EOD_TEMPLATE.format(
        dma=meta["dma"],
        yymmdd=prev_date.strftime("%y%m%d"),
    )

def maybe_make_signed_qty(tb_raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if tb_raw is None or tb_raw.empty:
        return tb_raw, TB_QTY_COL
    if "OrderSide" not in tb_raw.columns:
        return tb_raw, TB_QTY_COL

    side = tb_raw["OrderSide"].astype(str).str.upper()
    is_buy = side.isin(["BUY", "B"])
    qty_base = pd.to_numeric(tb_raw[TB_QTY_COL], errors="coerce")
    tb_raw = tb_raw.copy()
    tb_raw["signed_qty"] = qty_base.where(is_buy, -qty_base).fillna(0.0)
    return tb_raw, "signed_qty"

def _to_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _sf(x, default=None):
    try:
        if x is None:
            return default
        x = float(x)
        return x if np.isfinite(x) else default
    except Exception:
        return default

def extract_margin(margin_out: dict | None):
    """
    Supports BOTH formats:
      A) Raw worker output: {"result": {...}}
      B) Normalized output: {"has_output": True, "span":..., "exposure":..., "total":...}
    Returns: (total_margin, span, exposure)
    """
    if not margin_out or not isinstance(margin_out, dict):
        return 0.0, 0.0, 0.0

    if ("total" in margin_out) or ("span" in margin_out) or ("exposure" in margin_out):
        span = _sf(margin_out.get("span"), _sf(margin_out.get("span_broker_style"), 0.0))
        exposure = _sf(margin_out.get("exposure"), _sf(margin_out.get("exposure_total"), 0.0))
        total = _sf(margin_out.get("total"), _sf(margin_out.get("grand_total_broker_style"), span + exposure))
        return total, span, exposure

    result = margin_out.get("result") or {}
    span = _sf(result.get("span_broker_style"), 0.0)
    exposure = _sf(result.get("exposure_total"), 0.0)
    total = _sf(result.get("grand_total_broker_style"), span + exposure)
    return total, span, exposure

def update_pnl_minmax_hash(
    r: redis.Redis,
    username: str,
    now_dt: datetime,
    carry: float,
    day: float,
    net: float
) -> dict:
    dkey = f"risk:pnl_minmax:{username}:{now_dt.strftime('%Y%m%d')}"
    existing = r.hgetall(dkey) or {}

    def _iso(dt: datetime) -> str:
        return dt.isoformat(timespec="seconds")

    mm = {
        "carry_min": _sf(existing.get("carry_min")),
        "carry_max": _sf(existing.get("carry_max")),
        "day_min":   _sf(existing.get("day_min")),
        "day_max":   _sf(existing.get("day_max")),
        "net_min":   _sf(existing.get("net_min")),
        "net_max":   _sf(existing.get("net_max")),

        "carry_min_at": existing.get("carry_min_at"),
        "carry_max_at": existing.get("carry_max_at"),
        "day_min_at":   existing.get("day_min_at"),
        "day_max_at":   existing.get("day_max_at"),
        "net_min_at":   existing.get("net_min_at"),
        "net_max_at":   existing.get("net_max_at"),

        "last_carry": _sf(existing.get("last_carry")),
        "last_day":   _sf(existing.get("last_day")),
        "last_net":   _sf(existing.get("last_net")),
        "last_at":    existing.get("last_at"),

        "created_at": existing.get("created_at"),
        "updated_at": existing.get("updated_at"),
    }

    if not mm.get("created_at"):
        mm["created_at"] = _iso(now_dt)

    def _finite(x) -> bool:
        try:
            return math.isfinite(float(x))
        except Exception:
            return False

    carry_f = float(carry) if _finite(carry) else None
    day_f   = float(day)   if _finite(day) else None
    net_f   = float(net)   if _finite(net) else None

    now_iso = _iso(now_dt)

    def _upd(metric: str, val: float | None):
        if val is None:
            return

        min_k, max_k = f"{metric}_min", f"{metric}_max"
        min_at_k, max_at_k = f"{metric}_min_at", f"{metric}_max_at"

        cur_min = _sf(mm.get(min_k), None)
        cur_max = _sf(mm.get(max_k), None)

        if (cur_min is None) or (val < cur_min):
            mm[min_k] = float(val)
            mm[min_at_k] = now_iso

        if (cur_max is None) or (val > cur_max):
            mm[max_k] = float(val)
            mm[max_at_k] = now_iso

    _upd("carry", carry_f)
    _upd("day",   day_f)
    _upd("net",   net_f)

    if carry_f is not None:
        mm["last_carry"] = float(carry_f)
    if day_f is not None:
        mm["last_day"] = float(day_f)
    if net_f is not None:
        mm["last_net"] = float(net_f)

    mm["last_at"] = now_iso
    mm["updated_at"] = now_iso

    r.hset(dkey, mapping={k: str(v) for k, v in mm.items() if v is not None})
    r.expire(dkey, 2 * 24 * 3600)
    return mm

def publish_margin_inputs_if_changed(r: redis.Redis, username: str, payload: dict, now_dt: datetime) -> tuple[str, bool]:
    key = f"margin:inputs:latest:{username}"
    pid = payload_id(payload)
    prev = r.get(key)
    prev_pid = None
    if prev:
        try:
            prev_pid = json.loads(prev).get("payload_id")
        except Exception:
            prev_pid = None
    if pid == prev_pid:
        return pid, False
    wrapper = {"username": username, "as_of": now_dt.isoformat(), "payload_id": pid, "payload": payload}
    r.set(key, json.dumps(wrapper))
    return pid, True

def read_margin_outputs(r: redis.Redis, username: str) -> dict:
    out = {
        "has_output": False,
        "computed_at": "",
        "span": 0.0,
        "exposure": 0.0,
        "total": 0.0,
        "minmax": {},
    }
    key_out = f"margin:outputs:latest:{username}"
    raw = r.get(key_out)
    if not raw:
        return out
    try:
        obj = json.loads(raw)
    except Exception:
        return out

    out["has_output"] = True
    out["computed_at"] = obj.get("computed_at", "")
    result = obj.get("result") or {}
    out["span"] = _to_float(result.get("span_broker_style"))
    out["exposure"] = _to_float(result.get("exposure_total"))
    out["total"] = _to_float(result.get("grand_total_broker_style", out["span"] + out["exposure"]))

    span_date = obj.get("span_date")
    if span_date:
        mm_key = f"margin:outputs:minmax:{username}:{span_date}"
        mm = r.hgetall(mm_key) or {}
        out["minmax"] = mm
    return out

# -------------------------
# scenarios: existing engine output -> jsonable
# -------------------------
def _scenarios_to_jsonable(scenarios: dict[str, dict[str, pd.DataFrame]]) -> dict:
    out = {}
    for ul, packs in (scenarios or {}).items():
        out[ul] = {}
        for k, df in packs.items():
            if df is None or df.empty:
                out[ul][k] = {}
            else:
                out[ul][k] = {c: float(df.iloc[0][c]) for c in df.columns}
    return out

# -------------------------
# TRUE COMBINED SCENARIOS (spot + vol_pts + time_days) via joint repricing
# -------------------------

def _norm_cdf_approx(x: np.ndarray) -> np.ndarray:
    """
    Fast normal CDF approximation (Abramowitz-Stegun).
    Works without scipy/erf.
    """
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * ax)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    pdf = np.exp(-0.5 * ax * ax) / np.sqrt(2.0 * np.pi)
    cdf_pos = 1.0 - pdf * poly
    return np.where(x >= 0, cdf_pos, 1.0 - cdf_pos)

def _bs_price_vec(cp: np.ndarray, S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, q: float, vol: np.ndarray) -> np.ndarray:
    """
    Vectorized Black-Scholes (spot model, continuous dividend yield q).
    cp: array of +1 for call, -1 for put
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    vol = np.asarray(vol, dtype=float)
    cp = np.asarray(cp, dtype=float)

    out = np.zeros_like(S, dtype=float)

    # intrinsic if expired / invalid vol
    intrinsic = np.maximum(cp * (S - K), 0.0)
    mask = (T > 0.0) & (vol > 0.0) & np.isfinite(S) & np.isfinite(K) & (S > 0.0) & (K > 0.0)

    if not np.any(mask):
        return intrinsic

    Sm = S[mask]
    Km = K[mask]
    Tm = T[mask]
    vm = vol[mask]
    cpm = cp[mask]

    sqrtT = np.sqrt(Tm)
    d1 = (np.log(Sm / Km) + (r - q + 0.5 * vm * vm) * Tm) / (vm * sqrtT)
    d2 = d1 - vm * sqrtT

    Nd1 = _norm_cdf_approx(cpm * d1)
    Nd2 = _norm_cdf_approx(cpm * d2)

    dfq = np.exp(-q * Tm)
    dfr = np.exp(-r * Tm)

    # call: cp=+1 => dfq*S*N(d1) - dfr*K*N(d2)
    # put : cp=-1 => dfq*S*N(-d1) - dfr*K*N(-d2) = cp*(...) formulation below
    outm = cpm * (dfq * Sm * Nd1 - dfr * Km * Nd2)

    out[mask] = outm
    out[~mask] = intrinsic[~mask]
    return out

def _opt_cp_from_row(opt_val: str) -> int:
    o = (opt_val or "").upper()
    if o in ("CE", "CALL", "C"):
        return 1
    if o in ("PE", "PUT", "P"):
        return -1
    # fallback: treat unknown as call
    return 1

def _pick_mult_cols(df: pd.DataFrame) -> str | None:
    for c in ["mult", "lot_size", "lotsize", "lot", "multiplier"]:
        if c in df.columns:
            return c
    return None

def compute_combo_cube_from_tbm(
    tbm: pd.DataFrame,
    qty_col: str,
    rf: float,
    q_div: float,
    spot_shocks_frac: list[float],
    vol_pts: list[int],
    time_days: list[int],
) -> dict:
    """
    Returns:
      combo[ul][T+XD][V+Y][spot_key] = pnl_rupees

    TRUE combined repricing:
      S -> S*(1+spot_shock)
      iv -> iv + (vol_pts/100)
      T -> max(T - days/365, 0)
    """
    if tbm is None or tbm.empty or "ul" not in tbm.columns:
        return {}

    df = tbm.copy()

    # required-ish columns; be defensive
    if "S_used" not in df.columns:
        return {}

    # numeric coercions
    df["S_used"] = pd.to_numeric(df["S_used"], errors="coerce")
    if "strike" in df.columns:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    if "T" in df.columns:
        df["T"] = pd.to_numeric(df["T"], errors="coerce")
    else:
        df["T"] = np.nan
    if "iv" in df.columns:
        df["iv"] = pd.to_numeric(df["iv"], errors="coerce")
    else:
        df["iv"] = np.nan
    if qty_col in df.columns:
        df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0)
    else:
        return {}

    mult_col = _pick_mult_cols(df)
    if mult_col:
        df[mult_col] = pd.to_numeric(df[mult_col], errors="coerce").fillna(1.0)
    else:
        df["_mult_"] = 1.0
        mult_col = "_mult_"

    # option detection
    kind = df.get("kind")
    opt  = df.get("opt")

    kind_s = kind.astype(str).str.upper() if kind is not None else pd.Series([""] * len(df))
    opt_s  = opt.astype(str).str.upper() if opt is not None else pd.Series([""] * len(df))

    is_opt = opt_s.isin(["CE", "PE", "CALL", "PUT", "C", "P"]) | kind_s.str.contains("OPT", na=False)
    is_opt = is_opt & df["strike"].notna() & df["iv"].notna()

    # arrays
    S0 = df["S_used"].to_numpy(dtype=float)
    K0 = df["strike"].to_numpy(dtype=float) if "strike" in df.columns else np.full(len(df), np.nan)
    T0 = df["T"].to_numpy(dtype=float)
    iv0 = df["iv"].to_numpy(dtype=float)
    qty = df[qty_col].to_numpy(dtype=float)
    mult = df[mult_col].to_numpy(dtype=float)

    # cp (+1 call, -1 put)
    cp = np.array([_opt_cp_from_row(x) for x in opt_s.to_list()], dtype=float)

    # base prices: options via BS; non-options use LTP if present else S_used
    price_base = np.zeros(len(df), dtype=float)
    if np.any(is_opt.to_numpy()):
        m = is_opt.to_numpy()
        price_base[m] = _bs_price_vec(cp[m], S0[m], K0[m], T0[m], rf, q_div, iv0[m])

    if "LTP" in df.columns:
        ltp = pd.to_numeric(df["LTP"], errors="coerce").to_numpy(dtype=float)
        base_nonopt = np.where(np.isfinite(ltp), ltp, S0)
    else:
        base_nonopt = S0

    price_base[~is_opt.to_numpy()] = base_nonopt[~is_opt.to_numpy()]

    combo_by_ul: dict = {}

    # group by underlying
    for ul, g in df.groupby("ul", dropna=False):
        gi = g.index.to_numpy()
        if len(gi) == 0:
            continue

        # slice arrays
        S0g = S0[gi]
        K0g = K0[gi]
        T0g = T0[gi]
        iv0g = iv0[gi]
        qtyg = qty[gi]
        multg = mult[gi]
        cpg = cp[gi]
        base_pg = price_base[gi]

        is_opt_g = is_opt.loc[gi].to_numpy()

        # base portfolio value (for MTM-change style PnL)
        # pnl = sum((P_shock - P_base) * qty * mult)
        # (no need to compute base value separately)
        ul_combo: dict = {}

        for td in time_days:
            tkey = f"T+{int(td)}D"
            ul_combo.setdefault(tkey, {})
            T1g = np.maximum(T0g - (float(td) / 365.0), 0.0)

            for vp in vol_pts:
                vkey = f"V{int(vp):+d}"
                spot_map: dict = {}

                iv1g = iv0g + (float(vp) / 100.0)
                iv1g = np.maximum(iv1g, 1e-6)

                for sp in spot_shocks_frac:
                    # keys match your viewer sorter expectations (no leading +)
                    skey = f"{float(sp) * 100.0:.1f}%"
                    S1g = S0g * (1.0 + float(sp))

                    # shocked prices
                    p1 = np.zeros_like(base_pg)
                    if np.any(is_opt_g):
                        m = is_opt_g
                        p1[m] = _bs_price_vec(cpg[m], S1g[m], K0g[m], T1g[m], rf, q_div, iv1g[m])

                    # non-options: only spot shock matters (vol/time irrelevant)
                    p1[~is_opt_g] = base_pg[~is_opt_g] * (1.0 + float(sp))

                    pnl = float(np.nansum((p1 - base_pg) * qtyg * multg))
                    spot_map[skey] = pnl

                ul_combo[tkey][vkey] = spot_map

        combo_by_ul[str(ul)] = ul_combo

    # cleanup temp column
    if "_mult_" in df.columns:
        pass

    return combo_by_ul

def should_stop_for_day(now_dt: datetime) -> bool:
    """
    Stop the worker after STOP_TIME on trading days.
    Uses IST-aware now_dt.
    """
    try:
        if not STOP_AFTER_CUTOFF:
            return False
        return now_dt.time() > STOP_TIME
    except Exception:
        return False
# -------------------------
# MAIN LOOP
# -------------------------
def main():
    r = redis_client()
    try:
        r.ping()
    except Exception as e:
        raise RuntimeError(f"Redis not reachable: {e}")

    log.info("risk_worker started | users=%d | loop=%.2fs | redis=%s:%d/%d",
             len(USERDETAILS), LOOP_SECONDS, REDIS_HOST, REDIS_PORT, REDIS_DB)

    last_tb_mtime: dict[str, float] = {}
    prev_cache: dict[tuple[str, str], pd.DataFrame | None] = {}

    ts_store = DailyParquetStore(
        base_dir=BASE_DIR,
        flush_every_points=200,
        compression="zstd"
    )

    last_parquet_flush = 0.0

         # scenario write dedupe (per user)
    last_scn_pid: dict[str, str] = {}
 
    def append_scenarios_local(username: str, now_dt: datetime, scenarios_obj: dict):
        """
        Append one JSON line with the scenario grid to:
        {SCENARIO_LOCAL_DIR}/date=YYYYMMDD/user=<username>/scenarios.jsonl
        Dedupe: only writes when payload_id changes.
        """
        if not scenarios_obj:
            return

        try:
            pid = payload_id({"scenarios": scenarios_obj})
            if last_scn_pid.get(username) == pid:
                return
            last_scn_pid[username] = pid

            day = now_dt.strftime("%Y%m%d")
            out_dir = os.path.join(SCENARIO_LOCAL_DIR, f"date={day}", f"user={username}")
            os.makedirs(out_dir, exist_ok=True)

            ts_utc = now_dt.astimezone(timezone.utc).isoformat(timespec="seconds")
            rec = {
                "ts_utc": ts_utc,
                "as_of": now_dt.isoformat(timespec="seconds"),
                "username": username,
                "payload_id": pid,
                "scenarios": scenarios_obj,
            }

            out_path = os.path.join(out_dir, "scenarios.jsonl")
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=json_default, ensure_ascii=False) + "\n")

        except Exception:
            # don't let local logging break the worker
            log.debug("scenario local write failed for %s:\n%s", username, traceback.format_exc())
  
    while True:
        tick_start = _time.time()
        now_dt = datetime.now(IST)

        if should_stop_for_day(now_dt):
            log.info("Stopping risk_worker: now=%s reached stop_time=%02d:%02d IST",
                     now_dt.isoformat(timespec="seconds"), STOP_TIME.hour, STOP_TIME.minute)
            try:
                ts_store.flush_all()
            except Exception:
                pass
            return

        spot_nifty = 0.0
        spot_banknifty = 0.0
        spot_sensex = 0.0

        # 1) LTP once per tick
        try:
            ltp_raw = fetch_ltp_df(r)
            if ltp_raw.empty:
                log.warning("LTP empty; skipping tick")
                _time.sleep(LOOP_SECONDS)
                continue

            ltp_ctx = prepare_ltp_context(ltp_raw)
            spot_map = ltp_ctx.get("spot_map", {}) or {}

            def _spot(*keys):
                for k in keys:
                    v = spot_map.get(k)
                    try:
                        v = float(v)
                        if np.isfinite(v):
                            return v
                    except Exception:
                        pass
                return np.nan

            spot_nifty     = _spot("NIFTY", "NIFTY50", "NIFTY 50")
            spot_banknifty = _spot("BANKNIFTY", "NIFTY BANK", "BANK NIFTY")
            spot_sensex    = _spot("SENSEX", "BSE SENSEX")

        except Exception as e:
            log.error("LTP fetch/context failed: %s", e)
            _time.sleep(LOOP_SECONDS)
            continue

        prev_date = get_prev_valid_date(now_dt, DEFAULT_HOLIDAY_LIST)
        print("PREV DATE : ", prev_date)

        acc = CombinedAcc()

        # 2) Per user
        for username in USERDETAILS.keys():
            try:
                tb_path = build_tradebook_path(username, now_dt)
                if not tb_path or not os.path.exists(tb_path):
                    print(f"no tradebookfor {username}")
                    continue

                mtime = os.path.getmtime(tb_path)
                tb_raw = pd.read_csv(tb_path)
                last_tb_mtime[username] = mtime

                # Normalize compact symbols (e.g. NIFTY2640722650CE) to spaced format
                sym_col = get_user_col(username, "col_symbol")
                if sym_col in tb_raw.columns:
                    tb_raw[sym_col] = tb_raw[sym_col].astype(str).map(
                        lambda s: normalize_symbol(s, username)
                    )
                # Use per-user column names
                tb_raw.rename(columns={
                    get_user_col(username, "col_qty"):   TB_QTY_COL,
                    get_user_col(username, "col_price"): TB_AVG_COL,
                    get_user_col(username, "col_side"):  "OrderSide",
                    get_user_col(username, "col_symbol"): TB_SYMBOL_COL,
                }, inplace=True)
                tb_raw, qty_col = maybe_make_signed_qty(tb_raw)

                # prev eod cached per (user, prev_date)
                prev_key = (username, prev_date.strftime("%Y%m%d"))
                prev_raw = prev_cache.get(prev_key)
                if prev_raw is None:
                    prev_path = build_prev_eod_path(username, prev_date)
                    prev_raw = read_csv_safe(prev_path)
                    # print(prev_raw.head())
                    if prev_raw is not None and not prev_raw.empty:
                        prev_raw = filter_expired_positions(prev_raw, PREV_SYM_COL, now_dt, CUT)
                        prev_raw["PrevSymTradebookLike"] = (
                            prev_raw[PREV_SYM_COL]
                            .astype(str)
                            .map(lambda s: to_symbol_tradebook_spaced(parse_symbol(s)))
                        )
                        prev_raw = prev_raw.dropna(subset=["PrevSymTradebookLike"])
                    prev_cache[prev_key] = prev_raw

                sym_prev_for_roll = "PrevSymTradebookLike" if (prev_raw is not None and "PrevSymTradebookLike" in prev_raw.columns) else TB_SYMBOL_COL

                rolled, realized_all, mtm_components = roll_positions_weighted_average(
                    prev_df=prev_raw,
                    today_df=tb_raw,
                    sym_col=TB_SYMBOL_COL,
                    sym_col_prev=sym_prev_for_roll,
                    qty_col_today=qty_col,
                    avg_col_today=TB_AVG_COL,
                    prev_qty_col=PREV_QTY_COL,
                    prev_avg_col=PREV_AVG_COL,
                )

                rolled_ext = rolled.merge(mtm_components, on=TB_SYMBOL_COL, how="left")

                tb_for_engine = rolled_ext.rename(columns={"NetQty": qty_col, "CarryAvg": TB_AVG_COL})
                for c in ["Q_overnight", "Q_today", "PrevClose", "TodayBuyAvg"]:
                    if c not in tb_for_engine.columns:
                        tb_for_engine[c] = 0.0 if c in ["Q_overnight", "Q_today"] else np.nan

                tb_for_engine = tb_for_engine[[TB_SYMBOL_COL, qty_col, TB_AVG_COL, "Q_overnight", "Q_today", "PrevClose", "TodayBuyAvg", "ul", "RealizedPnL"]]

                tbm, pf_greeks, _scenarios, parity_dbg = run_engine_fast_from_ctx2(
                    tb_df=tb_for_engine,
                    ltp_ctx=ltp_ctx,
                    tb_symbol_col=TB_SYMBOL_COL,
                    qty_col=qty_col,
                    avg_col=TB_AVG_COL,
                    rf=RF,
                    q_div=QDIV,
                    spot_shocks=SPOT_SHOCKS,
                    vol_shocks=VOL_SHOCKS,
                    time_shocks=TIME_SHOCKS,
                    now_dt=now_dt,
                    cutoff=CUT,
                    spot_mode=SPOT_MODE,
                )
                opt_total = (tbm["kind"].astype(str).str.upper() == "OPT").sum()
                nan_iv_df = (tbm["kind"].astype(str).str.upper() == "OPT") & tbm["iv"].isna()
                opt_iv_nan = nan_iv_df.sum()
                log.info("UL=%s opt_total=%d opt_iv_nan=%d", username, int(opt_total), int(opt_iv_nan))
                print(tbm[nan_iv_df] if len(tbm[nan_iv_df])!=0 else None)
                # if opt_iv_nan!=0:
                #     try:
                #         tbm.to_csv(fr"Y:\Risk_dashboard_inputs\{username}_tbm.csv")
                #     except Exception as e:
                #         print("error saving file")
                # log.info(nan)

                # 4) PnL (carry + day + expenses)
                day_pnl_sym = compute_day_pnl_from_trades(tb_raw, ltp_ctx["ltp_raw"], TB_SYMBOL_COL, qty_col, TB_AVG_COL)
                carry_sym = pd.DataFrame(columns=["SymbolPrev", "ul", "expiry", "CarryPnL"])
                if prev_raw is not None and not prev_raw.empty:
                    carry_sym = compute_carry_pnl_from_prev(
                        prev_raw, ltp_ctx["ltp_raw"],
                        PREV_SYM_COL, PREV_QTY_COL, PREV_AVG_COL,
                        tradebook_like_col="PrevSymTradebookLike"
                    )

                exp_by_ul_exp = compute_trading_expenses(tb_raw, TB_SYMBOL_COL, qty_col, TB_AVG_COL, cost_per_cr=10000.0)

                day_ul_exp = day_pnl_sym.groupby(["ul", "expiry"], as_index=False)["DayPnL"].sum(min_count=1) if not day_pnl_sym.empty else pd.DataFrame(columns=["ul", "expiry", "DayPnL"])
                carry_ul_exp = carry_sym.groupby(["ul", "expiry"], as_index=False)["CarryPnL"].sum(min_count=1) if not carry_sym.empty else pd.DataFrame(columns=["ul", "expiry", "CarryPnL"])

                ul_exp_pnl = carry_ul_exp.merge(day_ul_exp, on=["ul", "expiry"], how="outer")
                ul_exp_pnl = ul_exp_pnl.merge(exp_by_ul_exp[["ul", "expiry", "PremiumTraded", "Expenses"]], on=["ul", "expiry"], how="outer")
                for c in ["CarryPnL", "DayPnL", "PremiumTraded", "Expenses"]:
                    if c in ul_exp_pnl.columns:
                        ul_exp_pnl[c] = ul_exp_pnl[c].fillna(0.0)

                ul_exp_pnl["TotalPnL"] = ul_exp_pnl["CarryPnL"] + ul_exp_pnl["DayPnL"]
                ul_exp_pnl["NetPnL"] = ul_exp_pnl["TotalPnL"] - ul_exp_pnl["Expenses"]
                ul_exp_pnl = ul_exp_pnl.sort_values("NetPnL", ascending=True)

                ul_pnl = (
                    ul_exp_pnl.groupby("ul", as_index=False)[["CarryPnL", "DayPnL", "PremiumTraded", "Expenses", "TotalPnL", "NetPnL"]]
                            .sum()
                            .sort_values("NetPnL", ascending=True)
                )

                total_carry = float(ul_exp_pnl["CarryPnL"].sum()) if not ul_exp_pnl.empty else 0.0
                total_day   = float(ul_exp_pnl["DayPnL"].sum()) if not ul_exp_pnl.empty else 0.0
                total_exp   = float(ul_exp_pnl["Expenses"].sum()) if not ul_exp_pnl.empty else 0.0
                total_net   = (total_carry + total_day) - total_exp

                pnl_mm = update_pnl_minmax_hash(r, username, now_dt, total_carry, total_day, total_net)

                # 5) Margin publish + read-back outputs
                margin_publish = {"enabled": PUBLISH_MARGIN, "payload_id": "", "changed": False}
                if PUBLISH_MARGIN:
                    try:
                        margin_payload = build_margin_inputs_from_tbm(tbm, ltp_ctx["ltp_raw"], qty_col=qty_col)
                        pid, changed = publish_margin_inputs_if_changed(r, username, margin_payload, now_dt)
                        margin_publish = {"enabled": True, "payload_id": pid, "changed": changed}
                    except Exception:
                        margin_publish = {"enabled": True, "payload_id": "", "changed": False, "error": "publish_failed"}

                margin_out = {"has_output": False, "computed_at": "", "span": 0.0, "exposure": 0.0, "total": 0.0, "minmax": {}}
                try:
                    margin_out = read_margin_outputs(r, username)
                    ts_store.append(
                        username,
                        now_dt,
                        pnl=total_net,
                        margin_total=margin_out["total"],
                        span=margin_out["span"],
                        exposure=margin_out["exposure"],
                        nifty_s=spot_nifty,
                        bn_s=spot_banknifty,
                        sx_s=spot_sensex
                    )
                except Exception:
                    print("ERROR WHILE storing pnl and margin in parquet")

                # 6) Top legs
                top = {}
                if tbm is not None and not tbm.empty:
                    tmp = tbm.copy()
                    tmp["abs_pos_delta"] = tmp["pos_delta"].abs()
                    tmp["abs_pos_vega"]  = tmp["pos_vega"].abs()
                    tmp["abs_pos_gamma"] = tmp["pos_gamma"].abs()

                    keep_cols = [
                        TB_SYMBOL_COL, "ul", "kind", "opt", "strike", "expiry", "S_used", "T", "iv",
                        qty_col, "LTP", "pos_delta", "pos_gamma", "pos_vega", "pos_theta", "MTM"
                    ]
                    keep_cols = [c for c in keep_cols if c in tmp.columns]

                    top["by_abs_delta"] = df_to_records(tmp.sort_values("abs_pos_delta", ascending=False)[keep_cols], TOP_LEGS_N)
                    top["by_abs_vega"]  = df_to_records(tmp.sort_values("abs_pos_vega",  ascending=False)[keep_cols], TOP_LEGS_N)
                    top["by_abs_gamma"] = df_to_records(tmp.sort_values("abs_pos_gamma", ascending=False)[keep_cols], TOP_LEGS_N)

                # 7) Publish snapshot
                pf_disp = pf_greeks.reset_index() if pf_greeks is not None and not pf_greeks.empty else pd.DataFrame()
                if not pf_disp.empty:
                    pf_disp["expiry"] = pf_disp["expiry"].astype(str).replace({"None": "", "NaT": ""})

                ul_exp_disp = ul_exp_pnl.copy()
                if "expiry" in ul_exp_disp.columns:
                    ul_exp_disp["expiry"] = ul_exp_disp["expiry"].astype(str).replace({"None": "", "NaT": ""})

                snapshot = {
                    "username": username,
                    "as_of": now_dt.isoformat(timespec="seconds"),
                    "config": {
                        "rf": RF, "q_div": QDIV, "cutoff": f"{CUT.hour:02d}:{CUT.minute:02d}",
                        "spot_mode": SPOT_MODE,
                    },
                    "kpis": {
                        "carry_pnl": total_carry,
                        "day_pnl": total_day,
                        "expenses": total_exp,
                        "net_pnl": total_net,
                        "legs_open": int(len(tbm)) if tbm is not None else 0,
                        "ltp_count": int(len(ltp_ctx["ltp_raw"])) if ltp_ctx.get("ltp_raw") is not None else 0,
                    },
                    "pnl_minmax": pnl_mm,
                    "margin_publish": margin_publish,
                    "margin": margin_out,
                    "tables": {
                        "pf_greeks": df_to_records(pf_disp),
                        "ul_exp_pnl": df_to_records(ul_exp_disp),
                        "ul_pnl": df_to_records(ul_pnl),
                    },
                    "top_legs": top,
                    "version": "fast_v1",
                }

                snapshot["payoff_pack"] = compute_payoff_pack(
                    tbm=tbm,
                    qty_col=qty_col,
                    avg_col=TB_AVG_COL,
                    spot_col="S_used",
                    expiry_grid_pct_wide=0.25,
                    expiry_n_points=301,
                    max_groups=12
                )

                # existing scenarios from engine
                scenarios_json = _scenarios_to_jsonable(_scenarios)

                # NEW: true combined cube
                if PUBLISH_COMBO_SCENARIOS and (tbm is not None) and (not tbm.empty):
                    try:
                        combo = compute_combo_cube_from_tbm(
                            tbm=tbm,
                            qty_col=qty_col,
                            rf=RF,
                            q_div=QDIV,
                            spot_shocks_frac=SPOT_SHOCKS_COMBO,
                            vol_pts=VOL_PTS_COMBO,
                            time_days=TIME_DAYS_COMBO,
                        )
                        # tbm.to_csv(fr"Y:\Risk_dashboard_inputs\strategy_ts\tbm_{username}.csv")
                        # with open(fr"Y:\Risk_dashboard_inputs\strategy_ts\grid_{username}.json",'w') as f:
                        #     json.dump(combo,f)
                        # attach per UL
                        for ul, cube in (combo or {}).items():
                            scenarios_json.setdefault(ul, {})
                            scenarios_json[ul]["combo"] = cube
                    except Exception as e:
                        log.error("combo scenarios failed user=%s: %s", username, e)

                snapshot["scenarios"] = scenarios_json

                payload = json.dumps(snapshot, default=json_default, ensure_ascii=False)
                r.set(f"risk:outputs:latest:{username}", payload)

                # print(snapshot.get("scenarios"))

                # ✅ NEW: store per-user scenarios locally (JSONL), deduped
                # append_scenarios_local(username, now_dt, snapshot.get("scenarios") or {})
  

                acc.add_user(
                    username=username,
                    snap=snapshot,
                    pf=pf_greeks,
                    ul_exp_pnl_df=ul_exp_disp,
                    ul_pnl_df=ul_pnl,
                    scenarios_json=snapshot.get("scenarios") or {},
                )

            except Exception as e:
                log.error("user=%s failed: %s\n%s", username, e, traceback.format_exc())
                continue

        ##################################################################
        # combined user
        ##################################################################
        if acc.users:
            pf_all = finalize_pf_with_total(acc.pf_sum)

            pf_disp_all = pf_all.reset_index() if pf_all is not None and not pf_all.empty else pd.DataFrame()
            if not pf_disp_all.empty and "expiry" in pf_disp_all.columns:
                pf_disp_all["expiry"] = pf_disp_all["expiry"].astype(str).replace({"None": "", "NaT": ""})

            ul_exp_all = acc.ul_exp_sum.reset_index() if acc.ul_exp_sum is not None and not acc.ul_exp_sum.empty else pd.DataFrame()
            if not ul_exp_all.empty and "expiry" in ul_exp_all.columns:
                ul_exp_all["expiry"] = ul_exp_all["expiry"].astype(str).replace({"None": "", "NaT": ""})

            ul_all = acc.ul_sum.reset_index() if acc.ul_sum is not None and not acc.ul_sum.empty else pd.DataFrame()

            total_carry_all = float(acc.kpis_sum.get("carry_pnl", 0.0))
            total_day_all   = float(acc.kpis_sum.get("day_pnl",   0.0))
            total_exp_all   = float(acc.kpis_sum.get("expenses",  0.0))
            total_net_all   = float(acc.kpis_sum.get("net_pnl",   0.0))

            pnl_mm_all = update_pnl_minmax_hash(r, COMBINED_USER, now_dt, total_carry_all, total_day_all, total_net_all)

            snapshot_all = {
                "username": COMBINED_USER,
                "as_of": now_dt.isoformat(timespec="seconds"),
                "config": {
                    "rf": RF, "q_div": QDIV, "cutoff": f"{CUT.hour:02d}:{CUT.minute:02d}",
                    "spot_mode": SPOT_MODE,
                    "note": "Combined = SUM of all accounts (not netted).",
                },
                "kpis": {
                    "carry_pnl": total_carry_all,
                    "day_pnl": total_day_all,
                    "expenses": total_exp_all,
                    "net_pnl": total_net_all,
                    "legs_open": int(acc.kpis_sum.get("legs_open", 0.0)),
                    "ltp_count": int(len(ltp_ctx["ltp_raw"])) if ltp_ctx.get("ltp_raw") is not None else 0,
                },
                "pnl_minmax": pnl_mm_all,
                "margin_publish": {"enabled": False, "payload_id": "", "changed": False},
                "margin": {
                    "has_output": True if acc.users else False,
                    "computed_at": "",
                    "span": float(acc.margin_sum.get("span", 0.0)),
                    "exposure": float(acc.margin_sum.get("exposure", 0.0)),
                    "total": float(acc.margin_sum.get("total", 0.0)),
                    "minmax": {},
                },
                "tables": {
                    "pf_greeks": df_to_records(pf_disp_all),
                    "ul_exp_pnl": df_to_records(ul_exp_all),
                    "ul_pnl": df_to_records(ul_all),
                },
                "top_legs": {},
                "payoff_pack": {},
                "scenarios": acc.scenarios_sum or {},
                "users_included": acc.users,
                "version": "fast_v1_combined",
            }

            payload_all = json.dumps(snapshot_all, default=json_default, ensure_ascii=False)
            r.set(f"risk:outputs:latest:{COMBINED_USER}", payload_all)

            # try:
            #     ts_store.append(
            #         "__ALL__", now_dt,
            #         pnl=total_net_all,
            #         margin_total=snapshot_all["margin"]["total"],
            #         span=snapshot_all["margin"]["span"],
            #         exposure=snapshot_all["margin"]["exposure"],
            #         nifty_s=spot_nifty,
            #         bn_s=spot_banknifty,
            #         sx_s=spot_sensex
            #     )
            # except Exception as e:
            #     print(f"failed while updating combined : {e}")

        # if (_time.time() - last_parquet_flush) > 30:
        #     ts_store.flush_all()
        #     last_parquet_flush = _time.time()

        elapsed = _time.time() - tick_start
        sleep_s = max(0.05, LOOP_SECONDS - elapsed)
        print("#" * 10, sleep_s, "#" * 10)
        _time.sleep(sleep_s)

if __name__ == "__main__":
    main()
