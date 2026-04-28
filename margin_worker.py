# margin_worker.py
# NOTE: This worker auto-discovers users by scanning Redis keys:
#   margin:inputs:latest:*
# No user list needed here — it processes whoever risk_worker.py has published.
# Adding/removing users in users.yaml is sufficient.

from __future__ import annotations

import json
import time
import ast
from pathlib import Path
from datetime import datetime
import pandas as pd
import redis
import traceback
import span_provider as get_spn
from typing import List,Optional,Sequence,Dict

from margin_calculator_v3 import (
    load_span_cached,
    build_fut_price_map_from_redis,
    build_opt_ltp_map_from_redis,
    build_underlying_spot_from_redis,
    compute_portfolio_margin_with_exposure,
    margin_breakdown,
)

# -----------------------------
# small helpers
# -----------------------------
def _dt_now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _log(msg: str):
    print(f"[{_dt_now_iso()}] {msg}", flush=True)

def _log_map_shape(name: str, d: dict):
    if not d:
        _log(f"{name}: empty")
        return
    k = next(iter(d.keys()))
    _log(f"{name}: size={len(d)} key_type={type(k).__name__} sample_key={k}")

def _summarize_positions(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"rows": 0}
    return {
        "rows": int(len(df)),
        "underlyings": sorted(df["underlying"].dropna().unique().tolist()),
        "kinds": df["kind"].value_counts(dropna=False).to_dict(),
        "net_qty": float(df["qty"].sum()),
        "gross_qty": float(df["qty"].abs().sum()),
        "series_count": int(df[["underlying", "expiry"]].drop_duplicates().shape[0]),
        "opt_count": int((df["kind"] == "OPT").sum()),
        "fut_count": int((df["kind"] == "FUT").sum()),
    }

def _coverage_check_positions_vs_maps(
    positions: pd.DataFrame,
    underlying_price: dict,
    fut_px_map: dict,
    opt_ltp_map: dict,
):
    """
    Prints coverage stats:
      - Underlyings missing from underlying_price
      - FUT series missing from fut_px_map (if fut map is tuple-keyed)
      - OPT contracts missing from opt_ltp_map (tuple-keyed)
    """
    if positions is None or positions.empty:
        _log("Coverage check: positions empty.")
        return

    # --- Underlying spot coverage ---
    uls = positions["underlying"].dropna().unique().tolist()
    missing_ul = [u for u in uls if u not in (underlying_price or {})]
    _log(f"Underlying spot map size={len(underlying_price or {})}, positions uls={len(uls)}, missing={len(missing_ul)}")
    if missing_ul:
        _log(f"Missing underlyings in underlying_price (sample up to 10): {missing_ul[:10]}")

    # --- FUT coverage (tuple-keyed expected) ---
    fut = positions[positions["kind"] == "FUT"][["underlying", "expiry"]].drop_duplicates()
    if not fut.empty:
        fut_keys = [(r["underlying"], int(r["expiry"])) for _, r in fut.iterrows()]
        fut_missing = [k for k in fut_keys if k not in (fut_px_map or {})]
        _log(f"FUT price map size={len(fut_px_map or {})}, fut series needed={len(fut_keys)}, missing={len(fut_missing)}")
        if fut_missing:
            _log(f"Missing FUT keys (sample up to 10): {fut_missing[:10]}")

    # --- OPT coverage (tuple-keyed expected) ---
    opt = positions[positions["kind"] == "OPT"][["underlying", "expiry", "strike", "option_type"]].drop_duplicates()
    if not opt.empty:
        opt_keys = [(r["underlying"], int(r["expiry"]), float(r["strike"]), str(r["option_type"]).upper()) for _, r in opt.iterrows()]
        opt_missing = [k for k in opt_keys if k not in (opt_ltp_map or {})]
        _log(f"OPT ltp map size={len(opt_ltp_map or {})}, opt contracts needed={len(opt_keys)}, missing={len(opt_missing)}")
        if opt_missing:
            _log(f"Missing OPT keys (sample up to 10): {opt_missing[:10]}")


#==============================================================================================================
# def _parse_payload_opt_map(opt_map: dict) -> dict:
#     """
#     Your dashboard publishes opt_ltp_by_contract_key with keys stringified like:
#       "('NIFTY', 20251230, 13700.0, 'CE')": 12.3

#     Convert back to tuple keys.
#     """
#     out = {}
#     if not isinstance(opt_map, dict):
#         return out
#     for k, v in opt_map.items():
#         try:
#             t = ast.literal_eval(k) if isinstance(k, str) else k
#             # normalize tuple shape: (ul, exp, strike, opt_type)
#             ul, exp, strike, opt_type = t
#             out[(str(ul).upper(), int(exp), float(strike), str(opt_type).upper())] = float(v)
#         except Exception:
#             continue
#     return out

def _parse_payload_opt_map(opt_map: dict) -> dict:
    """
    Supports BOTH formats:
      1) Pipe string: "NIFTY|20260106|26200.0|CE" -> (NIFTY, 20260106, 26200.0, CE)
      2) Tuple-string: "('NIFTY', 20260106, 26200.0, 'CE')" -> same
    Returns tuple-keyed dict:
      (ul, exp, strike, opt_type) -> ltp
    """
    out = {}
    if not isinstance(opt_map, dict):
        return out

    for k, v in opt_map.items():
        try:
            if isinstance(k, str) and "|" in k:
                ul, exp, strike, opt_type = k.split("|")
            else:
                t = ast.literal_eval(k) if isinstance(k, str) else k
                ul, exp, strike, opt_type = t

            out[(str(ul).upper(), int(exp), float(strike), str(opt_type).upper())] = float(v)
        except Exception:
            continue

    return out

def _parse_payload_fut_map(fut_map: dict) -> dict:
    """
    Supports:
      1) Pipe string: "NIFTY|20260106"
      2) Tuple-string: "('NIFTY', 20260106)"
    Returns tuple-keyed dict:
      (ul, exp) -> fut_px
    """
    out = {}
    if not isinstance(fut_map, dict):
        return out

    for k, v in fut_map.items():
        try:
            if isinstance(k, str) and "|" in k:
                ul, exp = k.split("|")
            else:
                t = ast.literal_eval(k) if isinstance(k, str) else k
                ul, exp = t

            out[(str(ul).upper(), int(exp))] = float(v)
        except Exception:
            continue

    return out

def _positions_df_from_payload(payload: dict) -> pd.DataFrame:
    rows = (payload or {}).get("positions_units", []) or []
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # engine expects columns: underlying, kind, expiry, option_type, strike, qty
    df["underlying"] = df["underlying"].astype(str).str.upper()
    df["kind"] = df["kind"].astype(str).str.upper()
    df["expiry"] = pd.to_numeric(df["expiry"], errors="coerce").astype("Int64")
    df["strike"] = pd.to_numeric(df.get("strike"), errors="coerce")
    df["option_type"] = df.get("option_type")
    df["option_type"] = df["option_type"].astype(str).str.upper().replace({"NONE": None, "NAN": None})

    # qty in UNITS
    df["qty"] = pd.to_numeric(df.get("qty_units"), errors="coerce").fillna(0.0)

    # keep only valid
    df = df[df["kind"].isin(["FUT", "OPT"]) & df["expiry"].notna() & (df["qty"].abs() > 1e-12)].copy()
    df["expiry"] = df["expiry"].astype(int)
    return df

def load_lot_size_by_series_csv(path: str) -> dict:
    df = pd.read_csv(path)
    df["underlying"] = df["underlying"].astype(str).str.upper()
    df["expiry"] = pd.to_numeric(df["expiry"], errors="coerce").astype(int)
    df["lot_size"] = pd.to_numeric(df["lot_size"], errors="coerce").astype(int)
    return {(r["underlying"], int(r["expiry"])): int(r["lot_size"]) for _, r in df.iterrows()}

def publish_margin_output(r: redis.Redis, username: str, wrapper: dict, prefix: str = "margin"):
    key = f"{prefix}:outputs:latest:{username}"
    r.set(key, json.dumps(wrapper))

# -----------------------------
# Peak margin (overall) tracking (atomic, Redis HASH + Lua)
# -----------------------------
# PEAK_LUA = r"""
#     local key = KEYS[1]
#     local new_total = tonumber(ARGV[1])
#     local peak_at = ARGV[2]
#     local inputs_as_of = ARGV[3]
#     local span_file = ARGV[4]
#     local span_date = ARGV[5]
#     local username = ARGV[6]
#     local ttl_sec = tonumber(ARGV[7])

#     local old_total_str = redis.call("HGET", key, "peak_total")
#     local old_total = tonumber(old_total_str)
#     if old_total == nil then old_total = -1e100 end

#     if new_total > old_total then
#     redis.call("HSET", key,
#         "peak_total", tostring(new_total),
#         "peak_at", peak_at,
#         "inputs_as_of", inputs_as_of,
#         "span_file", span_file,
#         "span_date", span_date,
#         "username", username
#     )
#     end

#     if ttl_sec ~= nil and ttl_sec > 0 then
#     redis.call("EXPIRE", key, ttl_sec)
#     end

#     return 1
#     """

# def update_peak_margin_overall(
#     r: redis.Redis,
#     *,
#     prefix: str,
#     username: str,
#     span_date: str,
#     span_file: str,
#     inputs_as_of: str,
#     computed_at: str,
#     total_margin: float,
#     ttl_days: int = 60,
#     also_alltime: bool = True,
# ):
#     """
#     Tracks peak of grand_total_broker_style (overall margin).
#     Daily key:  {prefix}:outputs:peak:{username}:{span_date}
#     All-time:   {prefix}:outputs:peak:{username}:alltime
#     Stored as Redis HASH. Update is atomic.
#     """
#     ttl_sec = int(ttl_days * 86400) if ttl_days and ttl_days > 0 else 0

#     # daily peak
#     key_daily = f"{prefix}:outputs:peak:{username}:{span_date}"
#     r.eval(
#         PEAK_LUA, 1, key_daily,
#         float(total_margin), computed_at, inputs_as_of or "", span_file or "", span_date or "", username or "",
#         ttl_sec
#     )

#     if also_alltime:
#         key_all = f"{prefix}:outputs:peak:{username}:alltime"
#         r.eval(
#             PEAK_LUA, 1, key_all,
#             float(total_margin), computed_at, inputs_as_of or "", span_file or "", span_date or "", username or "",
#             0
#         )
# -----------------------------
# Min/Max margin (overall) tracking (atomic, Redis HASH + Lua)
# -----------------------------
MINMAX_LUA = r"""
    local key = KEYS[1]

    local new_total   = tonumber(ARGV[1])
    local computed_at = ARGV[2]
    local inputs_as_of= ARGV[3]
    local span_file   = ARGV[4]
    local span_date   = ARGV[5]
    local username    = ARGV[6]
    local ttl_sec     = tonumber(ARGV[7])
    local skip_min    = tonumber(ARGV[8])  -- 1 => do not update min (e.g. positions empty)

    -- current stored values
    local max_str = redis.call("HGET", key, "max_total")
    local min_str = redis.call("HGET", key, "min_total")

    local max_val = tonumber(max_str)
    local min_val = tonumber(min_str)

    -- init if missing
    if max_val == nil then
    redis.call("HSET", key, "max_total", tostring(new_total), "max_at", computed_at)
    max_val = new_total
    elseif new_total > max_val then
    redis.call("HSET", key, "max_total", tostring(new_total), "max_at", computed_at)
    max_val = new_total
    end

    if skip_min == nil then skip_min = 0 end

    if skip_min == 0 then
    if min_val == nil then
        redis.call("HSET", key, "min_total", tostring(new_total), "min_at", computed_at)
        min_val = new_total
    elseif new_total < min_val then
        redis.call("HSET", key, "min_total", tostring(new_total), "min_at", computed_at)
        min_val = new_total
    end
    end

    -- always store metadata (last seen)
    redis.call("HSET", key,
    "username", username,
    "span_date", span_date,
    "span_file", span_file,
    "inputs_as_of", inputs_as_of,
    "last_total", tostring(new_total),
    "last_at", computed_at
    )

    if ttl_sec ~= nil and ttl_sec > 0 then
    redis.call("EXPIRE", key, ttl_sec)
    end

    return {tostring(max_val or ""), tostring(min_val or "")}
    """

MINMAX_LUA2 = r"""
local key = KEYS[1]

local new_total   = tonumber(ARGV[1])
local computed_at = ARGV[2]
local inputs_as_of= ARGV[3]
local span_file   = ARGV[4]
local span_date   = ARGV[5]
local username    = ARGV[6]
local ttl_sec     = tonumber(ARGV[7])
local skip_min    = tonumber(ARGV[8])  -- 1 => do not update min/avg (e.g. positions empty)

-- existing values
local max_val = tonumber(redis.call("HGET", key, "max_total"))
local min_val = tonumber(redis.call("HGET", key, "min_total"))
local sum_val = tonumber(redis.call("HGET", key, "sum_total"))
local cnt_val = tonumber(redis.call("HGET", key, "count"))

if sum_val == nil then sum_val = 0 end
if cnt_val == nil then cnt_val = 0 end
if skip_min == nil then skip_min = 0 end

-- MAX
if max_val == nil then
  redis.call("HSET", key, "max_total", tostring(new_total), "max_at", computed_at)
  max_val = new_total
elseif new_total > max_val then
  redis.call("HSET", key, "max_total", tostring(new_total), "max_at", computed_at)
  max_val = new_total
end

-- MIN (skip when empty)
if skip_min == 0 then
  if min_val == nil then
    redis.call("HSET", key, "min_total", tostring(new_total), "min_at", computed_at)
    min_val = new_total
  elseif new_total < min_val then
    redis.call("HSET", key, "min_total", tostring(new_total), "min_at", computed_at)
    min_val = new_total
  end

  -- AVG (same skip rule as MIN to avoid 0.0 pollution)
  sum_val = sum_val + new_total
  cnt_val = cnt_val + 1
  local avg_val = sum_val / cnt_val

  redis.call("HSET", key,
    "sum_total", tostring(sum_val),
    "count", tostring(cnt_val),
    "avg_total", tostring(avg_val),
    "avg_at", computed_at
  )
end

-- metadata / last seen
redis.call("HSET", key,
  "username", username,
  "span_date", span_date,
  "span_file", span_file,
  "inputs_as_of", inputs_as_of,
  "last_total", tostring(new_total),
  "last_at", computed_at
)

if ttl_sec ~= nil and ttl_sec > 0 then
  redis.call("EXPIRE", key, ttl_sec)
end

return {tostring(max_val or ""), tostring(min_val or "")}
"""

def update_minmax_margin_overall(
    r: redis.Redis,
    *,
    prefix: str,
    username: str,
    span_date: str,
    span_file: str,
    inputs_as_of: str,
    computed_at: str,
    total_margin: float,
    positions_empty: bool,
    ttl_days: int = 60,
    also_alltime: bool = True,
):
    ttl_sec = int(ttl_days * 86400) if ttl_days and ttl_days > 0 else 0
    skip_min = 1 if positions_empty else 0

    key_daily = f"{prefix}:outputs:minmax:{username}:{span_date}"
    r.eval(
        MINMAX_LUA2, 1, key_daily,
        float(total_margin), computed_at, inputs_as_of or "", span_file or "", span_date or "", username or "",
        ttl_sec, skip_min
    )

    if also_alltime:
        key_all = f"{prefix}:outputs:minmax:{username}:alltime"
        r.eval(
            MINMAX_LUA2, 1, key_all,
            float(total_margin), computed_at, inputs_as_of or "", span_file or "", span_date or "", username or "",
            0, skip_min
        )

    return key_daily


def load_instrument_master(
        filename: Optional[str] = None, exchanges: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load instrument master from zerodha for all segments
        filename
            load master from a file
        exchanges
            list of exchanges to load
        Note
        -----
        1) A column named exchange is assumed to be in the dataframe
        """
        url = "https://api.kite.trade/instruments"
        if filename is None:
            filename = url
        inst = pd.read_csv(filename, parse_dates=["expiry"])
        if exchanges:
            return inst[inst.exchange.isin(exchanges)].reset_index(drop=True)
        else:
            return inst

def read_redis_prices_zerodha(
    r: redis.Redis,
    *,
    hash_key: str = "last_price",
    drop_nonpositive: bool = True,
) -> Dict[str, float]:
    """
    Reads Zerodha LTP hash from Redis and returns {tradingsymbol: ltp}.

    Assumes redis client is created with decode_responses=True so keys/vals are str.
    If decode_responses=False, it still works (decodes bytes).

    drop_nonpositive=True: ignores 0 / negative ticks.
    """
    raw = r.hgetall(hash_key) or {}
    out: Dict[str, float] = {}

    for k, v in raw.items():
        sym = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
        try:
            px = float(v.decode() if isinstance(v, (bytes, bytearray)) else v)
        except Exception:
            continue

        if drop_nonpositive and px <= 0:
            continue

        out[sym] = px

    return out

def main():
    # ---- CONFIG ----
    REDIS_HOST, REDIS_PORT, REDIS_DB = "localhost", 6379, 0
    INPUT_PREFIX = "margin"
    PRICE_HASH_KEY = "last_price"        # <-- your Zerodha feed hash
    SPAN_DIR = Path(r"spanfiles")
    LOT_SIZE_CSV = r"/mnt/Quant_Research/Risk_dashboard_inputs/required_datas/lot_size_per_expiry.csv"  # <-- put correct path

    # index set + exposure rules
    IDX_UL = {"NIFTY", "BANKNIFTY", "SENSEX"}
    EXPOSURE_PCT = {"NIFTY": 0.02, "BANKNIFTY": 0.02, "SENSEX": 0.02}
    PREV_CLOSE_BY_UL = {"NIFTY": 25571.70, "BANKNIFTY": 60172.40, "SENSEX": 82814.71}

    # poll interval
    POLL_SECS = 15.0

    # ---- init ----
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

    # load master once (can refresh daily if you want)
    df_master = load_instrument_master()

    # lot size mapping
    lot_size_by_series = load_lot_size_by_series_csv(LOT_SIZE_CSV)

    last_seen_pid_by_user = {}  # avoid recompute when same payload

    print("Margin worker started.")

    while True:
        try:
            # Process ALL users who have a latest input key
            for key in r.scan_iter(match=f"{INPUT_PREFIX}:inputs:latest:*"):
                raw = r.get(key)
                if not raw:
                    continue

                wrapper_in = json.loads(raw)
                username = wrapper_in.get("username", "unknown")
                as_of = wrapper_in.get("as_of")
                payload_id = wrapper_in.get("payload_id")

                #if unchanged continue
                if payload_id and last_seen_pid_by_user.get(username) == payload_id:
                    continue
                # if payload_id:
                #     last_seen_pid_by_user = payload_id

                payload = wrapper_in.get("payload") or {}

                _log(f"IN key={key} user={username} as_of={as_of} payload_id={payload_id}")
                _log(f"Payload keys={list(payload.keys())}")
                _log(f"Payload sizes: positions_units={len((payload.get('positions_units') or []))} "
                    f"underlying_price={len(payload.get('underlying_price') or {})} "
                    f"fut_price_by_underlying_expiry={len(payload.get('fut_price_by_underlying_expiry') or {})} "
                    f"opt_ltp_by_contract_key={len(payload.get('opt_ltp_by_contract_key') or {})}")

                # skip if unchanged
                # if last_seen_asof_by_user.get(username) == as_of:
                #     continue

                # --- build positions df from payload (UNITS) ---
                positions = _positions_df_from_payload(payload)
                _log(f"Positions summary: {_summarize_positions(positions)}")

                if not positions.empty:
                    series = positions[["underlying", "expiry"]].drop_duplicates()
                    missing_ls = []
                    for _, r_ in series.iterrows():
                        key_ls = (str(r_["underlying"]).upper(), int(r_["expiry"]))
                        if key_ls not in lot_size_by_series:
                            missing_ls.append(key_ls)

                    _log(f"Lot-size coverage: needed={len(series)} missing={len(missing_ls)}")
                    if missing_ls:
                        _log(f"Missing lot sizes (sample up to 10): {missing_ls[:10]}")

                # if no positions, still publish zeros (nice UX)
                # if positions.empty:
                #     _log(f"Positions head (top 8):")
                #     _log(positions.head(8).to_string(index=False))
                #     out = {
                #         "username": username,
                #         "inputs_as_of": as_of,
                #         "computed_at": _dt_now_iso(),
                #         "result": {"totals": {"grand_total_broker_style": 0.0, "funds_required_cash": 0.0, "is_units": True}},
                #     }
                    
                #     publish_margin_output(r, username, wrapper_out, prefix=INPUT_PREFIX)

                #     totals_dict = wrapper_out.get("result") or {}
                #     grand_total = float(totals_dict.get("grand_total_broker_style") or 0.0)

                #     minmax_key = update_minmax_margin_overall(
                #         r,
                #         prefix=INPUT_PREFIX,
                #         username=username,
                #         span_date=str(wrapper_out.get("span_date") or ymd_used),
                #         span_file=str(wrapper_out.get("span_file") or span_path.name),
                #         inputs_as_of=str(wrapper_out.get("inputs_as_of") or as_of or ""),
                #         computed_at=str(wrapper_out.get("computed_at") or _dt_now_iso()),
                #         total_margin=grand_total,
                #         positions_empty=positions.empty,   # IMPORTANT: prevents min=0 on empty book
                #         ttl_days=60,
                #         also_alltime=False,
                #     )
                #     print(minmax_key)

                #     if payload_id:
                #         last_seen_pid_by_user[username] = payload_id
                #     continue
                if positions.empty:
                    computed_at = _dt_now_iso()
                    today = datetime.now().date()
                    ymd_used = today.strftime("%Y%m%d")

                    wrapper_out = {
                        "username": username,
                        "inputs_as_of": as_of,
                        "computed_at": computed_at,
                        "span_file": "",
                        "span_date": ymd_used,
                        "result": {
                            "grand_total_broker_style": 0.0,
                            "funds_required_cash": 0.0,
                            "is_units": True
                        },
                    }

                    publish_margin_output(r, username, wrapper_out, prefix=INPUT_PREFIX)

                    grand_total = float(wrapper_out["result"].get("grand_total_broker_style") or 0.0)

                    minmax_key = update_minmax_margin_overall(
                        r,
                        prefix=INPUT_PREFIX,
                        username=username,
                        span_date=ymd_used,
                        span_file="",
                        inputs_as_of=str(as_of or ""),
                        computed_at=computed_at,
                        total_margin=grand_total,
                        positions_empty=True,     # prevents min=0 pollution
                        ttl_days=60,
                        also_alltime=False,
                    )
                    _log(f"Updated minmax key (empty-book): {minmax_key} hgetall={r.hgetall(minmax_key)}")

                    if payload_id:
                        last_seen_pid_by_user[username] = payload_id
                    continue

                # --- live prices from redis (zerodha) ---
                prices = read_redis_prices_zerodha(r=r, hash_key=PRICE_HASH_KEY)  # returns dict[str,float]

                fut_px_map = build_fut_price_map_from_redis(last_price=prices, master=df_master)
                opt_ltp_map_live = build_opt_ltp_map_from_redis(last_price=prices, master=df_master)
                spot_map_live = build_underlying_spot_from_redis(last_price=prices, fut_map=fut_px_map)

                # --- prefer payload maps if they exist, else fallback to live ---
                # underlying_price = payload.get("underlying_price") or spot_map_live
                # opt_ltp_payload = _parse_payload_opt_map(payload.get("opt_ltp_by_contract_key") or {})
                # opt_ltp_map = opt_ltp_payload or opt_ltp_map_live
                underlying_price_payload = payload.get("underlying_price") or {}
                underlying_price = underlying_price_payload if underlying_price_payload else spot_map_live

                _log(f"Underlying price source: {'payload' if underlying_price_payload else 'live'} size={len(underlying_price or {})}")

                opt_ltp_payload = _parse_payload_opt_map(payload.get("opt_ltp_by_contract_key") or {})
                opt_ltp_map = opt_ltp_payload if opt_ltp_payload else opt_ltp_map_live
                _log(f"OPT LTP source: {'payload' if opt_ltp_payload else 'live'} size={len(opt_ltp_map or {})}")

                # Optional: FUT payload parsing + preference
                fut_payload_parsed = _parse_payload_fut_map(payload.get("fut_price_by_underlying_expiry") or {})
                if fut_payload_parsed:
                    _log(f"FUT payload parsed OK size={len(fut_payload_parsed)} (live size={len(fut_px_map)})")
                else:
                    _log("FUT payload missing/unparsed; using live FUT map.")

                fut_px_used = fut_payload_parsed if fut_payload_parsed else fut_px_map
                _log(f"FUT LTP source: {'payload' if fut_payload_parsed else 'live'} size={len(fut_px_used or {})}")

                ###==========================================================================================
                _log_map_shape("underlying_price", underlying_price)
                _log_map_shape("fut_px_used", fut_px_used)
                _log_map_shape("opt_ltp_map", opt_ltp_map)
                #============================================================================================
                _coverage_check_positions_vs_maps(positions=positions,
                                                  underlying_price=underlying_price,
                                                  fut_px_map=fut_px_used,
                                                  opt_ltp_map=opt_ltp_map
                                                  )
                ###==========================================================================================
                # --- span file ---
                # Use today's date for span (you can also use as_of date if you want)
                from datetime import date
                today = date.today()
                ymd_used, span_path = get_spn.ensure_latest_span_file(out_dir=SPAN_DIR, date_yyyymmdd=today.strftime("%Y%m%d"))
                span = load_span_cached(span_path=span_path)

                # --- compute ---
                res = compute_portfolio_margin_with_exposure(
                    positions=positions,
                    span=span,
                    underlying_price=underlying_price,
                    lot_size_by_series=lot_size_by_series,
                    lot_size_by_underlying=None,
                    futures_ra_is_per_unit=True,
                    options_ra_is_per_unit=True,
                    exposure_pct_by_underlying=EXPOSURE_PCT,
                    index_underlyings=IDX_UL,
                    somc_pct_index=0.03,
                    asof_date=int(today.strftime("%Y%m%d")),
                    fut_price_by_underlying_expiry=fut_px_used,
                    opt_ltp_by_contract_key=opt_ltp_map,
                    # prev_close_by_underlying=PREV_CLOSE_BY_UL,
                    is_units=True,  # IMPORTANT: your engine now takes UNITS
                )

                # bd = margin_breakdown(res, include_tables=False)

                wrapper_out = {
                    "username": username,
                    "inputs_as_of": as_of,
                    "computed_at": _dt_now_iso(),
                    "span_file": span_path.name,
                    "span_date": ymd_used,
                    "result": res['totals'],          # full result (if you want)
                    # "breakdown": bd,        # compact, UI friendly
                }
                
                print(wrapper_out)

                publish_margin_output(r, username, wrapper_out, prefix=INPUT_PREFIX)
                grand_total = float((res.get("totals") or {}).get("grand_total_broker_style") or 0.0)

                minmax_key = update_minmax_margin_overall(
                    r,
                    prefix=INPUT_PREFIX,
                    username=username,
                    span_date=str(wrapper_out.get("span_date") or ""),
                    span_file=str(wrapper_out.get("span_file") or ""),
                    inputs_as_of=str(wrapper_out.get("inputs_as_of") or as_of or ""),
                    computed_at=str(wrapper_out.get("computed_at") or _dt_now_iso()),
                    total_margin=grand_total,
                    positions_empty=False,
                    ttl_days=60,
                    also_alltime=False,
                )

                _log(f"Updated minmax key: {minmax_key} hgetall={r.hgetall(minmax_key)}")

                if payload_id:
                    last_seen_pid_by_user[username] = payload_id

                print(f"Computed margin for {username} at {as_of} using {span_path.name}")

        except Exception as e:
            print("Worker loop error:", e)
            traceback.print_exc()

        time.sleep(POLL_SECS)


if __name__ == "__main__":
    main()
