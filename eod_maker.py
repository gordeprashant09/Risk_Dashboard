# make_prev_eod_tagwise.py
"""
Generate tag-wise EOD open positions from historical orders.sqlite (table: orders).

Outputs a CSV with columns:
  tag, instrument, net_open_qty, carry_avg, eod_ts

Where:
- instrument = canonical "TradebookLike" symbol (so it matches your LTP join style)
- net_open_qty = signed net quantity for that (tag, instrument) as of cutoff
- carry_avg = weighted avg entry price for the remaining open net qty (FIFO-like approximation via WAP)
  * If you need true FIFO cost, we can extend it, but WAP is usually sufficient for carry.

Usage:
  python make_prev_eod_tagwise.py ^
    --db "Y:\Risk_dashboard_inputs\mock_DMA09\db\orders.sqlite" ^
    --out "Y:\Risk_dashboard_inputs\DMA09\eod_files\net_positions_eod_tagwise_260217.csv" ^
    --cutoff "2026-02-17 15:30:00" ^
    --tz "Asia/Kolkata"

Notes:
- Only uses rows with status COMPLETE and filled_quantity > 0 (fallback to quantity).
- BUY = +qty, SELL = -qty
- average_price preferred; fallback to price.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

# -----------------------------
# Symbol canonicalizer
# -----------------------------
def canonical_tradebooklike(symbol: str) -> str:
    """
    Convert DB symbol formats into the same "tradebook-like" representation you join LTPs with.
    Your DB examples look like: "SENSEX 09OCT2025 CE 82300"
    Your broker tradebook-like often uses: "SENSEX 09OCT2025 82300 CE" (or similar spacing)

    We’ll normalize a few common patterns conservatively:
    - Collapse multiple spaces
    - If ends with "... CE 82300" or "... PE 82300" reorder to "... 82300 CE/PE"
    If it doesn't match, leave as normalized spaces.
    """
    if symbol is None:
        return ""
    s = " ".join(str(symbol).strip().split())
    parts = s.split(" ")

    # Look for pattern: <UL> <DDMMMYYYY> <CE/PE> <STRIKE>
    # e.g. SENSEX 09OCT2025 CE 82300
    if len(parts) >= 4 and parts[-2] in ("CE", "PE") and parts[-1].replace(".", "", 1).isdigit():
        # move strike before opt type
        opt = parts[-2]
        strike = parts[-1]
        core = parts[:-2]
        return " ".join(core + [strike, opt])

    return s


# -----------------------------
# WAP position tracker per (tag, instrument)
# -----------------------------
@dataclass
class Pos:
    qty: float = 0.0
    avg: float = 0.0  # weighted avg cost of remaining open qty

def apply_fill(pos: Pos, signed_qty: float, px: float) -> Pos:
    """
    Update net position + weighted avg cost.
    - For increasing position in same direction: update WAP
    - For reducing/closing: reduce qty, keep avg for remaining
    - For flip: remaining qty takes the trade price as new avg (approx)
    """
    if not np.isfinite(px):
        px = 0.0

    q0 = pos.qty
    a0 = pos.avg

    q1 = q0 + signed_qty

    # If starting from flat
    if abs(q0) < 1e-12:
        pos.qty = q1
        pos.avg = float(px) if abs(q1) > 1e-12 else 0.0
        return pos

    # Same direction add
    if (q0 > 0 and signed_qty > 0) or (q0 < 0 and signed_qty < 0):
        # WAP update on absolute quantities
        new_abs = abs(q0) + abs(signed_qty)
        if new_abs > 1e-12:
            pos.avg = (abs(q0) * a0 + abs(signed_qty) * px) / new_abs
        pos.qty = q1
        return pos

    # Opposite direction trade reduces or flips
    # If reduces but doesn’t flip:
    if (q0 > 0 and q1 >= 0) or (q0 < 0 and q1 <= 0):
        pos.qty = q1
        # keep avg of remaining
        if abs(pos.qty) < 1e-12:
            pos.avg = 0.0
        return pos

    # Flip: remaining takes px as new avg (approx)
    pos.qty = q1
    pos.avg = float(px)
    return pos


# -----------------------------
# Main
# -----------------------------
def read_orders(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM orders", con)
    finally:
        con.close()
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to orders.sqlite")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--cutoff", required=True, help="Cutoff datetime, e.g. '2026-02-17 15:30:00'")
    ap.add_argument("--tz", default="Asia/Kolkata", help="Timezone for cutoff parsing")
    ap.add_argument("--only_exchange", default="", help="Optional filter exchange, e.g. 'NSEFO' or 'BSEFO'")
    args = ap.parse_args()

    db_path = args.db
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")

    tz = ZoneInfo(args.tz)
    cutoff_naive = datetime.strptime(args.cutoff, "%Y-%m-%d %H:%M:%S")
    cutoff = cutoff_naive.replace(tzinfo=tz)

    df = read_orders(db_path)
    if df.empty:
        print("No rows in orders table.")
        pd.DataFrame(columns=["tag","instrument","net_open_qty","carry_avg","eod_ts"]).to_csv(args.out, index=False)
        return

    # Basic filters
    # status COMPLETE, filled_quantity>0 (fallback to quantity)
    status = df.get("status")
    if status is not None:
        df = df[df["status"].astype(str).str.upper().eq("COMPLETE")]

    if args.only_exchange and "exchange" in df.columns:
        df = df[df["exchange"].astype(str).str.upper().eq(args.only_exchange.upper())]

    # Parse timestamp
    if "timestamp" not in df.columns:
        raise ValueError("orders table must have 'timestamp' column")

    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # ensure tz-aware
    if df["ts"].dt.tz is None:
        # if timestamps are naive, assume tz
        df["ts"] = df["ts"].dt.tz_localize(tz)
    else:
        df["ts"] = df["ts"].dt.tz_convert(tz)

    df = df[df["ts"] <= cutoff].copy()
    if df.empty:
        print("No orders up to cutoff; output will be empty.")
        pd.DataFrame(columns=["tag","instrument","net_open_qty","carry_avg","eod_ts"]).to_csv(args.out, index=False)
        return

    # Normalize columns
    if "symbol" not in df.columns:
        raise ValueError("orders table must have 'symbol' column")

    # Tag column: prefer 'tag' else strategy_id
    if "tag" in df.columns:
        df["tag_norm"] = df["tag"].astype(str).fillna("__NO_TAG__")
    elif "strategy_id" in df.columns:
        df["tag_norm"] = df["strategy_id"].astype(str).fillna("__NO_TAG__")
    else:
        df["tag_norm"] = "__NO_TAG__"

    # Quantity: prefer filled_quantity, fallback quantity
    fq = df.get("filled_quantity")
    q0 = df.get("quantity")
    df["qty"] = pd.to_numeric(fq if fq is not None else q0, errors="coerce").fillna(0.0)

    # Side
    side = df.get("side")
    if side is None:
        raise ValueError("orders table must have 'side' column")
    side = df["side"].astype(str).str.upper()
    df["signed_qty"] = df["qty"].where(side.isin(["BUY","B"]), -df["qty"])

    # Price: average_price preferred, fallback price
    px = df.get("average_price")
    px2 = df.get("price")
    df["px"] = pd.to_numeric(px if px is not None else px2, errors="coerce")
    if df["px"].isna().all():
        df["px"] = 0.0

    # Canonical instrument
    df["instrument"] = df["symbol"].astype(str).map(canonical_tradebooklike)

    # Sort by time (important)
    df = df.sort_values("ts")

    # Track positions
    pos_map: dict[tuple[str, str], Pos] = {}

    for row in df.itertuples(index=False):
        tag = getattr(row, "tag_norm")
        inst = getattr(row, "instrument")
        sq = float(getattr(row, "signed_qty", 0.0))
        pxr = float(getattr(row, "px", 0.0))

        if not inst:
            continue

        key = (tag, inst)
        p = pos_map.get(key)
        if p is None:
            p = Pos()
            pos_map[key] = p
        apply_fill(p, sq, pxr)

    # Build output: only open positions (non-zero)
    out_rows = []
    for (tag, inst), p in pos_map.items():
        if abs(p.qty) < 1e-9:
            continue
        out_rows.append(
            {
                "tag": tag,
                "instrument": inst,
                "net_open_qty": float(p.qty),
                "carry_avg": float(p.avg),
                "eod_ts": cutoff.isoformat(),
            }
        )

    out = pd.DataFrame(out_rows).sort_values(["tag", "instrument"]) if out_rows else pd.DataFrame(
        columns=["tag","instrument","net_open_qty","carry_avg","eod_ts"]
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} open rows -> {args.out}")

if __name__ == "__main__":
    print("kasdghfuiwgefui")
    main()
