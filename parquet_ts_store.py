# parquet_ts_store.py
import os
import time
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class TSPoint:
    ts_utc: datetime
    pnl: float
    margin_total: float
    span: float
    exposure: float
    nifty_s : float
    bn_s : float
    sx_s : float


class DailyParquetStore:
    """
    Writes time-series locally as partitioned Parquet parts:

      base_dir/
        date=YYYYMMDD/
          user=Sunny/
            part-<epoch_ms>-<pid>-<seq>.parquet

    - buffer points per user
    - flush in batches
    - rotate per IST day
    - no rewriting of existing parquet (append by writing new files)
    """

    def __init__(
        self,
        base_dir: str,
        flush_every_points: int = 200,
        compression: str = "zstd",   # "zstd" or "snappy"
    ):
        self.base_dir = base_dir
        self.flush_every_points = int(flush_every_points)
        self.compression = compression
        self._day: Optional[str] = None
        self._buf: dict[str, list[TSPoint]] = {}
        self._last_ts_ms: dict[str, int] = {}
        self._seq: int = 0

        os.makedirs(self.base_dir, exist_ok=True)

    def _daykey_ist(self, now_dt: datetime) -> str:
        return now_dt.astimezone(IST).strftime("%Y%m%d")

    def _ensure_day(self, daykey: str):
        if self._day != daykey:
            # rotate: flush everything before switching day
            self.flush_all()
            self._day = daykey

    def _user_dir(self, daykey: str, username: str) -> str:
        # hive partitioning style
        return os.path.join(self.base_dir, f"date={daykey}", f"user={username}")

    def append(
        self,
        username: str,
        now_dt: datetime,
        pnl: float,
        margin_total: float,
        span: float = 0.0,
        exposure: float = 0.0,
        nifty_s: float =0.0,
        bn_s: float = 0.0,
        sx_s: float = 0.0
    ):
        # store timestamps in UTC in the parquet itself
        now_utc = now_dt.astimezone(timezone.utc)
        daykey = self._daykey_ist(now_dt)
        self._ensure_day(daykey)

        ts_ms = int(now_utc.timestamp() * 1000)
        last = self._last_ts_ms.get(username)
        if last is not None and ts_ms <= last:
            return  # dedup / non-increasing timestamps

        self._last_ts_ms[username] = ts_ms

        b = self._buf.setdefault(username, [])
        b.append(
            TSPoint(
                ts_utc=now_utc,
                pnl=float(pnl),
                margin_total=float(margin_total),
                span=float(span),
                exposure=float(exposure),
                nifty_s= float(nifty_s),
                bn_s = float(bn_s),
                sx_s = float(sx_s)
            )
        )

        if len(b) >= self.flush_every_points:
            self.flush_user(username)

    # def flush_user(self, username: str):
    #     if not self._day:
    #         return
    #     b = self._buf.get(username)
    #     if not b:
    #         return

    #     out_dir = self._user_dir(self._day, username)
    #     os.makedirs(out_dir, exist_ok=True)

    #     # Build arrow table
    #     ts_arr = pa.array([p.ts_utc for p in b], type=pa.timestamp("ms", tz="UTC"))
    #     pnl_arr = pa.array([p.pnl for p in b], type=pa.float64())
    #     margin_arr = pa.array([p.margin_total for p in b], type=pa.float64())
    #     span_arr = pa.array([p.span for p in b], type=pa.float64())
    #     exp_arr = pa.array([p.exposure for p in b], type=pa.float64())

    #     table = pa.Table.from_arrays(
    #         [ts_arr, pnl_arr, margin_arr, span_arr, exp_arr],
    #         names=["ts_utc", "pnl", "margin_total", "span", "exposure"],
    #     )

    #     epoch_ms = int(time.time() * 1000)
    #     pid = os.getpid()
    #     self._seq += 1
    #     fname = f"part-{epoch_ms}-{pid}-{self._seq:06d}.parquet"
    #     path = os.path.join(out_dir, fname)

    #     pq.write_table(
    #         table,
    #         path,
    #         compression=self.compression,
    #         use_dictionary=True,
    #         write_statistics=True,
    #     )

    #     b.clear()

    def flush_user(self, username: str):
        if not self._day:
            return
        b = self._buf.get(username)
        if not b:
            return

        out_dir = self._user_dir(self._day, username)
        os.makedirs(out_dir, exist_ok=True)

        # Build arrow table
        ts_arr = pa.array([p.ts_utc for p in b], type=pa.timestamp("ms", tz="UTC"))
        pnl_arr = pa.array([p.pnl for p in b], type=pa.float64())
        margin_arr = pa.array([p.margin_total for p in b], type=pa.float64())
        span_arr = pa.array([p.span for p in b], type=pa.float64())
        exp_arr = pa.array([p.exposure for p in b], type=pa.float64())

        # 🔥 ADD THESE
        nifty_arr = pa.array([p.nifty_s for p in b], type=pa.float64())
        bn_arr = pa.array([p.bn_s for p in b], type=pa.float64())
        sx_arr = pa.array([p.sx_s for p in b], type=pa.float64())

        table = pa.Table.from_arrays(
            [
                ts_arr,
                pnl_arr,
                margin_arr,
                span_arr,
                exp_arr,
                nifty_arr,
                bn_arr,
                sx_arr,
            ],
            names=[
                "ts_utc",
                "pnl",
                "margin_total",
                "span",
                "exposure",
                "nifty_s",
                "bn_s",
                "sx_s",
            ],
        )

        epoch_ms = int(time.time() * 1000)
        pid = os.getpid()
        self._seq += 1
        fname = f"part-{epoch_ms}-{pid}-{self._seq:06d}.parquet"
        path = os.path.join(out_dir, fname)

        pq.write_table(
            table,
            path,
            compression=self.compression,
            use_dictionary=True,
            write_statistics=True,
        )

        b.clear()

    def flush_all(self):
        for u in list(self._buf.keys()):
            self.flush_user(u)
