from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124 Safari/537.36"
)
ARCHIVE_BASE = "https://nsearchives.nseindia.com/archives/nsccl/span/"

# Your ZIP patterns; keep what you already use
def candidate_zip_names(ymd: str) -> List[str]:
    return [
        f"nsccl.{ymd}.i1.zip",
        f"nsccl.{ymd}.i2.zip",
        f"nsccl.{ymd}.i3.zip",
        f"nsccl.{ymd}.i4.zip",
        f"nsccl.{ymd}.i5.zip",
        f"nsccl.{ymd}.i6.zip",
        # Older/alt patterns:
        f"nsccl_prism_fo_spn_{ymd}.zip",
        f"fo_prism_{ymd}.zip",
        f"prism_fo_{ymd}.zip",
        f"nsccl_fo_span_{ymd}.zip",
    ]

ZIP_I_RE = re.compile(r"\.i(\d+)\.zip$", re.IGNORECASE)

@dataclass(frozen=True)
class ZipHit:
    ymd: str
    url: str
    fname: str
    blob: bytes
    i_seq: int  # 0 if not inferable from name

def ist_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))

def default_ymd_candidates(preferred_yyyymmdd: Optional[str] = None) -> List[str]:
    if preferred_yyyymmdd:
        return [preferred_yyyymmdd]
    d0 = ist_now().date()
    return [d0.strftime("%Y%m%d"), (d0 - timedelta(days=1)).strftime("%Y%m%d")]

@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=8))
def _get(session: requests.Session, url: str) -> requests.Response:
    r = session.get(url, timeout=45, allow_redirects=True)
    r.raise_for_status()
    return r

def fetch_archives_for_date(ymd: str) -> List[ZipHit]:
    hits: List[ZipHit] = []
    with requests.Session() as s:
        s.headers.update({"User-Agent": UA, "Referer": "https://www.nseindia.com/"})
        # best-effort cookie priming
        try:
            s.get("https://www.nseindia.com/", timeout=10)
        except Exception:
            pass

        for fname in candidate_zip_names(ymd):
            url = ARCHIVE_BASE + fname
            try:
                resp = _get(s, url)
            except Exception:
                continue
            blob = resp.content
            # guard against HTML/403 bodies
            if not blob or len(blob) < 50_000:
                continue

            m = ZIP_I_RE.search(fname)
            i_seq = int(m.group(1)) if m else 0

            hits.append(ZipHit(ymd=ymd, url=url, fname=fname, blob=blob, i_seq=i_seq))

    return hits

def _extract_spn_from_zip(blob: bytes, out_dir: Path) -> List[Path]:
    """
    Extract SPN/XML/TXT files, but returns only extracted .spn paths (preferred).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for zi in zf.infolist():
            lname = zi.filename.lower()
            if not lname.endswith((".spn", ".xml", ".txt")):
                continue
            dst = out_dir / Path(zi.filename).name
            with zf.open(zi) as src, open(dst, "wb") as out:
                out.write(src.read())
            extracted.append(dst)

    return extracted

def _pick_best_spn(extracted: List[Path]) -> Optional[Path]:
    """
    Prefer .spn. If multiple .spn, pick the largest.
    If no .spn exists, return largest among extracted (fallback), but margin engine should warn.
    """
    if not extracted:
        return None
    spn_files = [p for p in extracted if p.suffix.lower() == ".spn"]
    if spn_files:
        return max(spn_files, key=lambda p: p.stat().st_size)
    return max(extracted, key=lambda p: p.stat().st_size)

def ensure_latest_span_file(
    *,
    out_dir: Path,
    date_yyyymmdd: Optional[str] = None,
) -> Tuple[str, Path]:
    """
    Returns (ymd_used, span_path) where span_path is the chosen extracted .spn (or fallback file).
    - Tries preferred date; else tries IST today then yesterday.
    - Downloads ZIPs, extracts, picks the latest i-seq (highest) then best .spn inside.
    - If file already extracted previously, it will reuse without re-downloading IF a matching .spn exists.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Decide candidate dates
    ymds = default_ymd_candidates(date_yyyymmdd)

    # 2) If we already have a .spn for today, reuse the newest local one (fast path)
    #    This prevents re-downloading on every run.
    def _local_best_for_ymd(ymd: str) -> Optional[Path]:
        cand = list(out_dir.glob(f"*{ymd}*.spn")) + list(out_dir.glob(f"nsccl.{ymd}.i*.spn"))
        if not cand:
            return None
        return max(cand, key=lambda p: p.stat().st_mtime_ns)

    for ymd in ymds:
        local = _local_best_for_ymd(ymd)
        if local and local.exists() and local.stat().st_size > 100_000:
            return ymd, local

        hits = fetch_archives_for_date(ymd)
        if not hits:
            continue

        # 3) Sort: highest i-seq first; if i-seq unknown, keep by blob size
        hits_sorted = sorted(
            hits,
            key=lambda h: (h.i_seq, len(h.blob)),
            reverse=True,
        )

        # 4) Try extracting in that order; first that yields a usable .spn wins
        for hit in hits_sorted:
            extracted = _extract_spn_from_zip(hit.blob, out_dir=out_dir)
            best = _pick_best_spn(extracted)
            if best is None:
                continue

            # sanity check: must look like SPAN XML-ish
            head = best.read_bytes()[:4096]
            if b"<spanFile" not in head:
                # Not ideal, but some packages differ; keep searching
                continue

            # Optionally rename to a stable name for caching/auditing
            # e.g. nsccl.<ymd>.iNN.spn
            stable_name = None
            if hit.i_seq > 0 and best.suffix.lower() == ".spn":
                stable_name = out_dir / f"nsccl.{ymd}.i{hit.i_seq:02d}.spn"
            if stable_name and stable_name.resolve() != best.resolve():
                try:
                    os.replace(best, stable_name)  # atomic replace on Windows too
                    best = stable_name
                except Exception:
                    pass

            return ymd, best

    raise FileNotFoundError(f"No valid SPAN found for dates: {ymds} (out_dir={out_dir})")
