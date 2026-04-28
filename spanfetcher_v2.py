# nsccl_prism_fetcher.py
# Requirements: requests>=2.32, tenacity>=8.2
#
# What this version fixes:
# - Idempotent downloads: won't re-download revisions already present in out_dir.
# - Incremental revisions: tries i1..i6 then s, downloads only missing ones, stops at first not-yet-published rev.
# - Supports final "s" file naming (e.g. nsccl.20260108.s.spn).
#
# Usage:
#   python nsccl_prism_fetcher.py --date 2026-01-08 --out "spanfiles"
#   python nsccl_prism_fetcher.py --date 20260108   --out "spanfiles"
#   python nsccl_prism_fetcher.py                   --out "spanfiles"
# (If --date omitted: tries IST today then yesterday)

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import zipfile
import gzip
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

import xml.etree.ElementTree as ET

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124 Safari/537.36"
)

ARCHIVE_BASE = "https://nsearchives.nseindia.com/archives/nsccl/span/"
REV_ORDER = [f"i{i}" for i in range(1, 7)] + ["s"]  # i1..i6 then final s


# ----------------------------
# Utilities: IST date helpers
# ----------------------------

def ist_today():
    return datetime.now(timezone(timedelta(hours=5, minutes=30))).date()


def ymd_list(preferred: Optional[str] = None) -> List[str]:
    # Try explicit date, else try IST today then yesterday
    if preferred:
        return [preferred]
    d0 = ist_today()
    return [d0.strftime("%Y%m%d"), (d0 - timedelta(days=1)).strftime("%Y%m%d")]


# ----------------------------
# Revision detection (local)
# ----------------------------

def _norm_rev_token(tok: str) -> Optional[str]:
    """
    Normalize tokens like:
      i04 -> i4
      I6  -> i6
      s   -> s
    """
    if not tok:
        return None
    t = tok.strip().lower()
    if t == "s":
        return "s"
    m = re.match(r"i0*(\d+)$", t)
    if m:
        return f"i{int(m.group(1))}"
    return None


def _local_revisions_for_date(out_dir: str, ymd: str) -> Set[str]:
    """
    Infer which revisions exist locally for a given date by scanning filenames:
      nsccl.YYYYMMDD.i04.spn
      nsccl.YYYYMMDD.i4.zip
      nsccl.YYYYMMDD.s.spn
      nsccl.YYYYMMDD.i1.spn.gz
    """
    revs: Set[str] = set()
    if not os.path.isdir(out_dir):
        return revs

    # Look for "nsccl.YYYYMMDD.<rev>." where rev is i\d+ or i\d{2} or s
    pat = re.compile(rf"^nsccl\.{re.escape(ymd)}\.(i\d+|i\d{{2}}|s)\.", re.IGNORECASE)
    for fn in os.listdir(out_dir):
        m = pat.match(fn)
        if not m:
            continue
        rev = _norm_rev_token(m.group(1))
        if rev:
            revs.add(rev)
    return revs


def _rev_rank(rev: str) -> int:
    """Higher is newer. s is newest."""
    r = _norm_rev_token(rev) or ""
    if r == "s":
        return 10_000
    if r.startswith("i"):
        try:
            return int(r[1:])
        except Exception:
            return -1
    return -1


def _find_best_local_main_file(out_dir: str, ymd: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Pick the best local extracted file for this date.
    Preference:
      1) highest revision (s > i6 > i5 ...)
      2) prefer .spn over .xml/.txt
      3) largest file within same preference group
    Returns: (filepath, revision)
    """
    if not os.path.isdir(out_dir):
        return None, None

    # Gather candidates by revision
    pat = re.compile(rf"^(nsccl\.{re.escape(ymd)}\.(i\d+|i\d{{2}}|s)\..+)$", re.IGNORECASE)

    buckets: Dict[str, List[str]] = {}
    for fn in os.listdir(out_dir):
        m = pat.match(fn)
        if not m:
            continue
        rev = _norm_rev_token(m.group(2))
        if not rev:
            continue
        buckets.setdefault(rev, []).append(os.path.join(out_dir, fn))

    if not buckets:
        return None, None

    best_rev = max(buckets.keys(), key=_rev_rank)
    files = buckets[best_rev]

    def score(p: str) -> Tuple[int, int]:
        # prefer .spn
        ext_score = 2 if p.lower().endswith(".spn") else (1 if p.lower().endswith(".xml") else 0)
        size = os.path.getsize(p)
        return (ext_score, size)

    best_file = max(files, key=score)
    return best_file, best_rev


# ----------------------------
# Remote candidates per revision
# ----------------------------

def _candidate_names_for_rev(ymd: str, rev: str) -> List[str]:
    """
    Try likely remote names.
    For i-revisions: try both i4 and i04.
    For s: only .s
    Also try .zip and direct .spn.gz/.spn, because NSE varies.
    """
    names: List[str] = []
    if rev == "s":
        bases = [f"nsccl.{ymd}.s"]
    else:
        n = int(rev[1:])
        bases = [
            f"nsccl.{ymd}.i{n}",       # i4
            f"nsccl.{ymd}.i{n:02d}",   # i04
        ]

    for b in bases:
        names.extend([f"{b}.zip", f"{b}.spn.gz", f"{b}.spn"])
    return names


# ----------------------------
# Download + extract
# ----------------------------

@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=8))
def _get(session: requests.Session, url: str) -> requests.Response:
    r = session.get(url, timeout=45, allow_redirects=True)
    r.raise_for_status()
    return r


def _download_blob(session: requests.Session, url: str) -> Optional[bytes]:
    """
    GET with sanity guard. Return bytes if it looks like a real archive/file.
    """
    try:
        r = _get(session, url)
    except Exception:
        return None

    blob = r.content
    # guard against tiny html/403 bodies
    if not blob or len(blob) < 50_000:
        return None
    return blob


def _save_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def extract_relevant_files(blob: bytes, fname: str, out_dir: str) -> List[str]:
    """
    If zip -> extract xml/txt/spn
    If gz  -> decompress into .spn
    Else   -> save as-is
    Returns list of saved/extracted paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []

    lower = fname.lower()

    if lower.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            names = [zi.filename for zi in zf.infolist()]
            for zi in zf.infolist():
                lname = zi.filename.lower()
                if lname.endswith((".xml", ".txt", ".spn")):
                    dst = os.path.join(out_dir, os.path.basename(zi.filename))
                    with zf.open(zi) as src, open(dst, "wb") as out:
                        out.write(src.read())
                    saved.append(dst)

            if not saved:
                raise ValueError(
                    "ZIP contained no XML/TXT/SPN files. Entries were:\n  " + "\n  ".join(names)
                )
        return saved

    if lower.endswith(".spn.gz"):
        out_name = os.path.basename(fname[:-3])  # strip .gz
        dst = os.path.join(out_dir, out_name)
        raw = gzip.decompress(blob)
        _save_bytes(dst, raw)
        saved.append(dst)
        return saved

    # raw .spn
    dst = os.path.join(out_dir, os.path.basename(fname))
    _save_bytes(dst, blob)
    saved.append(dst)
    return saved


# def fetch_only_missing_revisions_for_date(ymd: str, out_dir: str) -> List[str]:
#     """
#     Core behavior:
#       - Detect which revisions already exist locally
#       - Iterate i1..i6 then s
#       - For each missing revision:
#           probe remote candidates; if found, download+save+extract; then continue
#         If not found for that revision -> STOP (newer revisions won't be out yet)
#     Returns: list of newly saved/extracted files.
#     """
#     local_revs = _local_revisions_for_date(out_dir, ymd)
#     newly_saved: List[str] = []

#     with requests.Session() as s:
#         s.headers.update({"User-Agent": UA, "Referer": "https://www.nseindia.com/"})
#         try:
#             s.get("https://www.nseindia.com/", timeout=10)
#         except Exception:
#             pass

#         for rev in REV_ORDER:
#             if rev in local_revs:
#                 continue  # already have this revision on disk

#             found_any = False
#             for fname in _candidate_names_for_rev(ymd, rev):
#                 url = ARCHIVE_BASE + fname
#                 blob = _download_blob(s, url)
#                 if blob is None:
#                     continue

#                 found_any = True

#                 # Save remote payload (zip/spn/spn.gz) so it's cached for you
#                 archive_path = os.path.join(out_dir, fname)
#                 _save_bytes(archive_path, blob)

#                 # Extract/decompress usable files
#                 saved = extract_relevant_files(blob, fname, out_dir)
#                 newly_saved.extend(saved)

#                 # Mark this revision as present and move to next revision
#                 local_revs.add(rev)
#                 break

#             if not found_any:
#                 # This revision not available yet -> stop probing further revisions
#                 break

#     return newly_saved

def fetch_only_missing_revisions_for_date(ymd: str, out_dir: str) -> List[str]:
    """
    Fixed behavior:
      - Download missing i-revisions sequentially (i1..i6), but stop i-probing on first not-found.
      - ALWAYS try to fetch 's' afterwards (because 's' may appear even if i6 never appears).
    Returns: list of newly saved/extracted files.
    """
    local_revs = _local_revisions_for_date(out_dir, ymd)
    newly_saved: List[str] = []

    I_REVS = [f"i{i}" for i in range(1, 7)]
    FINAL_REV = "s"

    def _looks_like_html(data: bytes) -> bool:
        head = data[:512].lower()
        return (b"<html" in head) or (b"<!doctype html" in head)

    with requests.Session() as s:
        s.headers.update({"User-Agent": UA, "Referer": "https://www.nseindia.com/"})
        try:
            s.get("https://www.nseindia.com/", timeout=10)
        except Exception:
            pass

        # ---- 1) i1..i6 (sequential), stop at first missing ----
        for rev in I_REVS:
            if rev in local_revs:
                continue

            found_this_rev = False
            for fname in _candidate_names_for_rev(ymd, rev):
                url = ARCHIVE_BASE + fname
                blob = _download_blob(s, url)
                if blob is None:
                    continue
                if _looks_like_html(blob):
                    continue

                found_this_rev = True

                # cache remote payload
                archive_path = os.path.join(out_dir, fname)
                _save_bytes(archive_path, blob)

                # extract/decompress usable files
                saved = extract_relevant_files(blob, fname, out_dir)
                newly_saved.extend(saved)

                local_revs.add(rev)
                break

            if not found_this_rev:
                # i-revisions are sequential; if this one is not published yet,
                # don't waste time probing higher i-revisions in this run.
                break

        # ---- 2) ALWAYS try FINAL 's' even if an i-rev was missing ----
        if FINAL_REV not in local_revs:
            for fname in _candidate_names_for_rev(ymd, FINAL_REV):
                url = ARCHIVE_BASE + fname
                blob = _download_blob(s, url)
                if blob is None:
                    continue
                if _looks_like_html(blob):
                    continue

                archive_path = os.path.join(out_dir, fname)
                _save_bytes(archive_path, blob)

                saved = extract_relevant_files(blob, fname, out_dir)
                newly_saved.extend(saved)

                local_revs.add(FINAL_REV)
                break

    return newly_saved


# ----------------------------
# Parser (KEEP YOUR EXISTING parse_prism_any)
# ----------------------------
def parse_prism_any(path: str) -> Dict[str, Any]:
    # ---------- tiny helpers ----------
    def _xml(txt: str) -> Optional[ET.Element]:
        try:
            return ET.fromstring(txt)
        except Exception:
            return None

    def _grab(big: str, start: str, end: str) -> Optional[str]:
        i, j = big.find(start), big.rfind(end)
        if i == -1 or j == -1 or j < i:
            return None
        return big[i:j + len(end)]

    def _t(el: ET.Element, tag: str, default=None):
        if el is None:
            return default
        c = el.find(tag)
        return c.text.strip() if (c is not None and c.text) else default

    def _pack_ra(ra_el: Optional[ET.Element]) -> Optional[Dict[str, Any]]:
        if ra_el is None:
            return None
        a_vals = [(a.text.strip() if a.text else None) for a in ra_el.findall("a")]
        a_vals = (a_vals + [None] * 16)[:16]
        pack = {"r": _t(ra_el, "r"), "d": _t(ra_el, "d")}
        for i in range(1, 17):
            pack[f"a{i}"] = a_vals[i - 1]
        return pack

    def _tag_name(el: ET.Element) -> str:
        """Return local tag name without namespace, lowercased."""
        if el is None or not hasattr(el, "tag"):
            return ""
        tag = el.tag
        if "}" in tag:
            tag = tag.split("}", 1)[1]
        return tag.lower()

    # ---------- index filter helpers (NIFTY / BANKNIFTY / SENSEX only) ----------
    TARGET_PF_CODES = {"NIFTY", "BANKNIFTY", "SENSEX"}

    def _is_target_pf(
        pf_meta: Optional[Dict[str, Any]],
        und_meta: Optional[Dict[str, Any]],
    ) -> bool:
        """Return True if this PF looks like NIFTY / BANKNIFTY / SENSEX."""
        def _norm(val: Optional[str]) -> str:
            return val.strip().upper() if isinstance(val, str) else ""

        candidates: List[str] = []
        if pf_meta:
            candidates.append(_norm(pf_meta.get("pfCode")))
            candidates.append(_norm(pf_meta.get("name")))
        if und_meta:
            candidates.append(_norm(und_meta.get("pfCode")))
            # und_meta usually doesn't have a name, but this is cheap to check
            candidates.append(_norm(und_meta.get("name")))

        for v in candidates:
            if not v:
                continue
            # exact match on code or name
            if v in TARGET_PF_CODES:
                return True
            # handle common variants like "NIFTY 50", "NIFTY INDEX", etc.
            for tgt in TARGET_PF_CODES:
                if v == tgt or v.startswith(tgt + " "):
                    return True
        return False

    # ---------- read file ----------
    with open(path, "rb") as f:
        blob = f.read()
    text = blob.decode("utf-8", errors="ignore").replace("\x00", "")

    if "<spanFile" not in text:
        head = blob[:128]
        raise ValueError(
            f"'{os.path.basename(path)}' is not XML-ish (<spanFile> not found). "
            f"First bytes: {head!r}"
        )

    out: Dict[str, Any] = {
        "meta": {},
        "currencies": [],
        "account_types": [],
        "pb_rates": [],
        "scan_points": [],
        "delta_points": [],
        "products": [],          # <exchange>/<phyPf>
        "option_series": [],     # <oopPf>
        "futures_series": [],    # <futPf>  (direct or series-wrapped)
        "cc_defs": [],           # <ccDef> (calendar spreads, tiers, etc.)
    }

    # ---------- header + basic definitions (no ccDef here) ----------
    head = _grab(text, "<spanFile>", "</definitions>")
    if head:
        root = _xml(head + "</spanFile>")
        if root is not None:
            out["meta"]["fileFormat"] = _t(root, "fileFormat")
            out["meta"]["created"] = _t(root, "created")
            defs = root.find("definitions")
            if defs is not None:
                for c in defs.findall("currencyDef"):
                    out["currencies"].append(
                        {
                            "currency": _t(c, "currency"),
                            "symbol": _t(c, "symbol"),
                            "name": _t(c, "name"),
                            "decimalPos": _t(c, "decimalPos"),
                        }
                    )
                for a in defs.findall("acctTypeDef"):
                    out["account_types"].append(
                        {
                            "isCust": _t(a, "isCust"),
                            "acctType": _t(a, "acctType"),
                            "name": _t(a, "name"),
                            "isNetMargin": _t(a, "isNetMargin"),
                            "priority": _t(a, "priority"),
                        }
                    )

    # ---------- ccDef (calendar spreads, tiers, etc.) via regex over full text ----------
    out["cc_defs"] = []
    for m in re.finditer(r"<ccDef\b[^>]*>.*?</ccDef>", text, flags=re.S | re.I):
        snippet = "<root>" + m.group(0) + "</root>"
        root_cc = _xml(snippet)
        if root_cc is None:
            continue

        # get the ccDef element itself
        cc_el = root_cc.find("ccDef")
        if cc_el is None:
            # namespace / case-safe fallback
            for child in root_cc:
                if _tag_name(child) == "ccdef":
                    cc_el = child
                    break
        if cc_el is None:
            continue

        cc_pack: Dict[str, Any] = {
            "cc": _t(cc_el, "cc"),
            "name": _t(cc_el, "name"),
            "currency": _t(cc_el, "currency"),
            "riskExponent": _t(cc_el, "riskExponent"),
            "capAnov": _t(cc_el, "capAnov"),
            "procMeth": _t(cc_el, "procMeth"),
            "wfprMeth": _t(cc_el, "wfprMeth"),
            "spotMeth": _t(cc_el, "spotMeth"),
            "somMeth": _t(cc_el, "somMeth"),
            "cmbMeth": _t(cc_el, "cmbMeth"),
            "group": None,
            "pf_links": [],
            "adj_rates": [],
            "scan_tiers": [],
            "intra_tiers": [],
            "inter_tiers": [],
            "som_tiers": [],
            "d_spreads": [],
        }

        # group
        grp = cc_el.find("group")
        if grp is not None:
            cc_pack["group"] = {
                "id": _t(grp, "id"),
                "aVal": _t(grp, "aVal"),
            }

        # pfLink
        for pf in cc_el.findall("pfLink"):
            cc_pack["pf_links"].append(
                {
                    "exch": _t(pf, "exch"),
                    "pfId": _t(pf, "pfId"),
                    "pfCode": _t(pf, "pfCode"),
                    "pfType": _t(pf, "pfType"),
                    "sc": _t(pf, "sc"),
                    "cmbMeth": _t(pf, "cmbMeth"),
                    "applyBasisRisk": _t(pf, "applyBasisRisk"),
                }
            )

        # adjRate
        for ar in cc_el.findall("adjRate"):
            cc_pack["adj_rates"].append(
                {
                    "r": _t(ar, "r"),
                    "baseR": _t(ar, "baseR"),
                    "val": _t(ar, "val"),
                }
            )

        # scanTiers / tier
        for tier in cc_el.findall(".//scanTiers/tier"):
            cc_pack["scan_tiers"].append(
                {
                    "tn": _t(tier, "tn"),
                    "sPe": _t(tier, "sPe"),
                    "ePe": _t(tier, "ePe"),
                }
            )

        # intraTiers / tier
        for tier in cc_el.findall(".//intraTiers/tier"):
            cc_pack["intra_tiers"].append(
                {
                    "tn": _t(tier, "tn"),
                    "sPe": _t(tier, "sPe"),
                    "ePe": _t(tier, "ePe"),
                }
            )

        # interTiers / tier
        for tier in cc_el.findall(".//interTiers/tier"):
            cc_pack["inter_tiers"].append(
                {
                    "tn": _t(tier, "tn"),
                    "sPe": _t(tier, "sPe"),
                    "ePe": _t(tier, "ePe"),
                }
            )

        # somTiers
        for tier in cc_el.findall(".//somTiers/tier"):
            rates: List[Dict[str, Any]] = []
            for rt in tier.findall("rate"):
                rates.append(
                    {
                        "r": _t(rt, "r"),
                        "val": _t(rt, "val"),
                    }
                )
            cc_pack["som_tiers"].append(
                {
                    "tn": _t(tier, "tn"),
                    "rates": rates,
                }
            )

        # dSpread (calendar spreads) – this is the important part
        for ds in cc_el.findall(".//dSpread"):
            ds_rates: List[Dict[str, Any]] = []
            legs: List[Dict[str, Any]] = []

            for rate in ds.findall("rate"):
                ds_rates.append(
                    {
                        "r": _t(rate, "r"),
                        "val": float(_t(rate, "val", "0") or "0"),
                    }
                )

            for leg in ds.findall("pLeg"):
                pe_txt = _t(leg, "pe")
                legs.append(
                    {
                        "cc": _t(leg, "cc"),
                        "pe": int(pe_txt) if pe_txt is not None else None,
                        "rs": _t(leg, "rs"),
                        "i": float(_t(leg, "i", "0") or "0"),
                    }
                )

            cc_pack["d_spreads"].append(
                {
                    "spread": _t(ds, "spread"),
                    "chargeMeth": _t(ds, "chargeMeth"),
                    "rates": ds_rates,
                    "legs": legs,
                }
            )

        out["cc_defs"].append(cc_pack)

    # ---------- pointInTime / clearingOrg / pointDef ----------
    pit = _grab(text, "<pointInTime>", "</pointInTime>")
    if pit:
        root = _xml("<root>" + pit + "</root>")
        if root is not None:
            p = root.find("pointInTime")
            if p is not None:
                out["meta"]["date"] = _t(p, "date")
                out["meta"]["isSetl"] = _t(p, "isSetl")
                out["meta"]["setlQualifier"] = _t(p, "setlQualifier")
                co = p.find("clearingOrg")
                if co is not None:
                    out["meta"]["clearingOrg"] = {
                        "ec": _t(co, "ec"),
                        "name": _t(co, "name"),
                    }
                    for pb in co.findall("pbRateDef"):
                        out["pb_rates"].append(
                            {
                                "r": _t(pb, "r"),
                                "isCust": _t(pb, "isCust"),
                                "acctType": _t(pb, "acctType"),
                                "isM": _t(pb, "isM"),
                                "pbc": _t(pb, "pbc"),
                            }
                        )
                    pd = co.find("pointDef")
                    if pd is not None:
                        for sp in pd.findall("scanPointDef"):
                            ps, vs = sp.find("priceScanDef"), sp.find("volScanDef")
                            out["scan_points"].append(
                                {
                                    "point": _t(sp, "point"),
                                    "priceScan_mult": _t(ps, "mult") if ps is not None else None,
                                    "volScan_mult": _t(vs, "mult") if vs is not None else None,
                                    "weight": _t(sp, "weight"),
                                    "pairedPoint": _t(sp, "pairedPoint"),
                                }
                            )
                        for dp in pd.findall("deltaPointDef"):
                            ps, vs = dp.find("priceScanDef"), dp.find("volScanDef")
                            out["delta_points"].append(
                                {
                                    "point": _t(dp, "point"),
                                    "priceScan_mult": _t(ps, "mult") if ps is not None else None,
                                    "volScan_mult": _t(vs, "mult") if vs is not None else None,
                                    "weight": _t(dp, "weight"),
                                }
                            )

    # ---------- <exchange>/<phyPf> ----------
    exch = _grab(text, "<exchange>", "</exchange>")
    if exch:
        for m in re.finditer(r"<phyPf>.*?</phyPf>", exch, flags=re.S):
            root_pf = _xml("<root>" + m.group(0) + "</root>")
            if root_pf is None:
                continue
            pf = root_pf.find("phyPf")
            prod = {
                "pfId": _t(pf, "pfId"),
                "pfCode": _t(pf, "pfCode"),
                "name": _t(pf, "name"),
                "currency": _t(pf, "currency"),
                "cvf": _t(pf, "cvf"),
                "valueMeth": _t(pf, "valueMeth"),
                "priceMeth": _t(pf, "priceMeth"),
                "setlMeth": _t(pf, "setlMeth"),
                "contracts": [],
                "scan_rates": [],
                "ra": [],
            }
            for phy in pf.findall("phy"):
                prod["contracts"].append(
                    {
                        "cId": _t(phy, "cId"),
                        "p": _t(phy, "p"),
                        "d": _t(phy, "d"),
                        "v": _t(phy, "v"),
                        "cvf": _t(phy, "cvf"),
                        "sc": _t(phy, "sc"),
                    }
                )
                for sr in phy.findall("scanRate"):
                    prod["scan_rates"].append(
                        {
                            "r": _t(sr, "r"),
                            "priceScan": _t(sr, "priceScan"),
                            "volScan": _t(sr, "volScan"),
                        }
                    )
                for ra in phy.findall("ra"):
                    prod["ra"].append(_pack_ra(ra))
            out["products"].append(prod)

    # ---------- <oopPf> (options) ----------
    for m in re.finditer(r"<oopPf>.*?</oopPf>", text, flags=re.S):
        root_pf = _xml("<root>" + m.group(0) + "</root>")
        if root_pf is None:
            continue
        pf = root_pf.find("oopPf")
        pf_meta = {
            "pfId": _t(pf, "pfId"),
            "pfCode": _t(pf, "pfCode"),
            "name": _t(pf, "name"),
            "exercise": _t(pf, "exercise"),
            "currency": _t(pf, "currency"),
            "cvf": _t(pf, "cvf"),
            "valueMeth": _t(pf, "valueMeth"),
            "priceMeth": _t(pf, "priceMeth"),
            "setlMeth": _t(pf, "setlMeth"),
            "priceModel": _t(pf, "priceModel"),
        }
        undPf = pf.find("undPf")
        und_meta = None
        if undPf is not None:
            und_meta = {
                "exch": _t(undPf, "exch"),
                "pfId": _t(undPf, "pfId"),
                "pfCode": _t(undPf, "pfCode"),
                "pfType": _t(undPf, "pfType"),
                "s": _t(undPf, "s"),
                "i": _t(undPf, "i"),
            }

        # keep only NIFTY / BANKNIFTY / SENSEX option PFs
        if not _is_target_pf(pf_meta, und_meta):
            continue

        for series in pf.findall("series"):
            ser = {
                "pf": pf_meta,
                "underlying_pf": und_meta,
                "pe": _t(series, "pe"),
                "series_v": _t(series, "v"),
                "setlDate": _t(series, "setlDate"),
                "t": _t(series, "t"),
                "cvf": _t(series, "cvf"),
                "sc": _t(series, "sc"),
                "undC": None,
                "intrRate": None,
                "scanRate": None,
                "options": [],
            }
            undC = series.find("undC")
            if undC is not None:
                ser["undC"] = {
                    "exch": _t(undC, "exch"),
                    "pfId": _t(undC, "pfId"),
                    "cId": _t(undC, "cId"),
                    "s": _t(undC, "s"),
                    "i": _t(undC, "i"),
                }
            intr = series.find("intrRate")
            if intr is not None:
                ser["intrRate"] = {
                    "val": _t(intr, "val"),
                    "rl": _t(intr, "rl"),
                    "cpm": _t(intr, "cpm"),
                    "exm": _t(intr, "exm"),
                }
            sr = series.find("scanRate")
            if sr is not None:
                ser["scanRate"] = {
                    "r": _t(sr, "r"),
                    "priceScan": _t(sr, "priceScan"),
                    "volScan": _t(sr, "volScan"),
                }
            for opt in series.findall("opt"):
                ser["options"].append(
                    {
                        "cId": _t(opt, "cId"),
                        "type": _t(opt, "o"),
                        "strike": _t(opt, "k"),
                        "price": _t(opt, "p"),
                        "delta": _t(opt, "d"),
                        "iv": _t(opt, "v"),
                        "ra": _pack_ra(opt.find("ra")),
                    }
                )
            out["option_series"].append(ser)

    # ---------- <futPf> (futures) ----------
    for m in re.finditer(r"<futPf>.*?</futPf>", text, flags=re.S):
        root_pf = _xml("<root>" + m.group(0) + "</root>")
        if root_pf is None:
            continue
        pf = root_pf.find("futPf")
        pf_meta = {
            "pfId": _t(pf, "pfId"),
            "pfCode": _t(pf, "pfCode"),
            "name": _t(pf, "name"),
            "currency": _t(pf, "currency"),
            "cvf": _t(pf, "cvf"),
            "valueMeth": _t(pf, "valueMeth"),
            "priceMeth": _t(pf, "priceMeth"),
            "setlMeth": _t(pf, "setlMeth"),
        }
        undPf = pf.find("undPf")
        und_meta = None
        if undPf is not None:
            und_meta = {
                "exch": _t(undPf, "exch"),
                "pfId": _t(undPf, "pfId"),
                "pfCode": _t(undPf, "pfCode"),
                "pfType": _t(undPf, "pfType"),
                "s": _t(undPf, "s"),
                "i": _t(undPf, "i"),
            }

        # keep only NIFTY / BANKNIFTY / SENSEX futures PFs
        if not _is_target_pf(pf_meta, und_meta):
            continue

        # Shape A: direct fut
        direct_futs = pf.findall("fut")
        if direct_futs:
            ser = {
                "pf": pf_meta,
                "underlying_pf": und_meta,
                "pe": None,
                "setlDate": None,
                "t": None,
                "cvf": None,
                "sc": None,
                "undC": None,
                "intrRate": None,
                "scanRate": None,
                "futures": [],
            }
            for fut in direct_futs:
                undC = fut.find("undC")
                intr = fut.find("intrRate")
                sr = fut.find("scanRate")
                ser["futures"].append(
                    {
                        "cId": _t(fut, "cId"),
                        "pe": _t(fut, "pe"),
                        "price": _t(fut, "p"),
                        "delta": _t(fut, "d"),
                        "iv": _t(fut, "v"),
                        "cvf": _t(fut, "cvf"),
                        "sc": _t(fut, "sc"),
                        "setlDate": _t(fut, "setlDate"),
                        "t": _t(fut, "t"),
                        "undC": (
                            {
                                "exch": _t(undC, "exch"),
                                "pfId": _t(undC, "pfId"),
                                "cId": _t(undC, "cId"),
                                "s": _t(undC, "s"),
                                "i": _t(undC, "i"),
                            }
                            if undC is not None
                            else None
                        ),
                        "intrRate": (
                            {
                                "val": _t(intr, "val"),
                                "rl": _t(intr, "rl"),
                                "cpm": _t(intr, "cpm"),
                                "exm": _t(intr, "exm"),
                            }
                            if intr is not None
                            else None
                        ),
                        "scanRate": (
                            {
                                "r": _t(sr, "r"),
                                "priceScan": _t(sr, "priceScan"),
                                "volScan": _t(sr, "volScan"),
                            }
                            if sr is not None
                            else None
                        ),
                        "ra": _pack_ra(fut.find("ra")),
                    }
                )
            if ser["futures"]:
                out["futures_series"].append(ser)

        # Shape B: series-wrapped fut
        for series in pf.findall("series"):
            ser = {
                "pf": pf_meta,
                "underlying_pf": und_meta,
                "pe": _t(series, "pe"),
                "setlDate": _t(series, "setlDate"),
                "t": _t(series, "t"),
                "cvf": _t(series, "cvf"),
                "sc": _t(series, "sc"),
                "undC": None,
                "intrRate": None,
                "scanRate": None,
                "futures": [],
            }
            undC = series.find("undC")
            if undC is not None:
                ser["undC"] = {
                    "exch": _t(undC, "exch"),
                    "pfId": _t(undC, "pfId"),
                    "cId": _t(undC, "cId"),
                    "s": _t(undC, "s"),
                    "i": _t(undC, "i"),
                }
            intr = series.find("intrRate")
            if intr is not None:
                ser["intrRate"] = {
                    "val": _t(intr, "val"),
                    "rl": _t(intr, "rl"),
                    "cpm": _t(intr, "cpm"),
                    "exm": _t(intr, "exm"),
                }
            sr = series.find("scanRate")
            if sr is not None:
                ser["scanRate"] = {
                    "r": _t(sr, "r"),
                    "priceScan": _t(sr, "priceScan"),
                    "volScan": _t(sr, "volScan"),
                }
            for fut in series.findall("fut"):
                ser["futures"].append(
                    {
                        "cId": _t(fut, "cId"),
                        "pe": _t(fut, "pe"),
                        "price": _t(fut, "p"),
                        "delta": _t(fut, "d"),
                        "iv": _t(fut, "v"),
                        "cvf": _t(fut, "cvf"),
                        "sc": _t(fut, "sc"),
                        "setlDate": _t(fut, "setlDate"),
                        "t": _t(fut, "t"),
                        "undC": None,
                        "intrRate": None,
                        "scanRate": None,
                        "ra": _pack_ra(fut.find("ra")),
                    }
                )
            if ser["futures"]:
                out["futures_series"].append(ser)

    return out

# Paste your existing parse_prism_any(path: str) -> Dict[str, Any] here unchanged.
# (I am not repeating it here to avoid a very large message.)
def parse_prism_any2(path: str, instruments: Optional[List[str]] = None) -> Dict[str, Any]:
    # ---------- tiny helpers ----------
    def _xml(txt: str) -> Optional[ET.Element]:
        try:
            return ET.fromstring(txt)
        except Exception:
            return None

    def _grab(big: str, start: str, end: str) -> Optional[str]:
        i, j = big.find(start), big.rfind(end)
        if i == -1 or j == -1 or j < i:
            return None
        return big[i : j + len(end)]

    def _tag_name(el: ET.Element) -> str:
        """Return local tag name without namespace, lowercased."""
        if el is None or not hasattr(el, "tag"):
            return ""
        tag = el.tag
        if "}" in tag:
            tag = tag.split("}", 1)[1]
        return tag.lower()

    def _find_child_local(el: Optional[ET.Element], tag: str) -> Optional[ET.Element]:
        """Find direct child by localname (namespace-safe)."""
        if el is None:
            return None
        tgt = tag.lower()
        for ch in list(el):
            if _tag_name(ch) == tgt:
                return ch
        return None

    def _t(el: Optional[ET.Element], tag: str, default=None):
        """
        Namespace-safe-ish:
          - try el.find(tag)
          - else try direct-child localname scan
        """
        if el is None:
            return default
        c = el.find(tag)
        if c is None:
            c = _find_child_local(el, tag)
        return c.text.strip() if (c is not None and c.text) else default

    def _pack_ra(ra_el: Optional[ET.Element]) -> Optional[Dict[str, Any]]:
        if ra_el is None:
            return None
        a_vals = [(a.text.strip() if a.text else None) for a in ra_el.findall("a")]
        a_vals = (a_vals + [None] * 16)[:16]
        pack = {"r": _t(ra_el, "r"), "d": _t(ra_el, "d")}
        for i in range(1, 17):
            pack[f"a{i}"] = a_vals[i - 1]
        return pack

    # ---------- Instrument filter ----------
    instruments_set: Optional[Set[str]] = None
    if instruments:
        instruments_set = {str(x).strip().upper() for x in instruments if str(x).strip()}

    def _matches_requested(candidate: str) -> bool:
        """
        Match candidate against requested instruments.
        - exact match
        - prefix match: "NIFTY 50" startswith "NIFTY"
        - whole-word match
        """
        if not instruments_set:
            return True
        c = candidate.strip().upper()
        if not c:
            return False
        for ins in instruments_set:
            if c == ins:
                return True
            if c.startswith(ins + " ") or c.startswith(ins + "-"):
                return True
            if re.search(rf"\b{re.escape(ins)}\b", c):
                return True
        return False

    def _is_target_pf(
        pf_meta: Optional[Dict[str, Any]],
        und_meta: Optional[Dict[str, Any]],
    ) -> bool:
        """
        If instruments is None/empty => keep everything.
        Else keep only if pfCode/name or underlying pfCode/name matches requested list.
        """
        if not instruments_set:
            return True

        def _norm(val: Optional[str]) -> str:
            return val.strip().upper() if isinstance(val, str) else ""

        candidates: List[str] = []
        if pf_meta:
            candidates.append(_norm(pf_meta.get("pfCode")))
            candidates.append(_norm(pf_meta.get("name")))
        if und_meta:
            candidates.append(_norm(und_meta.get("pfCode")))
            candidates.append(_norm(und_meta.get("name")))

        return any(_matches_requested(v) for v in candidates if v)

    # ---------- read file ----------
    with open(path, "rb") as f:
        blob = f.read()
    text = blob.decode("utf-8", errors="ignore").replace("\x00", "")

    if "<spanFile" not in text:
        head = blob[:128]
        raise ValueError(
            f"'{os.path.basename(path)}' is not XML-ish (<spanFile> not found). "
            f"First bytes: {head!r}"
        )

    out: Dict[str, Any] = {
        "meta": {},
        "currencies": [],
        "account_types": [],
        "pb_rates": [],
        "scan_points": [],
        "delta_points": [],
        "products": [],          # <exchange>/<phyPf>
        "option_series": [],     # <oopPf>
        "futures_series": [],    # <futPf>
        "cc_defs": [],           # <ccDef>
    }

    # ---------- header + basic definitions ----------
    head = _grab(text, "<spanFile>", "</definitions>")
    if head:
        root = _xml(head + "</spanFile>")
        if root is not None:
            out["meta"]["fileFormat"] = _t(root, "fileFormat")
            out["meta"]["created"] = _t(root, "created")
            defs = root.find("definitions")
            if defs is not None:
                for c in defs.findall("currencyDef"):
                    out["currencies"].append(
                        {
                            "currency": _t(c, "currency"),
                            "symbol": _t(c, "symbol"),
                            "name": _t(c, "name"),
                            "decimalPos": _t(c, "decimalPos"),
                        }
                    )
                for a in defs.findall("acctTypeDef"):
                    out["account_types"].append(
                        {
                            "isCust": _t(a, "isCust"),
                            "acctType": _t(a, "acctType"),
                            "name": _t(a, "name"),
                            "isNetMargin": _t(a, "isNetMargin"),
                            "priority": _t(a, "priority"),
                        }
                    )

    # ---------- ccDef (calendar spreads etc.) ----------
    out["cc_defs"] = []
    for m in re.finditer(r"<ccDef\b[^>]*>.*?</ccDef>", text, flags=re.S | re.I):
        snippet = "<root>" + m.group(0) + "</root>"
        root_cc = _xml(snippet)
        if root_cc is None:
            continue

        cc_el = root_cc.find("ccDef")
        if cc_el is None:
            for child in root_cc:
                if _tag_name(child) == "ccdef":
                    cc_el = child
                    break
        if cc_el is None:
            continue

        cc_pack: Dict[str, Any] = {
            "cc": _t(cc_el, "cc"),
            "name": _t(cc_el, "name"),
            "currency": _t(cc_el, "currency"),
            "riskExponent": _t(cc_el, "riskExponent"),
            "capAnov": _t(cc_el, "capAnov"),
            "procMeth": _t(cc_el, "procMeth"),
            "wfprMeth": _t(cc_el, "wfprMeth"),
            "spotMeth": _t(cc_el, "spotMeth"),
            "somMeth": _t(cc_el, "somMeth"),
            "cmbMeth": _t(cc_el, "cmbMeth"),
            "group": None,
            "pf_links": [],
            "adj_rates": [],
            "scan_tiers": [],
            "intra_tiers": [],
            "inter_tiers": [],
            "som_tiers": [],
            "d_spreads": [],
        }

        grp = cc_el.find("group")
        if grp is not None:
            cc_pack["group"] = {"id": _t(grp, "id"), "aVal": _t(grp, "aVal")}

        for pf in cc_el.findall("pfLink"):
            cc_pack["pf_links"].append(
                {
                    "exch": _t(pf, "exch"),
                    "pfId": _t(pf, "pfId"),
                    "pfCode": _t(pf, "pfCode"),
                    "pfType": _t(pf, "pfType"),
                    "sc": _t(pf, "sc"),
                    "cmbMeth": _t(pf, "cmbMeth"),
                    "applyBasisRisk": _t(pf, "applyBasisRisk"),
                }
            )

        for ar in cc_el.findall("adjRate"):
            cc_pack["adj_rates"].append({"r": _t(ar, "r"), "baseR": _t(ar, "baseR"), "val": _t(ar, "val")})

        for tier in cc_el.findall(".//scanTiers/tier"):
            cc_pack["scan_tiers"].append({"tn": _t(tier, "tn"), "sPe": _t(tier, "sPe"), "ePe": _t(tier, "ePe")})

        for tier in cc_el.findall(".//intraTiers/tier"):
            cc_pack["intra_tiers"].append({"tn": _t(tier, "tn"), "sPe": _t(tier, "sPe"), "ePe": _t(tier, "ePe")})

        for tier in cc_el.findall(".//interTiers/tier"):
            cc_pack["inter_tiers"].append({"tn": _t(tier, "tn"), "sPe": _t(tier, "sPe"), "ePe": _t(tier, "ePe")})

        for tier in cc_el.findall(".//somTiers/tier"):
            rates: List[Dict[str, Any]] = []
            for rt in tier.findall("rate"):
                rates.append({"r": _t(rt, "r"), "val": _t(rt, "val")})
            cc_pack["som_tiers"].append({"tn": _t(tier, "tn"), "rates": rates})

        for ds in cc_el.findall(".//dSpread"):
            ds_rates: List[Dict[str, Any]] = []
            legs: List[Dict[str, Any]] = []

            for rate in ds.findall("rate"):
                ds_rates.append({"r": _t(rate, "r"), "val": float(_t(rate, "val", "0") or "0")})

            for leg in ds.findall("pLeg"):
                pe_txt = _t(leg, "pe")
                legs.append(
                    {
                        "cc": _t(leg, "cc"),
                        "pe": int(pe_txt) if pe_txt is not None else None,
                        "rs": _t(leg, "rs"),
                        "i": float(_t(leg, "i", "0") or "0"),
                    }
                )

            cc_pack["d_spreads"].append(
                {"spread": _t(ds, "spread"), "chargeMeth": _t(ds, "chargeMeth"), "rates": ds_rates, "legs": legs}
            )

        out["cc_defs"].append(cc_pack)

    # ---------- pointInTime ----------
    pit = _grab(text, "<pointInTime>", "</pointInTime>")
    if pit:
        root = _xml("<root>" + pit + "</root>")
        if root is not None:
            p = root.find("pointInTime")
            if p is not None:
                out["meta"]["date"] = _t(p, "date")
                out["meta"]["isSetl"] = _t(p, "isSetl")
                out["meta"]["setlQualifier"] = _t(p, "setlQualifier")
                co = p.find("clearingOrg")
                if co is not None:
                    out["meta"]["clearingOrg"] = {"ec": _t(co, "ec"), "name": _t(co, "name")}
                    for pb in co.findall("pbRateDef"):
                        out["pb_rates"].append(
                            {
                                "r": _t(pb, "r"),
                                "isCust": _t(pb, "isCust"),
                                "acctType": _t(pb, "acctType"),
                                "isM": _t(pb, "isM"),
                                "pbc": _t(pb, "pbc"),
                            }
                        )
                    pd = co.find("pointDef")
                    if pd is not None:
                        for sp in pd.findall("scanPointDef"):
                            ps, vs = sp.find("priceScanDef"), sp.find("volScanDef")
                            out["scan_points"].append(
                                {
                                    "point": _t(sp, "point"),
                                    "priceScan_mult": _t(ps, "mult") if ps is not None else None,
                                    "volScan_mult": _t(vs, "mult") if vs is not None else None,
                                    "weight": _t(sp, "weight"),
                                    "pairedPoint": _t(sp, "pairedPoint"),
                                }
                            )
                        for dp in pd.findall("deltaPointDef"):
                            ps, vs = dp.find("priceScanDef"), dp.find("volScanDef")
                            out["delta_points"].append(
                                {
                                    "point": _t(dp, "point"),
                                    "priceScan_mult": _t(ps, "mult") if ps is not None else None,
                                    "volScan_mult": _t(vs, "mult") if vs is not None else None,
                                    "weight": _t(dp, "weight"),
                                }
                            )

    # ---------- <exchange>/<phyPf> (products) ----------
    exch = _grab(text, "<exchange>", "</exchange>")
    if exch:
        for m in re.finditer(r"<phyPf>.*?</phyPf>", exch, flags=re.S):
            root_pf = _xml("<root>" + m.group(0) + "</root>")
            if root_pf is None:
                continue
            pf = root_pf.find("phyPf")
            if pf is None:
                continue

            pf_meta = {"pfId": _t(pf, "pfId"), "pfCode": _t(pf, "pfCode"), "name": _t(pf, "name")}
            if not _is_target_pf(pf_meta, und_meta=None):
                continue

            prod = {
                "pfId": pf_meta["pfId"],
                "pfCode": pf_meta["pfCode"],
                "name": pf_meta["name"],
                "currency": _t(pf, "currency"),
                "cvf": _t(pf, "cvf"),
                "valueMeth": _t(pf, "valueMeth"),
                "priceMeth": _t(pf, "priceMeth"),
                "setlMeth": _t(pf, "setlMeth"),
                "contracts": [],
                "scan_rates": [],
                "ra": [],
            }
            for phy in pf.findall("phy"):
                prod["contracts"].append(
                    {
                        "cId": _t(phy, "cId"),
                        "p": _t(phy, "p"),
                        "d": _t(phy, "d"),
                        "v": _t(phy, "v"),
                        "cvf": _t(phy, "cvf"),
                        "sc": _t(phy, "sc"),
                    }
                )
                for sr in phy.findall("scanRate"):
                    prod["scan_rates"].append(
                        {"r": _t(sr, "r"), "priceScan": _t(sr, "priceScan"), "volScan": _t(sr, "volScan")}
                    )
                for ra in phy.findall("ra"):
                    prod["ra"].append(_pack_ra(ra))
            out["products"].append(prod)

    # ---------- <oopPf> (options) ----------
    for m in re.finditer(r"<oopPf>.*?</oopPf>", text, flags=re.S):
        root_pf = _xml("<root>" + m.group(0) + "</root>")
        if root_pf is None:
            continue
        pf = root_pf.find("oopPf")
        if pf is None:
            continue

        pf_meta = {
            "pfId": _t(pf, "pfId"),
            "pfCode": _t(pf, "pfCode"),
            "name": _t(pf, "name"),
            "exercise": _t(pf, "exercise"),
            "currency": _t(pf, "currency"),
            "cvf": _t(pf, "cvf"),
            "valueMeth": _t(pf, "valueMeth"),
            "priceMeth": _t(pf, "priceMeth"),
            "setlMeth": _t(pf, "setlMeth"),
            "priceModel": _t(pf, "priceModel"),
        }
        undPf = pf.find("undPf")
        und_meta = None
        if undPf is not None:
            und_meta = {
                "exch": _t(undPf, "exch"),
                "pfId": _t(undPf, "pfId"),
                "pfCode": _t(undPf, "pfCode"),
                "pfType": _t(undPf, "pfType"),
                "s": _t(undPf, "s"),
                "i": _t(undPf, "i"),
                "name": _t(undPf, "name"),
            }

        if not _is_target_pf(pf_meta, und_meta):
            continue

        for series in pf.findall("series"):
            ser = {
                "pf": pf_meta,
                "underlying_pf": und_meta,
                "pe": _t(series, "pe"),
                "series_v": _t(series, "v"),
                "setlDate": _t(series, "setlDate"),
                "t": _t(series, "t"),
                "cvf": _t(series, "cvf"),
                "sc": _t(series, "sc"),
                "undC": None,
                "intrRate": None,
                "scanRate": None,
                "options": [],
            }
            undC = series.find("undC")
            if undC is not None:
                ser["undC"] = {
                    "exch": _t(undC, "exch"),
                    "pfId": _t(undC, "pfId"),
                    "cId": _t(undC, "cId"),
                    "s": _t(undC, "s"),
                    "i": _t(undC, "i"),
                }
            intr = series.find("intrRate")
            if intr is not None:
                ser["intrRate"] = {"val": _t(intr, "val"), "rl": _t(intr, "rl"), "cpm": _t(intr, "cpm"), "exm": _t(intr, "exm")}
            sr = series.find("scanRate")
            if sr is not None:
                ser["scanRate"] = {"r": _t(sr, "r"), "priceScan": _t(sr, "priceScan"), "volScan": _t(sr, "volScan")}

            for opt in series.findall("opt"):
                ser["options"].append(
                    {
                        "cId": _t(opt, "cId"),
                        "type": _t(opt, "o"),
                        "strike": _t(opt, "k"),
                        "price": _t(opt, "p"),
                        "delta": _t(opt, "d"),
                        "iv": _t(opt, "v"),
                        "ra": _pack_ra(opt.find("ra")),
                    }
                )
            out["option_series"].append(ser)

    # ---------- <futPf> (futures) ----------
    for m in re.finditer(r"<futPf>.*?</futPf>", text, flags=re.S):
        root_pf = _xml("<root>" + m.group(0) + "</root>")
        if root_pf is None:
            continue
        pf = root_pf.find("futPf")
        if pf is None:
            continue

        pf_meta = {
            "pfId": _t(pf, "pfId"),
            "pfCode": _t(pf, "pfCode"),
            "name": _t(pf, "name"),
            "currency": _t(pf, "currency"),
            "cvf": _t(pf, "cvf"),
            "valueMeth": _t(pf, "valueMeth"),
            "priceMeth": _t(pf, "priceMeth"),
            "setlMeth": _t(pf, "setlMeth"),
        }
        undPf = pf.find("undPf")
        und_meta = None
        if undPf is not None:
            und_meta = {
                "exch": _t(undPf, "exch"),
                "pfId": _t(undPf, "pfId"),
                "pfCode": _t(undPf, "pfCode"),
                "pfType": _t(undPf, "pfType"),
                "s": _t(undPf, "s"),
                "i": _t(undPf, "i"),
                "name": _t(undPf, "name"),
            }

        if not _is_target_pf(pf_meta, und_meta):
            continue

        direct_futs = pf.findall("fut")
        if direct_futs:
            ser = {
                "pf": pf_meta,
                "underlying_pf": und_meta,
                "pe": None,
                "setlDate": None,
                "t": None,
                "cvf": None,
                "sc": None,
                "undC": None,
                "intrRate": None,
                "scanRate": None,
                "futures": [],
            }
            for fut in direct_futs:
                undC = fut.find("undC")
                intr = fut.find("intrRate")
                sr = fut.find("scanRate")
                ser["futures"].append(
                    {
                        "cId": _t(fut, "cId"),
                        "pe": _t(fut, "pe"),
                        "price": _t(fut, "p"),
                        "delta": _t(fut, "d"),
                        "iv": _t(fut, "v"),
                        "cvf": _t(fut, "cvf"),
                        "sc": _t(fut, "sc"),
                        "setlDate": _t(fut, "setlDate"),
                        "t": _t(fut, "t"),
                        "undC": (
                            {"exch": _t(undC, "exch"), "pfId": _t(undC, "pfId"), "cId": _t(undC, "cId"), "s": _t(undC, "s"), "i": _t(undC, "i")}
                            if undC is not None
                            else None
                        ),
                        "intrRate": (
                            {"val": _t(intr, "val"), "rl": _t(intr, "rl"), "cpm": _t(intr, "cpm"), "exm": _t(intr, "exm")}
                            if intr is not None
                            else None
                        ),
                        "scanRate": (
                            {"r": _t(sr, "r"), "priceScan": _t(sr, "priceScan"), "volScan": _t(sr, "volScan")}
                            if sr is not None
                            else None
                        ),
                        "ra": _pack_ra(fut.find("ra")),
                    }
                )
            if ser["futures"]:
                out["futures_series"].append(ser)

        for series in pf.findall("series"):
            ser = {
                "pf": pf_meta,
                "underlying_pf": und_meta,
                "pe": _t(series, "pe"),
                "setlDate": _t(series, "setlDate"),
                "t": _t(series, "t"),
                "cvf": _t(series, "cvf"),
                "sc": _t(series, "sc"),
                "undC": None,
                "intrRate": None,
                "scanRate": None,
                "futures": [],
            }
            undC = series.find("undC")
            if undC is not None:
                ser["undC"] = {"exch": _t(undC, "exch"), "pfId": _t(undC, "pfId"), "cId": _t(undC, "cId"), "s": _t(undC, "s"), "i": _t(undC, "i")}
            intr = series.find("intrRate")
            if intr is not None:
                ser["intrRate"] = {"val": _t(intr, "val"), "rl": _t(intr, "rl"), "cpm": _t(intr, "cpm"), "exm": _t(intr, "exm")}
            sr = series.find("scanRate")
            if sr is not None:
                ser["scanRate"] = {"r": _t(sr, "r"), "priceScan": _t(sr, "priceScan"), "volScan": _t(sr, "volScan")}

            for fut in series.findall("fut"):
                ser["futures"].append(
                    {
                        "cId": _t(fut, "cId"),
                        "pe": _t(fut, "pe"),
                        "price": _t(fut, "p"),
                        "delta": _t(fut, "d"),
                        "iv": _t(fut, "v"),
                        "cvf": _t(fut, "cvf"),
                        "sc": _t(fut, "sc"),
                        "setlDate": _t(fut, "setlDate"),
                        "t": _t(fut, "t"),
                        "undC": None,
                        "intrRate": None,
                        "scanRate": None,
                        "ra": _pack_ra(fut.find("ra")),
                    }
                )
            if ser["futures"]:
                out["futures_series"].append(ser)

    return out


# ----------------------------
# Classification helpers
# ----------------------------

def classify_file(path: str) -> str:
    """
    Return 'series' if file seems to contain option-series rows, else 'product'.
    """
    size_mb = os.path.getsize(path) / (1024 * 1024)
    with open(path, "rb") as f:
        head = f.read(1_000_000).decode("utf-8", errors="ignore").lower()

    markers = ["<ooppf", "<opt", "</opt>", "series", "intrrate", "scanrate"]
    if any(m in head for m in markers) or size_mb > 40:
        return "series"
    return "product"


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


# ----------------------------
# Orchestrator
# ----------------------------

def run(date_str: Optional[str], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print("Output directory:", os.path.abspath(out_dir))
    ymds = ymd_list(date_str)

    chosen_ymd: Optional[str] = None
    downloaded_any = False

    # Try the first date that either:
    # - downloads something new, OR
    # - already has at least one local revision file
    for ymd in ymds:
        new_files = fetch_only_missing_revisions_for_date(ymd=ymd, out_dir=out_dir)
        if new_files:
            downloaded_any = True
            chosen_ymd = ymd
            break

    #     # If nothing new downloaded, but local exists, we can still parse latest local
    #     best_local, best_rev = _find_best_local_main_file(out_dir, ymd)
    #     if best_local:
    #         chosen_ymd = ymd
    #         break

    if not chosen_ymd:
        print("No archives found / no local files present for:", ymds)
        sys.exit(2)

    # # Pick latest local and parse it
    # best_local, best_rev = _find_best_local_main_file(out_dir, chosen_ymd)
    # if not best_local:
    #     print(f"Could not find any extracted/local file to parse for {chosen_ymd} in {out_dir}")
    #     sys.exit(2)

    # kind = classify_file(best_local)
    # print(f"Using date={chosen_ymd}, revision={best_rev}, file={os.path.basename(best_local)} ({kind})")
    if downloaded_any:
        print("Downloaded new revision(s) this run.")
    else:
        print("No new revisions downloaded; using existing local files.")

    # ---- Parse using your existing function ----
    # parsed = parse_prism_any(best_local)

    # Save with revision in name + also "latest" convenience
    # out_path = os.path.join(out_dir, f"{chosen_ymd}_{best_rev}_{kind}.json")
    # save_json(parsed, out_path)

    # latest_path = os.path.join(out_dir, f"{chosen_ymd}_{kind}_latest.json")
    # save_json(parsed, latest_path)

    # print("Saved:", out_path)
    # print("Saved:", latest_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch & parse NSCCL PRISM/SPAN archives incrementally (i1..i6..s)")
    ap.add_argument("--date", help="YYYY-MM-DD or YYYYMMDD (IST). Default: try today & yesterday.")
    ap.add_argument("--out", default=r"spanfiles", help="Output directory")
    args = ap.parse_args()

    ds = None
    if args.date:
        ds_in = args.date.strip()
        if re.match(r"^\d{8}$", ds_in):
            ds = ds_in
        elif re.match(r"^\d{4}-\d{2}-\d{2}$", ds_in):
            ds = ds_in.replace("-", "")
        else:
            print("--date must be YYYYMMDD or YYYY-MM-DD")
            sys.exit(2)

    run(ds, args.out)
