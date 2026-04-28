from __future__ import annotations
import utils
import pandas as pd
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import redis
# from dhanwebsocket import utils
import pandas as pd

from functools import lru_cache
from typing import Any, Dict, Optional, List
import os
from pathlib import Path

# from margin_calc import breakdown as bd_m
# from margin_calc.span_file import span_provider as get_spn
import json
# from margin_calc import margin_trace as mt

#=============================================================================================
#=============================================================================================
#=============================================================================================
# PRISM PARSER
#=============================================================================================
def parse_prism_any_indices(path: str) -> Dict[str, Any]:
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


#=======================================================================================
#updated function with lot and unit switch 
#=======================================================================================

# ============================================================
# 1. Small helpers
# ============================================================
def _parse_int_or_none(x) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _norm_opt_type(t: Optional[str]) -> Optional[str]:
    if t is None:
        return None
    t = str(t).strip().upper()
    if t in ("C", "CALL"):
        return "CE"
    if t in ("P", "PUT"):
        return "PE"
    if t in ("CE", "PE"):
        return t
    return t


def _ra_to_array(ra_dict: Optional[Dict[str, Any]]) -> Optional[List[float]]:
    if not ra_dict:
        return None
    arr: List[float] = []
    for i in range(1, 17):
        v = ra_dict.get(f"a{i}")
        if v is None or v == "":
            arr.append(0.0)
        else:
            try:
                arr.append(float(v))
            except Exception:
                arr.append(0.0)
    if all(a == 0.0 for a in arr):
        return None
    return arr


def _resolve_lot_size(
    pf_code: str,
    exp: int,
    lot_size_by_series: Optional[Dict[Tuple[str, int], int]] = None,
    lot_size_by_underlying: Optional[Dict[str, int]] = None,
) -> int:
    if lot_size_by_series is not None:
        lot = lot_size_by_series.get((pf_code, exp))
        if lot is not None:
            return int(lot)
    if lot_size_by_underlying is not None:
        lot = lot_size_by_underlying.get(pf_code)
        if lot is not None:
            return int(lot)
    return 1


# ============================================================
# 3. Build SPAN contract index (RA + Composite Delta)
# ============================================================

def build_span_contract_index(
    span: Dict[str, Any],
    lot_size_by_series: Optional[Dict[Tuple[str, int], int]] = None,
    lot_size_by_underlying: Optional[Dict[str, int]] = None,
    futures_ra_is_per_unit: bool = False,
    options_ra_is_per_unit: bool = True,
    negate_option_ra: bool = True,
    # NEW:
    is_units: bool = False,
) -> Dict[Tuple[str, str, int, Optional[float], Optional[str]], Dict[str, Any]]:
    """
    Build an index keyed by:
        (underlying, kind, expiry_int, strike, opt_type)

    IMPORTANT (basis of cinfo["ra"] depends on is_units):
      - is_units=False (legacy): cinfo["ra"] is per-LOT (per contract) for +1 LONG lot.
      - is_units=True          : cinfo["ra"] is per-UNIT for +1 LONG unit.

    We respect the raw SPAN RA basis flags:
      - futures_ra_is_per_unit / options_ra_is_per_unit tell whether raw RA is per-unit.
    """
    idx: Dict[
        Tuple[str, str, int, Optional[float], Optional[str]],
        Dict[str, Any]
    ] = {}

    def _convert_ra(raw: List[float], lot: int, raw_is_per_unit: bool) -> List[float]:
        lot_f = float(max(1, int(lot)))
        if is_units:
            # Want per-unit output
            if raw_is_per_unit:
                return raw
            # raw is per-lot -> divide to get per-unit
            return [v / lot_f for v in raw]
        else:
            # Want per-lot output (legacy)
            if raw_is_per_unit:
                return [v * lot_f for v in raw]
            return raw

    # ---------- Futures ----------
    for ser_idx, ser in enumerate(span.get("futures_series", [])):
        pf_meta = ser.get("pf", {}) or {}
        pf_code = pf_meta.get("pfCode")
        if not pf_code:
            continue

        for fut_idx, fut in enumerate(ser.get("futures", [])):
            exp = _parse_int_or_none(fut.get("pe") or ser.get("pe"))
            if exp is None:
                continue

            ra_arr_raw = _ra_to_array(fut.get("ra"))
            if ra_arr_raw is None:
                continue

            lot = _resolve_lot_size(
                pf_code,
                exp,
                lot_size_by_series=lot_size_by_series,
                lot_size_by_underlying=lot_size_by_underlying,
            )

            ra_arr = _convert_ra(ra_arr_raw, lot, futures_ra_is_per_unit)

            key = (pf_code, "FUT", exp, None, None)
            idx[key] = {
                "underlying": pf_code,
                "kind": "FUT",
                "expiry": exp,
                "strike": None,
                "opt_type": None,
                "ra": ra_arr,
                "delta_comp": 1.0,
                "fut_meta": fut,
                "source": {
                    "section": "futures_series",
                    "series_index": ser_idx,
                    "instrument_index": fut_idx,
                    "pfCode": pf_code,
                    "pe": exp,
                },
            }

    # ---------- Options ----------
    for ser_idx, ser in enumerate(span.get("option_series", [])):
        pf_meta = ser.get("pf", {}) or {}
        pf_code = pf_meta.get("pfCode")
        if not pf_code:
            continue

        exp = _parse_int_or_none(ser.get("pe"))
        if exp is None:
            continue

        for opt_idx, opt in enumerate(ser.get("options", [])):
            strike_raw = opt.get("strike")
            if strike_raw is None or strike_raw == "":
                continue
            try:
                strike = float(strike_raw)
            except Exception:
                continue

            opt_type = _norm_opt_type(opt.get("type"))
            ra_dict = opt.get("ra") or {}
            ra_arr_raw = _ra_to_array(ra_dict)
            if ra_arr_raw is None:
                continue

            # Keep your existing behavior (still commented out)
            # if negate_option_ra:
            #     ra_arr_raw = [-v for v in ra_arr_raw]

            lot = _resolve_lot_size(
                pf_code,
                exp,
                lot_size_by_series=lot_size_by_series,
                lot_size_by_underlying=lot_size_by_underlying,
            )

            ra_arr = _convert_ra(ra_arr_raw, lot, options_ra_is_per_unit)

            delta_comp = 0.0
            try:
                d_raw = ra_dict.get("d")
                if d_raw is not None and d_raw != "":
                    delta_comp = float(d_raw)
                else:
                    delta_comp = float(opt.get("delta") or 0.0)
            except Exception:
                delta_comp = 0.0

            key = (pf_code, "OPT", exp, strike, opt_type)
            idx[key] = {
                "underlying": pf_code,
                "kind": "OPT",
                "expiry": exp,
                "strike": strike,
                "opt_type": opt_type,
                "ra": ra_arr,
                "delta_comp": float(delta_comp),
                "opt_meta": opt,
                "source": {
                    "section": "option_series",
                    "series_index": ser_idx,
                    "instrument_index": opt_idx,
                    "pfCode": pf_code,
                    "pe": exp,
                    "strike": strike,
                    "type": opt_type,
                },
            }

    return idx


# ============================================================
# 4. ccDef helpers (calendar spreads)
# ============================================================

def build_ccdef_index(span: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for cc_def in span.get("cc_defs", []):
        cc = cc_def.get("cc")
        if cc:
            out[str(cc)] = cc_def
    return out


def extract_calendar_spreads(cc_def: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cc_code = cc_def.get("cc")

    for ds_index, ds in enumerate(cc_def.get("d_spreads", [])):
        legs = ds.get("legs", [])
        if len(legs) != 2:
            continue

        pe1 = legs[0].get("pe")
        pe2 = legs[1].get("pe")
        if pe1 is None or pe2 is None:
            continue

        rate_list = ds.get("rates", [])
        rate_val = 0.0
        if rate_list:
            try:
                rate_val = float(rate_list[0].get("val") or 0.0)
            except Exception:
                rate_val = 0.0

        out.append(
            {
                "spread_id": ds.get("spread"),
                "chargeMeth": ds.get("chargeMeth"),
                "pe1": int(pe1),
                "pe2": int(pe2),
                "rate_val": float(rate_val),
                "source": {
                    "section": "cc_defs",
                    "cc": cc_code,
                    "d_spread_index": ds_index,
                },
            }
        )

    return out


# ============================================================
# 5. Exchange-style SPAN: Scanning Risk + Calendar Spread Charge + NOV
# ============================================================

def compute_span_exchange_style(
    positions: pd.DataFrame,
    span: Dict[str, Any],
    contract_index: Dict[Tuple[str, str, int, Optional[float], Optional[str]], Dict[str, Any]],
    lot_size_by_series: Optional[Dict[Tuple[str, int], int]] = None,
    lot_size_by_underlying: Optional[Dict[str, int]] = None,
    assume_rate_is_per_delta_unit: bool = True,

    asof_date: Optional[int] = None,
    index_underlyings: Optional[set] = None,
    expiry_day_separate_scan: bool = True,
    expiry_day_disable_calendar_spreads: bool = True,
    opt_ltp_by_contract_key: Optional[Dict[Tuple[str, int, float, str], float]] = None,

    # NEW:
    is_units: bool = False,
) -> Dict[str, Any]:
    """
    If is_units=True:
      - qty is in units/contracts, NOT lots
      - delta_units = qty * delta_comp
      - NOV leg = price * qty
      - uses contract_index["ra"] already converted to per-unit (see build_span_contract_index(..., is_units=True))

    If is_units=False (legacy):
      - qty is in lots
      - delta_units = qty * lot_size * delta_comp
      - NOV leg = price * lot_size * qty
      - uses contract_index["ra"] per-lot (legacy)
    """
    df = positions.copy()
    opt_ltp_by_contract_key = opt_ltp_by_contract_key or {}

    df["kind"] = df["kind"].astype(str).str.upper()

    if pd.api.types.is_datetime64_any_dtype(df["expiry"]):
        df["expiry_int"] = df["expiry"].dt.strftime("%Y%m%d").astype(int)
    else:
        df["expiry_int"] = df["expiry"].astype(int)

    if "option_type" in df.columns:
        df["opt_type_norm"] = df["option_type"].apply(_norm_opt_type)
    else:
        df["opt_type_norm"] = None

    index_underlyings = index_underlyings or set()
    asof_date_i = int(asof_date) if asof_date is not None else None

    scenario_by_und: Dict[str, List[float]] = defaultdict(lambda: [0.0] * 16)
    scenario_expiring_by_und: Dict[str, List[float]] = defaultdict(lambda: [0.0] * 16)
    scenario_nonexp_by_und: Dict[str, List[float]] = defaultdict(lambda: [0.0] * 16)

    delta_units_by_und_exp: Dict[Tuple[str, int], float] = defaultdict(float)

    per_pos_rows: List[Dict[str, Any]] = []
    debug_per_position: Dict[int, Dict[str, Any]] = {}

    debug_calendar: List[Dict[str, Any]] = []
    debug_nov_legs: List[Dict[str, Any]] = []

    nov_total = 0.0

    for row_idx, row in df.iterrows():
        row_i = int(row_idx)
        und = str(row["underlying"]).strip()
        kind = str(row["kind"]).strip().upper()
        exp = int(row["expiry_int"])
        qty = float(row["qty"])

        strike_val: Optional[float] = None
        opt_type_val: Optional[str] = None

        if kind == "FUT":
            key = (und, "FUT", exp, None, None)
        else:
            try:
                strike_val = float(row.get("strike")) if row.get("strike") is not None else None
            except Exception:
                strike_val = None
            opt_type_val = _norm_opt_type(row.get("option_type"))
            key = (und, "OPT", exp, strike_val, opt_type_val)

        cinfo = contract_index.get(key)

        lean_row = {
            "row_index": row_i,
            "underlying": und,
            "kind": kind,
            "expiry_int": exp,
            "qty": qty,
            "strike": strike_val,
            "option_type": opt_type_val,
            "contract_key": key,
            "matched_span": bool(cinfo),
            "is_units": bool(is_units),
        }

        if not cinfo:
            lean_row["reason"] = "No matching contract in SPAN index"
            per_pos_rows.append(lean_row)
            debug_per_position[row_i] = {
                "matched_span": False,
                "reason": "No matching contract in SPAN index",
                "position": dict(lean_row),
            }
            continue

        lot = _resolve_lot_size(
            und, exp,
            lot_size_by_series=lot_size_by_series,
            lot_size_by_underlying=lot_size_by_underlying,
        )
        lot_mult = 1.0 if is_units else float(lot)

        ra_arr = cinfo.get("ra") or []
        delta_comp = float(cinfo.get("delta_comp", 1.0))
        src = cinfo.get("source")

        inst_meta = cinfo.get("fut_meta") if kind == "FUT" else (cinfo.get("opt_meta") or {})
        c_id = inst_meta.get("cId") if isinstance(inst_meta, dict) else None

        span_price = None
        span_iv = None
        span_delta_field = None
        ra_d_field = None

        if kind == "OPT" and isinstance(inst_meta, dict):
            try:
                span_price = float(inst_meta.get("price") or 0.0)
            except Exception:
                span_price = 0.0
            span_iv = inst_meta.get("iv")
            span_delta_field = inst_meta.get("delta")
            try:
                ra_d_field = float((inst_meta.get("ra") or {}).get("d")) if (inst_meta.get("ra") or {}).get("d") is not None else None
            except Exception:
                ra_d_field = None

        # 1) Per-leg scenario P&L vector
        if len(ra_arr) == 16:
            # RA is already in correct basis (per-lot or per-unit), aligned with qty basis.
            pnl_arr = [-qty * float(v) for v in ra_arr]

            for i in range(16):
                scenario_by_und[und][i] += pnl_arr[i]

            if (
                asof_date_i is not None
                and expiry_day_separate_scan
                and und in index_underlyings
            ):
                bucket = scenario_expiring_by_und if exp == asof_date_i else scenario_nonexp_by_und
                for i in range(16):
                    bucket[und][i] += pnl_arr[i]
        else:
            pnl_arr = [0.0] * 16

        # 2) Delta units (multiplies lot only in legacy mode)
        delta_units = qty * lot_mult * float(delta_comp)
        delta_units_by_und_exp[(und, exp)] += delta_units

        # 3) NOV leg (multiplies lot only in legacy mode)
        nov_leg = 0.0
        if kind == "OPT":
            span_price = float(inst_meta.get("price") or 0.0)

            live_key = (und, exp, float(cinfo.get("strike")), str(cinfo.get("opt_type")))
            live_ltp = opt_ltp_by_contract_key.get(live_key)
            if live_ltp is not None and live_ltp > 0:
                span_price = float(live_ltp)

            nov_leg = float(span_price) * lot_mult * qty
            nov_total += nov_leg

            debug_nov_legs.append(
                {
                    "row_index": row_i,
                    "underlying": und,
                    "expiry_int": exp,
                    "qty": qty,
                    "strike": cinfo.get("strike"),
                    "opt_type": cinfo.get("opt_type"),
                    "lot_size_actual": int(lot),
                    "lot_multiplier_used": float(lot_mult),
                    "span_price": float(span_price or 0.0),
                    "nov_leg": float(nov_leg),
                    "delta_comp_used": float(delta_comp),
                    "delta_field": span_delta_field,
                    "ra_d_field": ra_d_field,
                    "contract_key": key,
                    "span_source": src,
                    "is_units": bool(is_units),
                }
            )

        lean_row.update(
            {
                "lot_size_actual": int(lot),
                "lot_multiplier_used": float(lot_mult),
                "delta_comp_used": float(delta_comp),
                "delta_units_contrib": float(delta_units),
                "nov_leg": float(nov_leg),
                "cId": c_id,
                "span_source": src,
            }
        )
        per_pos_rows.append(lean_row)

        debug_per_position[row_i] = {
            "matched_span": True,
            "position": {
                "underlying": und,
                "kind": kind,
                "expiry_int": exp,
                "qty": qty,
                "strike": strike_val,
                "option_type": opt_type_val,
                "contract_key": key,
                "is_units": bool(is_units),
            },
            "span_picked": {
                "cId": c_id,
                "span_source": src,
                "instrument_meta": inst_meta,
                "ra": list(ra_arr) if ra_arr else None,
                "delta_comp_used": float(delta_comp),
                "span_price_used": float(span_price) if span_price is not None else None,
                "span_iv": span_iv,
                "span_delta_field": span_delta_field,
                "span_ra_d_field": ra_d_field,
            },
            "derived": {
                "lot_size_actual": int(lot),
                "lot_multiplier_used": float(lot_mult),
                "delta_units_contrib": float(delta_units),
                "scenario_pnls_leg": list(pnl_arr),
                "nov_leg": float(nov_leg),
            },
        }

    # --- Scanning risk per underlying ---
    scan_by_und: Dict[str, float] = {}
    active_scen_by_und: Dict[str, int] = {}
    worst_pnl_by_und: Dict[str, float] = {}

    scan_expiring_by_und: Dict[str, float] = {}
    scan_nonexp_by_und: Dict[str, float] = {}
    active_scen_expiring_by_und: Dict[str, int] = {}
    active_scen_nonexp_by_und: Dict[str, int] = {}
    worst_pnl_expiring_by_und: Dict[str, float] = {}
    worst_pnl_nonexp_by_und: Dict[str, float] = {}

    for und, vec_total in scenario_by_und.items():
        worst_idx_total, worst_pnl_total = min(enumerate(vec_total), key=lambda x: x[1])
        active_scen_by_und[und] = int(worst_idx_total + 1)
        worst_pnl_by_und[und] = float(worst_pnl_total)

        if (
            asof_date_i is not None
            and expiry_day_separate_scan
            and und in index_underlyings
        ):
            vec_e = scenario_expiring_by_und.get(und, [0.0] * 16)
            vec_n = scenario_nonexp_by_und.get(und, [0.0] * 16)

            widx_e, wpnl_e = min(enumerate(vec_e), key=lambda x: x[1])
            widx_n, wpnl_n = min(enumerate(vec_n), key=lambda x: x[1])

            scan_e = max(0.0, -float(wpnl_e))
            scan_n = max(0.0, -float(wpnl_n))

            scan_by_und[und] = float(scan_e + scan_n)

            scan_expiring_by_und[und] = float(scan_e)
            scan_nonexp_by_und[und] = float(scan_n)
            active_scen_expiring_by_und[und] = int(widx_e + 1)
            active_scen_nonexp_by_und[und] = int(widx_n + 1)
            worst_pnl_expiring_by_und[und] = float(wpnl_e)
            worst_pnl_nonexp_by_und[und] = float(wpnl_n)
        else:
            scan_by_und[und] = max(0.0, -float(worst_pnl_total))

    # --- Calendar spread charge per underlying using cc_defs.d_spreads ---
    cc_index = build_ccdef_index(span)
    cal_charge_by_und: Dict[str, float] = defaultdict(float)

    for und in scan_by_und.keys():
        cc_def = cc_index.get(und)
        if not cc_def:
            continue

        cal_spreads = extract_calendar_spreads(cc_def)
        if not cal_spreads:
            continue

        remaining: Dict[int, float] = {}
        for (u, e), units in delta_units_by_und_exp.items():
            if u != und:
                continue
            if abs(units) > 1e-12:
                remaining[int(e)] = float(units)

        for sp in cal_spreads:
            pe1 = int(sp["pe1"])
            pe2 = int(sp["pe2"])
            rate_val = float(sp.get("rate_val") or 0.0)

            if (
                asof_date_i is not None
                and expiry_day_disable_calendar_spreads
                and und in index_underlyings
                and (pe1 == asof_date_i or pe2 == asof_date_i)
            ):
                continue

            u1 = float(remaining.get(pe1, 0.0))
            u2 = float(remaining.get(pe2, 0.0))
            if abs(u1) < 1e-12 or abs(u2) < 1e-12:
                continue
            if u1 * u2 >= 0.0:
                continue

            matched_units = min(abs(u1), abs(u2))
            if matched_units <= 1e-12:
                continue

            if not assume_rate_is_per_delta_unit:
                raise ValueError(
                    "Calendar rate conversion not implemented. "
                    "Set assume_rate_is_per_delta_unit=True or implement conversion."
                )

            charge = rate_val * matched_units
            if charge <= 0.0:
                continue

            cal_charge_by_und[und] += charge

            remaining[pe1] = u1 - (matched_units if u1 > 0 else -matched_units)
            remaining[pe2] = u2 - (matched_units if u2 > 0 else -matched_units)

            debug_calendar.append(
                {
                    "underlying": und,
                    "pe1": pe1,
                    "pe2": pe2,
                    "rate_val": float(rate_val),
                    "matched_units": float(matched_units),
                    "charge": float(charge),
                    "net_units_pe1_before": float(u1),
                    "net_units_pe2_before": float(u2),
                    "remaining_units_pe1_after": float(remaining[pe1]),
                    "remaining_units_pe2_after": float(remaining[pe2]),
                    "spread_id": sp.get("spread_id"),
                    "chargeMeth": sp.get("chargeMeth"),
                    "span_source": sp.get("source"),
                    "is_units": bool(is_units),
                }
            )

    per_underlying: Dict[str, Dict[str, Any]] = {}
    scan_total = 0.0
    cal_total = 0.0

    for und, scan in scan_by_und.items():
        cal = float(cal_charge_by_und.get(und, 0.0))
        scan_total += float(scan)
        cal_total += float(cal)

        row = {
            "scan_risk": float(scan),
            "calendar_spread_charge": float(cal),
            "span_risk_requirement_before_nov": float(scan + cal),

            "active_scenario": int(active_scen_by_und.get(und, 1)),
            "worst_pnl": float(worst_pnl_by_und.get(und, 0.0)),
            "scenario_pnls": scenario_by_und.get(und, [0.0] * 16),

            "is_units": bool(is_units),
        }

        if (
            asof_date_i is not None
            and und in index_underlyings
            and expiry_day_separate_scan
        ):
            row.update(
                {
                    "scan_risk_expiring": float(scan_expiring_by_und.get(und, 0.0)),
                    "scan_risk_nonexpiring": float(scan_nonexp_by_und.get(und, 0.0)),
                    "active_scenario_expiring": int(active_scen_expiring_by_und.get(und, 1)),
                    "active_scenario_nonexpiring": int(active_scen_nonexp_by_und.get(und, 1)),
                    "worst_pnl_expiring": float(worst_pnl_expiring_by_und.get(und, 0.0)),
                    "worst_pnl_nonexpiring": float(worst_pnl_nonexp_by_und.get(und, 0.0)),
                    "scenario_pnls_expiring": scenario_expiring_by_und.get(und, [0.0] * 16),
                    "scenario_pnls_nonexpiring": scenario_nonexp_by_und.get(und, [0.0] * 16),
                }
            )

        per_underlying[und] = row

    span_risk_before_nov = float(scan_total + cal_total)

    net_premium_payable = max(0.0, float(nov_total))
    net_premium_receivable = max(0.0, -float(nov_total))
    span_required = span_risk_before_nov

    per_position_df = pd.DataFrame(per_pos_rows)

    return {
        "per_position": per_position_df,
        "per_underlying": per_underlying,
        "totals": {
            "scan_risk_total": float(scan_total),
            "calendar_spread_charge_total": float(cal_total),
            "span_risk_requirement_before_nov": float(span_risk_before_nov),
            "net_option_value": float(nov_total),
            "net_premium_payable": float(net_premium_payable),
            "net_premium_receivable": float(net_premium_receivable),
            "span_requirement_excl_premium": float(span_required),
            "is_units": bool(is_units),
        },
        "debug_calendar": debug_calendar,
        "debug_nov_legs": debug_nov_legs,
        "delta_units_by_und_exp": dict(delta_units_by_und_exp),
        "debug per position": debug_per_position,
        "debug_per_position": debug_per_position,
    }


# ============================================================
# 6. Exposure (ELM) margin
# ============================================================

def _is_deep_otm(
    opt_type: Optional[str],
    strike: Optional[float],
    prev_close: Optional[float],
    threshold: float = 0.10,
) -> bool:
    if opt_type is None or strike is None or prev_close is None:
        return False
    if prev_close <= 0:
        return False

    ot = _norm_opt_type(opt_type)
    k = float(strike)
    s = float(prev_close)

    if ot == "PE":
        return k <= s * (1.0 - threshold)
    if ot == "CE":
        return k >= s * (1.0 + threshold)
    return False


def compute_exposure_margin(
    positions: pd.DataFrame,
    underlying_price: Dict[str, float],
    lot_size_by_series: Optional[Dict[Tuple[str, int], int]] = None,
    lot_size_by_underlying: Optional[Dict[str, int]] = None,
    exposure_pct_by_underlying: Optional[Dict[str, float]] = None,
    index_underlyings: Optional[set] = None,
    calendar_far_month_notional_fraction: float = 1.0 / 3.0,
    calendar_fraction_by_underlying: Optional[Dict[str, float]] = None,
    asof_date: Optional[int] = None,
    fut_price_by_underlying_expiry: Optional[Dict[Tuple[str, int], float]] = None,

    expiry_day_addl_elm_index_opt_pct: float = 0.02,
    expiry_day_addl_elm_only_for_expiring_series: bool = True,

    prev_close_by_underlying: Optional[Dict[str, float]] = None,
    deep_otm_threshold: float = 0.10,
    deep_otm_exposure_pct: float = 0.03,

    # NEW:
    is_units: bool = False,
) -> Dict[str, Any]:
    """
    If is_units=True:
      notional_abs = abs(qty) * underlying_px
      (no lot multiplier anywhere)

    If is_units=False (legacy):
      notional_abs = abs(qty) * lot_size * underlying_px
    """
    df = positions.copy()
    df["kind"] = df["kind"].astype(str).str.upper()

    mask = (df["kind"] == "FUT") | ((df["kind"] == "OPT") & (df["qty"] < 0))
    df = df[mask].copy()

    if df.empty:
        return {
            "per_position": df.assign(
                underlying_px=0.0,
                lot_size=0.0,
                exposure_pct=0.0,
                notional_abs=0.0,
                exposure_margin=0.0,
            ),
            "totals": {"exposure_total": 0.0, "is_units": bool(is_units)},
        }

    if pd.api.types.is_datetime64_any_dtype(df["expiry"]):
        df["expiry_int"] = df["expiry"].dt.strftime("%Y%m%d").astype(int)
    else:
        df["expiry_int"] = df["expiry"].astype(int)

    df["underlying_px"] = df["underlying"].map(underlying_price).astype(float)

    lot_sizes: List[float] = []
    for _, row in df.iterrows():
        und = str(row["underlying"])
        exp = int(row["expiry_int"])
        lot = _resolve_lot_size(
            und,
            exp,
            lot_size_by_series=lot_size_by_series,
            lot_size_by_underlying=lot_size_by_underlying,
        )
        lot_sizes.append(float(lot))
    df["lot_size"] = lot_sizes
    df["lot_multiplier_used"] = 1.0 if is_units else df["lot_size"]

    exposure_pct_by_underlying = exposure_pct_by_underlying or {}
    df["exposure_pct"] = df["underlying"].map(exposure_pct_by_underlying).fillna(0.0).astype(float)

    prev_close_by_underlying = prev_close_by_underlying or {}
    df["exposure_pct_base"] = df["exposure_pct"]

    if "strike" in df.columns and "option_type" in df.columns:
        df["prev_close_px"] = df["underlying"].map(prev_close_by_underlying).astype(float)

        m_short_opt = (df["kind"] == "OPT") & (df["qty"] < 0)

        def _deep_otm_row(r) -> bool:
            try:
                return _is_deep_otm(
                    opt_type=r.get("option_type"),
                    strike=float(r.get("strike")) if r.get("strike") is not None else None,
                    prev_close=float(r.get("prev_close_px")) if r.get("prev_close_px") is not None else None,
                    threshold=float(deep_otm_threshold),
                )
            except Exception:
                return False

        df["is_deep_otm"] = False
        df.loc[m_short_opt, "is_deep_otm"] = df.loc[m_short_opt].apply(_deep_otm_row, axis=1)

        df.loc[m_short_opt & (df["is_deep_otm"]), "exposure_pct"] = df.loc[
            m_short_opt & (df["is_deep_otm"]), "exposure_pct"
        ].apply(lambda x: max(float(x), float(deep_otm_exposure_pct)))
    else:
        df["prev_close_px"] = 0.0
        df["is_deep_otm"] = False

    if asof_date is not None and float(expiry_day_addl_elm_index_opt_pct) > 0:
        asof_i = int(asof_date)
        idx_set = index_underlyings or set()

        m = (
            (df["kind"] == "OPT")
            & (df["qty"] < 0)
            & (df["underlying"].astype(str).isin(idx_set))
        )
        if expiry_day_addl_elm_only_for_expiring_series:
            m = m & (df["expiry_int"] == asof_i)

        df["exposure_pct_base"] = df["exposure_pct"]
        df.loc[m, "exposure_pct"] = df.loc[m, "exposure_pct"] + float(expiry_day_addl_elm_index_opt_pct)

    # Raw exposure (no calendar benefit)
    df["notional_abs"] = df["qty"].abs() * df["lot_multiplier_used"] * df["underlying_px"]
    df["exposure_margin_raw"] = df["notional_abs"] * df["exposure_pct"]
    df["exposure_margin"] = df["exposure_margin_raw"].copy()

    index_underlyings = index_underlyings or set()
    calendar_fraction_by_underlying = calendar_fraction_by_underlying or {}
    fut_price_by_underlying_expiry = fut_price_by_underlying_expiry or {}

    fut_mask = df["kind"] == "FUT"
    if fut_mask.any():
        df_fut = df[fut_mask].copy()

        raw_fut_by_und: Dict[Any, float] = df_fut.groupby("underlying")["exposure_margin_raw"].sum().to_dict()
        new_fut_by_und: Dict[Any, float] = {}

        for und, df_u in df_fut.groupby("underlying"):
            und_str = str(und)
            raw_total_und = float(raw_fut_by_und.get(und, 0.0))

            if und_str not in index_underlyings or raw_total_und <= 0.0:
                new_fut_by_und[und] = raw_total_und
                continue

            grouped = (
                df_u.groupby("expiry_int", as_index=False)
                .agg(
                    net_qty=("qty", "sum"),
                    lot_size=("lot_size", "first"),
                    underlying_px=("underlying_px", "first"),
                    exposure_pct=("exposure_pct", "first"),
                )
            )

            if grouped.shape[0] <= 1:
                new_fut_by_und[und] = raw_total_und
                continue

            remaining_qty: Dict[int, float] = {}
            lot_by_exp: Dict[int, float] = {}
            px_by_exp: Dict[int, float] = {}

            for _, row_g in grouped.iterrows():
                exp_i = int(row_g["expiry_int"])
                q_i = float(row_g["net_qty"])
                if abs(q_i) < 1e-12:
                    continue
                remaining_qty[exp_i] = q_i
                lot_by_exp[exp_i] = float(row_g["lot_size"])
                px_by_exp[exp_i] = float(
                    fut_price_by_underlying_expiry.get((und_str, exp_i), row_g["underlying_px"])
                )

            if not remaining_qty:
                new_fut_by_und[und] = 0.0
                continue

            signs = {1 if q > 0 else -1 for q in remaining_qty.values()}
            if len(signs) == 1:
                new_fut_by_und[und] = raw_total_und
                continue

            exposure_pct = float(grouped["exposure_pct"].iloc[0])
            frac = float(calendar_fraction_by_underlying.get(und_str, calendar_far_month_notional_fraction))
            calendar_exposure_pct = exposure_pct * frac

            exposure_calendar_und = 0.0
            expiries_sorted = sorted(remaining_qty.keys())

            for i, e1 in enumerate(expiries_sorted):
                q1 = remaining_qty.get(e1, 0.0)
                if abs(q1) < 1e-12:
                    continue

                for j in range(i + 1, len(expiries_sorted)):
                    e2 = expiries_sorted[j]
                    q2 = remaining_qty.get(e2, 0.0)
                    if abs(q2) < 1e-12:
                        continue
                    if q1 * q2 >= 0:
                        continue

                    if asof_date is not None:
                        near_exp = e1 if e1 < e2 else e2
                        if near_exp == int(asof_date):
                            continue

                    hedged_qty = min(abs(q1), abs(q2))
                    if hedged_qty <= 0.0:
                        continue

                    far_exp = e1 if e1 > e2 else e2
                    lot_far = lot_by_exp[far_exp]
                    px_far = px_by_exp[far_exp]
                    lot_mult_far = 1.0 if is_units else float(lot_far)

                    exposure_pair = calendar_exposure_pct * hedged_qty * lot_mult_far * px_far
                    exposure_calendar_und += exposure_pair

                    if q1 > 0:
                        q1 -= hedged_qty
                    else:
                        q1 += hedged_qty

                    if q2 > 0:
                        q2 -= hedged_qty
                    else:
                        q2 += hedged_qty

                    remaining_qty[e1] = q1
                    remaining_qty[e2] = q2

                    if abs(q1) < 1e-12:
                        break

            exposure_naked_und = 0.0
            for exp_i, q_i in remaining_qty.items():
                if abs(q_i) < 1e-12:
                    continue
                lot_i = lot_by_exp[exp_i]
                px_i = px_by_exp[exp_i]
                lot_mult_i = 1.0 if is_units else float(lot_i)
                exposure_naked_und += exposure_pct * abs(q_i) * lot_mult_i * px_i

            exposure_total_und = min(exposure_calendar_und + exposure_naked_und, raw_total_und)
            new_fut_by_und[und] = float(exposure_total_und)

        for und, raw_total_und in raw_fut_by_und.items():
            new_total_und = float(new_fut_by_und.get(und, raw_total_und))
            ratio = new_total_und / raw_total_und if raw_total_und > 0 else 1.0
            mask_u = fut_mask & (df["underlying"] == und)
            df.loc[mask_u, "exposure_margin"] = df.loc[mask_u, "exposure_margin_raw"] * ratio

    total_exposure = float(df["exposure_margin"].sum())

    return {
        "per_position": df.drop(columns=["exposure_margin_raw"]),
        "totals": {"exposure_total": total_exposure, "is_units": bool(is_units)},
    }


# ============================================================
# 7. Short Option Minimum Charge (SOMC)
# ============================================================

def compute_short_option_min_charge(
    positions: pd.DataFrame,
    underlying_price: Dict[str, float],
    lot_size_by_series: Optional[Dict[Tuple[str, int], int]] = None,
    lot_size_by_underlying: Optional[Dict[str, int]] = None,
    index_underlyings: Optional[set] = None,
    somc_pct_index: float = 0.10,
    somc_pct_stock: float = 0.20,
    # NEW:
    is_units: bool = False,
) -> Dict[str, Any]:
    """
    If is_units=True:
      notional_abs = net_short_qty * underlying_px

    If is_units=False (legacy):
      notional_abs = net_short_qty * lot_size * underlying_px
    """
    df = positions.copy()

    if pd.api.types.is_datetime64_any_dtype(df["expiry"]):
        df["expiry_int"] = df["expiry"].dt.strftime("%Y%m%d").astype(int)
    else:
        df["expiry_int"] = df["expiry"].astype(int)

    mask_opts = df["kind"].str.upper() == "OPT"
    df = df[mask_opts].copy()
    if df.empty:
        return {
            "per_position": df.assign(
                underlying_px=0.0,
                lot_size=0.0,
                notional_abs=0.0,
                somc_leg=0.0,
            ),
            "totals": {"somc_total": 0.0, "is_units": bool(is_units)},
        }

    df["underlying_px"] = df["underlying"].map(underlying_price).astype(float)

    lot_sizes: List[float] = []
    for _, row in df.iterrows():
        und = str(row["underlying"])
        exp = int(row["expiry_int"])
        lot = _resolve_lot_size(
            und,
            exp,
            lot_size_by_series=lot_size_by_series,
            lot_size_by_underlying=lot_size_by_underlying,
        )
        lot_sizes.append(float(lot))
    df["lot_size"] = lot_sizes
    df["lot_multiplier_used"] = 1.0 if is_units else df["lot_size"]

    index_underlyings = index_underlyings or set()

    def _somc_pct(und: str) -> float:
        return somc_pct_index if und in index_underlyings else somc_pct_stock

    df["somc_pct"] = df["underlying"].apply(_somc_pct).astype(float)

    grouped = (
        df.groupby(
            ["underlying", "expiry_int", "option_type", "underlying_px", "lot_size", "lot_multiplier_used", "somc_pct"],
            as_index=False,
        )["qty"]
        .sum()
        .rename(columns={"qty": "net_qty"})
    )

    grouped["net_short_qty"] = grouped["net_qty"].apply(lambda q: -q if q < 0 else 0.0)
    grouped["notional_abs"] = grouped["net_short_qty"] * grouped["lot_multiplier_used"] * grouped["underlying_px"]
    grouped["somc_leg"] = grouped["notional_abs"] * grouped["somc_pct"]

    net_opt_by_und = df.groupby("underlying")["qty"].sum().astype(float).to_dict()

    def _adjust_somc(row):
        und = row["underlying"]
        net_total = float(net_opt_by_und.get(und, 0.0))
        if net_total >= -1e-9:
            return 0.0
        return float(row["somc_leg"])

    grouped["somc_leg"] = grouped.apply(_adjust_somc, axis=1)

    somc_total = float(grouped["somc_leg"].sum())

    return {
        "per_position": grouped,
        "totals": {"somc_total": somc_total, "is_units": bool(is_units)},
    }


# ============================================================
# 8. Full portfolio margin: Exchange SPAN + SOMC + Exposure
# ============================================================

def compute_portfolio_margin_with_exposure(
    positions: pd.DataFrame,
    span: Dict[str, Any],
    underlying_price: Dict[str, float],
    lot_size_by_series: Optional[Dict[Tuple[str, int], int]] = None,
    lot_size_by_underlying: Optional[Dict[str, int]] = None,
    futures_ra_is_per_unit: bool = True,
    options_ra_is_per_unit: bool = True,
    exposure_pct_by_underlying: Optional[Dict[str, float]] = None,
    index_underlyings: Optional[set] = None,
    somc_pct_index: float = 0.10,
    somc_pct_stock: float = 0.20,
    calendar_far_month_notional_fraction: float = 1.0 / 3.0,
    calendar_fraction_by_underlying: Optional[Dict[str, float]] = None,
    asof_date: Optional[int] = None,
    fut_price_by_underlying_expiry: Optional[Dict[Tuple[str, int], float]] = None,
    opt_ltp_by_contract_key=None,
    prev_close_by_underlying: Optional[Dict[str, float]] = None,

    # NEW:
    is_units: bool = False,
) -> Dict[str, Any]:
    """
    If is_units=True, you pass qty as units/contracts everywhere.
    If is_units=False, behavior is identical to your original script (qty as lots).
    """
    pos = positions.copy()
    pos["kind"] = pos["kind"].astype(str).str.upper()

    only_options = (pos["kind"] == "OPT").all()
    all_long_or_flat = (pos["qty"] >= 0).all()
    if only_options and all_long_or_flat:
        return {
            "span": None,
            "somc": None,
            "exposure": None,
            "contract_index": None,
            "totals": {
                "scan_risk_total": 0.0,
                "calendar_spread_charge_total": 0.0,
                "scan_cal_total": 0.0,
                "somc_total": 0.0,
                "scan_cal_floored": 0.0,
                "net_option_value": 0.0,
                "premium_payable": 0.0,
                "premium_receivable": 0.0,
                "span_broker_style": 0.0,
                "exposure_total": 0.0,
                "grand_total_broker_style": 0.0,
                "funds_required_cash": 0.0,
                "is_units": bool(is_units),
            },
        }

    # 1) Contract index (RA basis aligned to qty basis)
    contract_index = build_span_contract_index(
        span,
        lot_size_by_series=lot_size_by_series,
        lot_size_by_underlying=lot_size_by_underlying,
        futures_ra_is_per_unit=futures_ra_is_per_unit,
        options_ra_is_per_unit=options_ra_is_per_unit,
        is_units=is_units,
    )

    # 2) SPAN
    span_res = compute_span_exchange_style(
        positions=positions,
        span=span,
        contract_index=contract_index,
        lot_size_by_series=lot_size_by_series,
        lot_size_by_underlying=lot_size_by_underlying,
        assume_rate_is_per_delta_unit=True,

        asof_date=asof_date,
        index_underlyings=index_underlyings,
        expiry_day_separate_scan=True,
        expiry_day_disable_calendar_spreads=True,
        opt_ltp_by_contract_key=opt_ltp_by_contract_key,

        is_units=is_units,
    )
    span_tot = span_res["totals"]

    scan_total = float(span_tot["scan_risk_total"])
    cal_total = float(span_tot["calendar_spread_charge_total"])
    scan_cal_total = float(span_tot["span_risk_requirement_before_nov"])
    nov_total = float(span_tot.get("net_option_value", 0.0))

    premium_payable = max(0.0, nov_total)
    premium_receivable = max(0.0, -nov_total)

    # 3) SOMC
    somc_res = compute_short_option_min_charge(
        positions=positions,
        underlying_price=underlying_price,
        lot_size_by_series=lot_size_by_series,
        lot_size_by_underlying=lot_size_by_underlying,
        index_underlyings=index_underlyings,
        somc_pct_index=somc_pct_index,
        somc_pct_stock=somc_pct_stock,
        is_units=is_units,
    )
    somc_total = float(somc_res["totals"]["somc_total"])
    scan_cal_floored = max(scan_cal_total, somc_total)

    # 4) Broker-style SPAN applies NOV as subtraction with sign
    span_broker_style = max(0.0, scan_cal_floored - nov_total)

    # 5) Exposure (ELM)
    exp_res = compute_exposure_margin(
        positions=positions,
        underlying_price=underlying_price,
        lot_size_by_series=lot_size_by_series,
        lot_size_by_underlying=lot_size_by_underlying,
        exposure_pct_by_underlying=exposure_pct_by_underlying,
        index_underlyings=index_underlyings,
        calendar_far_month_notional_fraction=calendar_far_month_notional_fraction,
        calendar_fraction_by_underlying=calendar_fraction_by_underlying,
        asof_date=asof_date,
        fut_price_by_underlying_expiry=fut_price_by_underlying_expiry,

        expiry_day_addl_elm_index_opt_pct=0.02,
        expiry_day_addl_elm_only_for_expiring_series=True,
        prev_close_by_underlying=prev_close_by_underlying,

        is_units=is_units,
    )
    exposure_total = float(exp_res["totals"]["exposure_total"])

    grand_total_broker_style = float(span_broker_style + exposure_total)
    funds_required_cash = float(grand_total_broker_style + premium_payable - premium_receivable)

    return {
        "span": span_res,
        "somc": somc_res,
        "exposure": exp_res,
        "contract_index": contract_index,
        "totals": {
            "scan_risk_total": scan_total,
            "calendar_spread_charge_total": cal_total,
            "scan_cal_total": scan_cal_total,
            "somc_total": somc_total,
            "scan_cal_floored": scan_cal_floored,

            "net_option_value": nov_total,
            "premium_payable": premium_payable,
            "premium_receivable": premium_receivable,

            "span_broker_style": span_broker_style,
            "exposure_total": exposure_total,

            "grand_total_broker_style": grand_total_broker_style,
            "funds_required_cash": funds_required_cash,

            "is_units": bool(is_units),
        },
    }


#===================================================================================
#===================================================================================
##helper funtions to read zerodha prices and map them to symbolnames
import pandas as pd
from typing import Dict, Tuple, Any, Optional

def _expiry_to_int(exp) -> Optional[int]:
    # master expiry looks like "27-01-2026" in your snippet
    ts = pd.to_datetime(exp, errors="coerce", dayfirst=True)
    if pd.isna(ts):
        return None
    return int(ts.strftime("%Y%m%d"))

def build_fut_price_map_from_redis(last_price: Dict[str, str], master: pd.DataFrame):
    """
    Returns: {(UNDERLYING, EXPIRY_YYYYMMDD): fut_ltp}
    last_price keys are Zerodha tradingsymbols.
    """
    if not last_price:
        return {}

    m = master.copy()
    m["instrument_type"] = m["instrument_type"].astype(str).str.upper()
    m = m[m["instrument_type"].eq("FUT")].copy()

    # keep only rows that exist in redis hash
    m = m[m["tradingsymbol"].isin(last_price.keys())].copy()
    if m.empty:
        return {}

    m["ul"] = m["name"].astype(str).str.upper()
    m["expiry_int"] = m["expiry"].map(_expiry_to_int)

    out = {}
    for _, r in m.dropna(subset=["ul", "expiry_int"]).iterrows():
        ts = r["tradingsymbol"]
        try:
            out[(r["ul"], int(r["expiry_int"]))] = float(last_price[ts])
        except Exception:
            continue
    return out

def build_opt_ltp_map_from_redis(last_price: Dict[str, str], master: pd.DataFrame):
    """
    Returns: {(UNDERLYING, EXPIRY_YYYYMMDD, STRIKE, OPT_TYPE): opt_ltp}
    OPT_TYPE is "CE"/"PE"
    """
    if not last_price:
        return {}

    m = master.copy()
    m["instrument_type"] = m["instrument_type"].astype(str).str.upper()
    m = m[m["instrument_type"].isin(["CE", "PE"])].copy()

    m = m[m["tradingsymbol"].isin(last_price.keys())].copy()
    if m.empty:
        return {}

    m["ul"] = m["name"].astype(str).str.upper()
    m["expiry_int"] = m["expiry"].map(_expiry_to_int)
    m["strike"] = pd.to_numeric(m["strike"], errors="coerce")
    m["opt_type"] = m["instrument_type"].astype(str).str.upper()

    out = {}
    for _, r in m.dropna(subset=["ul", "expiry_int", "strike", "opt_type"]).iterrows():
        ts = r["tradingsymbol"]
        try:
            out[(r["ul"], int(r["expiry_int"]), float(r["strike"]), r["opt_type"])] = float(last_price[ts])
        except Exception:
            continue
    return out

def build_underlying_spot_from_redis(last_price: Dict[str, str], fut_map: Dict[Tuple[str,int], float]):
    """
    Try true spot keys first (NIFTY/BANKNIFTY/SENSEX/etc), else fallback to nearest-expiry FUT.
    """
    spot = {}
    for ul in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY", "BANKEX"]:
        if ul in last_price:
            try:
                spot[ul] = float(last_price[ul])
            except Exception:
                pass

    # FUT fallback: nearest expiry
    if fut_map:
        by_ul = {}
        for (ul, exp), px in fut_map.items():
            by_ul.setdefault(ul, []).append((exp, px))
        for ul, rows in by_ul.items():
            if ul in spot:
                continue
            rows.sort(key=lambda x: x[0])
            spot[ul] = float(rows[0][1])
    return spot


#helper functions
def read_redis_prices(r, key="dhan_prices") -> dict[str, float]:
    raw = r.hgetall(key)
    out = {}
    for k, v in raw.items():
        sym = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
        try:
            ltp = float(v.decode() if isinstance(v, (bytes, bytearray)) else v)
        except Exception:
            continue
        if ltp > 0:          # ignore 0.00 ticks
            out[sym] = ltp
    return out

def build_live_price_inputs(prices: dict[str, float], master: pd.DataFrame):
    # Make sure master has consistent dtypes
    m = master.copy()
    if "expiry" in m.columns:
        m["expiry"] = pd.to_datetime(m["expiry"], errors="coerce")

    # 1) underlying/index spot
    underlying_price_live = {}
    for idx_name in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY", "BANKEX"]:
        if idx_name in prices:
            underlying_price_live[idx_name] = float(prices[idx_name])

    # 2) futures by (underlying, expiry_int)
    fut_price_by_underlying_expiry = {}
    fut_rows = m[m["instrument_type"].eq("FUT") & m["tradingsymbol"].isin(prices.keys())]
    for _, row in fut_rows.iterrows():
        und = str(row["name"]).upper()
        exp = row["expiry"]
        if pd.isna(exp):
            continue
        exp_i = int(pd.Timestamp(exp).strftime("%Y%m%d"))
        fut_price_by_underlying_expiry[(und, exp_i)] = float(prices[row["tradingsymbol"]])

    # 3) options by (underlying, expiry_int, strike, opt_type)
    opt_ltp_by_contract_key = {}
    opt_rows = m[m["instrument_type"].isin(["CE", "PE"]) & m["tradingsymbol"].isin(prices.keys())]
    for _, row in opt_rows.iterrows():
        und = str(row["name"]).upper()
        exp = row["expiry"]
        if pd.isna(exp):
            continue
        exp_i = int(pd.Timestamp(exp).strftime("%Y%m%d"))
        strike = float(row["strike"])
        opt_type = str(row["instrument_type"]).upper()  # CE/PE
        opt_ltp_by_contract_key[(und, exp_i, strike, opt_type)] = float(prices[row["tradingsymbol"]])

    return underlying_price_live, fut_price_by_underlying_expiry, opt_ltp_by_contract_key

def build_live_prices_inputs_zerodha(prices: dict[str,float],master:pd.DataFrame):

    fut_px_map = build_fut_price_map_from_redis(last_price=prices, master=master)
    underlying_price_live = build_underlying_spot_from_redis(last_price=prices, fut_map=fut_px_map)
    opt_ltp = build_opt_ltp_map_from_redis(last_price=prices, master=master)

    return underlying_price_live,fut_px_map,opt_ltp

#Margin breakdown function 

def margin_breakdown(
    res: Dict[str, Any],
    *,
    include_tables: bool = True,
    round_to: Optional[int] = 2,
) -> Dict[str, Any]:
    """
    Build a clean, audit-friendly breakdown of your portfolio margin result.

    Expects the output dict from compute_portfolio_margin_with_exposure(...).

    Returns a dict with:
      - "components": ordered list of line-items (name, value, formula, source)
      - "totals": the key totals (same numbers, normalized)
      - Optional tables:
          * "span_per_underlying"
          * "span_per_position"
          * "exposure_per_position"
          * "somc_per_position"
          * "span_debug_calendar"
          * "span_debug_nov_legs"
    """

    def _r(x: Any) -> Any:
        if round_to is None:
            return x
        try:
            return round(float(x), round_to)
        except Exception:
            return x

    # ---- pull sections safely ----
    totals = (res or {}).get("totals", {}) or {}
    span_res = (res or {}).get("span")
    somc_res = (res or {}).get("somc")
    exposure_res = (res or {}).get("exposure")

    # ---- normalize key totals ----
    scan_risk_total = float(totals.get("scan_risk_total", 0.0) or 0.0)
    calendar_total = float(totals.get("calendar_spread_charge_total", 0.0) or 0.0)
    scan_cal_total = float(totals.get("scan_cal_total", 0.0) or 0.0)
    somc_total = float(totals.get("somc_total", 0.0) or 0.0)
    scan_cal_floored = float(totals.get("scan_cal_floored", 0.0) or 0.0)

    net_option_value = float(totals.get("net_option_value", 0.0) or 0.0)  # signed
    premium_payable = float(totals.get("premium_payable", 0.0) or 0.0)
    premium_receivable = float(totals.get("premium_receivable", 0.0) or 0.0)

    span_broker_style = float(totals.get("span_broker_style", 0.0) or 0.0)
    exposure_total = float(totals.get("exposure_total", 0.0) or 0.0)
    grand_total_broker_style = float(totals.get("grand_total_broker_style", 0.0) or 0.0)
    funds_required_cash = float(totals.get("funds_required_cash", 0.0) or 0.0)

    # ---- line-item components (ordered) ----
    components = [
        {
            "name": "Scan Risk (Total)",
            "value": _r(scan_risk_total),
            "formula": "Worst loss across 16 scenarios (per underlying) summed",
            "source": "SPAN.scanning_risk",
        },
        {
            "name": "Calendar Spread Charge (Total)",
            "value": _r(calendar_total),
            "formula": "Matched opposite delta-units across expiries × calendar rate",
            "source": "SPAN.calendar_spread_charge",
        },
        {
            "name": "Scan + Calendar (Before SOMC & NOV)",
            "value": _r(scan_cal_total),
            "formula": "scan_risk_total + calendar_spread_charge_total",
            "source": "Computed",
        },
        {
            "name": "SOMC (Short Option Minimum Charge)",
            "value": _r(somc_total),
            "formula": "Min charge on net short options (rules per index/stock)",
            "source": "SOMC",
        },
        {
            "name": "Scan+Calendar Floored by SOMC",
            "value": _r(scan_cal_floored),
            "formula": "max(scan_cal_total, somc_total)",
            "source": "Computed",
        },
        {
            "name": "Net Option Value (NOV) (Signed)",
            "value": _r(net_option_value),
            "formula": "Σ(option_price_used × lot_size × qty)",
            "source": "SPAN.NOV (with live LTP override if provided)",
        },
        {
            "name": "SPAN (Broker Style) After NOV",
            "value": _r(span_broker_style),
            "formula": "max(0, scan_cal_floored - net_option_value)",
            "source": "Computed",
        },
        {
            "name": "Exposure / ELM (Total)",
            "value": _r(exposure_total),
            "formula": "ELM on FUT + short OPT; includes calendar benefit logic for index FUT",
            "source": "Exposure/ELM",
        },
        {
            "name": "Grand Total (Broker Style)",
            "value": _r(grand_total_broker_style),
            "formula": "span_broker_style + exposure_total",
            "source": "Computed",
        },
        {
            "name": "Premium Payable (Cash)",
            "value": _r(premium_payable),
            "formula": "max(0, net_option_value)",
            "source": "Computed",
        },
        {
            "name": "Premium Receivable (Cash)",
            "value": _r(premium_receivable),
            "formula": "max(0, -net_option_value)",
            "source": "Computed",
        },
        {
            "name": "Funds Required (Cash Style)",
            "value": _r(funds_required_cash),
            "formula": "grand_total_broker_style + premium_payable - premium_receivable",
            "source": "Computed",
        },
    ]

    out: Dict[str, Any] = {
        "components": components,
        "totals": {k: _r(v) for k, v in totals.items()},
    }

    # ---- optional tables for audit ----
    if include_tables:
        # SPAN tables
        if isinstance(span_res, dict):
            out["span_per_underlying"] = span_res.get("per_underlying")
            out["span_per_position"] = span_res.get("per_position")
            out["span_debug_calendar"] = span_res.get("debug_calendar")
            out["span_debug_nov_legs"] = span_res.get("debug_nov_legs")

        # Exposure tables
        if isinstance(exposure_res, dict):
            out["exposure_per_position"] = exposure_res.get("per_position")
            out["exposure_totals"] = exposure_res.get("totals")

        # SOMC tables
        if isinstance(somc_res, dict):
            out["somc_per_position"] = somc_res.get("per_position")
            out["somc_totals"] = somc_res.get("totals")

    return out


def margin_breakdown_table(breakdown: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert margin_breakdown(...) output into a single easy-to-read table.
    """
    rows = []
    for c in breakdown.get("components", []):
        rows.append(
            {
                "Component": c.get("name"),
                "Value": c.get("value"),
                "Formula": c.get("formula"),
                "Source": c.get("source"),
            }
        )
    return pd.DataFrame(rows)


#margin breakdown text file
def _fmt_money(x: float, nd: int = 2) -> str:
    # Indian-style commas are nice, but keep it simple/portable:
    return f"{x:,.{nd}f}"


def margin_calc_trace(
    res: Dict[str, Any],
    *,
    nd: int = 2,
    include_per_underlying: bool = True,
    include_expiry_day_split_if_present: bool = True,
    show_zero_underlyings: bool = False,
) -> str:
    """
    Returns a compact multi-line “calculation trace” for broker-style margin.

    Uses:
      - res["totals"] (from compute_portfolio_margin_with_exposure)
      - optional: res["span"]["per_underlying"] for drilldown

    Notes:
      - NOV is signed: +NOV reduces margin (premium receivable), -NOV increases margin (premium payable)
      - Broker-style span = max(0, max(scan+cal, somc) - NOV)
      - Grand total broker style = span_broker + exposure_total
      - Funds required cash = grand_total + premium_payable - premium_receivable
    """
    totals = (res or {}).get("totals", {}) or {}

    def f(key: str, default: float = 0.0) -> float:
        try:
            return float(totals.get(key, default) or 0.0)
        except Exception:
            return float(default)

    scan = f("scan_risk_total")
    cal = f("calendar_spread_charge_total")
    scan_cal = f("scan_cal_total")  # should equal scan+cal
    somc = f("somc_total")
    floored = f("scan_cal_floored")
    nov = f("net_option_value")  # signed
    span_broker = f("span_broker_style")
    exposure = f("exposure_total")
    grand = f("grand_total_broker_style")
    prem_pay = f("premium_payable")
    prem_recv = f("premium_receivable")
    funds = f("funds_required_cash")

    # recompute locally as a guardrail / sanity check
    scan_cal_local = scan + cal
    floored_local = max(scan_cal_local, somc)
    span_broker_local = max(0.0, floored_local - nov)
    grand_local = span_broker_local + exposure
    funds_local = grand_local + max(0.0, nov) - max(0.0, -nov)

    warn: List[str] = []
    # tolerate tiny float drift
    def _close(a: float, b: float) -> bool:
        return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))

    if not _close(scan_cal, scan_cal_local):
        warn.append(f"scan_cal_total mismatch: totals={scan_cal} vs local={scan_cal_local}")
    if not _close(floored, floored_local):
        warn.append(f"scan_cal_floored mismatch: totals={floored} vs local={floored_local}")
    if not _close(span_broker, span_broker_local):
        warn.append(f"span_broker_style mismatch: totals={span_broker} vs local={span_broker_local}")
    if not _close(grand, grand_local):
        warn.append(f"grand_total_broker_style mismatch: totals={grand} vs local={grand_local}")
    if not _close(funds, funds_local):
        warn.append(f"funds_required_cash mismatch: totals={funds} vs local={funds_local}")

    lines: List[str] = []
    lines.append("BROKER-STYLE MARGIN TRACE")
    lines.append("-" * 72)

    lines.append(f"1) Scan Risk Total                = {_fmt_money(scan, nd)}")
    lines.append(f"2) Calendar Spread Charge Total   = {_fmt_money(cal, nd)}")
    lines.append(f"3) Scan + Calendar                = {_fmt_money(scan_cal_local, nd)}"
                 f"    (scan {_fmt_money(scan, nd)} + cal {_fmt_money(cal, nd)})")

    lines.append(f"4) SOMC Total                     = {_fmt_money(somc, nd)}")
    lines.append(f"5) Floored (max of 3 & 4)          = {_fmt_money(floored_local, nd)}"
                 f"    = max({_fmt_money(scan_cal_local, nd)}, {_fmt_money(somc, nd)})")

    nov_note = "reduces" if nov > 0 else ("increases" if nov < 0 else "no change")
    lines.append(f"6) Net Option Value (NOV, signed)  = {_fmt_money(nov, nd)}   -> {nov_note} margin")
    lines.append(f"7) SPAN After NOV (broker style)   = {_fmt_money(span_broker_local, nd)}"
                 f"    = max(0, {_fmt_money(floored_local, nd)} - {_fmt_money(nov, nd)})")

    lines.append(f"8) Exposure / ELM Total            = {_fmt_money(exposure, nd)}")
    lines.append(f"9) Grand Total (broker style)      = {_fmt_money(grand_local, nd)}"
                 f"    = {_fmt_money(span_broker_local, nd)} + {_fmt_money(exposure, nd)}")

    lines.append(f"10) Premium payable                = {_fmt_money(max(0.0, nov), nd)} = max(0, NOV)")
    lines.append(f"11) Premium receivable             = {_fmt_money(max(0.0, -nov), nd)} = max(0, -NOV)")
    lines.append(f"12) Funds required (cash style)    = {_fmt_money(funds_local, nd)}"
                 f"    = grand + payable - receivable")

    if include_per_underlying:
        span = (res or {}).get("span") or {}
        per_u = span.get("per_underlying")
        if isinstance(per_u, dict) and per_u:
            lines.append("")
            lines.append("SPAN PER-UNDERLYING (audit)")
            lines.append("-" * 72)
            # show all underlyings with non-zero scan/cal unless user wants all
            for und in sorted(per_u.keys()):
                row = per_u[und] or {}
                scan_u = float(row.get("scan_risk", 0.0) or 0.0)
                cal_u = float(row.get("calendar_spread_charge", 0.0) or 0.0)
                total_u = scan_u + cal_u

                if (not show_zero_underlyings) and abs(total_u) < 1e-12:
                    continue

                lines.append(f"{und}: scan={_fmt_money(scan_u, nd)}  "
                             f"cal={_fmt_money(cal_u, nd)}  "
                             f"scan+cal={_fmt_money(total_u, nd)}  "
                             f"(active_scen={row.get('active_scenario')}, worst_pnl={_fmt_money(float(row.get('worst_pnl', 0.0) or 0.0), nd)})")

                if include_expiry_day_split_if_present:
                    # If expiry-day split fields exist, show them
                    if "scan_risk_expiring" in row or "scan_risk_nonexpiring" in row:
                        se = float(row.get("scan_risk_expiring", 0.0) or 0.0)
                        sn = float(row.get("scan_risk_nonexpiring", 0.0) or 0.0)
                        ase = row.get("active_scenario_expiring")
                        asn = row.get("active_scenario_nonexpiring")
                        wpe = float(row.get("worst_pnl_expiring", 0.0) or 0.0)
                        wpn = float(row.get("worst_pnl_nonexpiring", 0.0) or 0.0)
                        lines.append(f"  └─ expiry-day split: expiring_scan={_fmt_money(se, nd)} "
                                     f"(scen={ase}, worst={_fmt_money(wpe, nd)}), "
                                     f"nonexp_scan={_fmt_money(sn, nd)} "
                                     f"(scen={asn}, worst={_fmt_money(wpn, nd)})")

    if warn:
        lines.append("")
        lines.append("WARNINGS (sanity-check mismatches)")
        lines.append("-" * 72)
        for w in warn:
            lines.append(f"- {w}")

    return "\n".join(lines)


def print_margin_calc_trace(*args, **kwargs) -> None:
    """
    Convenience: prints the output of margin_calc_trace(...).
    """
    print(margin_calc_trace(*args, **kwargs))

def save_margin_calc_trace(
    res: Dict[str, Any],
    out_path: str | os.PathLike,
    *,
    nd: int = 2,
    include_per_underlying: bool = True,
    include_expiry_day_split_if_present: bool = True,
    show_zero_underlyings: bool = False,
    encoding: str = "utf-8",
    mkdirs: bool = True,
) -> str:
    """
    Builds margin_calc_trace(...) and saves it to out_path.

    Returns the absolute path of the saved file.
    """
    txt = margin_calc_trace(
        res,
        nd=nd,
        include_per_underlying=include_per_underlying,
        include_expiry_day_split_if_present=include_expiry_day_split_if_present,
        show_zero_underlyings=show_zero_underlyings,
    )

    p = Path(out_path).expanduser()
    if mkdirs:
        p.parent.mkdir(parents=True, exist_ok=True)

    p.write_text(txt, encoding=encoding)
    return str(p.resolve())


#cached parse prism any 
@lru_cache(maxsize=8)
def parse_prism_cached(path: str, mtime_ns: int):
    return parse_prism_any_indices(path)

def load_span_cached(span_path: Path):
    st = span_path.stat()
    return parse_prism_cached(str(span_path), st.st_mtime_ns)


if __name__ =='__main__':
    from datetime import date

    # span = parse_prism_cached(r"Z:\abhijeet\positions_dashboard\nsccl.20260105.i04.spn")
    today = date.today()

    r = redis.Redis()
    df_master = utils.load_instrument_master()

    prices = read_redis_prices(r=r,key='dhan_prices')
    # print(f"redis Prices : {prices}")

    #-------previous day close price to identify the deep otms to charge 1% extra elm on them -------
    prev_close_by_ul = {'NIFTY':26178.70,'BANKNIFTY':60118.40,'SENSEX':85063.34} 

    # underlying_price_live, fut_px_map, opt_ltp_map = build_live_price_inputs(prices=prices,
    #                                                                        master=df_master)
    
    underlying_price_live, fut_px_map, opt_ltp_map = build_live_prices_inputs_zerodha(prices=prices,master = df_master)
    # print(underlying_price_live)
    # print(fut_px_map)
    # print(opt_ltp_map)

    positions = pd.DataFrame([
        # {'underlying':"BANKNIFTY","kind":"FUT","expiry":20260127,"qty":1},
        {'underlying':"NIFTY","kind":"OPT","option_type":"CE","strike":25900,"expiry":20260113,"qty":-65},
        {'underlying':"NIFTY","kind":"OPT","option_type":"PE","strike":26800,"expiry":20260113,"qty":-65},
        # {'underlying':"NIFTY","kind":"OPT","option_type":"CE","strike":25900,"expiry":20260113,"qty":-1},
        # {'underlying':"NIFTY","kind":"OPT","option_type":"PE","strike":26800,"expiry":20260113,"qty":-1}
        ])

    lot_size_by_series = {
        ("BANKNIFTY", 20251230): 35,
        ("BANKNIFTY", 20260127): 30,
        ("BANKNIFTY", 20260224): 30,
        ("NIFTY",     20251230): 75,
        ("NIFTY",     20260127): 65,
        ("NIFTY",     20260113): 65,
        ("NIFTY",     20260127): 65,
        ("NIFTY",     20260106): 65,        
        ("NIFTY",     20260224): 65,
    }

    idx_ul = {"NIFTY","BANKNIFTY","SENSEX"}

    # 4. Underlying prices & exposure %
    # underlying_price = {"BANKNIFTY": 59348.25, "NIFTY": 25986.00}
    exposure_pct_by_underlying = {"BANKNIFTY": 0.02, "NIFTY": 0.02}

    span_dir = Path(r"spanfiles")

    ymd_used,span_path = get_spn.ensure_latest_span_file(out_dir=span_dir,date_yyyymmdd=today.strftime("%Y%m%d"))

    span = load_span_cached(span_path=span_path)
    print("Using SPAN:", span_path.name, "for date:", ymd_used)

    
    # # 5. Compute total margin
    res = compute_portfolio_margin_with_exposure(
            positions=positions,
            span=span,
            underlying_price=underlying_price_live,
            index_underlyings= idx_ul,
            lot_size_by_series=lot_size_by_series,
            lot_size_by_underlying= None,
            futures_ra_is_per_unit=True,  # if your futures RA is per unit
            options_ra_is_per_unit=True,
            exposure_pct_by_underlying=exposure_pct_by_underlying,
            somc_pct_index= 0.03,
            asof_date=int(today.strftime("%Y%m%d")),
            fut_price_by_underlying_expiry=fut_px_map,
            opt_ltp_by_contract_key=opt_ltp_map,
            # somc_pct_stock=0.10,
            prev_close_by_underlying= prev_close_by_ul,
            is_units=False
        )
    print(res['totals'])
    
    
    # ATM_BN = round(underlying_price_live['BANKNIFTY']/100)*100
    # ATM_NF = round(underlying_price_live['NIFTY']/50)*50

    # exp1 = 20260127
    # exp2 = 20260224
    # exp3 = 20260113

    # # #normal BN
    # # portfolio_cases = {

    # #     # 1) Single FUT outright (baseline)
    # #     "BN_FUT_OUTRIGHT_LONG": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"FUT","expiry":exp1,"qty": 1},
    # #     ]),

    # #     # 2) FUT calendar spread (tests CCDEF calendar margin recognition)
    # #     "BN_FUT_CALENDAR_1x1": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"FUT","expiry":exp1,"qty":-1},
    # #         {"underlying":"BANKNIFTY","kind":"FUT","expiry":exp2,"qty": 1},
    # #     ]),

    # #     # 3) FUT calendar unequal lots (tests leftover outright legs after matching)
    # #     "BN_FUT_CALENDAR_3x1": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"FUT","expiry":exp1,"qty":-3},
    # #         {"underlying":"BANKNIFTY","kind":"FUT","expiry":exp2,"qty": 1},
    # #     ]),

    # #     # 4) Single short option (baseline short option risk)
    # #     "BN_SHORT_CALL_ATM": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN,"expiry":exp1,"qty":-1},
    # #     ]),

    # #     # 5) Single long option (baseline long option / NOV debit handling)
    # #     "BN_LONG_PUT_ATM": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"PE","strike":ATM_BN,"expiry":exp1,"qty": 1},
    # #     ]),

    # #     # 6) Vertical credit spread (defined risk; tests same-expiry spread matching)
    # #     "BN_CALL_CREDIT_SPREAD": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN,    "expiry":exp1,"qty":-1},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN+100,"expiry":exp1,"qty": 1},
    # #     ]),

    # #     # 7) Vertical debit spread
    # #     "BN_PUT_DEBIT_SPREAD": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"PE","strike":ATM_BN,    "expiry":exp1,"qty": 1},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"PE","strike":ATM_BN-100,"expiry":exp1,"qty":-1},
    # #     ]),

    # #     # 8) Short straddle (high risk; tests worst-case scan + NOV credit sign)
    # #     "BN_SHORT_STRADDLE_ATM": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN,"expiry":exp1,"qty":-1},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"PE","strike":ATM_BN,"expiry":exp1,"qty":-1},
    # #     ]),

    # #     # 9) Long strangle (low risk; tests NOV debit sign)
    # #     "BN_LONG_STRANGLE": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN+200,"expiry":exp1,"qty": 1},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"PE","strike":ATM_BN-200,"expiry":exp1,"qty": 1},
    # #     ]),

    # #     # 10) Iron condor (defined risk, 4 legs; tests netting + multi-leg stability)
    # #     "BN_IRON_CONDOR": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"PE","strike":ATM_BN-300,"expiry":exp1,"qty": 1},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"PE","strike":ATM_BN-200,"expiry":exp1,"qty":-1},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN+200,"expiry":exp1,"qty":-1},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN+300,"expiry":exp1,"qty": 1},
    # #     ]),

    # #     # 11) Ratio spread (tests non-1:1 hedging + leftover short options)
    # #     "BN_CALL_RATIO_1x2": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN,    "expiry":exp1,"qty": 1},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN+100,"expiry":exp1,"qty":-2},
    # #     ]),

    # #     # 12) Covered call (FUT + short call)
    # #     "BN_COVERED_CALL": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"FUT","expiry":exp1,"qty": 1},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN+200,"expiry":exp1,"qty":-1},
    # #     ]),

    # #     # 13) Protective put (FUT + long put)
    # #     "BN_PROTECTIVE_PUT": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"FUT","expiry":exp1,"qty": 1},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"PE","strike":ATM_BN-200,"expiry":exp1,"qty": 1},
    # #     ]),

    # #     # 14) Synthetic future long (C long + P short, same strike/expiry)
    # #     "BN_SYNTH_FUTURE_LONG": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN,"expiry":exp1,"qty": 1},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"PE","strike":ATM_BN,"expiry":exp1,"qty":-1},
    # #     ]),

    # #     # 15) Options calendar (same strike/type across expiries; tests calendar recognition for OPT)
    # #     "BN_OPT_CALENDAR_CALL": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN,"expiry":exp1,"qty":-1},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN,"expiry":exp2,"qty": 1},
    # #     ]),

    # #     # 16) Diagonal (different strike + expiry; tests partial offsets / classification)
    # #     "BN_DIAGONAL_CALL": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN+100,"expiry":exp1,"qty":-1},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN+200,"expiry":exp2,"qty": 1},
    # #     ]),

    # #     # 17) Multi-underlying isolation (ensure no unintended cross-underlying offsets)
    # #     "BN_PLUS_NIFTY_ISOLATION": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN,"expiry":exp1,"qty":-1},
    # #         {"underlying":"NIFTY","kind":"OPT","option_type":"PE","strike":ATM_NF,"expiry":exp1,"qty":-1},
    # #     ]),

    # #     # 18) NIFTY baseline vertical spread (tests different underlying + different strike step)
    # #     "NF_CALL_CREDIT_SPREAD": pd.DataFrame([
    # #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF,    "expiry":exp1,"qty":-10},
    # #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF+50, "expiry":exp1,"qty": 10},
    # #     ]),

    # #     # 19) Duplicate rows (tests your grouping/aggregation logic)
    # #     "NF_DUPLICATE_ROWS_AGG_TEST": pd.DataFrame([
    # #         {"underlying":"NIFTY","kind":"FUT","expiry":exp1,"qty": 1},
    # #         {"underlying":"NIFTY","kind":"FUT","expiry":exp1,"qty": 1},
    # #     ]),

    # #     # 20) Mixed FUT + OPT multi-expiry stress (tests stability in complex books)
    # #     "BN_STRESS_MIXED_MULTIEXP": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"FUT","expiry":exp1,"qty": 2},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN+200,"expiry":exp1,"qty":-2},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN+300,"expiry":exp1,"qty": 1},
    # #         {"underlying":"BANKNIFTY","kind":"FUT","expiry":exp2,"qty":-1},
    # #     ]),
    # # }

    # # #normal Nifty
    # # portfolio_cases = {

    # #     # =========================
    # #     # NIFTY – Futures (NO exp3)
    # #     # =========================

    # #     "NF_FUT_OUTRIGHT": pd.DataFrame([
    # #         {"underlying":"NIFTY","kind":"FUT","expiry":exp1,"qty": 1},
    # #     ]),

    # #     "NF_FUT_CALENDAR": pd.DataFrame([
    # #         {"underlying":"NIFTY","kind":"FUT","expiry":exp1,"qty":-1},
    # #         {"underlying":"NIFTY","kind":"FUT","expiry":exp1,"qty": 1},  # net zero (sanity)
    # #     ]),

    # #     # =========================
    # #     # NIFTY – Options (weekly + monthly)
    # #     # =========================

    # #     "NF_SHORT_CALL_WEEKLY": pd.DataFrame([
    # #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF,"expiry":exp3,"qty":-10},
    # #     ]),

    # #     "NF_LONG_PUT_WEEKLY": pd.DataFrame([
    # #         {"underlying":"NIFTY","kind":"OPT","option_type":"PE","strike":ATM_NF,"expiry":exp3,"qty": 10},
    # #     ]),

    # #     "NF_CALL_CREDIT_SPREAD_WEEKLY": pd.DataFrame([
    # #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF,    "expiry":exp3,"qty":-10},
    # #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF+50, "expiry":exp3,"qty": 10},
    # #     ]),

    # #     "NF_PUT_DEBIT_SPREAD_WEEKLY": pd.DataFrame([
    # #         {"underlying":"NIFTY","kind":"OPT","option_type":"PE","strike":ATM_NF,    "expiry":exp3,"qty": 10},
    # #         {"underlying":"NIFTY","kind":"OPT","option_type":"PE","strike":ATM_NF-50, "expiry":exp3,"qty":-10},
    # #     ]),

    # #     "NF_SHORT_STRADDLE_WEEKLY": pd.DataFrame([
    # #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF,"expiry":exp3,"qty":-10},
    # #         {"underlying":"NIFTY","kind":"OPT","option_type":"PE","strike":ATM_NF,"expiry":exp3,"qty":-10},
    # #     ]),

    # #     # =========================
    # #     # NIFTY – Calendar / Diagonal (OPT only)
    # #     # =========================

    # #     "NF_OPT_CALENDAR_CALL": pd.DataFrame([
    # #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF,"expiry":exp3,"qty":-10},
    # #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF,"expiry":exp1,"qty": 10},
    # #     ]),

    # #     "NF_DIAGONAL_CALL": pd.DataFrame([
    # #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF+50,"expiry":exp3,"qty":-10},
    # #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF+100,"expiry":exp1,"qty": 10},
    # #     ]),

    # #     # =========================
    # #     # BANKNIFTY – Core
    # #     # =========================

    # #     "BN_FUT_CALENDAR": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"FUT","expiry":exp1,"qty":-1},
    # #         {"underlying":"BANKNIFTY","kind":"FUT","expiry":exp2,"qty": 1},
    # #     ]),

    # #     "BN_CALL_CREDIT_SPREAD": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN,     "expiry":exp1,"qty":-1},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN+100, "expiry":exp1,"qty": 1},
    # #     ]),

    # #     "BN_OPT_CALENDAR_CALL": pd.DataFrame([
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN,"expiry":exp1,"qty":-1},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN,"expiry":exp2,"qty": 1},
    # #     ]),

    # #     # =========================
    # #     # Mixed NIFTY + BANKNIFTY
    # #     # =========================

    # #     "NF_BN_SHORT_OPTIONS": pd.DataFrame([
    # #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF,"expiry":exp3,"qty":-10},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"PE","strike":ATM_BN,"expiry":exp1,"qty":-1},
    # #     ]),

    # #     "NF_BN_HEDGED_BOOK": pd.DataFrame([
    # #         {"underlying":"NIFTY","kind":"FUT","expiry":exp1,"qty": 1},
    # #         {"underlying":"NIFTY","kind":"OPT","option_type":"PE","strike":ATM_NF-100,"expiry":exp3,"qty": 10},
    # #         {"underlying":"BANKNIFTY","kind":"OPT","option_type":"CE","strike":ATM_BN+200,"expiry":exp1,"qty":-1},
    # #     ]),
    # # }

    # #extreme cases nifty banknifty
    # portfolio_cases = {

    #     # =========================
    #     # 1) Naked Future (Baseline)
    #     # =========================

    #     "NF_NAKED_FUT_LONG": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"FUT","expiry":exp1,"qty": 1},
    #     ]),

    #     # =========================
    #     # 2) Naked Option Shorts
    #     # =========================

    #     "NF_NAKED_SHORT_CALL_WEEKLY": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF,"expiry":exp3,"qty":-10},
    #     ]),

    #     "NF_NAKED_SHORT_PUT_WEEKLY": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"PE","strike":ATM_NF,"expiry":exp3,"qty":-10},
    #     ]),

    #     # =========================
    #     # 3) Covered Calls
    #     # FUT long + short call
    #     # =========================

    #     "NF_COVERED_CALL_MONTHLY": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"FUT","expiry":exp1,"qty": 1},
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF+100,"expiry":exp1,"qty":-10},
    #     ]),

    #     "NF_COVERED_CALL_WEEKLY": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"FUT","expiry":exp1,"qty": 1},
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF+50,"expiry":exp3,"qty":-10},
    #     ]),

    #     # =========================
    #     # 4) Protective Puts
    #     # FUT long + long put
    #     # =========================

    #     "NF_PROTECTIVE_PUT_MONTHLY": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"FUT","expiry":exp1,"qty": 1},
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"PE","strike":ATM_NF-100,"expiry":exp1,"qty": 10},
    #     ]),

    #     "NF_PROTECTIVE_PUT_WEEKLY": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"FUT","expiry":exp1,"qty": 1},
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"PE","strike":ATM_NF-100,"expiry":exp3,"qty": 10},
    #     ]),

    #     # =========================
    #     # 5) Calendar Spreads
    #     # =========================

    #     # FUT calendar (monthly only, sanity)
    #     "NF_FUT_CALENDAR": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"FUT","expiry":exp1,"qty":-1},
    #         {"underlying":"NIFTY","kind":"FUT","expiry":exp1,"qty": 1},
    #     ]),

    #     # Option calendar – naked short weekly vs long monthly
    #     "NF_OPT_CALENDAR_CALL": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF,"expiry":exp3,"qty":-10},
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF,"expiry":exp1,"qty": 10},
    #     ]),

    #     "NF_OPT_CALENDAR_PUT": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"PE","strike":ATM_NF,"expiry":exp3,"qty":-10},
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"PE","strike":ATM_NF,"expiry":exp1,"qty": 10},
    #     ]),
    #     "NF_EXTREME_DEEP_OTM_SHORT_CALL": pd.DataFrame([
    #     {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF+800,"expiry":exp3,"qty":-10},
    #     ]),

    #     "NF_EXTREME_DEEP_OTM_SHORT_PUT": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"PE","strike":ATM_NF-800,"expiry":exp3,"qty":-10},
    #     ]),
    #     "NF_EXTREME_SHORT_STRADDLE_WEEKLY": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF,"expiry":exp3,"qty":-20},
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"PE","strike":ATM_NF,"expiry":exp3,"qty":-20},
    #     ]),
    #     "NF_EXTREME_SHORT_STRADDLE_WEEKLY": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF,"expiry":exp3,"qty":-20},
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"PE","strike":ATM_NF,"expiry":exp3,"qty":-20},
    #     ]),
    #     "NF_EXTREME_COVERED_CALL_ITM": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"FUT","expiry":exp1,"qty": 1},
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF-300,"expiry":exp1,"qty":-10},
    #     ]),
    #     "NF_EXTREME_PROTECTIVE_PUT_ITM": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"FUT","expiry":exp1,"qty": 1},
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"PE","strike":ATM_NF+200,"expiry":exp1,"qty": 10},
    #     ]),
    #     "NF_EXTREME_BROKEN_CALENDAR": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF,"expiry":exp3,"qty":-20},
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF,"expiry":exp1,"qty": 10},
    #     ]),
    #     "NF_EXTREME_FUT_PLUS_SHORT_STRADDLE": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"FUT","expiry":exp1,"qty": 1},
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF,"expiry":exp3,"qty":-10},
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"PE","strike":ATM_NF,"expiry":exp3,"qty":-10},
    #     ]),
    #     "NF_EXTREME_STRIKE_COLLISION": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF,"expiry":exp3,"qty":-10},
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF,"expiry":exp3,"qty": 10},
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"CE","strike":ATM_NF,"expiry":exp3,"qty": -5},
    #     ]),
    #     "NF_EXTREME_LARGE_LOT_SHORT": pd.DataFrame([
    #         {"underlying":"NIFTY","kind":"OPT","option_type":"PE","strike":ATM_NF,"expiry":exp3,"qty":-200},
    #     ]),
    # }

    # print(portfolio_cases.keys())
    # positions = portfolio_cases['NF_FUT_OUTRIGHT'].copy()
    
    # print(positions)


    # # 5. Compute total margin
    # res = compute_portfolio_margin_with_exposure(
    #         positions=positions,
    #         span=span,
    #         underlying_price=underlying_price_live,
    #         index_underlyings= idx_ul,
    #         lot_size_by_series=lot_size_by_series,
    #         lot_size_by_underlying= None,
    #         futures_ra_is_per_unit=True,  # if your futures RA is per unit
    #         options_ra_is_per_unit=True,
    #         exposure_pct_by_underlying=exposure_pct_by_underlying,
    #         somc_pct_index= 0.03,
    #         asof_date=int(today.strftime("%Y%m%d")),
    #         fut_price_by_underlying_expiry=fut_px_map,
    #         opt_ltp_by_contract_key=opt_ltp_map,
    #         # somc_pct_stock=0.10,
    #         prev_close_by_underlying= prev_close_by_ul
    #     )
    # print(res['totals'])

    # # margin breakdown 
    # # bd = margin_breakdown(res,include_tables=True,round_to=2)
    # trace_txt = mt.margin_calc_trace(
    #     res=res,
    #     positions= positions,
    #     exposure_pct_by_underlying=exposure_pct_by_underlying,
    #     prev_close_by_underlying=prev_close_by_ul,
    #     asof_date=int(today.strftime("%Y%m%d")),
    #     index_underlyings=idx_ul,
    #     save_path=r"Z:\abhijeet\positions_dashboard\trace_files\margin_calc_trace.txt"
    #     )

    # summary_df = margin_breakdown_table(bd)
    # summary_df.to_csv(fr"Z:\abhijeet\positions_dashboard\margin_outputs\{str(today)}_breakdown_niftyceshort.csv")

    # trace_txt = bd_m.margin_calc_trace_detailed(res,include_full_instrument_meta=False)

    # trace_file = bd_m.save_margin_calc_trace_detailed(res,fr"Z:\abhijeet\positions_dashboard\margin_outputs\niftyexp.txt",include_full_instrument_meta=False)

    # print("saved",trace_file)


    #multiple cases for margin testing
    # res_list = []
    # res_all = dict()
    # for k,v in portfolio_cases.items():
    #     res = compute_portfolio_margin_with_exposure(
    #         positions=v,
    #         span=span,
    #         underlying_price=underlying_price_live,
    #         index_underlyings= idx_ul,
    #         lot_size_by_series=lot_size_by_series,
    #         lot_size_by_underlying= None,
    #         futures_ra_is_per_unit=True,  # if your futures RA is per unit
    #         options_ra_is_per_unit=True,
    #         exposure_pct_by_underlying=exposure_pct_by_underlying,
    #         somc_pct_index= 0.03,
    #         asof_date=today.strftime("%Y%m%d"),
    #         fut_price_by_underlying_expiry=fut_px_map,
    #         opt_ltp_by_contract_key=opt_ltp_map
    #         # somc_pct_stock=0.10
    #     )
    #     res['totals']['position'] = k
    #     res_list.append(res['totals'])
    #     res_all[k] = res
    
    # pd.DataFrame(res_list).to_csv(fr"Z:\abhijeet\positions_dashboard\margin_outputs\{str(today)}_margins.csv")


        

