# =============================================================================
# user_config.py  —  Shared loader for users.yaml
# =============================================================================
# Usage:
#   from user_config import USERDETAILS, USERNAMES, get_user_col
#   from user_config import normalize_symbol, build_tradebook_path, build_mock_paths
# =============================================================================

from __future__ import annotations
import os
import re
import pathlib
import yaml

_HERE = pathlib.Path(__file__).parent
_YAML_PATH = pathlib.Path(os.getenv("USERS_YAML", str(_HERE / "users.yaml")))

# ---- defaults so old users without new fields still work ----
_DEFAULTS = {
    "tradebook_prefix":   "tradebook_T611",
    "symbol_format":      "spaced",
    "col_symbol":         "TradingSymbol",
    "col_qty":            "LastTradedQuantity",
    "col_price":          "OrderAverageTradedPrice",
    "col_side":           "OrderSide",
    "col_tag":            "OrderUniqueIdentifier",
    "mock_intraday_dir":  None,   # None = use "mock_{dma}"
    "mock_positional_dir": None,  # None = use "mock_{dma}_P"
}

_BASE_DIR = "/mnt/Quant_Research/Risk_dashboard_inputs"

def _load() -> dict:
    if not _YAML_PATH.exists():
        raise FileNotFoundError(
            f"[user_config] users.yaml not found at: {_YAML_PATH}\n"
            f"Set env var USERS_YAML=<path> to override."
        )
    with open(_YAML_PATH, "r") as f:
        data = yaml.safe_load(f)

    users_list = data.get("users") or []
    if not users_list:
        raise ValueError("[user_config] users.yaml loaded but 'users' list is empty.")

    userdetails: dict[str, dict] = {}
    for entry in users_list:
        uname = str(entry["username"])
        merged = {**_DEFAULTS, **{k: v for k, v in entry.items()}}
        # fill mock dirs with defaults if not set
        dma = merged["dma"]
        if not merged["mock_intraday_dir"]:
            merged["mock_intraday_dir"] = f"mock_{dma}"
        if not merged["mock_positional_dir"]:
            merged["mock_positional_dir"] = f"mock_{dma}_P"
        userdetails[uname] = merged
    return userdetails


# ---- Module-level exports ----
USERDETAILS: dict[str, dict] = _load()
USERNAMES:   list[str]       = list(USERDETAILS.keys())


# ---- Helper: get per-user column name ----
def get_user_col(username: str, field: str) -> str:
    return USERDETAILS.get(username, {}).get(field, _DEFAULTS.get(field, field))


# ---- Compact symbol → spaced format converter ----
# Handles NSE:  NIFTY2640722650CE    → NIFTY 07APR2026 22650 CE
# Handles BSE:  SENSEX2640973400PE   → SENSEX 09APR2026 73400 PE
_COMPACT_RE = re.compile(
    r'^([A-Z]+?)'          # underlying (non-greedy): NIFTY, BANKNIFTY, SENSEX etc
    r'(\d{2})'             # year last 2 digits: 26
    r'(\d{2})'             # month: 04
    r'(\d{2})'             # day: 07
    r'(\d+)'               # strike: 22650
    r'(CE|PE)$',
    re.IGNORECASE
)

_MONTH_MAP = {
    '01': 'JAN', '02': 'FEB', '03': 'MAR', '04': 'APR',
    '05': 'MAY', '06': 'JUN', '07': 'JUL', '08': 'AUG',
    '09': 'SEP', '10': 'OCT', '11': 'NOV', '12': 'DEC',
}

def compact_to_spaced(symbol: str) -> str:
    """
    NIFTY2640722650CE   → NIFTY 07APR2026 22650 CE
    SENSEX2640973400PE  → SENSEX 09APR2026 73400 PE
    Returns original if it does not match compact pattern.
    """
    if not symbol:
        return symbol
    s = str(symbol).strip()
    m = _COMPACT_RE.match(s)
    if not m:
        return s
    ul, yy, mm, dd, strike, opt = m.groups()
    month_str = _MONTH_MAP.get(mm, mm)
    expiry_str = f"{dd}{month_str}20{yy}"
    return f"{ul.upper()} {expiry_str} {strike} {opt.upper()}"


def normalize_symbol(symbol: str, username: str) -> str:
    """
    Convert symbol to spaced format if user's symbol_format is 'compact'.
    """
    fmt = USERDETAILS.get(username, {}).get("symbol_format", "spaced")
    if fmt == "compact":
        return compact_to_spaced(symbol)
    return str(symbol).strip() if symbol else symbol


# ---- Path builders ----
def build_tradebook_path(username: str, base_dir: str, date_str: str) -> str:
    """Build tradebook CSV path using per-user prefix. date_str: YYYY-MM-DD"""
    meta = USERDETAILS[username]
    prefix = meta.get("tradebook_prefix", "tradebook_T611")
    dma = meta["dma"]
    return f"{base_dir}/{dma}/tradebook/{prefix}_{date_str}.csv"


def build_mock_paths(username: str, base_dir: str) -> tuple[str, str]:
    """
    Returns (intraday_db_path, positional_db_path) using per-user mock dirs.
    Old style:  mock_DMA09/db/orders.sqlite
    New style:  QMOCK1/db/orders.sqlite
    """
    meta = USERDETAILS[username]
    intraday_dir  = meta.get("mock_intraday_dir",  f"mock_{meta['dma']}")
    positional_dir = meta.get("mock_positional_dir", f"mock_{meta['dma']}_P")
    intraday  = f"{base_dir}/{intraday_dir}/db/orders.sqlite"
    positional = f"{base_dir}/{positional_dir}/db/orders.sqlite"
    return intraday, positional
