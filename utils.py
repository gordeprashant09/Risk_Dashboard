import os
import pandas as pd
import numpy as np
import pendulum
from logzero import logger
from omspy.order import create_db, Order
from typing import List, Union, Optional, Any, Dict, Set, Tuple, Callable
from sqlite_utils import Database
import requests
from joblib import Memory, expires_after
from dataclasses import dataclass
from omspy.simulation.models import Instrument
from itertools import accumulate
from pydantic import BaseModel
from pydantic_extra_types.pendulum_dt import DateTime
import platform
import yaml
import math
import conf as conf
import redis
# from py_vollib.black_scholes.greeks.analytical import delta


memory = Memory("/tmp")


class PivotPoint(BaseModel):
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float


class Candle(BaseModel):
    timestamp: DateTime
    open: float
    high: float
    low: float
    close: float


class CandleSticks(BaseModel):
    candles: Optional[List[Candle]] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.candles is None:
            self.candles = []

    def add(self, candle: Candle):
        self.candles.append(candle)


def _intersection(set1: Set, set2: Set) -> Set[str]:
    """
    get the intersection of 2 sets and print the missing ones
    """
    s3 = set1.intersection(set2)
    logger.warning(f"Total symbols = {len(set1)}; Allowed = {len(s3)}")
    logger.warning(f"Symbols not allowed = {set1-set2}")
    return s3


def create_database(name="orders.sqlite") -> Optional[Database]:
    """
    create a sqlite database if it doesn't exist
    returns the existing database if already exists
    """
    if os.path.exists(name):
        logger.info("Database already exists")
        return Database(name)
    else:
        try:
            db = create_db(name)
            logger.info("New database created")
            return db
        except Exception as e:
            logger.error(e)
            return None


def get_nearest_options(
    spot: float, step: Union[float, int], num: int = 1
) -> List[Union[float, int]]:
    """
    Get the nearest options given the number of strikes
    spot
        spot price of the underlying
    step
        the step size of the strike
    num
        number of strikes to return
    Note
    ----
    1)The function returns the number of strikes on both the
    sides. So the number of options would be (n-1)*2
    >>> get_nearest_options(17128,100,2)
    [17000,17100,17200]
    2) The nearest option is calcuated by rounding off
    """
    start = round(spot / step) * step
    num = abs(num)
    if num == 0:
        return [start]
    strikes = []
    for i in range(-num, num + 1):
        strikes.append(start + i * step)
    return strikes


def get_latest_expiry(
    master: pd.DataFrame, name: str = "NIFTY", n: int = 0
) -> pendulum.DateTime:
    """
    Get the expiry date by number. By default
    the nearest expiry date is returned
    master
        instrument master as a dataframe
    name
        instrument name to search
    n
        nth expiry to return
    """
    name = name.upper()
    df = master.query(f"name=='{name}'").copy()
    expiry = pd.Timestamp(sorted(df.expiry.unique())[n])
    return pendulum.instance(expiry, tz="local")


def get_latest_monthly_expiry(
    master: pd.DataFrame, name: str = "NIFTY"
) -> pendulum.DateTime:
    """
    Get the monthly expiry
    master
        instrument master as a dataframe
    name
        instrument name to search
    """
    name = name.upper()
    df = master.query(f"name=='{name}'").copy()
    df["year"] = df.expiry.dt.year
    df["month"] = df.expiry.dt.month
    expiries = df.groupby(["year", "month"]).expiry.max().sort_index()
    expiry = pd.Timestamp(sorted(expiries)[0])
    return pendulum.instance(expiry, tz="local")


def get_atm(spot: float, step: float = 100.0) -> float:
    """
    Get the at the money option; the most nearest option.
    This is common for both put and call options
    spot
        spot price of the underlying
    step
        option price step
    """
    return round(spot / step) * step


def get_itm(spot: float, opt: str, step: float = 100.0) -> float:
    """
    Get in the money option
    spot
        spot price of the underlying
    opt
        put or call - only the first character is taken
    step
        option price step
    Note
    ----
    1) If opt is neither call or put, 0 is returned
    """
    opt = opt.lower()[0]
    if spot % step == 0:
        return spot
    elif opt == "c":
        return int(spot / step) * step
    elif opt == "p":
        return (int(spot / step) * step) + step
    else:
        return 0.0


def is_last_day(expiry_date: pendulum.DateTime) -> bool:
    """
    Check whether the given date is the last weekday in the month
    expiry_date
        date of expiry
    returns True if it is the last occuring weekday in the month
    Note
    -----
    If the next weekday falls in the next month returns True.
    If the next weekday falls in the same month returns False
    """
    if expiry_date.month == 1 and expiry_date.day == 31:
        return False
    if expiry_date.month == expiry_date.add(days=7).month:
        return False
    else:
        return True


def get_option_string(expiry_date: pendulum.DateTime, opt="w") -> str:
    """
    Get the 3-letter option string for the option symbol
    date
        expiry date
    opt
        weekly or monthly option
        w for weekly; m for monthly
    Note
    ----
    1) if weekly returns the month and day, if monthly returns the 3 letter month
    2) if monthly is the current weekly, then the month is returned
    3) This is not adjusted for monthly rollover since this is based on expiry date
    4) If opt is other than m or w, monthly option is returned
    """
    is_last_day(expiry_date)
    # TODO: Adjust for oct,nov,dec
    m = "%b"
    # Specifics
    if platform.system() == "Windows":
        w = "%#m%d"
    else:
        w = "%-m%d"
    char_map = {10: "O", 11: "N", 12: "D"}
    monthly = expiry_date.strftime(m).upper()
    if expiry_date.month > 9:
        weekly = f"{char_map[expiry_date.month]}{expiry_date.strftime('%d')}"
    else:
        weekly = str(expiry_date.strftime(w))
    if opt.lower()[0] == "m":
        return monthly
    elif opt.lower()[0] == "w":
        if is_last_day(expiry_date):
            return monthly
        else:
            return weekly
    else:
        return monthly


def orders_as_dict(broker: Any) -> Dict[str, Dict]:
    """
    return orders as a dictionary with key being
    order number and value the full order details
    Note
    ----
    1) returns and empty dict when there are no orders
    or an error is raised
    """
    try:
        orders = broker.orders
        if not orders:
            return dict()
        if len(orders) == 0:
            return dict()  # Return an empty dict in case of no orders
        orders = {str(o["order_id"]): o for o in orders if o["order_id"] is not None}
        return orders
    except Exception as e:
        logger.error(f"Error when fetching orders {e}")
        return dict()


def get_nearest_premium(
    premium: float, ltps: Dict[str, float], minimum: float = 0
) -> str:
    """
    Get the symbol with the nearest premium from the given list of instruments
    premium
        premium to search
    ltps
        dictionary of ltps with symbols as keys and
        last price as values
    minimum
        minimum value above which premium should be searched
    Note
    ----
    1. nearest premium is calculated on the basis of absolute difference
    """
    diff = 1e10
    latest_symbol = None
    for k, v in ltps.items():
        d = abs(premium - v)
        if (d < diff) and (v > minimum):
            diff = d
            latest_symbol = k
    return latest_symbol

# def compute_deltas(strikes, S, t, r, sigma):
#     """
#     Precompute delta values for all strikes for both call and put options.
    
#     Parameters:
#         strikes (list): List of available strike prices.
#         S (float): Current price of the underlying asset.
#         t (float): Time to maturity (in years).
#         r (float): Risk-free interest rate.
#         sigma (float): Volatility of the underlying asset.
    
#     Returns:
#         dict: A dictionary with strikes as keys and a tuple (call_delta, put_delta) as values.
#     """
#     delta_map = {}
#     for K in strikes:
#         call_delta = delta('c', S, K, t, r, sigma)
#         put_delta = delta('p', S, K, t, r, sigma)
#         delta_map[K] = (call_delta, put_delta)
#     return delta_map

# def get_nearest_delta_strike():
#     print(NUM)

def get_option_symbol_zerodha(
    symbol: str, expiry: pendulum.DateTime, strike: Union[int, float], opt: str
) -> Union[str, None]:
    """
    Get option symbol for zerodha broker
    """

    if not (all([symbol, expiry, strike, opt])):
        return None
    opt_string = get_option_string(expiry)
    # BANKNIFTY hack for weekly expiry after monthly expiry
    if symbol.lower() == "banknifty":
        if expiry.date() == pendulum.date(2024, 4, 30):
            opt_string = "430"
        elif expiry.date() == pendulum.date(2024, 10, 1):
            opt_string = "O01"
        elif expiry.date() == pendulum.date(2024, 12, 24):
            opt_string = "DEC"
    elif symbol.lower() == "finnifty":
        # Add finnifty dates and holiday mappings here
        pass
    elif symbol.lower() == "nifty":
        # Add nifty dates and holiday mappings here
        pass
    elif symbol.lower() == "sensex":
        if expiry.date() == pendulum.date(2025, 12, 24):
            opt_string = "DEC"
    opt = opt.lower()[0]
    opt_type = {"p": "PE", "c": "CE"}.get(opt, "PE")
    year = expiry.strftime("%y")
    return f"{symbol}{year}{opt_string}{strike}{opt_type}".upper()


def create_dict_on_values(
    dict1: Dict[str, str], dict2: Dict[str, str]
) -> Dict[str, str]:
    """
    create a dictionary out of the values of both the dictionaries based on keys
    Note
    ----
    1) We assume both the dictionaries have the same keys but different values
    >>> a = {1:'m', 2:'n'}
    >>> b = {1:'x', 2:'y'}
    >>> create_dict_on_values(a,b)
    >>> {'m':'x', 'n':'y'}
    >>> create_dict_on_values(b,a)
    >>> {'x':'m', 'y':'n'}
    """
    dct = {}
    for k, v in dict1.items():
        v2 = dict2.get(k)
        if v2:
            dct[v] = v2
    return dct


def strip_none(dct: Dict[Any, Any], empty=False) -> Dict[Any, Any]:
    """
    Strip keys with value None and return the dictionary
    dct
        dictionary with key and values
    """
    if empty:
        return {k: v for k, v in dct.items() if v}
    else:
        return {k: v for k, v in dct.items() if v is not None}


def dict_from_text(
    text: str, d1: str = ",", d2: str = ":", strip: bool = True
) -> Dict[Any, Any]:
    """
    Create a dictionary from the given text
    text
        text to be converted to dictionary
    d1
        delimiter 1 - to split the dictionaries
    d2
        delimiter 2 - to split key and value
    strip
        strip unnecessary spaces from key and values
        default True
        pass False to retain spaces from original text
    >>> dict_from_text('a:10,b:20')
    >>> {'a':'10', 'b': '20'}
    >>> dict_from_text('a;10|b;20', '|', ';')
    >>> {'a':'10', 'b': '20'}
    >>> dict_from_text('a:10, b:20', strip=False)
    >>> {'a':'10', ' b': '20'}
    """
    dct = {}
    dicts = text.split(d1)
    for d in dicts:
        k = d.split(d2)
        key, value = k[0], k[-1]
        if strip:
            dct[key.strip()] = value.strip()
        else:
            dct[key] = value
    return dct


def extract_from_quotes(quotes: Dict[str, Any], key="open") -> Dict[str, Any]:
    """
    extract necessary data from zerodha quotes
    quotes
        quotes from zerodha could be either `quote` or `ohlc` method
    key
        key to extract
    """
    dct = {}
    for k, v in quotes.items():
        if key in ("open", "high", "low", "close"):
            ohlc = v.get("ohlc")
            if ohlc:
                dct[k[4:]] = ohlc.get(key)
        else:
            dct[k[4:]] = v.get(key)
    return dct


def extract(quote: Dict, key="open") -> Optional[Any]:
    """
    extract necessary data from zerodha quote
    quote
        quotes from zerodha could be either `quote` or `ohlc` method
    key
        key to extract
    """
    if key in ("open", "high", "low", "close"):
        ohlc = quote.get("ohlc")
        if ohlc:
            return ohlc.get(key)
    else:
        return quote.get(key)


def filter_models(models: List[object], **kwargs) -> List[object]:
    """
    Filter models recursively based on kwargs
    models
        list of models
    kwargs
        kwargs with key being the attribute and value the value to match
    Note
    ----
    1) The attribute must be available in the model
    2) Only equality filter is applied
    """
    if len(kwargs) == 0:
        return models
    for k, v in kwargs.items():
        models = [model for model in models if getattr(model, k) == v]
    return models


def list_splitter(lst: List, n: int = None) -> List[List]:
    """
    Splits the list into equal lists of size n
    and returns a list of lists as copy
    lst
        list
    n
        number of items in each sub-list
    >>> list_splitter([1,2,3,4,5,6],2)
    >>> [[1,2],[3,4], [5,6]]
    >>> list_splitter([1,2,3,4,5,6],5)
    >>> [[1,2,3,4,5], [6]]
    """
    # Return as an empty list of list
    if lst == []:
        return [[]]
    elif len(lst) <= n:
        return [lst]
    else:
        if len(lst) % n == 0:
            chunks = len(lst) // n
        else:
            chunks = (len(lst) // n) + 1
        new_lst = []
        for i in range(chunks):
            new_lst.append(lst[i * n : (i + 1) * n][:])
        return new_lst


def get_strikes(spot: float, step: float = 100, num: int = 10) -> List[float]:
    """
    get the number of strikes
    """
    div = (spot // step) * step
    strikes = []
    N = 25
    num1 = min(num, N)
    for n in range(-num1, num1 + 1):
        strike = div + n * step
        strikes.append(strike)
    if num > N:
        plus = ((div + step * N) // 500) * 500
        minus = ((div - step * N) // 500) * 500
        for n in range(num - num1):
            strike = plus + n * 500
            strikes.append(strike)
            strike = minus - n * 500
            strikes.append(strike)
    return sorted(set(strikes))


def convert_dict_to_instrument(quotes: Dict[str, Dict]) -> List[Instrument]:
    """
    Convert dictionary to instrument
    """
    instruments = []
    for k, v in quotes.items():
        try:
            keys = ("instrument_token", "last_price", "ohlc")
            if type(v) == dict:
                if all([key in v for key in keys]):
                    inst = Instrument(
                        name=k,
                        token=v["instrument_token"],
                        last_price=v["last_price"],
                        open=v["ohlc"]["open"],
                        high=v["ohlc"]["high"],
                        low=v["ohlc"]["low"],
                        close=v["ohlc"]["close"],
                    )
                    if "volume_traded" in v:
                        inst.volume = v["volume_traded"]
                    if "oi" in v:
                        inst.open_interest = v["oi"]
                    instruments.append(inst)
        except Exception as e:
            logger.error(f"Error {e} in symbol {k}")
    return instruments


# @memory.cache(cache_validation_callback=expires_after(hours=3))
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


def get_contract_names(
    underlying: str, spot: float, expiry=None, step: float = 50, num: int = 10
):
    if expiry is None:
        expiry = pendulum.today()
    strikes = get_strikes(spot=spot, step=step, num=num)
    strikes = [int(x) for x in strikes]
    puts = [
        get_option_symbol_zerodha(symbol=underlying, expiry=expiry, strike=s, opt="PE")
        for s in strikes
    ]
    calls = [
        get_option_symbol_zerodha(symbol=underlying, expiry=expiry, strike=s, opt="CE")
        for s in strikes
    ]
    return calls + puts


def generate_strategies(strategy: Dict, **kwargs) -> List:
    """
    Generate the list of parent and child strategies given a dictionary
    strategy
        dict of options
    """
    import algowise.strategies as st

    lot_sizes = conf.lot_sizes
    freeze_limts = conf.freeze_limits
    tokens = conf.instrument_tokens
    steps = conf.step

    strategies = []
    children = int(strategy.pop("children"))
    strat_type = strategy.pop("strategy")
    name = strategy.pop("name")
    module = getattr(st, strat_type.upper())
    child_quantity = strategy.pop("child_qty", None)
    child_hedge_quantity = strategy.pop("child_hedge_qty")
    if child_quantity:
        child_quantity = [int(x) for x in child_quantity.split("|")]
    if child_hedge_quantity:
        child_hedge_quantity = [int(x) for x in child_hedge_quantity.split("|")]
    iv_increase = strategy.pop("iv_increase", None)
    if iv_increase:
        iv_increase = [float(x) for x in iv_increase.split("|")]
        iv_increase = list(accumulate(iv_increase))
    iv_list = strategy.pop("iv_list", None)
    if iv_list:
        iv_lst = [float(x) for x in iv_list.split("|")]
        strategy["iv_one"] = iv_lst[0]
        strategy["iv_two"] = iv_lst[1]
        strategy["iv_three"] = iv_lst[2]
        strategy["iv_four"] = iv_lst[3]
    for key in strategy.keys():
        if key.endswith("time"):
            strategy[key] = pendulum.parse(strategy[key], tz="local", strict=False)
    for i in range(children + 1):
        try:
            for k, v in kwargs.items():
                strategy[k] = v
            strategy["name"] = f"{name}:{i}"
            strategy["priority"] = i
            strategy["children"] = children
            if i > 0:
                strategy["quantity"] = child_quantity[i - 1]
                strategy["hedge_quantity"] = child_hedge_quantity[i-1]
                if "iv" in strat_type:
                    strategy["iv_increase"] = iv_increase[i - 1]
            # Add default lot sizes, tokens and step size
            underlying = strategy["underlying"]
            strategy["lot_size"] = lot_sizes[underlying]
            strategy["instrument_token"] = tokens[underlying]
            strategy["step"] = steps[underlying]
            # For second underlying
            if "underlying2" in strategy:
                underlying2 = strategy["underlying2"]
                strategy["instrument_token2"] = tokens[underlying2]
                strategy["step2"] = steps[underlying2]
                strategy["expiry2"] = strategy["expiry2"]
                strategy["quantity2"] = strategy["quantity2"]
            strategies.append(module(**strategy))
        except Exception as e:
            logger.error(f"Error in generating strategy {name}: {e}")
    return strategies


def historical_data(
    broker, exchangeSegment=1, exchangeInstrumentID=26000, interval="D", **kwargs
):
    """
    get historical market data from broker
    """
    ohlc = broker.get_historical_data(
        exchangeSegment=exchangeSegment,
        exchangeInstrumentID=exchangeInstrumentID,
        interval=interval,
        **kwargs,
    )
    return ohlc


def _fibonacci_pivot_points(high, low, close) -> PivotPoint:
    """
    Calculates the Fibonacci pivot points for a given set of high, low, and close prices.

    Args:
      high: A NumPy array of high prices.
      low: A NumPy array of low prices.
      close: A NumPy array of close prices.

    Returns:
      A NumPy array of Fibonacci pivot points.
    """

    # Calculate the pivot point.
    pivot_point = (high + low + close) / 3
    pivot_point = round(pivot_point, 2)

    # Calculate the Fibonacci retracement levels.
    s1 = pivot_point - 0.382 * (high - low)
    s2 = pivot_point - 0.618 * (high - low)
    s3 = pivot_point - 1 * (high - low)
    r1 = pivot_point + 0.382 * (high - low)
    r2 = pivot_point + 0.618 * (high - low)
    r3 = pivot_point + 1 * (high - low)
    s1 = round(s1, 2)
    s2 = round(s2, 2)
    s3 = round(s3, 3)
    r1 = round(r1, 2)
    r2 = round(r2, 2)
    r3 = round(r3, 2)
    return PivotPoint(pivot=pivot_point, r1=r1, r2=r2, r3=r3, s1=s1, s2=s2, s3=s3)


def get_fibonacci_pivot(broker, **kwargs):
    """
    get historical market data from broker
    """
    ohlc = broker.get_historical_data(**kwargs)[-1]
    high = ohlc[2]
    low = ohlc[3]
    close = ohlc[4]
    return _fibonacci_pivot_points(high, low, close)


def get_candles(broker, **kwargs) -> Optional[CandleSticks]:
    """
    get historical market data from broker
    """
    raw = broker.get_historical_data(**kwargs)
    if len(raw) > 0:
        cdl_stk = CandleSticks()
        for cdl in raw:
            keys = ("timestamp", "open", "high", "low", "close")
            kwargs = dict(zip(keys, cdl))
            candle = Candle(**kwargs)
            cdl_stk.add(candle)
        return cdl_stk
    else:
        return None


def simple_scheduler(
    start: pendulum.DateTime,
    end: pendulum.DateTime,
    interval: int = 10,
    func: Optional[Callable] = None,
) -> Optional[List[pendulum.DateTime]]:
    duration = end - start
    periods = []
    for dt in duration.range("minutes", interval):
        periods.append(dt)
    if func is None:
        func = lambda: periods[0]
    now = pendulum.now(tz="local")
    while periods[0] < now:
        periods.pop(0)

    while len(periods) > 0:
        now = pendulum.now(tz="local")
        first = periods[0]
        if now > first:
            periods.pop(0)
            logger.info(f"Running function {func}")
            yield func()
        else:
            yield None


def get_lot_quantities(quantity: int, lot_size: int, n: int = 3) -> List[int]:
    """
    given total quantity and lot sizes, return the list of quantities for individual splits
    """
    logger.info(f"quantity: {quantity}, lot_size: {lot_size}, n: {n}")
    per_lot = math.ceil(quantity // lot_size / n)

    total = quantity
    lots = []
    if per_lot == 0:
        logger.warning(
            f"Something wrong with lot size {lot_size} and quantity {quantity}"
        )
        return [quantity]
    for i in range(n - 1):
        qty = per_lot * lot_size
        q = min(qty, total)
        lots.append(q)
        total -= qty
        if total <= 0:
            break
    if total > 0:
        lots.append(total)
    return lots


def get_userid(filename: str) -> str:
    """
    Get the user_id for the user
    """
    with open(filename) as f:
        config = yaml.safe_load(f)
    return config["user_id"]


def get_ltp_datasource(key: str, r: redis.Redis) -> str:
    """
    Get ltp data source
    """
    try:
        ticker = r.hget("api_status", "TEST#ticker")
        # if "api41" or "api45" in key.lower():
        #     ticker = r.hget("api_status", f"{key}")
        # else:
        #     ticker = r.hget("api_status", "PA1602#ticker")
        zticker = r.hget("api_status", "z_ticker")
        if ticker is None:
            return "backup_price"
        elif zticker is None:
            return "last_price"
        else:
            ticker_time = pendulum.parse(ticker.decode("utf-8"))
            zticker_time = pendulum.parse(zticker.decode("utf-8"))
            if ticker_time > zticker_time:
                return "last_price"
            diff = (pendulum.now(tz="local") - ticker_time).total_seconds()
            if diff > 5:
                return "backup_price"
            else:
                return "last_price"
    except Exception as e:
        logger.error(
            f"Error {e} in fetching prices from redis, using backup prices only"
        )
        return "backup_price"


def get_time_decay(symbol: str, price: float, spot: float) -> float:
    """
    get the time decay
    symbol
        symbol as string in zerodha/XTS format
    price
        current ltp for the symbol
    spot
        spot price of the underlying
    Note
    ----
    1) We calculate decay assuming the option is sold
    2) If there is no time value we return 0
    """
    strike = int(symbol[-7:-2])
    opt = symbol[-2:].lower()[0]
    if opt == "c":
        diff = max(spot - strike, 0)
        return max(price - diff, 0)
    elif opt == "p":
        diff = max(strike - spot, 0)
        return max(price - diff, 0)
    else:
        return 0.0


def get_monthly_future_contracts(master: pd.DataFrame, names: List[str]) -> List[str]:
    """
    get the list of monthly futures contracts for the given names
    master
        instrument master as a dataframe containing all instruments
    names
        list of names/symbols to fetch future contracts
    """
    filtered = master[master.name.isin(names)]
    filtered = filtered.query("instrument_type=='FUT'")
    result = filtered.groupby("name").expiry.min()
    symbols = []
    for k, v in result.items():
        month = v.strftime("%b").upper()
        year = v.strftime("%y")
        symbol = f"{k}{year}{month}FUT"
        symbols.append(symbol)
    return symbols
