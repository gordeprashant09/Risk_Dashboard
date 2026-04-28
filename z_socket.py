from omspy.brokers.zerodha import Zerodha

import yaml
import os

import time
import pendulum
from logzero import logger
from collections import defaultdict
import utils
import conf
from concurrent.futures import ThreadPoolExecutor
import redis
from typing import Optional, List

try:
    import dotenv

    logger.info(dotenv.load_dotenv())
except Exception as e:
    pass

ENV = os.environ.get("ENV")
CONFIG = "config.yaml"
INTERVAL = 1
NUM = 35 if ENV == "DEV" else 35
NEXT_UPDATE_TIME = pendulum.now(tz="Asia/Kolkata").add(seconds=INTERVAL)
NEXT_ORDER_UPDATE_TIME = pendulum.now(tz="Asia/Kolkata").add(seconds=3)
FLAG = True
COUNTER = 0
LTPS = defaultdict(str)
MAX_CONTRACTS = (
    45 if ENV == "DEV" else 100
)  # maximum number of contracts that could be fetched

ENABLE_WEBSOCKET = True

r = redis.Redis()
df = utils.load_instrument_master()
symbol_mapper = {
    int(k): v for k, v in zip(df.instrument_token.values, df.tradingsymbol.values)
}
indices = {256265: "NIFTY", 260105: "BANKNIFTY", 257801: "FINNIFTY", 265: "SENSEX"}
symbol_mapper.update(indices)
rev_mapper = {v: k for k, v in symbol_mapper.items()}

executor = ThreadPoolExecutor(max_workers=10)
price_data_file = open("price_data_zerodha.csv", "w")
price_data_file.write("timestamp,symbol,ltp\n")

def get_all_contracts(
    expiries: dict[int, pendulum.DateTime],
    open_prices: dict[int, float],
    mapper: dict[int, str],
    num=20,
):
    contracts = []
    step_size = conf.step
    for k, v in open_prices.items():
        symbol = mapper.get(k)
        expiry = expiries.get(symbol)
        if symbol and expiry:
            n = num
            c = utils.get_contract_names(
                symbol, spot=v, expiry=expiry, num=n, step=step_size.get(symbol, 50)
            )
            contracts.extend(c)
    return contracts


def get_expiries(n: int = 0, weekly: bool = True):
    """
    Get the expiries for instruments
    n
        n is the week of expiry, 0 means current week expiry
    """
    if weekly:
        nifty_expiry = utils.get_latest_expiry(df, name="NIFTY", n=n)
        bn_expiry = utils.get_latest_expiry(df, name="BANKNIFTY", n=n)
        fn_expiry = utils.get_latest_expiry(df, name="FINNIFTY", n=n)
        sensex_expiry = utils.get_latest_expiry(df, name="SENSEX", n=n)
    else:
        nifty_expiry = utils.get_latest_monthly_expiry(df, name="NIFTY")
        bn_expiry = utils.get_latest_monthly_expiry(df, name="BANKNIFTY")
        fn_expiry = utils.get_latest_monthly_expiry(df, name="FINNIFTY")
        sensex_expiry = utils.get_latest_monthly_expiry(df, name="SENSEX")
    expiries = dict(
        NIFTY=nifty_expiry,
        BANKNIFTY=bn_expiry,
        # FINNIFTY=fn_expiry,
        SENSEX=sensex_expiry,
    )
    return expiries


def get_open_prices(broker:Zerodha, instruments):
    """
    Get the open price of all the given instruments
    """
    if instruments is None:
        logger.info("no instrument to fetch quotes")    
        return None
    response = broker.kite.quote(instruments)
    quotes = {}
    for msg_dict in response.values():
        try:
            key = int(msg_dict["instrument_token"])
            value = msg_dict["last_price"]
            quotes[key] = value
        except Exception as e:
            logger.error(e)
    return quotes


def update():
    global NEXT_UPDATE_TIME
    now = pendulum.now(tz="Asia/Kolkata")
    if now > NEXT_UPDATE_TIME:
        NEXT_UPDATE_TIME = now.add(seconds=5)


def load_all_contracts(index_names: Optional[List] = None) -> List[str]:
    """
    load all contracts for the given indices
    """
    if index_names is None:
        index_names = ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"]
    open_prices = get_open_prices(broker, instruments=Instruments)
    logger.info(open_prices)
    contracts = []
    weekly_expiries = get_expiries(n=0)
    logger.info(weekly_expiries)
    this_week_expiries = get_all_contracts(
        weekly_expiries, open_prices, symbol_mapper, num=NUM
    )
    only_calls = []
    for contract in this_week_expiries:
        if contract[-1] == "C":
            only_calls.append(contract)
    contracts.extend(only_calls)
    contracts.extend(this_week_expiries)
    next_week_expiries = get_expiries(n=1)
    logger.info(next_week_expiries)
    contracts.extend(
        get_all_contracts(next_week_expiries, open_prices, symbol_mapper, num=NUM)
    )
    monthly_expiries = get_expiries(n=0, weekly=False)
    contracts.extend(
        get_all_contracts(monthly_expiries, open_prices, symbol_mapper, num=NUM)
    )
    logger.info(contracts)
    return contracts

def on_ticks(ws, ticks):  # noqa
    # Callback to receive ticks.
    try:
        for tick in ticks:
            token = tick["instrument_token"]
            ltp = tick["last_price"]
            symbol = symbol_mapper.get(token)
            # print(symbol, ltp)
            r.hset("last_price", symbol, ltp)
            price_data_file.write(f"{pendulum.now(tz='Asia/Kolkata').time()},{symbol},{ltp}\n")
    except Exception as e:
        logger.error(e)
    # print("Ticks: {}".format(ticks))

def on_connect(ws, response):  # noqa
    print("=======================Connected=======================")
    # Callback on successful connect.
    # Subscribe to a list of instrument_tokens (RELIANCE and ACC here).
    # ws.subscribe([738561, 5633])
    for i in range(0, len(Instruments), 49):
        _insts = Instruments[i : i + 49]
        broker.ticker.subscribe(_insts)
        broker.ticker.set_mode(broker.ticker.MODE_LTP, _insts)
        time.sleep(1)
    # Set RELIANCE to tick in `full` mode.
    # ws.set_mode(ws.MODE_FULL, [738561])

    

if __name__ == "__main__":
    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)[0]["config"]
    broker = Zerodha(**config)
    broker.authenticate()
    
    # r.flushdb()
    # logger.info("Flushing all old data")

    
    """
    if os.path.exists("market_token.tok"):
        with open("market_token.tok", "r") as f:
            mkt_token = f.read()
        broker.marketdata_login(mkt_token=mkt_token)
    else:
        broker.marketdata_login()
    """
    insts = ["NSE:NIFTY 50", "NSE:NIFTY BANK", "NSE:NIFTY FIN SERVICE", "BSE:SENSEX"]
    Instruments = [256265, 265,260105]

    time.sleep(10)
    open_prices = get_open_prices(broker, instruments=insts)
    logger.info(open_prices)
    contracts = []
    futures = utils.get_monthly_future_contracts(df, names=indices.values())

    contracts.extend(futures)
    logger.info(contracts)
    weekly_expiries = get_expiries(n=0)
    logger.info(weekly_expiries)
    this_week_expiries = get_all_contracts(
        weekly_expiries, open_prices, symbol_mapper, num=NUM
    )
    print("current week",this_week_expiries)
    # only_calls = []
    # for contract in this_week_expiries:
    #     if contract[-2] == "C":
    #         only_calls.append(contract)
    # contracts.extend(only_calls)
    contracts.extend(this_week_expiries)
    next_week_expiries = get_expiries(n=1)
    logger.info(next_week_expiries)
    contracts.extend(
        get_all_contracts(next_week_expiries, open_prices, symbol_mapper, num=NUM)
    )
    monthly_expiries = get_expiries(n=0, weekly=False)
    contracts.extend(
        get_all_contracts(monthly_expiries, open_prices, symbol_mapper, num=NUM)
    )
    logger.info(monthly_expiries)

    # contracts = contracts[:MAX_CONTRACTS]
    print("TOTAL number of contracts",len(contracts))
    print(contracts)
    for contract in contracts:
        token = rev_mapper.get(contract)
        if token:
            Instruments.append(token)

    print(len(Instruments))


    # try:
    #     fib = utils.get_fibonacci_pivot(
    #         broker=broker, exchangeInstrumentID=26001, interval="D"
    #     )
    #     with open("fib.json", "w") as f:
    #         json.dump(fib.dict(), f)
    # except Exception as e:
    #     logger.exception(e)
    if ENABLE_WEBSOCKET:
        broker.ticker.on_ticks = on_ticks
        broker.ticker.on_connect = on_connect
        broker.ticker.connect()
    else:
        print("websocket is disabled")
        