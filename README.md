# Risk Dashboard

Real-time multi-user **options risk monitoring system** for NSE/BSE derivatives. Streams live LTPs via Zerodha WebSocket into Redis, computes IV/Greeks/P&L/Margin per user every few seconds, and serves an interactive Streamlit dashboard with scenario analysis, payoff curves, and strategy-level breakdowns.

---

## Project Files

| File | Role |
|---|---|
| `z_socket.py` | Zerodha WebSocket feed — subscribes to all contracts, writes LTPs to Redis |
| `risk_worker.py` | Core risk engine — reads LTPs from Redis, computes greeks/P&L/scenarios per user, publishes snapshots back to Redis |
| `margin_worker.py` | SPAN margin calculator — reads margin inputs from Redis, computes portfolio margin with exposure |
| `strategy_pnl_worker.py` | Strategy-level P&L worker — breaks down P&L by strategy tag, tracks min/max, computes slippage vs simulation |
| `risk_viewer.py` | Streamlit UI — reads precomputed snapshots from Redis, renders all tables and charts |

---

## Architecture Overview

```
Zerodha WebSocket
        │  z_socket.py
        │  - subscribes: NIFTY, BANKNIFTY, SENSEX options
        │    (current week + next week + monthly expiries)
        ▼
Redis  last_price  {symbol → LTP}
        │
        ├──────────────────────────────────────────────┐
        │  risk_worker.py (every N seconds)            │
        │  For each user:                              │
        │  1. Read tradebook CSV                       │
        │  2. Read prev EOD net positions CSV          │
        │  3. FIFO roll → effective open book          │
        │  4. Vectorized IV + Greeks engine            │
        │  5. Compute carry/day/net P&L                │
        │  6. Track min/max P&L in Redis               │
        │  7. Compute spot/vol/time/combo scenarios    │
        │  8. Publish snapshot → Redis                 │
        │  9. Aggregate combined __ALL__ user          │
        └──────────────────────────────────────────────┘
        │
        │  Redis keys written per user:
        │  risk:outputs:latest:{username}       (JSON snapshot)
        │  risk:pnl_minmax:{username}:{date}    (HASH)
        │  margin:inputs:latest:{username}      (JSON)
        │
        ├──────────────────────────────────────────────┐
        │  margin_worker.py (independent loop)         │
        │  - Auto-discovers users via Redis key scan   │
        │  - Loads SPAN files                          │
        │  - Computes portfolio margin + exposure      │
        │  - Publishes → margin:outputs:latest:{user}  │
        └──────────────────────────────────────────────┘
        │
        ├──────────────────────────────────────────────┐
        │  strategy_pnl_worker.py (independent loop)   │
        │  - Reads tradebook + prev EOD per user       │
        │  - Groups positions by strategy tag          │
        │  - Computes carry/day/net P&L per tag        │
        │  - Tracks min/max with timestamps            │
        │  - Compares live vs mock simulation          │
        │  - Publishes → risk:strategy_pnl:latest:{u}  │
        └──────────────────────────────────────────────┘
        │
        ▼
  risk_viewer.py  (Streamlit, auto-refresh)
  - Reads all Redis snapshots
  - Renders: KPIs, greeks table, scenario heatmaps,
    payoff curves, strategy P&L table, margin breakdown
```

---

## Component Details

### `z_socket.py` — WebSocket Feed
Connects to Zerodha's KiteTicker and streams LTPs for all subscribed instruments into Redis hash `last_price`.

**Instruments subscribed:**
- Index spots: NIFTY 50, NIFTY BANK, FINNIFTY, SENSEX
- Monthly futures for all indices
- **Current week options:** ±35 strikes around spot for NIFTY, BANKNIFTY, SENSEX
- **Next week options:** same range
- **Monthly options:** same range

Each tick writes: `r.hset("last_price", symbol, ltp)` and logs to `price_data_zerodha.csv`.

---

### `risk_worker.py` — Core Risk Engine
Runs a tight loop (configurable `LOOP_SECONDS`) processing every user in sequence.

**Per-user processing:**

| Step | Action |
|---|---|
| 1 | Read tradebook CSV (`tradebook_T611_{date}.csv`) |
| 2 | Read prev EOD net positions CSV (optional) |
| 3 | FIFO roll — build effective open book from intraday trades + EOD carry |
| 4 | Filter expired positions |
| 5 | Attach LTPs from Redis, compute IV + full Greeks (vectorized) |
| 6 | Compute carry P&L, day P&L, trading expenses, net P&L |
| 7 | Update min/max P&L tracking in Redis hash |
| 8 | Compute scenario grid: spot shocks / vol shocks / time decay / combo cube |
| 9 | Compute payoff pack (expiry P&L curves per underlying) |
| 10 | Publish full JSON snapshot to `risk:outputs:latest:{username}` |

**Combined user (`__ALL__`):** after all individual users are processed, numerically sums KPIs, margin, greeks tables, and scenario cubes across all users and publishes to `risk:outputs:latest:__ALL__`.

**Redis keys written:**

| Key | Type | Content |
|---|---|---|
| `risk:outputs:latest:{username}` | String (JSON) | Full risk snapshot |
| `risk:pnl_minmax:{username}:{YYYYMMDD}` | Hash | min/max carry/day/net P&L + timestamps |
| `margin:inputs:latest:{username}` | String (JSON) | Position inputs for margin worker |

---

### `margin_worker.py` — SPAN Margin Calculator
Runs independently. Auto-discovers users by scanning `margin:inputs:latest:*` Redis keys — no user list needed.

**Per-user processing:**
1. Reads position inputs published by `risk_worker.py`
2. Loads SPAN parameter files
3. Builds futures price map and options LTP map from Redis
4. Computes portfolio SPAN margin + exposure margin
5. Publishes result to `margin:outputs:latest:{username}`

---

### `strategy_pnl_worker.py` — Strategy P&L Breakdown
Runs independently. Groups positions and trades by **strategy tag** column in the tradebook.

**Output columns per tag:**

| Column | Description |
|---|---|
| `CarryPnL` | P&L from overnight carry positions |
| `DayPnL` | P&L from intraday trades |
| `Expenses` | Brokerage + transaction costs |
| `NetPnL` | CarryPnL + DayPnL − Expenses |
| `allocated_margin` | From multiplier CSV per strategy |
| `net_pnl/margin (%)` | Return on allocated margin |
| `MinPnL / MaxPnL` | Intraday min/max with timestamps |
| `SimCarryPnL / SimDayPnL` | From mock sqlite simulation DB |
| `Slippage (%)` | Live vs simulation difference |

Redis keys written:
- `risk:strategy_pnl:latest:{username}` — JSON table
- `risk:strategy_pnl_minmax:{username}:{YYYYMMDD}` — HASH with tag-level min/max

---

### `risk_viewer.py` — Streamlit Dashboard
Lightweight viewer — reads all precomputed data from Redis, renders the UI. No heavy computation runs here.

**Dashboard sections:**

| Section | Content |
|---|---|
| KPI bar | Carry P&L, Day P&L, Expenses, Net P&L, Legs open |
| P&L min/max | Intraday high/low tracking |
| Margin | SPAN, Exposure, Total with min/max |
| Portfolio Greeks | Delta, Gamma, Vega, Theta by underlying + expiry |
| Scenario heatmaps | Spot shock / Vol shock / Time decay / Combo cube |
| Payoff curves | Expiry P&L curves per underlying group |
| Top legs | Largest positions by abs delta / vega / gamma |
| Strategy P&L | Tag-level breakdown with slippage vs simulation |

Auto-refreshes every N seconds via `streamlit_autorefresh`.

---

## Input File Structure

```
/mnt/Quant_Research/Risk_dashboard_inputs/
└── {dma}/
    ├── tradebook/
    │   └── tradebook_T611_{YYYY_MM_DD}.csv
    ├── eod_files/
    │   └── net_positions_eod_{YYMMDD}.csv
    └── multiplier/
        └── multiplier_{dma}_{YYYYMMDD}.csv

/mnt/Quant_Research/Risk_dashboard_inputs/mock_{dma}/
└── db/
    └── orders.sqlite
```

---

## Redis Key Reference

| Key Pattern | Writer | Reader |
|---|---|---|
| `last_price` (hash) | `z_socket.py` | `risk_worker.py`, `margin_worker.py` |
| `risk:outputs:latest:{user}` | `risk_worker.py` | `risk_viewer.py` |
| `risk:pnl_minmax:{user}:{date}` | `risk_worker.py` | `risk_viewer.py` |
| `margin:inputs:latest:{user}` | `risk_worker.py` | `margin_worker.py` |
| `margin:outputs:latest:{user}` | `margin_worker.py` | `risk_worker.py`, `risk_viewer.py` |
| `risk:strategy_pnl:latest:{user}` | `strategy_pnl_worker.py` | `risk_viewer.py` |
| `risk:strategy_pnl_minmax:{user}:{date}` | `strategy_pnl_worker.py` | `risk_viewer.py` |

---

## Usage

Run all processes in separate terminals (or as systemd services):

```bash
# 1. Start WebSocket feed (run first — populates last_price in Redis)
python z_socket.py

# 2. Start core risk engine
python risk_worker.py

# 3. Start margin calculator
python margin_worker.py

# 4. Start strategy P&L worker
python strategy_pnl_worker.py

# 5. Start Streamlit dashboard
streamlit run risk_viewer.py
```

---

## Configuration

| Parameter | Location | Description |
|---|---|---|
| User list | `users.yaml` / `user_config.py` | Usernames and DMA mapping |
| Zerodha credentials | `config.yaml` | API key, secret, access token |
| Redis connection | env vars `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB` | Default: localhost:6379 |
| Risk-free rate | `risk_worker.py` `RF` | Used for IV/Greeks computation |
| Loop interval | `risk_worker.py` `LOOP_SECONDS` | How often risk engine runs |
| Instruments base path | `z_socket.py` | Instrument master CSV path |

---

## Prerequisites

```bash
pip install omspy redis streamlit streamlit-autorefresh plotly pandas numpy \
            pyarrow pendulum logzero pyyaml python-dotenv
```

- Python 3.10+
- Redis server running locally
- Zerodha Kite Connect API credentials
- NAS mount at `/mnt/Quant_Research`
- Valid Zerodha session token (refreshed daily)
