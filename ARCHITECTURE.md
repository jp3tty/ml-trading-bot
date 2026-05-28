# System Architecture

Automated stock trading system using two independent binary ML classifiers — one to detect entry opportunities, one to detect early exits — executed via Alpaca's brokerage API on a twice-daily GitHub Actions schedule.

---

## Table of Contents

1. [Core Strategy](#1-core-strategy)
2. [Full Pipeline](#2-full-pipeline)
3. [Data Layer](#3-data-layer)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Design](#5-model-design)
6. [Execution Engine](#6-execution-engine)
7. [Automation & Monitoring](#7-automation--monitoring)
8. [Key Design Decisions](#8-key-design-decisions)

---

## 1. Core Strategy

The system uses an **early-exit short-term trading** strategy built around two independent binary detectors:

| Detector | Question | Tuned For |
|----------|----------|-----------|
| **BUY** | Is this a valid entry? | Recall — catch as many real entries as possible |
| **SELL** | Is this position turning? | Recall — exit at the first sign of decline |

The system defaults to **HOLD**. A position is only entered when the BUY detector fires above a confidence threshold, and only exited when the SELL detector fires on a currently held position.

### Why Two Detectors?

A single 3-class model (BUY / HOLD / SELL) cannot tune entry and exit independently. Separating them allows:

- The **BUY detector** to cast a wide net (high recall), accepting some false positives
- The **SELL detector** to act as the risk manager, cutting losing positions quickly before they compound

Bad BUY entries become short, bounded losses rather than catastrophes. This shifts the optimization surface: instead of chasing BUY precision, the system maximizes BUY recall and relies on fast SELL exits to manage downside.

---

## 2. Full Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA COLLECTION                                │
│                                                                         │
│   Alpaca Markets API                                                    │
│   ├── 872 US equities                                                   │
│   ├── 4-hour OHLCV bars  ──▶  saved_data/historical_4h/*.parquet       │
│   └── Daily bars         ──▶  saved_data/historical/*.parquet          │
│                                                                         │
│   FinViz Screener  ──▶  Momentum watchlist (live, per run)             │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        FEATURE ENGINEERING                              │
│                                                                         │
│   Triple-Barrier Labeling (Lopez de Prado)                              │
│   ├── BUY label:  take-profit hit before stop-loss within horizon       │
│   └── SELL label: stop-loss hit before take-profit within horizon       │
│                                                                         │
│   Feature Modes                                                         │
│   ├── indicators  — RSI, momentum, candlestick patterns, MACD, BBands  │
│   ├── catch22     — 22 canonical time series statistics (pycatch22)     │
│   └── combined    — indicators + catch22  [used in production]         │
└──────────────┬────────────────────────────────────┬─────────────────────┘
               │                                    │
               ▼                                    ▼
┌──────────────────────────┐          ┌─────────────────────────────┐
│      BUY DETECTOR        │          │       SELL DETECTOR         │
│                          │          │                             │
│  binary_search.py        │          │  binary_sell_search.py      │
│  Grid search:            │          │  Grid search:               │
│  · window, horizon       │          │  · window, horizon          │
│  · take_profit / SL      │          │  · sell_threshold           │
│  · RF or XGBoost         │          │  · RF or XGBoost            │
│  · n_estimators, depth   │          │  · n_estimators, depth      │
│                          │          │                             │
│  Champion selection:     │          │  Champion selection:        │
│  precision ≥ 35–40%      │          │  precision ≥ 40%            │
│  ranked by recall        │          │  recall ≥ 20%               │
│                          │          │  ranked by F1               │
│  champion_buy.pkl        │          │  champion_sell.pkl          │
└──────────────┬───────────┘          └──────────────┬──────────────┘
               │                                     │
               └──────────────────┬──────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          EXECUTION ENGINE                               │
│                            ml_trader.py                                 │
│                                                                         │
│  Pass 1 — SELL                                                          │
│  ├── Fetch all open Alpaca positions                                    │
│  ├── Run SELL detector on each held ticker                              │
│  └── Close position if signal fires above confidence floor              │
│                                                                         │
│  Pass 2 — BUY                                                           │
│  ├── Fetch FinViz momentum watchlist                                    │
│  ├── Skip any ticker already held                                       │
│  ├── Run BUY detector on each candidate                                 │
│  └── Place bracket order if confidence ≥ threshold                     │
│      (ATR-based stop loss, no TP ceiling by default)                   │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          ALPACA BROKERAGE                               │
│   Paper trading (default) · Live trading (--live flag)                  │
│   Bracket orders with ATR stop · Position size: equal-weight            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Layer

### Historical Data

| Property | Value |
|----------|-------|
| Source | Alpaca Markets REST API |
| Universe | ~872 US equities |
| Timeframe | **4-hour bars** (training), daily bars (legacy) |
| Format | Parquet, one file per ticker |
| Fields | Open, High, Low, Close, Volume |
| Adjustment | Split- and dividend-adjusted |

Training uses 4-hour bars exclusively. Each ticker is stored as a self-contained `.parquet` file under `saved_data/historical_4h/`, making it trivial to add tickers or refresh data independently.

### Live Watchlist

At runtime, the system scrapes **FinViz** for a fresh momentum watchlist using these filters:

| Filter | Value |
|--------|-------|
| Market cap | Small cap and above |
| Relative volume | Over 2× average |
| 5-day performance | Positive |
| Sort | Market cap descending |

This produces a focused list of liquid, momentum-driven candidates for each trading session — no static ticker list to maintain.

---

## 4. Feature Engineering

### Labeling: Triple-Barrier Method

Labels are generated using the **triple-barrier method** (Lopez de Prado, *Advances in Financial Machine Learning*). For each candle, the next `horizon` bars are scanned using actual high/low prices:

```
                    ┌─────────────────────────────────┐
                    │  Take-Profit Barrier (+TP%)      │  ← Label 1 (BUY) if hit first
       entry ──▶    │─────────────────────────────────│
                    │  Stop-Loss Barrier (–SL%)        │  ← Label 0 if hit first
                    │                   time barrier   │  ← Label 0 if neither hit
                    └─────────────────────────────────┘
                         horizon candles
```

This produces meaningfully balanced labels tied to real reward:risk outcomes — not a simple return threshold that ignores path.

### Feature Modes

Three feature modes are supported:

**`indicators`** — Technical analysis features computed over a rolling window:
- Normalized OHLCV (open, high, low, close, volume relative to window mean)
- RSI, momentum strength
- Candlestick patterns: hammer, engulfing, doji
- MACD and MACD signal/histogram (normalized)
- Bollinger Band position and width
- Weekly momentum, trend SMA
- Composite bullish/bearish momentum scores

**`catch22`** — 22 canonical time series statistics from [pycatch22](https://github.com/DynamicsAndNeuralSystems/pycatch22), computed independently on the close price and volume series (44 features total). These capture distributional, autocorrelation, and nonlinear dynamics properties that technical indicators do not.

**`combined`** *(production default)* — Flattened indicator window matrix concatenated with catch22 statistics. Gives the model both interpretable pattern features and time series structure features.

---

## 5. Model Design

### BUY Detector

| Property | Value |
|----------|-------|
| Algorithm | Random Forest |
| Window size | 21 bars (4h) |
| Horizon | 6 bars (4h) |
| Take profit / Stop loss | 0.8% / 0.5% |
| Decision threshold | 0.009 |
| Precision / Recall / F1 | 39.2% / 100.0% / 0.563 |
| Input | Combined features over rolling window |
| Training data | 4-hour bars, 872 tickers |
| Labeling | Triple-barrier (take-profit vs stop-loss) |
| Champion selection | `precision ≥ 35–40%` → ranked by **recall** |
| Search date | 2026-05-28 (post-fix) |

**Why maximize recall?** The SELL detector has 100% recall and cuts losing positions quickly. A bad BUY entry becomes a short, bounded loss. Missing a real BUY opportunity has no recovery path. So the BUY model is tuned to catch as many real entries as possible, accepting a precision floor of ~35–40% to avoid excessive commission drag.

### SELL Detector

| Property | Value |
|----------|-------|
| Algorithm | XGBoost |
| Window size | 20 bars (4h) |
| Horizon | 5 bars (4h) |
| Sell threshold | 0.5% |
| Decision threshold | 0.040 |
| Precision / Recall / F1 | 40.1% / 100.0% / 0.573 |
| Runtime confidence floor | 0.30 (`--sell-confidence`) |
| Input | Combined features (inverted labels — SELL is the positive class) |
| Training data | 4-hour bars, same universe |
| Champion selection | `precision ≥ 40%` and `recall ≥ 20%` → ranked by **F1** |
| Search date | 2026-04-15 (re-validated post-fix) |

The model's baked-in decision threshold is intentionally sensitive. A **runtime confidence floor** is layered on top so the SELL signal only fires when the model's probability exceeds both the model threshold *and* this floor — preventing hairpin exits on marginal signals without requiring a full retrain.

### Hyperparameter Search

Both detectors are trained via **grid search** across:

```
window_size    · how many historical bars the feature window covers
horizon        · how many bars ahead to scan for the barrier outcome
take_profit    · TP barrier height (BUY) or sell threshold (SELL)
stop_loss      · SL barrier depth
classifier     · random_forest or xgboost
n_estimators   · 100 or 200 trees
max_depth      · tree depth (5, 8, 10, 12, or unlimited)
min_precision  · precision floor for champion eligibility
```

Data is loaded once per unique `(window, horizon, take_profit, stop_loss)` combination, then reused across all classifier variants — this is the most expensive part of the search and the key optimization that makes a 240-combo quick search feasible in ~3 hours.

Champion models are saved as `.pkl` files and promoted to production automatically when a new search completes.

---

## 6. Execution Engine

### Trading Loop (`ml_trader.py`)

Each run executes two independent passes:

```
┌─────────────────────────────────────────────────────┐
│  Pass 1 — SELL (exit management)                    │
│                                                     │
│  1. Fetch all open positions from Alpaca            │
│  2. For each held ticker:                           │
│     a. Pull latest 4h bars                         │
│     b. Build SELL features                         │
│     c. If SELL probability ≥ confidence floor:     │
│        → Close position at market                  │
│                                                     │
│  Note: SELL pass is independent of the watchlist.  │
│  Held tickers are never missed regardless of       │
│  whether FinViz returns them.                      │
└─────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  Pass 2 — BUY (entry scanning)                      │
│                                                     │
│  1. Fetch FinViz momentum watchlist                 │
│  2. Check current positions (max 20)                │
│  3. For each candidate not already held:            │
│     a. Pull latest 4h bars                         │
│     b. Build BUY features                          │
│     c. If BUY probability ≥ confidence threshold:  │
│        → Fetch live ask price                      │
│        → Calculate ATR-based stop loss             │
│        → Place bracket order on Alpaca             │
│        → Log order with RSI, momentum, order ID    │
└─────────────────────────────────────────────────────┘
```

### Order Parameters

| Parameter | Value |
|-----------|-------|
| Order type | Bracket (entry + stop loss) |
| Stop loss | ATR × 2.0 below entry (14-period ATR) |
| Take profit | Disabled by default (`USE_TAKE_PROFIT = False`) |
| Entry price | Live ask price at order time |
| Max positions | 20 concurrent |
| Position sizing | Equal-weight (Alpaca notional) |

ATR-based stops adapt to each stock's volatility rather than using a fixed percentage — a stock with a wide daily range gets a wider stop; a low-volatility stock gets a tighter one.

---

## 7. Automation & Monitoring

### GitHub Actions Schedule

The trading loop runs automatically twice per market day via GitHub Actions:

| Run | UTC | Eastern |
|-----|-----|---------|
| Morning | 14:00 | ~10:00 AM EDT |
| Afternoon | 17:00 | ~1:00 PM EDT |

Runs execute Monday–Friday only. A 2-hour buffer is built into the schedule to account for GitHub Actions queue lag while staying safely within market hours (9:30 AM–4:00 PM ET).

After each run, the workflow commits updated `orders.csv` and `signals.csv` back to the repository so the dashboard always reflects current data.

### Streamlit Monitoring Dashboard

A live dashboard at `dashboard/app.py` provides real-time visibility into paper trading performance:

| Section | Content |
|---------|---------|
| **Account Summary** | Portfolio value, buying power, day P&L |
| **Active Positions** | Open positions with entry price, current price, unrealized P&L, RSI and momentum at entry |
| **Trade History** | Entry/exit date, entry/exit price, realized P&L, exit type (Stop Loss / SELL Signal) |
| **Signal Log** | Full history of every ticker scored per run |

Deployed on **Streamlit Community Cloud**, connected directly to this repository. No separate data pipeline — the dashboard reads `orders.csv` and `signals.csv` from the repo, which are kept current by the GitHub Actions commit step.

### Trade Logging

Every order placed is appended to `paper_trade_log/orders.csv` with:
- Timestamp, symbol, side, quantity, entry price
- Stop loss and take profit levels
- BUY confidence score
- RSI and momentum strength at entry
- Alpaca order ID (for reconciliation)

Every signal scored (including non-triggers) is appended to `signals.csv` for post-hoc analysis.

---

## 8. Key Design Decisions

**Triple-barrier labeling over fixed-horizon returns.**
A simple "did price go up 1% in 5 days?" label ignores path and creates asymmetric risk. Triple-barrier labels tie directly to actual trade outcomes — a BUY label means the take-profit would have been hit before the stop-loss, so the model learns setups that produce real positive reward:risk, not just directional moves.

**Recall-first BUY, F1-first SELL.**
These objectives reflect what matters at each stage. A missed BUY is a missed opportunity with no recovery. A bad SELL is also bad, but the ATR stop provides a floor. So BUY is tuned to miss as few real entries as possible; SELL is balanced between not missing exits and not creating excessive churn.

**Independent champion models with runtime overrides.**
Each model is promoted to production as a `.pkl` file, independent of the other. The runtime `--sell-confidence` flag provides a soft tuning layer between retrains — adjusting sensitivity without touching the model. This decouples operational tuning from the (expensive) search-and-retrain cycle.

**4-hour bars for training, FinViz for candidate selection.**
4h bars strike a balance between intraday noise and the multi-day momentum that FinViz screeners capture. The screener narrows the universe to high-momentum, high-volume candidates at runtime, so the model only needs to make a binary decision on pre-filtered setups rather than scanning the entire market.

**Two-pass execution (SELL before BUY).**
Running SELL before BUY ensures losing positions are cut before new capital is deployed. It also decouples exit logic from entry logic — the SELL pass operates on Alpaca's actual held positions, not the FinViz watchlist, so a held ticker is never missed even if it drops out of the momentum screener.
