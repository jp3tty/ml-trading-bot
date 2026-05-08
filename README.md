# Auto Trade - ML Stock Trading System

Automated stock trading system using separate binary ML models for entry and exit signal detection. Scans momentum stocks via FinViz, trains independent BUY and SELL detectors on historical candlestick data, and executes bracket orders through Alpaca's paper or live trading API.

## Strategy

**Early-exit short-term trading** using two independent binary classifiers:

| Detector | Objective | Tuned For |
|----------|-----------|-----------|
| **BUY** | Catch as many valid entries as possible | Recall — SELL handles bad entries |
| **SELL** | Detect early signs of decline | Recall — exit fast, miss fewer turns |

The system defaults to **HOLD**. A position is only entered when the BUY detector fires, and only exited when the SELL detector triggers on a held position.

### Architectural Decision (2026-05-06)

The BUY detector is optimized for **recall over precision**. Because the SELL detector has 100% recall and cuts losing positions quickly, a bad BUY entry becomes a short, bounded loss rather than a catastrophe. Chasing high BUY precision was causing the model to miss the majority of real opportunities (near-zero recall). The precision floor is now set to 35–40% (enough to avoid excessive commission drag) while the threshold optimizer maximizes recall within that constraint. Champion selection ranks by recall, not F1.

### BUY Labeling: Triple-Barrier Method

BUY labels are generated using the **triple-barrier method** (Lopez de Prado), not a simple fixed-horizon return. For each candle, the next `horizon` bars are scanned using high/low prices:

- **Label 1 (BUY):** take-profit barrier (`+take_profit%`) is touched *before* the stop-loss barrier
- **Label 0 (NOT BUY):** stop-loss is hit first, both barriers hit the same candle (conservative tie-break), or neither barrier is hit within `horizon` candles (time barrier)

This produces meaningfully balanced labels and filters for setups with an explicit positive reward:risk ratio.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA COLLECTION                             │
│  Alpaca API  ──▶  HistoricalCollector  ──▶  Parquet (per ticker)   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FEATURE ENGINEERING                           │
│  BinaryFeatureBuilder / BinarySellFeatureBuilder                    │
│  Modes: combined (primary) · catch22 · indicators                  │
└──────────────────┬──────────────────────────┬───────────────────────┘
                   │                          │
          ┌────────▼────────┐       ┌─────────▼───────┐
          │  BUY Detector   │       │  SELL Detector  │
          │  binary_search  │       │  sell_search    │
          │  XGBoost / RF   │       │  XGBoost / RF   │
          └────────┬────────┘       └─────────┬───────┘
                   │                          │
                   └────────────┬─────────────┘
                                │
               ┌────────────────▼────────────────┐
               │           ML Trader             │
               │   Pass 1: SELL all held tickers │
               │   Pass 2: BUY watchlist tickers │
               └────────────────┬────────────────┘
                                │
                       Alpaca bracket orders
```

## Project Structure

```
ml-trading-bot/
├── alpaca_trading.py                    # Alpaca REST client
├── ml_trader.py                         # Main trading orchestrator (cron job)
├── paper_trade_validator.py             # Paper trading validation + signal log
├── techAnalysis.py                      # RSI, momentum, candlestick patterns
├── data_collection/
│   ├── historical_collector.py          # Bulk daily data fetching
│   └── historical_collector_4h.py       # 4-hour data fetching
├── ml/
│   ├── binary_feature_builder.py        # BUY features (catch22 / indicators)
│   ├── binary_sell_feature_builder.py   # SELL features (inverted labels)
│   ├── binary_search.py                 # BUY hyperparameter grid search
│   ├── binary_sell_search.py            # SELL hyperparameter grid search
│   ├── binary_predictor.py              # BUY detector — live inference
│   ├── binary_sell_predictor.py         # SELL detector — live inference
│   ├── feature_builder.py               # (legacy — 3-class)
│   ├── predictor.py                     # (legacy — 3-class)
│   └── trainer.py                       # (legacy — ROCKET)
├── models/
│   ├── binary_search_results/           # BUY champion models + search CSVs
│   └── sell_search_results/             # SELL champion models + search CSVs
├── paper_trade_log/
│   ├── signals.csv                      # All BUY/SELL signals logged per run
│   └── orders.csv                       # All orders placed (paper or live)
├── saved_data/
│   ├── historical/                      # Daily parquet files
│   └── historical_4h/                   # 4-hour parquet files
├── stock_picker/
│   └── stock_screener.py                # FinViz momentum screener
├── Streaming_Method/                    # Legacy pattern scanner
├── ML_TRADING_PLAN.md                   # Implementation roadmap
├── pyproject.toml
└── README.md
```

## Setup

### Prerequisites
- Python 3.12
- Alpaca account (paper or live)
- Poetry

### Installation

```bash
git clone https://github.com/jp3tty/ml-trading-bot.git
cd ml-trading-bot
pip install poetry
poetry install
```

### Environment Variables

Create a `.env` file in the project root:

```env
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
```

## Usage

### 1. Collect Historical Data

```bash
# Daily bars (used for live inference)
poetry run python data_collection/historical_collector.py

# 4-hour bars (used for model training)
poetry run python data_collection/historical_collector_4h.py
```

### 2. Train the BUY Detector

```bash
# Quick search (~30 min)
poetry run python ml/binary_search.py --quick

# Full grid search (~2-4 hours)
poetry run python ml/binary_search.py
```

Champion model saved to `models/binary_search_results/champion_binary_<timestamp>.pkl`.

### 3. Train the SELL Detector

```bash
# Quick search — use --max-files to limit data loaded per combination (recommended)
poetry run python ml/binary_sell_search.py --quick --max-files 60

# Full grid search
poetry run python ml/binary_sell_search.py --max-files 60
```

Champion model saved to `models/sell_search_results/champion_sell_<timestamp>.pkl`.

> **Note:** Without `--max-files`, the search loads all 400+ parquet files per combination
> and can take many hours. 60 files gives a representative sample and completes in ~15 min.

### 4. Validate with Paper Trading

```bash
# Dry run — scan signals without placing orders
poetry run python paper_trade_validator.py --dry-run

# Live paper trading — places real bracket orders on Alpaca paper account
poetry run python paper_trade_validator.py

# View cumulative signal log summary
poetry run python paper_trade_validator.py --report

# Scan specific symbols
poetry run python paper_trade_validator.py --dry-run --symbols AAPL TSLA NVDA
```

### 5. Run the ML Trader

```bash
# Dry run
poetry run python ml_trader.py --dry-run

# Paper trading (default)
poetry run python ml_trader.py

# Specify confidence threshold
poetry run python ml_trader.py --confidence 0.7

# Live trading
poetry run python ml_trader.py --live

# Use specific model files
poetry run python ml_trader.py \
  --model models/binary_search_results/champion_binary_20260114_113724.pkl \
  --sell-model models/sell_search_results/champion_sell_<timestamp>.pkl
```

#### ML Trader Options

| Flag | Description |
|------|-------------|
| `--dry-run` | Preview trades without executing |
| `--symbols` | Specific tickers to scan for BUY signals |
| `--confidence` | Min BUY confidence threshold (default: 0.6) |
| `--model` | Path to BUY champion `.pkl` (default: latest) |
| `--sell-model` | Path to SELL champion `.pkl` (default: latest) |
| `--live` | Use live trading instead of paper |

## Trading Loop

Each run executes two passes:

**Pass 1 — SELL** (held positions from Alpaca):
- Fetches all open positions once via `get_held_positions()`
- Runs SELL detector on every held ticker
- Closes position if SELL signal fires
- Independent of the FinViz watchlist — held tickers are never missed

**Pass 2 — BUY** (FinViz watchlist):
- Fetches momentum stocks from FinViz screener
- Skips any ticker already held
- Runs BUY detector; places bracket order if signal fires above confidence threshold

## Models

### BUY Detector (Current Champion)

| Parameter | Value |
|-----------|-------|
| Classifier | Random Forest |
| Window size | 30 bars |
| Horizon | 9 bars |
| Take profit | 0.5% |
| Stop loss | 0.3% |
| Decision threshold | 0.507 |
| Precision | 48.0% |
| Recall | 31.0% |
| F1 | 0.377 |
| Feature mode | combined |
| Labeling | Triple-barrier |

> **Note:** The above champion was produced before triple-barrier labeling was introduced and does not yet reflect the new search. A new search with triple-barrier labels is the active next step.

### SELL Detector (Current Champion)

| Parameter | Value |
|-----------|-------|
| Classifier | XGBoost |
| Window size | 20 bars |
| Horizon | 5 bars |
| Sell threshold | 0.5% |
| Decision threshold | 0.040 |
| Precision | 40.1% |
| Recall | 100.0% |
| F1 | 0.573 |
| Feature mode | catch22 |

Optimised for recall (fast exits). Champion selection requires `recall ≥ 20%` and `precision ≥ 40%`, ranked by F1.

### Feature Modes

| Mode | Description | Features |
|------|-------------|----------|
| `catch22` | 22 canonical time series statistics on close + volume | 44 total |
| `indicators` | RSI, momentum, hammer, doji, engulfing, volatility | 11 × window |
| `combined` | Flattened indicators + catch22 | 11×window + 22 |

### Champion Selection

**BUY**: `precision ≥ min_precision` (searched: 0.35–0.40) → ranked by **recall**
**SELL**: `recall ≥ 20%` and `precision ≥ 40%` → ranked by F1

The BUY threshold is chosen to maximize recall subject to the precision floor. The SELL threshold is chosen to maximize F1 subject to both floors.

## Output Files

| File | Description |
|------|-------------|
| `paper_trade_log/signals.csv` | Every BUY/SELL signal scored per run |
| `paper_trade_log/orders.csv` | Every order placed with entry/TP/SL |
| `models/binary_search_results/` | BUY champion `.pkl` + search results `.csv` |
| `models/sell_search_results/` | SELL champion `.pkl` + search results `.csv` |
| `saved_data/historical_4h/*.parquet` | 4-hour OHLCV data per ticker |

## FinViz Screening Criteria

Default filters (configurable in `stock_picker/stock_screener.py`):
- Market cap: Small cap and over
- Relative volume: Over 2× average
- Performance: Up over 5 days
- Sorted by: Market cap descending

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data collection (daily + 4h parquet) | ✅ Complete |
| 2 | Feature engineering (catch22 + indicators + combined) | ✅ Complete |
| 3 | BUY detector training + champion selection | 🔄 In progress |
| 4 | BUY detector integrated into trading loop | ✅ Complete |
| 5 | SELL detector training + champion selection | ✅ Complete |
| 6 | Full BUY + SELL integration + paper trading | ⏳ Blocked on Phase 3 |
| 7 | Refinement (ensemble, market context, walk-forward) | ⏳ Planned |

**Active work:** Switched BUY labeling to triple-barrier method (Lopez de Prado). Running new
hyperparameter search with `take_profit` + `stop_loss` barriers, dual precision/recall constraints,
and combined feature mode. Previous fixed-horizon approach produced near-zero recall across all
parameter combinations, making the champion model effectively non-functional for live trading.

## Development Roadmap

See [ML_TRADING_PLAN.md](ML_TRADING_PLAN.md) for the full implementation plan and phase checklist.

## License

MIT
