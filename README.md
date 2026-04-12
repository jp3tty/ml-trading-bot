# Auto Trade - ML Stock Trading System

Automated stock trading system using separate binary ML models for entry and exit signal detection. Scans momentum stocks via FinViz, trains independent BUY and SELL detectors on historical candlestick data, and executes bracket orders through Alpaca's paper or live trading API.

## Strategy

**Early-exit short-term trading** using two independent binary classifiers:

| Detector | Objective | Tuned For |
|----------|-----------|-----------|
| **BUY** | Identify high-probability entries | Precision — minimize false positives |
| **SELL** | Detect early signs of decline | Recall — exit fast, miss fewer turns |

The system defaults to **HOLD**. A position is only entered when the BUY detector fires with sufficient confidence, and only exited when the SELL detector triggers on a held position.

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
│  Modes: catch22 (primary) · indicators · combined                  │
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
# Quick search
poetry run python ml/binary_sell_search.py --quick

# Full grid search
poetry run python ml/binary_sell_search.py
```

Champion model saved to `models/sell_search_results/champion_sell_<timestamp>.pkl`.

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
| Classifier | XGBoost |
| Window size | 40 bars |
| Horizon | 9 bars |
| Buy threshold | 1.5% |
| Decision threshold | 0.826 |
| Precision | 58.6% |
| Feature mode | catch22 |

### SELL Detector

Optimised for recall (fast exits). Champion selection requires `recall ≥ 20%` and `precision ≥ 40%`, ranked by F1. Train with `ml/binary_sell_search.py`.

### Feature Modes

| Mode | Description | Features |
|------|-------------|----------|
| `catch22` | 22 canonical time series statistics on close + volume | 44 total |
| `indicators` | RSI, momentum, hammer, doji, engulfing, volatility | 11 × window |
| `combined` | Flattened indicators + catch22 | 11×window + 22 |

### Champion Selection

**BUY**: `precision ≥ 50%` and `recall ≥ 10%` → ranked by F1
**SELL**: `recall ≥ 20%` and `precision ≥ 40%` → ranked by F1

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

## Development Roadmap

See [ML_TRADING_PLAN.md](ML_TRADING_PLAN.md) for the full implementation plan and phase checklist.

## License

MIT
