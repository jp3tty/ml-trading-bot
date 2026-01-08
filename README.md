# Auto Trade - ML Stock Trading System

Automated stock trading system combining technical analysis with machine learning. Scans stocks via FinViz, trains ML models on historical candlestick patterns, and executes trades through Alpaca.

## Overview

This project has two modes of operation:

1. **Stock Scanner** - Screens stocks using technical indicators and candlestick patterns
2. **ML Trader** - Uses trained ML models to predict buy/sell signals and execute trades

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA COLLECTION                                     │
│  ┌──────────────────┐      ┌─────────────────┐      ┌─────────────────┐         │
│  │  Alpaca API      │ ──▶  │  Historical     │ ──▶  │  Parquet Files  │         │
│  │  (multi-symbol)  │      │  Collector      │      │  (per ticker)   │         │
│  └──────────────────┘      └─────────────────┘      └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ML PIPELINE                                         │
│  ┌──────────────────┐      ┌─────────────────┐      ┌─────────────────┐         │
│  │  Feature Builder │ ──▶  │  Aeon Trainer   │ ──▶  │  Trained Model  │         │
│  │  (indicators)    │      │  (ROCKET, etc)  │      │  (.pkl file)    │         │
│  └──────────────────┘      └─────────────────┘      └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TRADING (CRON)                                      │
│  ┌──────────────────┐      ┌─────────────────┐      ┌─────────────────┐         │
│  │  ML Trader       │ ──▶  │  Predictions    │ ──▶  │  Alpaca Orders  │         │
│  │  (scheduled)     │      │  (buy/sell/hold)│      │  (bracket)      │         │
│  └──────────────────┘      └─────────────────┘      └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
auto_trade/
├── alpaca_trading.py           # Stock scanner + Alpaca connection
├── ml_trader.py                # ML-based trading (cron job)
├── techAnalysis.py             # Technical indicators & patterns
├── data_collection/
│   ├── __init__.py
│   └── historical_collector.py # Bulk historical data fetching
├── ml/
│   ├── __init__.py
│   ├── feature_builder.py      # Candlestick → ML features
│   ├── trainer.py              # Aeon model training
│   └── predictor.py            # Live inference
├── models/                     # Saved trained models
├── saved_data/
│   ├── historical/             # Parquet files (per ticker)
│   ├── FinVizData.csv
│   └── scan_results.csv
├── stock_picker/
│   └── stock_screener.py       # FinViz scraper
├── notebooks/
│   └── eda.ipynb               # Exploratory analysis
├── ML_TRADING_PLAN.md          # ML implementation roadmap
├── pyproject.toml
└── README.md
```

## Components

| File | Description |
|------|-------------|
| `alpaca_trading.py` | Main scanner. Fetches data, runs tech analysis, outputs results |
| `ml_trader.py` | Cron-based ML trader. Fetches data, predicts, executes trades |
| `techAnalysis.py` | Technical analysis: RSI, momentum, candlestick patterns |
| `data_collection/historical_collector.py` | Bulk fetch 2+ years of data for ML training |
| `ml/feature_builder.py` | Converts OHLCV + indicators into ML-ready sliding windows |
| `ml/trainer.py` | Trains aeon time series classifiers (ROCKET, InceptionTime) |
| `ml/predictor.py` | Loads trained model for live predictions |
| `stock_picker/stock_screener.py` | Scrapes FinViz for momentum stocks |

## Technical Indicators

### Momentum Indicators
- **RSI** (14-period) - Relative Strength Index
- **Momentum** - 10-period price momentum
- **SMA 20/50** - Simple moving averages
- **Bullish/Bearish Momentum** - Composite trend signal

### Candlestick Patterns
- **Hammer/Inverted Hammer** - Reversal patterns
- **Doji** (standard, dragonfly, gravestone, long-legged) - Indecision patterns
- **Engulfing** (bullish/bearish) - Reversal patterns

## ML Models

Using [aeon](https://github.com/aeon-toolkit/aeon) time series classifiers:

| Model | Description | Speed |
|-------|-------------|-------|
| RocketClassifier | Convolution-based, excellent accuracy | Fast |
| MiniRocket | Lightweight ROCKET variant | Very Fast |
| InceptionTimeClassifier | Deep learning approach | Slower (GPU) |

## Setup

### Prerequisites
- Python 3.13+
- Alpaca account (paper or live)
- Poetry (dependency management)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd auto_trade

# Install dependencies with Poetry
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

### 1. Stock Scanner (Technical Analysis)

```bash
# Run the scanner - outputs to saved_data/scan_results.csv
poetry run python alpaca_trading.py
```

### 2. Collect Historical Data (ML Training)

```bash
# Fetch 2 years of daily data for 500+ stocks
poetry run python data_collection/historical_collector.py
```

### 3. Train ML Model

```python
from ml.feature_builder import FeatureBuilder
from ml.trainer import TradingModelTrainer

feature_builder = FeatureBuilder(window_size=20, horizon=5)
trainer = TradingModelTrainer()

X, y = trainer.prepare_dataset("saved_data/historical", feature_builder)
model = trainer.train(X, y, model_type='rocket')
trainer.save_model("models/rocket_trading_model.pkl")
```

### 4. ML Trader (Automated Trading)

```bash
# Dry run - preview without trading
poetry run python ml_trader.py --dry-run

# Test with specific symbols
poetry run python ml_trader.py --dry-run --symbols AAPL MSFT GOOGL

# Paper trading with 70% confidence threshold
poetry run python ml_trader.py --confidence 0.7

# Live trading (use with caution!)
poetry run python ml_trader.py --live
```

#### ML Trader Options

| Flag | Description |
|------|-------------|
| `--dry-run` | Preview trades without executing |
| `--symbols` | Specific tickers to analyze |
| `--confidence` | Min ML confidence threshold (default: 0.6) |
| `--model` | Path to trained model (default: models/rocket_trading_model.pkl) |
| `--live` | Use live trading instead of paper |

## GitHub Actions

### Stock Scanner Schedule

| Schedule | Time (ET) | Description |
|----------|-----------|-------------|
| `30 14 * * 1-5` | 9:30 AM | Market open scan |
| `30 18 * * 1-5` | 1:30 PM | Mid-day scan |

### ML Trader Schedule

| Schedule | Time (ET) | Description |
|----------|-----------|-------------|
| `0 14-21 * * 1-5` | 9AM-4PM | Hourly during market hours |

### Required Secrets

Add these secrets to your GitHub repository:
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`

### Manual Trigger

Both workflows support `workflow_dispatch` for manual triggering from the GitHub Actions tab.

## Scanner Results Table

| Column | Description |
|--------|-------------|
| `Ticker` | Stock symbol |
| `Latest Price` | Latest closing price |
| `Engulfing Signal` | Latest engulfing pattern (bullish/bearish/neutral) |
| `Momentum Trend` | Composite momentum (Bullish/Bearish/Neutral) |
| `Hammer Signal` | Count of hammer patterns in last 5 days |
| `Doji Signal` | Type of latest doji pattern |

## FinViz Screening Criteria

Default filters (configurable in `stock_screener.py`):
- Market cap: Small cap and over
- Relative volume: Over 2x average
- Performance: Up over 5 days
- Sorted by: Market cap (descending)

## Output Files

| File | Description |
|------|-------------|
| `saved_data/scan_results.csv` | Scanner results table |
| `saved_data/historical/*.parquet` | Historical OHLCV data per ticker |
| `models/*.pkl` | Trained ML models |

## Development Roadmap

See [ML_TRADING_PLAN.md](ML_TRADING_PLAN.md) for the detailed ML implementation plan including:
- Data collection strategy
- Feature engineering approach
- Model training workflow
- Live integration steps

## License

MIT