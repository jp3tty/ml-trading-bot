# Auto Trade - ML Stock Trading System

Automated stock trading system combining technical analysis with machine learning. Scans stocks via FinViz, trains ML models on historical candlestick patterns, and executes trades through Alpaca.

## Overview

This project has two modes of operation:

1. **Stock Scanner** - Screens stocks using technical indicators and candlestick patterns
2. **ML Trader** - Uses trained ML models to predict buy/sell signals and execute trades

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA COLLECTION                                    │
│  ┌──────────────────┐      ┌─────────────────┐      ┌─────────────────┐         │
│  │  Alpaca API      │ ──▶ │  Historical     │ ──▶  │  Parquet Files  │         │
│  │  (multi-symbol)  │      │  Collector      │      │  (per ticker)   │         │
│  └──────────────────┘      └─────────────────┘      └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ML PIPELINE                                         │
│  ┌──────────────────┐      ┌─────────────────┐      ┌─────────────────┐         │
│  │  Feature Builder │ ──▶ │  Model Trainer  │ ──▶  │  Trained Model  │         │
│  │  (indicators +   │      │  (ROCKET, RF,   │      │  (.pkl file)    │         │
│  │   catch22)       │      │   XGBoost)      │      │                 │         │
│  └──────────────────┘      └─────────────────┘      └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TRADING (CRON)                                      │
│  ┌──────────────────┐      ┌─────────────────┐      ┌─────────────────┐         │
│  │  ML Trader       │ ──▶ │  Predictions    │ ──▶  │  Alpaca Orders  │         │
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
│   ├── feature_builder.py      # Candlestick → ML features (indicators + catch22)
│   ├── trainer.py              # ROCKET model training
│   ├── predictor.py            # Live inference
│   ├── hyperparameter_search.py # Grid search for ROCKET
│   └── catch22_search.py       # Grid search comparing ROCKET vs catch22
├── models/
│   ├── search_results/         # ROCKET hyperparameter search results
│   └── catch22_results/        # catch22 comparison results
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
| `ml/feature_builder.py` | Converts OHLCV + indicators + catch22 into ML features |
| `ml/trainer.py` | Trains ROCKET time series classifier |
| `ml/predictor.py` | Loads trained model for live predictions |
| `ml/hyperparameter_search.py` | Grid search for ROCKET hyperparameters |
| `ml/catch22_search.py` | Compares ROCKET vs catch22 with RF/XGBoost |
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

## Feature Modes

The feature builder supports three modes:

| Mode | Features | Use Case |
|------|----------|----------|
| `indicators` | RSI, momentum, hammer, doji, engulfing (9 features × window) | Time series with ROCKET |
| `catch22` | 22 canonical time series features for close + volume (44 total) | Tabular ML (RF, XGBoost) |
| `combined` | Flattened indicators + catch22 | Best of both worlds |

## ML Models

### Time Series Classifiers (aeon)

| Model | Description | Speed |
|-------|-------------|-------|
| RocketClassifier | Convolution-based, excellent accuracy | Fast |
| MiniRocket | Lightweight ROCKET variant | Very Fast |

### Tabular Classifiers (for catch22/combined)

| Model | Description | Best For |
|-------|-------------|----------|
| Random Forest | Ensemble of decision trees | Interpretability |
| XGBoost | Gradient boosted trees | Accuracy |

## Setup

### Prerequisites
- Python 3.12
- Alpaca account (paper or live)
- Poetry (dependency management)
- Microsoft C++ Build Tools (for pycatch22 on Windows)

### Installation

```bash
# Clone the repository
git clone https://github.com/jp3tty/ml-trading-bot.git
cd ml-trading-bot

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

```bash
# Train ROCKET classifier directly
poetry run python ml/trainer.py
```

Or in Python:

```python
from ml.feature_builder import FeatureBuilder
from ml.trainer import TradingModelTrainer

feature_builder = FeatureBuilder(window_size=20, horizon=5)
trainer = TradingModelTrainer()

X, y = trainer.prepare_dataset("saved_data/historical", feature_builder)
model = trainer.train(X, y, model_type='rocket')
trainer.save_model("models/rocket_trading_model.pkl")
```

### 4. Hyperparameter Search

#### ROCKET Search
```bash
# Find best ROCKET parameters
poetry run python ml/hyperparameter_search.py
```

#### catch22 Comparison Search
```bash
# Quick search (fewer combinations, ~5-10 min)
poetry run python ml/catch22_search.py --quick

# Full search (~1-2 hours)
poetry run python ml/catch22_search.py
```

### 5. ML Trader (Automated Trading)

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

## Hyperparameters

### Feature Engineering

| Parameter | Values | Description |
|-----------|--------|-------------|
| `window_size` | 10, 20, 30 | Days of history model sees |
| `horizon` | 3, 5, 7 | Days ahead to predict |
| `label_threshold` | 0.01, 0.02, 0.03 | % threshold for BUY/SELL |
| `feature_mode` | indicators, catch22, combined | Which features to use |

### Model Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `n_kernels` | 1000-10000 | ROCKET kernel count |
| `n_estimators` | 100, 200 | Trees in RF/XGBoost |
| `max_depth` | 5, 10, None | Tree depth |

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

## Output Files

| File | Description |
|------|-------------|
| `saved_data/scan_results.csv` | Scanner results table |
| `saved_data/historical/*.parquet` | Historical OHLCV data per ticker |
| `models/*.pkl` | Trained ML models |
| `models/search_results/*.csv` | ROCKET hyperparameter search results |
| `models/catch22_results/*.csv` | catch22 comparison results |

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

## Development Roadmap

See [ML_TRADING_PLAN.md](ML_TRADING_PLAN.md) for the detailed ML implementation plan including:
- Data collection strategy
- Feature engineering approach
- Model training workflow
- Live integration steps

## License

MIT
