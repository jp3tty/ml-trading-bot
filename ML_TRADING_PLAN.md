
# ML Trading Model Plan

## Overview

Build separate **binary classification models** for BUY and SELL signal detection, trained on historical stock data from Alpaca. This modular approach allows independent optimization of entry (BUY) and exit (SELL) decisions.

### Strategy: Early-Exit Short-Term Trading
- **Entry**: High-precision BUY signals (minimize false positives)
- **Exit**: Fast SELL signals at first sign of decline
- **Hold**: Default state when neither detector triggers

### Why Separate Detectors?

| Approach | Pros | Cons |
|----------|------|------|
| 3-class (SELL/HOLD/BUY) | Simpler, one model | Can't tune entry/exit independently |
| **Separate BUY/SELL** | Independent optimization, better control | More complex, two models to maintain |

For an early-exit strategy, separate detectors allow:
- **BUY detector**: Tuned for high precision (55%+ win rate)
- **SELL detector**: Tuned for sensitivity (quick exits)

---

## Architecture

```
                    ┌──────────────────┐
                    │  Data Collection │
                    └────────┬─────────┘
                             │
                   ┌─────────▼───────────┐
                   │ Feature Engineering │
                   └─────────┬───────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼────────┐     │     ┌────────▼────────┐
     │  BUY Detector   │     │     │  SELL Detector  │
     │  (Binary: 0/1)  │     │     │  (Binary: 0/1)  │
     └────────┬────────┘     │     └────────┬────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │   ML Trader     │
                    │  (Orchestrator) │
                    └─────────────────┘
```

### Signal Flow
1. **No position**: Only BUY detector runs → triggers entry if confident
2. **In position**: Only SELL detector runs → triggers exit if declining
3. **Default**: HOLD (no action)

---

## Step 1: Gather Historical Data at Scale

### Goals
- Fetch historical OHLCV data for **500+ stocks**
- Collect **2-3 years** of daily data minimum
- Store efficiently using **Parquet** format

### Implementation

Create a new module: `data_collection/historical_collector.py`

```python
import pandas as pd
from datetime import datetime, timedelta
from alpaca_trade_api.rest import TimeFrame
import time
import os

class HistoricalDataCollector:
    def __init__(self, alpaca_conn, data_dir="saved_data/historical"):
        self.api = alpaca_conn.api
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def get_all_tradeable_symbols(self, min_price=5, max_price=500):
        """Get all active, tradeable US equities from Alpaca"""
        assets = self.api.list_assets(status='active', asset_class='us_equity')
        symbols = [
            a.symbol for a in assets 
            if a.tradable and a.shortable  # liquid stocks
            and not a.symbol.isdigit()  # filter out weird tickers
            and '.' not in a.symbol  # no preferred shares
        ]
        return symbols
    
    def fetch_and_save_historical(self, symbols, start_date, end_date, 
                                   timeframe=TimeFrame.Day, batch_size=200):
        """
        Fetch historical data for many symbols in batches.
        Alpaca allows up to 200 symbols per multi-bar request.
        """
        all_data = {}
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            print(f"Fetching batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")
            
            try:
                # Multi-symbol request (much faster than one-by-one)
                bars = self.api.get_bars(
                    batch,
                    timeframe,
                    start=start_date,
                    end=end_date,
                    feed="iex",
                    adjustment='all'  # Include splits/dividends
                ).df
                
                if not bars.empty:
                    # Group by symbol and save
                    for symbol in bars.index.get_level_values('symbol').unique():
                        symbol_data = bars.xs(symbol, level='symbol')
                        all_data[symbol] = symbol_data
                        
            except Exception as e:
                print(f"Error fetching batch: {e}")
            
            time.sleep(0.5)  # Rate limiting
        
        # Save as parquet (efficient storage)
        for symbol, df in all_data.items():
            df.to_parquet(f"{self.data_dir}/{symbol}.parquet")
        
        print(f"Saved {len(all_data)} symbols to {self.data_dir}")
        return all_data
```

### Usage

```python
from alpaca_trading import AlpacaConnection
from data_collection.historical_collector import HistoricalDataCollector

conn = AlpacaConnection(paper=True)
collector = HistoricalDataCollector(conn)

# Get tradeable symbols
symbols = collector.get_all_tradeable_symbols()
print(f"Found {len(symbols)} tradeable symbols")

# Fetch 2 years of daily data
collector.fetch_and_save_historical(
    symbols[:500],  # Start with top 500
    start_date="2022-01-01",
    end_date="2024-12-01",
    timeframe=TimeFrame.Day
)
```

---

## Step 2: Feature Engineering

### Goals
- Convert raw OHLCV data into ML-ready features
- Leverage existing indicators from `techAnalysis.py`
- Create sliding windows for time series classification

### Features to Include

| Feature | Source | Description |
|---------|--------|-------------|
| `open_norm` | OHLCV | Normalized open price (% change) |
| `high_norm` | OHLCV | Normalized high price |
| `low_norm` | OHLCV | Normalized low price |
| `close_norm` | OHLCV | Normalized close price |
| `rsi` | `TechnicalAnalysis` | Relative Strength Index (14-period) |
| `momentum_strength` | `TechnicalAnalysis` | Price momentum as % |
| `hammer` | `TechnicalAnalysis` | Hammer/Inverted Hammer (0/1) |
| `engulfing` | `TechnicalAnalysis` | Engulfing pattern (0/1/2) |
| `doji_num` | `TechnicalAnalysis` | Doji type (0-4) |

### Implementation

Create: `ml/feature_builder.py`

```python
import numpy as np
import pandas as pd
from techAnalysis import TechnicalAnalysis

class FeatureBuilder:
    def __init__(self, window_size=20, horizon=5):
        """
        window_size: Number of candles to look back (input sequence)
        horizon: Days ahead to determine buy/sell label
        """
        self.window_size = window_size
        self.horizon = horizon
        self.ta = TechnicalAnalysis()
    
    def build_features(self, df):
        """
        Build feature matrix from OHLCV data.
        Returns: X (n_samples, n_channels, window_size), y (n_samples,)
        """
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Add technical indicators
        df = self.ta.momentum_trend(df)
        df['hammer'] = self.ta.hammer_signal(df).astype(int)
        df['engulfing'] = self.ta.engulfing_signal(df)
        
        # Doji as numeric (0=none, 1=standard, 2=dragonfly, 3=gravestone, 4=long_legged)
        doji_map = {'standard': 1, 'dragonfly': 2, 'gravestone': 3, 'long_legged': 4}
        df['doji_num'] = self.ta.doji_signal(df).map(doji_map).fillna(0)
        
        # Normalize OHLCV (percent change from window start)
        for col in ['open', 'high', 'low', 'close']:
            df[f'{col}_norm'] = df[col].pct_change(self.window_size)
        
        # Feature columns for model
        feature_cols = [
            'open_norm', 'high_norm', 'low_norm', 'close_norm',
            'rsi', 'momentum_strength', 'hammer', 'engulfing', 'doji_num'
        ]
        
        # Create labels: future return classification
        df['future_return'] = df['close'].shift(-self.horizon) / df['close'] - 1
        df['label'] = pd.cut(
            df['future_return'], 
            bins=[-np.inf, -0.02, 0.02, np.inf],
            labels=[0, 1, 2]  # 0=Sell, 1=Hold, 2=Buy
        )
        
        # Build sliding windows
        X, y = [], []
        valid_idx = df.dropna(subset=feature_cols + ['label']).index
        
        for i in range(len(valid_idx) - self.window_size):
            window = df.loc[valid_idx[i:i+self.window_size], feature_cols].values
            label = df.loc[valid_idx[i+self.window_size-1], 'label']
            
            if not np.isnan(window).any():
                X.append(window.T)  # Shape: (n_channels, window_size)
                y.append(int(label))
        
        return np.array(X), np.array(y)
```

---

## Step 3: Labeling Strategy

### Binary Classification Approach (Current)

Instead of 3-class classification, we use **separate binary detectors**:

#### BUY Detector Labels
```
Label 1 (BUY):     future_return > buy_threshold (e.g., +1.5%)
Label 0 (NOT_BUY): Everything else
```

#### SELL Detector Labels (Future)
```
Label 1 (SELL):     future_return < sell_threshold (e.g., -1%)
Label 0 (NOT_SELL): Everything else
```

### Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `window_size` | Candles of history for features | 15-30 |
| `horizon` | Candles ahead for label calculation | 3-9 |
| `buy_threshold` | Min return to label as BUY | 0.015-0.03 (1.5-3%) |
| `sell_threshold` | Max return to label as SELL | -0.01 to -0.02 |

### Metric Targets for Early-Exit Strategy

| Metric | Target | Why |
|--------|--------|-----|
| **Precision** | ≥ 55% | Win rate must exceed 50% + transaction costs |
| **Recall** | ≥ 10% | Enough signals to make trading worthwhile |
| **F1 Score** | Maximize (with floors) | Balance precision and recall |

---

## Step 4: Model Training (Binary Detectors)

### Recommended Models for Binary Classification

| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| **RandomForest** | Fast, handles imbalance well | Less pattern recognition | Quick iterations |
| **XGBoost** | Excellent performance, handles imbalance | Requires tuning | Production |
| **catch22 + RF** | Time series features + fast training | Feature extraction overhead | Current approach |

### Feature Modes

| Mode | Features | Speed | Accuracy |
|------|----------|-------|----------|
| `indicators` | Technical indicators (RSI, momentum, patterns) | Fast | Good |
| `catch22` | 22 time series statistics on price + volume | Medium | Better |
| `combined` | Both indicator and catch22 features | Slow | Best |

### Implementation

Implemented in: `ml/binary_search.py` and `ml/binary_feature_builder.py`

```python
from ml.binary_feature_builder import BinaryFeatureBuilder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Build features
feature_builder = BinaryFeatureBuilder(
    window_size=15,
    horizon=6,
    buy_threshold=0.02,
    feature_mode='catch22'
)

X, y = feature_builder.build_features(df)

# Scale features (required for sklearn models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train with class balancing
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)
```

### Hyperparameter Search

Use `ml/binary_search.py` for grid search:

```python
from ml.binary_search import BinaryHyperparameterSearch

search = BinaryHyperparameterSearch(
    data_dir="saved_data/historical_4h",
    results_dir="models/binary_search_results"
)

# Run search
champion = search.run_search(quick=True)

# Champion saved to:
# - models/binary_search_results/champion_binary_{timestamp}.pkl
# - models/binary_search_results/champion_params_{timestamp}.json
```

### Champion Selection Criteria

Champions are selected by **highest F1 score** with minimum thresholds:

```python
# Only consider models that meet both criteria:
if precision >= 0.5 and recall >= 0.1:
    score = f1  # Rank by F1 among qualifying models
```

---

## Step 5: Live Inference / Backtesting

### Binary Predictor

Create: `ml/binary_predictor.py`

```python
import numpy as np
import joblib
import glob
import os

from ml.binary_feature_builder import BinaryFeatureBuilder

class BinaryBuyPredictor:
    """
    Loads a trained binary BUY detector and makes predictions.
    """
    
    def __init__(self, model_path=None):
        """
        Args:
            model_path: Path to champion .pkl file. If None, loads latest.
        """
        if model_path is None:
            model_path = self._find_latest_champion()
        
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.threshold = self.model_data['threshold']
        self.params = self.model_data['params']
        
        # Create feature builder with same params used in training
        self.feature_builder = BinaryFeatureBuilder(
            window_size=self.params['window_size'],
            horizon=self.params.get('horizon', 6),
            buy_threshold=self.params.get('buy_threshold', 0.02),
            feature_mode='catch22'
        )
    
    def _find_latest_champion(self):
        """Find most recent champion model"""
        pattern = "models/binary_search_results/champion_binary_*.pkl"
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No champion models found")
        return files[-1]
    
    def predict(self, df):
        """
        Predict BUY/NOT_BUY for most recent data point.
        
        Returns:
            dict with 'signal', 'probability', 'threshold', 'is_buy'
        """
        X, _ = self.feature_builder.build_features(df)
        
        if len(X) == 0:
            return None
        
        # Use most recent window, flatten if needed
        latest = X[-1:].reshape(1, -1) if len(X[-1:].shape) > 2 else X[-1:]
        
        # Scale and predict
        latest_scaled = self.scaler.transform(latest)
        proba = self.model.predict_proba(latest_scaled)[0, 1]
        
        return {
            'signal': 'BUY' if proba >= self.threshold else 'NOT_BUY',
            'probability': float(proba),
            'threshold': self.threshold,
            'is_buy': proba >= self.threshold
        }
    
    def get_model_info(self):
        """Return model metadata"""
        return {
            'params': self.params,
            'threshold': self.threshold,
            'precision': self.model_data.get('precision'),
            'recall': self.model_data.get('recall'),
            'f1': self.model_data.get('f1'),
            'win_rate': self.model_data.get('win_rate')
        }
```

### Integration with ML Trader

```python
class MLTrader:
    def __init__(self, paper=True):
        self.conn = AlpacaConnection(paper=paper)
        
        # Load binary detectors
        self.buy_detector = BinaryBuyPredictor()  # Loads latest champion
        # self.sell_detector = BinarySellPredictor()  # Future
    
    def should_buy(self, df):
        """Check if BUY detector triggers"""
        prediction = self.buy_detector.predict(df)
        if prediction and prediction['is_buy']:
            return True, prediction['probability']
        return False, 0.0
    
    def should_sell(self, df):
        """Check if SELL detector triggers (placeholder)"""
        # TODO: Implement SELL detector
        # For now, use simple stop-loss / take-profit
        return False, 0.0
    
    def run(self, symbols):
        for symbol in symbols:
            df = self.fetch_recent_data(symbol)
            
            if not self.check_existing_position(symbol):
                # No position - check BUY
                should_buy, confidence = self.should_buy(df)
                if should_buy:
                    self.execute_trade(symbol, 'BUY', confidence)
            else:
                # In position - check SELL
                should_sell, confidence = self.should_sell(df)
                if should_sell:
                    self.execute_trade(symbol, 'SELL', confidence)
```

---

## Project Structure

```
ml-trading-bot/
├── data_collection/
│   ├── __init__.py
│   └── historical_collector.py       # Bulk data fetching
├── ml/
│   ├── __init__.py
│   ├── feature_builder.py            # Original 3-class features (legacy)
│   ├── binary_feature_builder.py     # Binary classification features
│   ├── binary_search.py              # Hyperparameter grid search
│   ├── binary_predictor.py           # BUY detector inference
│   ├── trainer.py                    # Original trainer (legacy)
│   └── predictor.py                  # Original predictor (legacy)
├── saved_data/
│   ├── historical/                   # Daily parquet files
│   ├── historical_4h/                # 4-hour parquet files
│   └── scan_results.csv
├── models/
│   ├── binary_search_results/        # Grid search outputs
│   │   ├── champion_binary_*.pkl     # Best model + scaler + params
│   │   ├── champion_params_*.json    # Human-readable params
│   │   └── binary_search_*.csv       # All search results
│   └── rocket_trading_model.pkl      # Legacy 3-class model
├── stock_picker/
│   └── stock_screener.py
├── techAnalysis.py                   # Technical indicators
├── alpaca_trading.py                 # Alpaca REST (ML + collectors)
├── Streaming_Method/                 # Pattern scanner, streaming, legacy IBKR
├── ml_trader.py                      # Main trading orchestrator
├── ML_TRADING_PLAN.md                # This document
├── pyproject.toml
└── README.md
```

---

## Dependencies

### Required for Binary Detectors

```toml
[tool.poetry.dependencies]
scikit-learn = "^1.3.0"    # RandomForest, StandardScaler
xgboost = "^2.0.0"         # XGBoost classifier
pycatch22 = "^0.4.0"       # Time series features
pyarrow = "^14.0.0"        # Parquet support
joblib = "^1.3.0"          # Model serialization
pandas = "^2.0.0"
numpy = "^1.24.0"
```

### Optional (for alternative approaches)

```toml
aeon = "^0.7.0"            # Time series ML (RocketClassifier, etc.)
```

### Install via pip:

```bash
pip install scikit-learn xgboost pycatch22 pyarrow joblib pandas numpy
```

---

## Implementation Checklist

### Phase 1: Data Collection ✅
- [x] Create `data_collection/historical_collector.py`
- [x] Fetch symbols list from Alpaca
- [x] Download historical data (daily and 4-hour)
- [x] Save to parquet files

### Phase 2: Feature Engineering ✅
- [x] Create `ml/binary_feature_builder.py`
- [x] Implement indicator mode (technical indicators)
- [x] Implement catch22 mode (time series features)
- [x] Implement combined mode

### Phase 3: BUY Detector Training ✅
- [x] Create `ml/binary_search.py` (hyperparameter grid search)
- [x] Implement precision/recall optimization
- [x] Champion selection by F1 with precision/recall floors
- [x] Save champion model with scaler and threshold

### Phase 4: BUY Detector Integration ✅
- [x] Create `ml/binary_predictor.py`
- [x] Update `ml_trader.py` to use binary predictor
- [x] Paper trade for validation (`paper_trade_validator.py`)

### Phase 5: SELL Detector 🔄
- [x] Create `ml/binary_sell_feature_builder.py`
- [x] Create `ml/binary_sell_search.py`
- [ ] Train SELL detector (optimize for speed/sensitivity)
- [x] Create `ml/binary_sell_predictor.py`
- [x] Integrate with ml_trader.py

### Phase 6: Full Integration 🔄
- [x] Combine BUY and SELL detectors in ml_trader.py
- [x] Implement position management logic
  - [x] `get_held_positions()` — fetch all open Alpaca positions once per run as `{symbol: position}`
  - [x] Two-pass run loop: SELL pass over held positions first, then BUY pass over watchlist
  - [x] BUY pass skips any symbol already held; SELL pass is independent of the watchlist
  - [x] `paper_trade_validator.py` mirrors the same two-pass logic with per-side signal logging
- [ ] Paper trade complete system (BUY + SELL together, pending SELL champion)
- [ ] Build backtesting framework

### Phase 7: Refinement ⏳
- [ ] Ensemble multiple models
- [ ] Add market context features (SPY correlation)
- [ ] Walk-forward validation
- [ ] Live trading (with small position sizes)

---

## Tips & Considerations

1. **Class Imbalance**: BUY signals are rare. Use `class_weight='balanced'` or `scale_pos_weight`.

2. **Data Leakage**: Never shuffle time series data. Always use temporal splits (`shuffle=False`).

3. **Transaction Costs**: A 50% win rate is break-even before fees. Target 55%+ precision.

4. **Overfitting**: Financial data is noisy. Watch for high recall with near-zero precision.

5. **Vacuously True Results**: 100% precision with <1% recall = model barely fires. Useless.

6. **Threshold Optimization**: The optimal threshold from training is saved with the model. Use it!

---

## Precision vs Recall Trade-off

For early-exit trading strategy:

| Precision | Recall | Trades/Month | Profitability |
|-----------|--------|--------------|---------------|
| 100% | 0.01% | ~1 | ❌ Statistically meaningless |
| 60% | 20% | Many | ✅ Sweet spot |
| 55% | 35% | Very many | ✅ Good if fees are low |
| 50% | 50% | Too many | ❌ Break-even before fees |

**Target: 55-60% precision with 20-40% recall**

---

## Improving the BUY Detector

After initial training, you may find precision is acceptable but recall is low (model is too selective). Here are strategies to improve performance:

---

### 1. Lower the Decision Threshold (Quick Win)

Your model outputs probabilities. Trade precision for recall by lowering the threshold:

```python
# Current: threshold ~0.5 or optimized for precision
y_pred = (y_proba >= 0.5).astype(int)

# Try lower threshold for more signals:
y_pred = (y_proba >= 0.35).astype(int)  # More BUYs, lower precision
```

Control this in `find_optimal_threshold()` via the `min_precision` parameter.

---

### 2. Use All Available Data

The default `max_files=200` limits training data. Change to use all parquet files:

```python
# In load_data(), change:
def load_data(self, window_size, horizon, buy_threshold, max_files=None):  # Use all

# Or set explicitly:
files = glob.glob(f"{self.data_dir}/*.parquet")  # No slicing
```

More data → better generalization → potentially better recall.

---

### 3. Relax the Buy Threshold

Lower `buy_threshold` creates more BUY labels in training:

```python
# Current search space
'buy_threshold': [0.01, 0.015, 0.02, 0.025]  # 1-2.5% gains

# More relaxed (try these)
'buy_threshold': [0.005, 0.008, 0.01, 0.012]  # 0.5-1.2% gains
```

With 4h candles and `horizon=6`, that's targeting 0.5-1.2% gains over 24 hours.

---

### 4. Add More Technical Indicators

Expand beyond the current feature set:

| Indicator | Description | Implementation |
|-----------|-------------|----------------|
| **MACD** | Momentum/trend crossovers | `ta-lib` or custom |
| **Bollinger Bands** | Volatility breakouts | Rolling mean ± 2*std |
| **ATR** | Average True Range (volatility) | `ta-lib` or custom |
| **OBV** | On-Balance Volume | Cumulative volume direction |
| **Stochastic** | Overbought/oversold | %K and %D lines |
| **ADX** | Trend strength | Directional movement index |

Example implementation:

```python
# Bollinger Bands
df['bb_middle'] = df['close'].rolling(20).mean()
df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(20).std()
df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(20).std()
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

# MACD
ema12 = df['close'].ewm(span=12).mean()
ema26 = df['close'].ewm(span=26).mean()
df['macd'] = ema12 - ema26
df['macd_signal'] = df['macd'].ewm(span=9).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']
```

---

### 5. Multi-Timeframe Features

Add longer-term context to short-term signals:

```python
# For 4-hour data
df['daily_trend'] = df['close'].rolling(6).mean() > df['close'].rolling(24).mean()
df['weekly_momentum'] = df['close'].pct_change(42)  # ~1 week of 4h bars
df['monthly_trend'] = df['close'].rolling(180).mean().pct_change(42)
```

---

### 6. Filter for Quality Stocks

Exclude low-volume or penny stocks that are harder to predict:

```python
# In data loading or feature building
if df['volume'].mean() < 500000:  # Skip illiquid
    continue
if df['close'].mean() < 5:  # Skip penny stocks
    continue
```

---

### 7. Class Weights Tuning

Adjust class weights to prioritize catching BUY signals:

```python
# RandomForestClassifier
RandomForestClassifier(
    class_weight={0: 1, 1: 3},  # 3x importance on BUY
    ...
)

# XGBoost
XGBClassifier(
    scale_pos_weight=10,  # Higher = more weight on minority class
    ...
)
```

---

### 8. Ensemble Multiple Models

Combine predictions from different configurations:

```python
# Train multiple models with different settings
model_rf = RandomForestClassifier(...)
model_xgb = XGBClassifier(...)
model_catch22 = RandomForestClassifier(...)  # Trained on catch22 features

# Ensemble predictions
pred1 = model_rf.predict_proba(X)[:, 1]
pred2 = model_xgb.predict_proba(X)[:, 1]
pred3 = model_catch22.predict_proba(X)[:, 1]

ensemble_proba = (pred1 + pred2 + pred3) / 3
buy_signal = ensemble_proba > 0.4
```

---

### 9. Add Market Context Features

Include broader market conditions:

```python
# Fetch SPY or sector ETF data
spy_df = fetch_spy_data()

# Add market momentum
df['market_trend'] = spy_df['close'].pct_change(10)
df['market_above_sma'] = (spy_df['close'] > spy_df['close'].rolling(50).mean()).astype(int)

# Sector relative strength
df['rel_strength'] = df['close'].pct_change(20) - spy_df['close'].pct_change(20)
```

---

### Improvement Priority Order

1. **Use all data files** - Free improvement, no code changes needed
2. **Lower `buy_threshold` to 0.008-0.01** - More training signal
3. **Adjust `min_precision` to 0.4** - Trade some precision for recall
4. **Add MACD and Bollinger Bands** - Proven short-term indicators
5. **Ensemble models** - Combine multiple approaches

---

## Next Steps

### Immediate — Complete Phase 5
1. Run `poetry run python ml/binary_sell_search.py --quick` to train the SELL champion
2. Confirm `models/sell_search_results/champion_sell_*.pkl` is saved with recall ≥ 20%
3. Run `paper_trade_validator.py --dry-run` on a live trading day to confirm both detectors fire

### Short-term — Complete Phase 6
1. Paper trade the full BUY + SELL system for several weeks
2. Monitor `paper_trade_log/signals.csv` and open positions daily
3. Build a simple backtesting framework against the existing parquet files

### Medium-term — Phase 7 Refinement
1. Tune BUY threshold if signal rate is too low (lower `min_precision` in search)
2. Add market context features (SPY trend, VIX level, sector ETF performance)
3. Implement walk-forward validation to check for overfitting over time
4. Ensemble multiple BUY or SELL champion models

### Long-term — Production
1. Paper trade for 1+ months with consistent positive P&L before going live
2. Switch to live trading with small position sizes
3. Schedule model retraining monthly as new data accumulates

