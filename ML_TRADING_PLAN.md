# ML Trading Model Plan

## Overview

Build a machine learning model using **aeon** (time series ML library) trained on historical stock data from Alpaca. The model will interpret candlestick indicators to make buy/sell/hold decisions.

---

## Architecture

```
1. Data Collection → 2. Feature Engineering → 3. Labeling → 4. Model Training → 5. Backtesting
```

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

### Current Approach: Future Return Classification

```
Label 0 (SELL):  future_return < -2%
Label 1 (HOLD):  -2% <= future_return <= +2%
Label 2 (BUY):   future_return > +2%
```

### Alternative Labeling Strategies to Explore

1. **Triple Barrier Method** (Lopez de Prado)
   - Take profit barrier (upper)
   - Stop loss barrier (lower)
   - Time barrier (max holding period)

2. **Trend Following**
   - Label based on SMA crossovers
   - Label based on price breaking support/resistance

3. **Volatility-Adjusted Returns**
   - Adjust thresholds based on recent ATR
   - Higher volatility = wider bands

---

## Step 4: Model Training with Aeon

### Recommended Models

| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| **RocketClassifier** | Fast, excellent accuracy | Less interpretable | Quick iterations |
| **MiniRocket** | Very fast, good accuracy | Less flexible | Large datasets |
| **InceptionTimeClassifier** | Deep learning, captures complex patterns | Slower, needs GPU | Final model |
| **HIVE-COTE 2** | State-of-the-art ensemble | Very slow | Benchmarking |

### Implementation

Create: `ml/trainer.py`

```python
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.deep_learning import InceptionTimeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import glob
import joblib

class TradingModelTrainer:
    def __init__(self):
        self.model = None
    
    def prepare_dataset(self, data_dir, feature_builder):
        """Load all parquet files and build combined dataset"""
        all_X, all_y = [], []
        files = glob.glob(f"{data_dir}/*.parquet")
        
        for f in files:
            df = pd.read_parquet(f)
            if len(df) > 100:  # Need enough data
                X, y = feature_builder.build_features(df)
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
        
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        
        return X, y
    
    def train(self, X, y, model_type='rocket'):
        """Train the classifier"""
        # IMPORTANT: Don't shuffle time series data!
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        if model_type == 'rocket':
            self.model = RocketClassifier(num_kernels=10000, random_state=42)
        elif model_type == 'inception':
            self.model = InceptionTimeClassifier(n_epochs=100, random_state=42)
        
        print(f"Training {model_type} on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred, 
                                    target_names=['Sell', 'Hold', 'Buy']))
        
        return self.model
    
    def save_model(self, path="models/trading_model.pkl"):
        """Save trained model"""
        joblib.dump(self.model, path)
    
    def load_model(self, path="models/trading_model.pkl"):
        """Load trained model"""
        self.model = joblib.load(path)
        return self.model
```

### Training Script

```python
from ml.feature_builder import FeatureBuilder
from ml.trainer import TradingModelTrainer

# Initialize
feature_builder = FeatureBuilder(window_size=20, horizon=5)
trainer = TradingModelTrainer()

# Load and prepare data
X, y = trainer.prepare_dataset("saved_data/historical", feature_builder)
print(f"Dataset shape: X={X.shape}, y={y.shape}")

# Train model
model = trainer.train(X, y, model_type='rocket')

# Save model
trainer.save_model("models/rocket_trading_model.pkl")
```

---

## Step 5: Live Inference / Backtesting

### Live Prediction

Create: `ml/predictor.py`

```python
import numpy as np
import joblib

class TradingPredictor:
    def __init__(self, model_path, feature_builder):
        self.model = joblib.load(model_path)
        self.feature_builder = feature_builder
    
    def predict(self, recent_bars_df):
        """
        Predict buy/sell/hold for current market state.
        recent_bars_df: DataFrame with last `window_size` candles
        """
        X, _ = self.feature_builder.build_features(recent_bars_df)
        
        if len(X) == 0:
            return None
        
        # Use only the most recent window
        latest_window = X[-1:] 
        
        prediction = self.model.predict(latest_window)[0]
        probabilities = self.model.predict_proba(latest_window)[0]
        
        labels = ['SELL', 'HOLD', 'BUY']
        return {
            'signal': labels[prediction],
            'confidence': probabilities[prediction],
            'probabilities': dict(zip(labels, probabilities))
        }
```

### Integration with Alpaca Streaming

```python
async def on_bar(self, bar):
    """Enhanced on_bar with ML prediction"""
    # Fetch recent history for this symbol
    recent_df = self.get_historical_data(bar.symbol, days=30)
    
    # Get ML prediction
    prediction = self.predictor.predict(recent_df)
    
    if prediction and prediction['confidence'] > 0.7:
        if prediction['signal'] == 'BUY':
            self.place_bracket_order(bar.symbol, qty=1, ...)
        elif prediction['signal'] == 'SELL':
            # Close position or short
            pass
```

---

## Project Structure

```
auto_trade/
├── data_collection/
│   ├── __init__.py
│   └── historical_collector.py    # Bulk data fetching
├── ml/
│   ├── __init__.py
│   ├── feature_builder.py         # Candlestick → ML features
│   ├── trainer.py                 # Aeon model training
│   └── predictor.py               # Live inference
├── saved_data/
│   ├── historical/                # Parquet files per ticker
│   ├── FinVizData.csv
│   └── scan_results.csv
├── models/                        # Saved trained models
│   └── rocket_trading_model.pkl
├── notebooks/
│   └── eda.ipynb
├── stock_picker/
│   └── stock_screener.py
├── techAnalysis.py                # Existing indicators
├── alpaca_trading.py              # Existing trading code
├── pyproject.toml
└── README.md
```

---

## Dependencies to Add

Add to `pyproject.toml`:

```toml
[tool.poetry.dependencies]
aeon = "^0.7.0"
scikit-learn = "^1.3.0"
pyarrow = "^14.0.0"  # For parquet support
joblib = "^1.3.0"    # Model serialization
```

Or install via pip:

```bash
pip install aeon scikit-learn pyarrow joblib
```

---

## Implementation Checklist

- [ ] **Phase 1: Data Collection**
  - [ ] Create `data_collection/historical_collector.py`
  - [ ] Fetch symbols list from Alpaca
  - [ ] Download 2+ years of daily data for 500+ stocks
  - [ ] Save to parquet files

- [ ] **Phase 2: Feature Engineering**
  - [ ] Create `ml/feature_builder.py`
  - [ ] Integrate existing `TechnicalAnalysis` indicators
  - [ ] Test feature generation on sample data

- [ ] **Phase 3: Model Training**
  - [ ] Create `ml/trainer.py`
  - [ ] Train initial RocketClassifier
  - [ ] Evaluate with classification report
  - [ ] Experiment with hyperparameters

- [ ] **Phase 4: Live Integration**
  - [ ] Create `ml/predictor.py`
  - [ ] Integrate with `on_bar` handler
  - [ ] Paper trade for validation

- [ ] **Phase 5: Refinement**
  - [ ] Try alternative labeling strategies
  - [ ] Add more features (volume patterns, etc.)
  - [ ] Test InceptionTime for comparison
  - [ ] Build backtesting framework

---

## Tips & Considerations

1. **Class Imbalance**: Markets are often sideways. Expect many "HOLD" labels. Use stratified sampling or class weights.

2. **Data Leakage**: Never shuffle time series data. Always use temporal splits.

3. **Transaction Costs**: Include commissions/slippage in backtesting.

4. **Overfitting**: Financial data is noisy. Use walk-forward validation.

5. **Feature Importance**: After training, analyze which indicators matter most.

---

## Next Steps

1. Start with data collection - this takes the longest
2. Run feature builder on a few stocks to validate
3. Train a quick RocketClassifier to establish a baseline
4. Iterate on features and labeling based on results

