
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

### Triple-Barrier Method (Current)

BUY labels use the **triple-barrier method** (Lopez de Prado) instead of a fixed-horizon return check. For each candle, the subsequent `horizon` candles are scanned using their high/low prices:

- **Label 1 (BUY):** `high >= entry * (1 + take_profit)` before `low <= entry * (1 - stop_loss)`
- **Label 0 (NOT BUY):** stop-loss hit first, both barriers on the same candle (tie → stop-loss wins), or neither hit within `horizon` candles (time barrier)

This is strictly better than the fixed-horizon approach because:
- It defines success by actual risk/reward, not just direction at a fixed future point
- Labels are path-aware — a price that spikes then crashes is correctly labeled 1 only if the spike comes first
- Positive-class balance is naturally higher and more meaningful

#### SELL Detector Labels
```
Label 1 (SELL):     future_return < sell_threshold (e.g., -0.5%)
Label 0 (NOT_SELL): Everything else
```
SELL labels remain fixed-horizon for now. Triple-barrier could be applied here too if the SELL detector underperforms.

### Key Parameters

| Parameter | Description | Search Range |
|-----------|-------------|--------------|
| `window_size` | Candles of history for features | 15–30 |
| `horizon` | Max candles to wait for a barrier to be hit | 6–9 |
| `take_profit` | Upper barrier — % gain to label as BUY | 0.005–0.015 |
| `stop_loss` | Lower barrier — % loss that cancels the BUY label | 0.003–0.008 |
| `min_precision` | Minimum precision floor for threshold optimizer | 0.48–0.52 |
| `min_recall` | Minimum recall floor for threshold optimizer | 0.05–0.10 |

Combinations where `stop_loss >= take_profit` are filtered out (must have positive reward:risk).

### Metric Targets — Architectural Decision (2026-05-06)

The BUY detector is now optimized for **recall over precision**. Key reasoning:

- The SELL detector has 100% recall — it catches every declining position and exits quickly
- A bad BUY entry becomes a short, bounded loss; it does not need to be avoided at all costs
- Chasing high BUY precision (≥ 48%) was causing near-zero recall — the model missed almost all real opportunities
- Enough precision to avoid excessive commission drag is the only hard requirement

| Metric | Target | Why |
|--------|--------|-----|
| **Recall** | Maximize | Catch as many real entries as possible |
| **Precision** | ≥ 35–40% floor | Avoid excessive commission drag from constant bad entries |
| **F1** | Not the primary objective | Replaced by recall as the ranking metric |

Champion selection ranks by **recall** (not F1) among models that meet the precision floor.
Threshold optimization picks the lowest threshold that still meets the precision floor, maximizing recall.

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

# Build features with triple-barrier labels
feature_builder = BinaryFeatureBuilder(
    window_size=15,
    horizon=6,
    take_profit=0.010,   # +1% take-profit barrier
    stop_loss=0.005,     # -0.5% stop-loss barrier (2:1 reward:risk)
    feature_mode='combined'
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

Champions are selected by **highest recall** subject to a minimum precision floor (0.35–0.40). The decision threshold is chosen by `find_optimal_threshold()`, which picks the lowest threshold that still meets the precision floor, directly maximizing recall:

```python
# Precision floor enforced during threshold selection:
valid_idx = precisions >= min_precision
# Lowest valid threshold chosen → maximizes recall
# Champion requires: precision >= min_precision, ranked by recall
```

Search space for the floor: `min_precision ∈ [0.35, 0.40]`. (`min_recall` was removed from the search space — the optimizer maximizes recall directly.)

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
- [x] Incremental CSV + JSON saving after every combination (crash-safe)
- [x] Switch labeling to triple-barrier method (`take_profit` + `stop_loss` barriers)
- [x] Switch default feature mode to `combined` (indicators + catch22)
- [x] Add MACD, Bollinger Bands, and multi-timeframe features to feature builder
- [x] **2026-05-06: Reframe BUY optimization target from precision → recall**
  - Threshold optimizer now maximizes recall subject to a minimum precision floor (0.35–0.40)
  - Champion selection now ranks by recall, not F1
  - `min_recall` removed from search space (redundant — optimizer directly maximizes it)
  - Precision floor lowered from 0.48 → 0.35–0.40 (enough to avoid commission drag)
- [x] Run search with recall-optimized objective — champion: Random Forest, window=21, horizon=9, take_profit=0.8%, stop_loss=0.5%, threshold=0.005, precision=37.1%, recall=100%, F1=0.541
- [ ] **Re-run search** — a timeframe mismatch bug (daily bars in live inference vs 4h training) was fixed 2026-05-27; this champion was never validated against correct inference. Quick search space updated to expand horizon=[6,9,12]. Run: `poetry run python ml/binary_search.py --quick --max-files 200`

### Phase 4: BUY Detector Integration ✅
- [x] Create `ml/binary_predictor.py`
- [x] Update `ml_trader.py` to use binary predictor
- [x] Paper trade for validation (`paper_trade_validator.py`)

### Phase 5: SELL Detector ✅
- [x] Create `ml/binary_sell_feature_builder.py`
- [x] Create `ml/binary_sell_search.py`
  - Fixed import fallback (`try ml.binary_sell_feature_builder / except binary_sell_feature_builder`)
  - Added `--max-files` CLI flag to limit parquet files per combination
  - Refactored `run_search()` to cache feature builds per data config — reduced 6-hour run to ~15 min
- [x] Train SELL detector — champion: XGBoost, window=20, horizon=5, sell_thresh=0.5%, precision=40.1%, recall=100%, F1=0.573
- [x] Create `ml/binary_sell_predictor.py`
- [x] Integrate with ml_trader.py
- [ ] **Re-run SELL search** — same timeframe mismatch applied; re-train after new BUY champion is confirmed

### Phase 6: Full Integration ✅ / 🔄 Active
- [x] Combine BUY and SELL detectors in ml_trader.py
- [x] Implement position management logic
  - [x] `get_held_positions()` — fetch all open Alpaca positions once per run as `{symbol: position}`
  - [x] Two-pass run loop: SELL pass over held positions first, then BUY pass over watchlist
  - [x] BUY pass skips any symbol already held; SELL pass is independent of the watchlist
  - [x] `paper_trade_validator.py` mirrors the same two-pass logic with per-side signal logging
- [x] Dry run confirmed — both detectors load and scan without errors (2026-04-15)
- [x] BUY threshold fixed — lowered `--confidence` to 0.45 in CI workflow to match model's actual probability range (2026-05-08)
- [x] Paper trading active — full BUY + SELL system running via GitHub Actions (2026-05-08)
- [x] Trade logging enhanced — `ml_trader.py` writes RSI, momentum, and order ID to `orders.csv` at BUY/SELL time (2026-05-09)
- [x] Streamlit dashboard built — `dashboard/app.py` shows account value, active positions, trade history with entry/exit indicators, and signal log (2026-05-09)
- [x] CI auto-commits trade logs — workflow commits updated `orders.csv` and `signals.csv` after each run so dashboard data stays current (2026-05-09)
- [x] Switched dashboard to `alpaca-py` SDK — resolved persistent auth errors on Streamlit Cloud caused by the older `alpaca-trade-api` library (2026-05-11)
- [x] Dashboard deployed to Streamlit Community Cloud — live and accessible (2026-05-11)
- [x] Added runtime SELL confidence floor — `--sell-confidence 0.3` overrides the model's hairpin 0.040 threshold without retraining; SELL only fires when `probability ≥ 0.30` (2026-05-13)
- [x] Replaced fixed 1% stop loss with ATR-based safety net — stop set at `entry − 2.0 × ATR(14)`, giving each position room proportional to its actual volatility; take profit widened to 20% ceiling so the ML sell model handles normal exits; falls back to 3% if ATR is unavailable (2026-05-14)
- [x] Disabled take-profit ceiling exit — `USE_TAKE_PROFIT = False` in `ml_trader.py`; orders now use `oto` (stop-loss only) so all exits go via ML SELL signal or ATR stop; TP can be re-enabled by flipping the flag (2026-05-19)
- [x] Fixed stale-price stop-loss trigger — BUY orders previously anchored limit price and ATR stop to prior day's close; bot now fetches live ask via `get_live_price()` (quote → last trade → close fallback) immediately before each order so the stop is correctly placed relative to the actual fill price (2026-05-19)
- [x] Fixed duplicate buys and unlogged bracket exits — deduplication guard added so the same ticker isn't bought twice in a single run; bracket exit orders are now logged correctly to `orders.csv` (2026-05-26)
- [x] Fixed sync_bracket_exits() re-logging same exits every run — now checks existing order IDs in orders.csv before writing; also logs all unrecorded fills per symbol (not just the most recent), handling cases where a symbol had multiple buys with separate bracket orders (2026-05-27)
- [x] Fixed timeframe mismatch — live inference was fetching daily bars (TimeFrame.Day) but the models were trained on 4-hour bars; changed fetch_recent_data() to TimeFrame(4h), lookback reduced to 60 days (~300 4h bars, well above minimum) (2026-05-27)
- [x] Fixed paper_trade_validator.py bracket order parameters — was using hardcoded 1% SL / 2% TP; now uses ATR-based stop (2×ATR, 3% fallback) and TP disabled, matching ml_trader.py intent (2026-05-27)
- [x] Cleaned orders.csv — removed 24 duplicate SELL rows created by the old sync function (2026-05-27)
- [ ] Monitor paper trade results and P&L via dashboard
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
| **37%** | **100%** | **Very many** | **⚠️ Current — SELL detector limits downside** |

**Current approach (2026-05-06):** Maximize recall subject to a 35–40% precision floor. The SELL detector's fast exits bound losses from bad BUY entries, making high recall viable even at 37% precision.

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

1. ~~**Use all data files**~~ — `--max-files 200` is the recommended balance (~3 hrs); full 872 files for final validation
2. ~~**Lower `buy_threshold`**~~ — Superseded by triple-barrier labeling
3. ~~**Adjust `min_precision`**~~ — Done (0.35–0.40 recall-first approach)
4. ~~**Triple-barrier labeling**~~ — Done; replaces fixed-horizon approach
5. **Retrain BUY + SELL models** — timeframe mismatch bug fixed; new search pending (quick search: `--quick --max-files 200`)
6. **Add MACD and Bollinger Bands** — after retraining validates the corrected inference
7. **Ensemble models** — combine multiple approaches

---

## Next Steps

### Immediate — Retrain Models (2026-05-27)
Three bugs were discovered and fixed that invalidate the current champions as live performance benchmarks:
1. **Timeframe mismatch** — live inference was fetching daily bars; models trained on 4h. Fixed in `fetch_recent_data()`.
2. **Wrong bracket order parameters** — `paper_trade_validator.py` used hardcoded 1% SL / 2% TP instead of ATR-based. Fixed.
3. **Duplicate sync logging** — `sync_bracket_exits()` re-logged same exits every run. Fixed.

Net P&L as of 2026-05-27 (50 closed trades): **-$190** — bracket stop exits (-$297) offset by ML SELL exits (+$107). Losses driven primarily by the bugs above.

Retrain order:
1. Run BUY search: `poetry run python ml/binary_search.py --quick --max-files 200` (~3 hrs)
2. Validate new BUY champion; replace `.pkl` if improved
3. Run SELL search: `poetry run python ml/binary_sell_search.py --quick --max-files 60` (~15 min)
4. Resume paper trading with corrected models and monitor P&L

### Short-term — Phase 6 Completion
1. Paper trade with corrected models for several weeks to validate end-to-end behavior
2. Track P&L, exit types (ATR stop vs SELL Signal), and indicator context per trade
3. Build a simple backtesting framework against existing parquet files
4. Assess live precision vs training precision to detect overfitting

### Medium-term — Phase 7 Refinement
1. Add market context features (SPY trend, VIX level, sector ETF performance)
2. Implement walk-forward validation to check for overfitting over time
3. Ensemble multiple BUY champion models
4. Add MACD and Bollinger Bands to the feature set
5. Consider replacing SELL ML model with trailing stop (simpler, no retraining needed)

### Long-term — Production
1. Paper trade for 1+ months with consistent positive P&L before going live
2. Switch to live trading with small position sizes
3. Schedule model retraining monthly as new data accumulates

