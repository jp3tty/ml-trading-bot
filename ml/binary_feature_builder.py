"""
Binary Feature Builder for BUY Signal Detection
Creates binary classification features using triple-barrier labeling:
  - Label 1 (BUY): Price hits take_profit barrier before stop_loss within horizon candles
  - Label 0 (NOT BUY): Stop loss hit first, or neither barrier hit (time barrier)

Optimized for 4-hour candlestick data and high-precision trading signals.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from techAnalysis import TechnicalAnalysis

try:
    import pycatch22
    CATCH22_AVAILABLE = True
except ImportError:
    CATCH22_AVAILABLE = False

class BinaryFeatureBuilder:
    MODE_INDICATORS = 'indicators'
    MODE_CATCH22 = 'catch22'
    MODE_COMBINED = 'combined'

    def __init__(self, window_size=20, horizon=6, buy_threshold=None, take_profit=0.01,
                 stop_loss=0.005, feature_mode='indicators', min_volume=0):
        self.window_size = window_size
        self.horizon = horizon
        # buy_threshold is a legacy alias for take_profit
        self.take_profit = buy_threshold if buy_threshold is not None else take_profit
        self.stop_loss = stop_loss
        self.feature_mode = feature_mode
        self.min_volume = min_volume
        self.ta = TechnicalAnalysis()

        self.indicator_cols = [
            'open_norm', 'high_norm', 'low_norm', 'close_norm',
            'rsi', 'momentum_strength', 'hammer', 'engulfing', 'doji_num',
            'volume_norm', 'volatility',
            # MACD
            'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
            # Bollinger Bands
            'bb_position', 'bb_width',
            # Multi-timeframe context
            'weekly_momentum', 'trend_sma',
            # Composite momentum signals
            'bullish_momentum', 'bearish_momentum',
        ]

        self.catch22_names = [
            'DN_HistogramMode_5', 'DN_HistogramMode_10',
            'CO_f1ecac', 'CO_FirstMin_ac', 'CO_HistogramAMI_even_2_5',
            'CO_trev_1_num', 'MD_hrv_classic_pnn40',
            'SB_BinaryStats_mean_longstretch1', 'SB_TransitionMatrix_3ac_sumdiagcov',
            'PD_PeriodicityWang_th0_01', 'CO_Embed2_Dist_tau_d_expfit_meandiff',
            'IN_AutoMutualInfoStats_40_gaussian_fmmi', 'FC_LocalSimple_mean1_tauresrat',
            'DN_OutlierInclude_p_001_mdrmd', 'DN_OutlierInclude_n_001_mdrmd',
            'SP_Summaries_welch_rect_area_5_1', 'SB_BinaryStats_diff_longstretch0',
            'SB_MotifThree_quantile_hh', 'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
            'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1', 'SP_Summaries_welch_rect_centroid',
            'FC_LocalSimple_mean3_stderr'
        ]

        if feature_mode in [self.MODE_CATCH22, self.MODE_COMBINED] and not CATCH22_AVAILABLE:
            raise ImportError("pycatch22 required. Run: poetry add pycatch22")

    def build_features(self, df):
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Need enough data for indicators + window + horizon
        min_required = self.window_size + self.horizon + 50
        if len(df) < min_required:
            return np.array([]), np.array([])

        # add technical indicators
        df = self._add_indicators(df)

        # Create triple-barrier BUY labels
        df = self._create_triple_barrier_labels(df)

        # Build features based on mode
        if self.feature_mode == self.MODE_CATCH22:
            return self._build_catch22_features(df)
        elif self.feature_mode == self.MODE_INDICATORS:
            return self._build_indicator_features(df)
        else:  # combined
            return self._build_combined_features(df)

    def _add_indicators(self, df):
        """Add technical indicators to dataframe"""
        # Momentum
        df = self.ta.momentum_trend(df)
                
        # Candlestick patterns
        df['hammer'] = self.ta.hammer_signal(df).astype(int)
        df['engulfing'] = self.ta.engulfing_signal(df)

        # Doji as numeric
        doji_map = {'standard': 1, 'dragonfly': 2, 'gravestone': 3, 'long_legged': 4}
        df['doji_num'] = self.ta.doji_signal(df).map(doji_map).fillna(0)

        # Normalize OHLCV (percent change over window)
        for col in ['open', 'high', 'low', 'close']:
            df[f'{col}_norm'] = df[col].pct_change(self.window_size)

        # Volume normalization
        df['volume_avg'] = df['volume'].rolling(self.window_size).mean()
        df['volume_norm'] = df['volume'] / df['volume_avg'].replace(0, np.nan)

        # Volatility (rolling std of returns)
        df['return'] = df['close'].pct_change()
        df['volatility'] = df['return'].rolling(window=self.window_size).std()

        # MACD (normalized by close price to keep scale-invariant)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd_raw = ema12 - ema26
        macd_signal_raw = macd_raw.ewm(span=9, adjust=False).mean()
        df['macd_norm'] = macd_raw / df['close']
        df['macd_signal_norm'] = macd_signal_raw / df['close']
        df['macd_hist_norm'] = (macd_raw - macd_signal_raw) / df['close']

        # Bollinger Bands (20-period)
        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_range = (bb_upper - bb_lower).replace(0, np.nan)
        df['bb_position'] = (df['close'] - bb_lower) / bb_range  # 0=at lower, 1=at upper
        df['bb_width'] = bb_range / bb_mid                        # wider = more volatile

        # Multi-timeframe context (4h bars: 6/bar/day → ~42 bars/week)
        df['weekly_momentum'] = df['close'].pct_change(42)
        df['trend_sma'] = (df['sma_20'] > df['sma_50']).astype(float)

        # Composite momentum signals (cast bool → float for window builder)
        df['bullish_momentum'] = df['bullish_momentum'].astype(float)
        df['bearish_momentum'] = df['bearish_momentum'].astype(float)


        return df

    def _create_triple_barrier_labels(self, df):
        """Triple-barrier labeling (Lopez de Prado).

        For each candle, scan the next `horizon` candles using high/low prices:
          - Label 1: take_profit barrier touched before stop_loss barrier
          - Label 0: stop_loss hit first, both hit the same candle, or time barrier
        Conservative tie-breaking: when both barriers are touched on the same candle,
        the stop loss wins (label 0).
        """
        closes = df['close'].values
        highs  = df['high'].values
        lows   = df['low'].values
        n      = len(closes)
        labels = np.full(n, np.nan)

        for i in range(n - 1):
            entry = closes[i]
            upper = entry * (1.0 + self.take_profit)
            lower = entry * (1.0 - self.stop_loss)

            end          = min(i + 1 + self.horizon, n)
            future_highs = highs[i + 1:end]
            future_lows  = lows[i + 1:end]

            upper_hits = np.where(future_highs >= upper)[0]
            lower_hits = np.where(future_lows  <= lower)[0]

            first_upper = upper_hits[0] if len(upper_hits) > 0 else self.horizon
            first_lower = lower_hits[0] if len(lower_hits) > 0 else self.horizon

            # Strict less-than: ties (same candle) go to stop loss (conservative)
            labels[i] = 1 if first_upper < first_lower else 0

        df['label'] = labels
        return df

    def _build_indicator_features(self, df):
        """Build sliding window features technical indicators"""
        X, y = [], []

        # Get valid indices (no NaN in features or labels)
        feature_cols_available = [c for c in self.indicator_cols if c in df.columns]
        valid_mask = df[feature_cols_available + ['label']].notna().all(axis=1)

        # Optional volume filter
        if self.min_volume > 0 and 'volume_avg' in df.columns:
            valid_mask &= df['volume_avg'] > self.min_volume

        valid_indices = df.index[valid_mask].tolist()

        for i in range(len(valid_indices) - self.window_size):
            window_indices = valid_indices[i:i + self.window_size]

            # Extract feature window
            window = df.loc[window_indices, feature_cols_available].values

            # Get label at end of window
            label = df.loc[window_indices[-1], 'label']

            # Skip if any NaN values
            if not np.isnan(window).any() and pd.notna(label):
                X.append(window.T) # Transpose to (n_channels, window_size)
                y.append(int(label))

        return np.array(X), np.array(y)

    def _build_catch22_features(self, df):
        """build catch22 features from price windows"""
        X, y = [], []

        valid_mask = df['label'].notna()
        if self.min_volume > 0 and 'volume_avg' in df.columns:
            valid_mask &= df['volume_avg'] > self.min_volume

        valid_indices = df.index[valid_mask].tolist()

        for i in range(len(valid_indices) - self.window_size):
            window_indices = valid_indices[i:i + self.window_size]
            label = df.loc[window_indices[-1], 'label']

            if pd.notna(label):
                try:
                    # Compute catch22 on close prices
                    close_window = df.loc[window_indices, 'close'].values
                    catch22_close = pycatch22.catch22_all(close_window)['values']

                    # catch22 on volume
                    volume_window = df.loc[window_indices, 'volume'].values.astype(float)
                    catch22_volume = pycatch22.catch22_all(volume_window)['values']

                    # Combine (44 features total)
                    features = catch22_close + catch22_volume

                    if not any(np.isnan(features)):
                        X.append(features)
                        y.append(int(label))
                except Exception:
                    continue

        return np.array(X), np.array(y)

    def _build_combined_features(self, df):
        """build combined indicator + catch22 features"""
        X, y = [], []

        feature_cols_available = [c for c in self.indicator_cols if c in df.columns]
        valid_mask = df[feature_cols_available + ['label']].notna().all(axis=1)

        if self.min_volume > 0 and 'volume_avg' in df.columns:
            valid_mask &= df['volume_avg'] > self.min_volume

        valid_indices = df.index[valid_mask].tolist()

        for i in range(len(valid_indices) - self.window_size):
            window_indices = valid_indices[i:i + self.window_size]
            label = df.loc[window_indices[-1], 'label']

            if pd.notna(label):
                try:
                    # Flattened indicators
                    indicator_window = df.loc[window_indices, feature_cols_available].values
                    indicator_flat = indicator_window.flatten()

                    # catch22 features
                    close_window = df.loc[window_indices, 'close'].values
                    catch22_features = pycatch22.catch22_all(close_window)['values']

                    # Combine (indicator_flat + catch22_features)
                    combined = np.concatenate([indicator_flat, catch22_features])

                    if not any(np.isnan(combined)):
                        X.append(combined)
                        y.append(int(label))
                except Exception:
                    continue

        return np.array(X), np.array(y)
        
    def get_label_distribution(self, y):
        """get distribution of labels"""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        return {
            'total': total,
            'buy': int(counts[unique == 1][0]) if 1 in unique else 0,
            'not_buy': int(counts[unique == 0][0]) if 0 in unique else 0,
            'buy_pct': float(counts[unique == 1][0] / total * 100) if 1 in unique else 0,
        }

    def get_feature_names(self):
        """Return list of feature names based on mode"""
        if self.feature_mode == self.MODE_INDICATORS:
            names = []
            feature_cols = [c for c in self.indicator_cols]
            for t in range(self.window_size):
                for col in feature_cols:
                    names.append(f"{col}_t{t}")
            return names
        elif self.feature_mode == self.MODE_CATCH22:
            return ([f"close_{n}" for n in self.catch22_names] + 
                    [f"volume_{n}" for n in self.catch22_names])
        else:  # combined
            indicator_names = []
            feature_cols = [c for c in self.indicator_cols]
            for t in range(self.window_size):
                for col in feature_cols:
                    indicator_names.append(f"{col}_t{t}")
            return indicator_names + [f"catch22_{n}" for n in self.catch22_names]