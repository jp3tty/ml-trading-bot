"""
Binary Feature Builder for BUY Signal Detection
Creates binary classification features:
  - Label 1 (BUY): Future return exceeds buy_threshold
  - Label 0 (NOT BUY): Everything else

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

    def __init__(self, window_size=20, horizon=6, buy_threshold=0.02, feature_mode='indicators', min_volume=0):
        self.window_size = window_size
        self.horizon = horizon
        self.buy_threshold = buy_threshold
        self.feature_mode = feature_mode
        self.min_volume = min_volume
        self.ta = TechnicalAnalysis()

        self.indicator_cols = [
            'open_norm', 'high_norm', 'low_norm', 'close_norm',
            'rsi', 'momentum_strength', 'hammer', 'engulfing', 'doji_num',
            'volume_norm', 'volatility'
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

        # Create binary BUY labels
        df = self._create_binary_labels(df)

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

        return df

    def _create_binary_labels(self, df):
        """Create binary BUY labels based on future returns"""
        # Calculate future returns
        df['future_return'] = df['close'].shift(-self.horizon) / df['close'] - 1

        # Binary label: 1 if return exceeds threshold, 0 otherwise
        df['label'] = (df['future_return'] > self.buy_threshold).astype(int)

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