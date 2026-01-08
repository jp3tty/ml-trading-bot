import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from techAnalysis import TechnicalAnalysis

# Optional catch22 import
try:
    import pycatch22
    CATCH22_AVAILABLE = True
except ImportError:
    CATCH22_AVAILABLE = False


class FeatureBuilder:
    """
    Converts OHLCV data into ML-ready feature matrices with labels.
    Uses sliding windows of candlestick data + technical indicators.
    
    Supports three feature modes:
    - 'indicators': Your technical indicators only (RSI, hammer, etc.)
    - 'catch22': catch22 time series features only
    - 'combined': Both indicators and catch22 features
    """

    # Feature modes
    MODE_INDICATORS = 'indicators'
    MODE_CATCH22 = 'catch22'
    MODE_COMBINED = 'combined'

    def __init__(self, window_size=20, horizon=5, feature_mode='indicators', 
                 label_threshold=0.02):
        """
        Args:
            window_size: Number of candles to look back (input sequence length)
            horizon: Days ahead to determine buy/sell label
            feature_mode: 'indicators', 'catch22', or 'combined'
            label_threshold: Percentage threshold for BUY/SELL labels (default 2%)
        """
        self.window_size = window_size
        self.horizon = horizon
        self.feature_mode = feature_mode
        self.label_threshold = label_threshold
        self.ta = TechnicalAnalysis()

        # Technical indicator feature columns
        self.indicator_cols = [
            'open_norm', 'high_norm', 'low_norm', 'close_norm',
            'rsi', 'momentum_strength', 'hammer', 'engulfing', 'doji_num'
        ]
        
        # catch22 feature names
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

        # Validate catch22 availability
        if feature_mode in [self.MODE_CATCH22, self.MODE_COMBINED] and not CATCH22_AVAILABLE:
            raise ImportError("pycatch22 is required for catch22 features. Run: poetry add pycatch22")

    @property
    def feature_cols(self):
        """Return feature column names based on mode"""
        if self.feature_mode == self.MODE_INDICATORS:
            return self.indicator_cols
        elif self.feature_mode == self.MODE_CATCH22:
            return self.catch22_names
        else:  # combined
            return self.indicator_cols + self.catch22_names

    def compute_catch22(self, series):
        """
        Compute catch22 features for a single time series.
        
        Args:
            series: 1D numpy array or list of values
            
        Returns:
            List of 22 feature values
        """
        if not CATCH22_AVAILABLE:
            raise ImportError("pycatch22 not installed")
        
        result = pycatch22.catch22_all(series)
        return result['values']

    def build_features(self, df):
        """
        Build feature matrix from OHLCV dataframe.

        Args:
            df: DataFrame with OHLCV columns
        
        Returns:
            X: numpy array of shape (n_samples, n_features, window_size) for indicators
               or (n_samples, n_features) for catch22-only mode
            y: numpy array of shape (n_samples,) with labels 0=Sell, 1=Hold, 2=Buy
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Need enough data for indicators + window + horizon
        if len(df) < self.window_size + self.horizon + 50:
            return np.array([]), np.array([])

        # Add technical indicators from TechnicalAnalysis
        df = self.ta.momentum_trend(df)
        df['hammer'] = self.ta.hammer_signal(df).astype(int)
        df['engulfing'] = self.ta.engulfing_signal(df)

        # Convert doji to numeric (0=none, 1=standard, 2=dragonfly, 3=gravestone, 4=long_legged)
        doji_map = {'standard': 1, 'dragonfly': 2, 'gravestone': 3, 'long_legged': 4}
        df['doji_num'] = self.ta.doji_signal(df).map(doji_map).fillna(0)

        # Normalize OHLCV (percent change over window)
        for col in ['open', 'high', 'low', 'close']:
            df[f'{col}_norm'] = df[col].pct_change(self.window_size)

        # Create labels based on future returns with configurable threshold
        df['future_return'] = df['close'].shift(-self.horizon) / df['close'] - 1
        df['label'] = pd.cut(
            df['future_return'],
            bins=[-np.inf, -self.label_threshold, self.label_threshold, np.inf],
            labels=[0, 1, 2]  # 0=Sell, 1=Hold, 2=Buy
        )

        # Build features based on mode
        if self.feature_mode == self.MODE_CATCH22:
            return self._build_catch22_features(df)
        elif self.feature_mode == self.MODE_INDICATORS:
            return self._build_indicator_features(df)
        else:  # combined
            return self._build_combined_features(df)

    def _build_indicator_features(self, df):
        """Build sliding window features from technical indicators"""
        X, y = [], []

        # Get valid indices (no NaN in features or labels)
        valid_mask = df[self.indicator_cols + ['label']].notna().all(axis=1)
        valid_indices = df.index[valid_mask].tolist()

        for i in range(len(valid_indices) - self.window_size):
            window_indices = valid_indices[i:i + self.window_size]

            # Extract feature window
            window = df.loc[window_indices, self.indicator_cols].values

            # Get label at end of window
            label = df.loc[window_indices[-1], 'label']

            # Skip if any NaN values
            if not np.isnan(window).any() and pd.notna(label):
                X.append(window.T)  # Transpose to (n_channels, window_size)
                y.append(int(label))

        return np.array(X), np.array(y)

    def _build_catch22_features(self, df):
        """Build catch22 features from price windows"""
        X, y = [], []

        # Get valid indices where we have labels
        valid_mask = df['label'].notna()
        valid_indices = df.index[valid_mask].tolist()

        for i in range(len(valid_indices) - self.window_size):
            window_indices = valid_indices[i:i + self.window_size]

            # Get label at end of window
            label = df.loc[window_indices[-1], 'label']

            if pd.notna(label):
                try:
                    # Compute catch22 on close prices
                    close_window = df.loc[window_indices, 'close'].values
                    catch22_close = self.compute_catch22(close_window)
                    
                    # Optionally add catch22 for other series
                    volume_window = df.loc[window_indices, 'volume'].values
                    catch22_volume = self.compute_catch22(volume_window)
                    
                    # Combine features (22 close + 22 volume = 44 features)
                    features = catch22_close + catch22_volume
                    
                    if not any(np.isnan(features)):
                        X.append(features)
                        y.append(int(label))
                except Exception:
                    continue

        return np.array(X), np.array(y)

    def _build_combined_features(self, df):
        """Build combined indicator + catch22 features"""
        X, y = [], []

        # Get valid indices (no NaN in indicator features or labels)
        valid_mask = df[self.indicator_cols + ['label']].notna().all(axis=1)
        valid_indices = df.index[valid_mask].tolist()

        for i in range(len(valid_indices) - self.window_size):
            window_indices = valid_indices[i:i + self.window_size]

            # Get label at end of window
            label = df.loc[window_indices[-1], 'label']

            if pd.notna(label):
                try:
                    # Get indicator features (flattened from window)
                    indicator_window = df.loc[window_indices, self.indicator_cols].values
                    indicator_flat = indicator_window.flatten()  # 9 features × 20 window = 180
                    
                    # Get catch22 features
                    close_window = df.loc[window_indices, 'close'].values
                    catch22_features = self.compute_catch22(close_window)
                    
                    # Combine all features
                    combined = np.concatenate([indicator_flat, catch22_features])
                    
                    if not np.isnan(combined).any():
                        X.append(combined)
                        y.append(int(label))
                except Exception:
                    continue

        return np.array(X), np.array(y)

    def get_feature_names(self):
        """Return list of feature names based on mode"""
        if self.feature_mode == self.MODE_INDICATORS:
            # For sliding window, features repeat for each timestep
            names = []
            for t in range(self.window_size):
                for col in self.indicator_cols:
                    names.append(f"{col}_t{t}")
            return names
        elif self.feature_mode == self.MODE_CATCH22:
            return [f"close_{n}" for n in self.catch22_names] + \
                   [f"volume_{n}" for n in self.catch22_names]
        else:  # combined
            indicator_names = []
            for t in range(self.window_size):
                for col in self.indicator_cols:
                    indicator_names.append(f"{col}_t{t}")
            return indicator_names + [f"catch22_{n}" for n in self.catch22_names]

    def get_n_features(self):
        """Return number of features based on mode"""
        if self.feature_mode == self.MODE_INDICATORS:
            return len(self.indicator_cols)
        elif self.feature_mode == self.MODE_CATCH22:
            return 44  # 22 for close + 22 for volume
        else:  # combined
            return len(self.indicator_cols) * self.window_size + 22
