"""
Binary SELL Feature Builder for Exit Signal Detection

Creates binary classification features:
  - Label 1 (SELL): Future return falls below -sell_threshold
  - Label 0 (NOT_SELL): Everything else

Mirrors BinaryFeatureBuilder but with inverted labeling.
Optimized for 4-hour candlestick data and fast exit detection.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from techAnalysis import TechnicalAnalysis

try:
    import pycatch22
    CATCH22_AVAILABLE = True
except ImportError:
    CATCH22_AVAILABLE = False


class BinarySellFeatureBuilder:
    MODE_INDICATORS = 'indicators'
    MODE_CATCH22    = 'catch22'
    MODE_COMBINED   = 'combined'

    def __init__(self, window_size=20, horizon=3, sell_threshold=0.01,
                 feature_mode='catch22', min_volume=0):
        """
        Args:
            window_size:    Candles of history used as input features.
            horizon:        Candles ahead to measure future return.
                            Shorter horizon suits the early-exit strategy.
            sell_threshold: Decline magnitude that constitutes a SELL signal.
                            Label 1 when future_return < -sell_threshold.
            feature_mode:   'indicators', 'catch22', or 'combined'.
            min_volume:     Skip bars where rolling avg volume is below this.
        """
        self.window_size    = window_size
        self.horizon        = horizon
        self.sell_threshold = sell_threshold
        self.feature_mode   = feature_mode
        self.min_volume     = min_volume
        self.ta             = TechnicalAnalysis()

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

        min_required = self.window_size + self.horizon + 50
        if len(df) < min_required:
            return np.array([]), np.array([])

        df = self._add_indicators(df)
        df = self._create_sell_labels(df)

        if self.feature_mode == self.MODE_CATCH22:
            return self._build_catch22_features(df)
        elif self.feature_mode == self.MODE_INDICATORS:
            return self._build_indicator_features(df)
        else:
            return self._build_combined_features(df)

    def _add_indicators(self, df):
        df = self.ta.momentum_trend(df)

        df['hammer']   = self.ta.hammer_signal(df).astype(int)
        df['engulfing'] = self.ta.engulfing_signal(df)

        doji_map = {'standard': 1, 'dragonfly': 2, 'gravestone': 3, 'long_legged': 4}
        df['doji_num'] = self.ta.doji_signal(df).map(doji_map).fillna(0)

        for col in ['open', 'high', 'low', 'close']:
            df[f'{col}_norm'] = df[col].pct_change(self.window_size)

        df['volume_avg']  = df['volume'].rolling(self.window_size).mean()
        df['volume_norm'] = df['volume'] / df['volume_avg'].replace(0, np.nan)

        df['return']     = df['close'].pct_change()
        df['volatility'] = df['return'].rolling(window=self.window_size).std()

        return df

    def _create_sell_labels(self, df):
        """Label 1 (SELL) when future return drops below -sell_threshold."""
        df['future_return'] = df['close'].shift(-self.horizon) / df['close'] - 1
        df['label'] = (df['future_return'] < -self.sell_threshold).astype(int)
        return df

    def _build_indicator_features(self, df):
        X, y = [], []

        feature_cols = [c for c in self.indicator_cols if c in df.columns]
        valid_mask   = df[feature_cols + ['label']].notna().all(axis=1)

        if self.min_volume > 0 and 'volume_avg' in df.columns:
            valid_mask &= df['volume_avg'] > self.min_volume

        valid_indices = df.index[valid_mask].tolist()

        for i in range(len(valid_indices) - self.window_size):
            window_indices = valid_indices[i:i + self.window_size]
            window = df.loc[window_indices, feature_cols].values
            label  = df.loc[window_indices[-1], 'label']

            if not np.isnan(window).any() and pd.notna(label):
                X.append(window.T)
                y.append(int(label))

        return np.array(X), np.array(y)

    def _build_catch22_features(self, df):
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
                    close_window  = df.loc[window_indices, 'close'].values
                    catch22_close = pycatch22.catch22_all(close_window)['values']

                    volume_window  = df.loc[window_indices, 'volume'].values.astype(float)
                    catch22_volume = pycatch22.catch22_all(volume_window)['values']

                    features = catch22_close + catch22_volume

                    if not any(np.isnan(features)):
                        X.append(features)
                        y.append(int(label))
                except Exception:
                    continue

        return np.array(X), np.array(y)

    def _build_combined_features(self, df):
        X, y = [], []

        feature_cols = [c for c in self.indicator_cols if c in df.columns]
        valid_mask   = df[feature_cols + ['label']].notna().all(axis=1)

        if self.min_volume > 0 and 'volume_avg' in df.columns:
            valid_mask &= df['volume_avg'] > self.min_volume

        valid_indices = df.index[valid_mask].tolist()

        for i in range(len(valid_indices) - self.window_size):
            window_indices = valid_indices[i:i + self.window_size]
            label = df.loc[window_indices[-1], 'label']

            if pd.notna(label):
                try:
                    indicator_flat  = df.loc[window_indices, feature_cols].values.flatten()
                    close_window    = df.loc[window_indices, 'close'].values
                    catch22_features = pycatch22.catch22_all(close_window)['values']
                    combined        = np.concatenate([indicator_flat, catch22_features])

                    if not any(np.isnan(combined)):
                        X.append(combined)
                        y.append(int(label))
                except Exception:
                    continue

        return np.array(X), np.array(y)

    def get_label_distribution(self, y):
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        return {
            'total':    total,
            'sell':     int(counts[unique == 1][0]) if 1 in unique else 0,
            'not_sell': int(counts[unique == 0][0]) if 0 in unique else 0,
            'sell_pct': float(counts[unique == 1][0] / total * 100) if 1 in unique else 0,
        }

    def get_feature_names(self):
        if self.feature_mode == self.MODE_INDICATORS:
            names = []
            feature_cols = list(self.indicator_cols)
            for t in range(self.window_size):
                for col in feature_cols:
                    names.append(f"{col}_t{t}")
            return names
        elif self.feature_mode == self.MODE_CATCH22:
            return (
                [f"close_{n}" for n in self.catch22_names] +
                [f"volume_{n}" for n in self.catch22_names]
            )
        else:
            indicator_names = []
            feature_cols = list(self.indicator_cols)
            for t in range(self.window_size):
                for col in feature_cols:
                    indicator_names.append(f"{col}_t{t}")
            return indicator_names + [f"catch22_{n}" for n in self.catch22_names]
