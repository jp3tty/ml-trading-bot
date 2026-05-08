"""
Binary SELL Predictor for Live Trading

Loads a trained binary SELL detector champion and makes predictions on new data.
Designed to work with models trained by binary_sell_search.py.

The SELL detector is tuned for sensitivity — it fires more readily than the
BUY detector, prioritising fast exits over avoiding false alarms.
"""

import glob
import logging
import os

import joblib
import numpy as np

from ml.binary_sell_feature_builder import BinarySellFeatureBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class BinarySellPredictor:
    """
    Loads a trained binary SELL detector and makes predictions on new data.

    Usage:
        predictor = BinarySellPredictor()   # Loads latest champion
        prediction = predictor.predict(df)  # df is OHLCV DataFrame

        if prediction['is_sell']:
            # Execute exit / close position
    """

    def __init__(self, model_path=None, results_dir="models/sell_search_results"):
        """
        Args:
            model_path:  Path to specific champion .pkl file.
                         If None, loads the most recent champion.
            results_dir: Directory containing champion SELL models.
        """
        self.results_dir = results_dir

        if model_path is None:
            model_path = self._find_latest_champion()

        self.model_path = model_path
        logging.info(f"Loading SELL champion model from: {model_path}")

        self.model_data = joblib.load(model_path)
        self.model      = self.model_data['model']
        self.scaler     = self.model_data['scaler']
        self.threshold  = self.model_data['threshold']
        self.params     = self.model_data['params']

        self.feature_builder = BinarySellFeatureBuilder(
            window_size=self.params['window_size'],
            horizon=self.params.get('horizon', 3),
            sell_threshold=self.params.get('sell_threshold', 0.01),
            feature_mode=self.params.get('feature_mode', 'catch22')
        )

        logging.info(
            f"SELL model loaded | threshold={self.threshold:.3f} | "
            f"precision={self.model_data.get('precision', 'N/A')} | "
            f"recall={self.model_data.get('recall', 'N/A')} | "
            f"f1={self.model_data.get('f1', 'N/A')}"
        )

    def _find_latest_champion(self):
        """Find champion SELL model file."""
        fixed = os.path.join(self.results_dir, "champion_sell.pkl")
        if os.path.exists(fixed):
            return fixed
        # Fall back to legacy timestamped files
        pattern = os.path.join(self.results_dir, "champion_sell_*.pkl")
        files   = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No champion model found at {fixed} and no timestamped fallback.\n"
                "Run ml/binary_sell_search.py first to train a champion model."
            )
        return files[-1]

    def predict(self, df):
        """
        Predict SELL/NOT_SELL for the most recent data point.

        Args:
            df: OHLCV DataFrame with enough history for feature calculation.

        Returns:
            dict with keys: 'signal', 'probability', 'threshold', 'is_sell'
            Returns None if features cannot be built.
        """
        X, _ = self.feature_builder.build_features(df)

        if len(X) == 0:
            logging.warning("SELL predictor: no features could be built.")
            return None

        latest = X[-1:].reshape(1, -1) if X[-1:].ndim > 2 else X[-1:]

        latest_scaled = self.scaler.transform(latest)
        proba         = self.model.predict_proba(latest_scaled)[0, 1]

        return {
            'signal':      'SELL' if proba >= self.threshold else 'NOT_SELL',
            'probability': float(proba),
            'threshold':   self.threshold,
            'is_sell':     proba >= self.threshold,
        }

    def get_model_info(self):
        """Return model metadata."""
        return {
            'params':     self.params,
            'threshold':  self.threshold,
            'precision':  self.model_data.get('precision'),
            'recall':     self.model_data.get('recall'),
            'f1':         self.model_data.get('f1'),
            'model_path': self.model_path,
        }
