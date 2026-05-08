"""
Binary BUY Predictor for Live Trading

Loads a trained binary BUY detector champion and makes predictions on new data.
Designed to work with models trained by binary_search.py.
"""

import numpy as np
import joblib
import logging
import os
import glob

from ml.binary_feature_builder import BinaryFeatureBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BinaryBuyPredictor:
    """
    Loads a trained binary BUY detector and makes predictions on new data.

    Usage:
        predictor = BinaryBuyPredictor() # Loads latest champion
        prediction = predictor.predict(df) # df is OHLCV DataFrame

        if prediction['is_buy']:
            # Execute BUY signal
    """

    def __init__(self, model_path=None, results_dir="models/binary_search_results"):
        """
        Initialize the predictor.

        Args:
            model_path: Path to specific champion .pkl file.
                        If None, loads the most recent champion.
            results_dir: Directory containing champion models.
        """
        self.results_dir = results_dir

        if model_path is None:
            model_path = self._find_latest_champion()

        self.model_path = model_path
        logging.info(f"Loading champion model from: {model_path}")

        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.threshold = self.model_data['threshold']
        self.params = self.model_data['params']

        # Create feature builder with same params used in training
        self.feature_builder = BinaryFeatureBuilder(
            window_size=self.params['window_size'],
            horizon=self.params.get('horizon', 6),
            take_profit=self.params.get('take_profit', self.params.get('buy_threshold', 0.01)),
            stop_loss=self.params.get('stop_loss', 0.005),
            feature_mode=self.params.get('feature_mode', 'combined')
        )

        logging.info(
            f"Model loaded | threshold={self.threshold:.3f} | "
            f"precision={self.model_data.get('precision', 'N/A')} | "
            f"recall={self.model_data.get('recall', 'N/A')} | "
            f"f1={self.model_data.get('f1', 'N/A')}"
        )

    def _find_latest_champion(self):
        """Find most recent champion model file."""
        fixed = os.path.join(self.results_dir, "champion_buy.pkl")
        if os.path.exists(fixed):
            return fixed
        # Fall back to legacy timestamped files
        pattern = os.path.join(self.results_dir, "champion_binary_*.pkl")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No champion model found at {fixed} and no timestamped fallback.\n"
                "Run ml/binary_search.py first to train a champion model."
            )
        return files[-1]

    def predict(self, df):
        """
        Predict BUY/NOT_BUY for the most recent data point.

        Args:
            df: OHLCV DataFrame with enough history for feature calculation.

        Returns:
            dict with keys: 'signal', 'probability', 'threshold', 'is_buy'
            Returns None if features cannot be built from the data.
        """
        X, _ = self.feature_builder.build_features(df)

        if len(X) == 0:
            logging.warning("No features could be built from the provided data.")
            return None

        # Use most recent window, flatten to 2D for sklearn
        latest = X[-1:].reshape(1, -1) if X[-1:].ndim > 2 else X[-1:]

        latest_scaled = self.scaler.transform(latest)
        proba = self.model.predict_proba(latest_scaled)[0, 1]

        return {
            'signal': 'BUY' if proba >= self.threshold else 'NOT_BUY',
            'probability': float(proba),
            'threshold': self.threshold,
            'is_buy': proba >= self.threshold
        }

    def get_model_info(self):
        """Return model metadata."""
        return {
            'params': self.params,
            'threshold': self.threshold,
            'precision': self.model_data.get('precision'),
            'recall': self.model_data.get('recall'),
            'f1': self.model_data.get('f1'),
            'win_rate': self.model_data.get('win_rate'),
            'model_path': self.model_path
        }
