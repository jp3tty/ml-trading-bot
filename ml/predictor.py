import numpy as np
from sklearn import config_context
import pandas as pd
import joblib
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TradingPredictor:
    """
    Loads a trained model and makes buy/sell/hold predicitons on new data.
    """
    
    # Label mapping
    LABEL_MAP = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

    def __init__(self, model_path, feature_builder):
        """
        Args:
            model_path: Path to saved model (.pkl file)
            feature_builder: FeatureBuilder instance (must mathc training config)
        """
        self.model = self._load_model(model_path)
        self.feature_builder = feature_builder

    def _load_model(self, path):
        """Load trained model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        
        model = joblib.load(path)
        logging.info(f"Model loaded from {path}")
        return model

    def predict(self, df):
        """
        Predict buy/sell/hold signal for the most recent data point.

        Args:
            df: DataFrame with OHLCV data (needs at least window_size + 50 rows)

        Returns:
            dict with 'signal', 'confidence', 'probabilities' or None if insufficient data
        """
        # Build features (this creates sliding windows)
        X, _ = self.feature_builder.build_features(df)

        if len(X) == 0:
            logging.warning("Insufficient data to make prediction")
            return None

        # Use only the most recent window
        latest_window = X[-1:] # Shape: (1, n_channels, window_size)

        # Get prediction
        prediction = self.model.predict(latest_window)[0]

        # Get probabilities if available
        probabilities = None
        confidence = None

        if hasattr(self.model, 'predict_proba'):
            try:
                proba = self.model.predict_proba(latest_window)[0]
                probabilities = {
                    'SELL': float(proba[0]),
                    'HOLD': float(proba[1]),
                    'BUY': float(proba[2]),
                }
                confidence = float(proba[prediction])
            except Exception as e:
                logging.warning(f"Error getting probabilities: {e}")
                confidence = None

        return {
            'signal': self.LABELS[prediction],
            'confidence': confidence,
            'probabilities': probabilities,
            'prediction_raw': int(prediction)
        }

    def predict_batch(self, df):
        """
        Get predictions for all valid windows in the data.
        Useful for backtesting.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            List of prediction dicts, one per window
        """
        X, _ = self.feature_builder.build_features(df)

        if len(X) == 0:
            return []

        predictions = self.model.predict(X)

        results = []

        # Get probabilities if available
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            try:
                probabilities = self.model.predict_proba(X)
            except:
                pass

        for i, pred in enumerate(predictions):
            result = {
                'signal': self.LABELS[pred],
                'prediction_raw': int(pred)
            }

            if probabilities is not None:
                result['confidence'] = float(probabilities[i, pred])
                result['probabilities'] = {
                    'SELL': float(probabilities[i, 0]),
                    'HOLD': float(probabilities[i, 1]),
                    'BUY': float(probabilities[i, 2])
                }

            results.append(result)

        return results

    def get_signal_summary(self, predictions):
        """
        Summarize a batch of predictions.

        Args:
            predictions: List of prediction dicts from predict_batch()

        Returns:
            dict with counts and percentages
        """
        if not predictions:
            return None

        signals = [p['signal'] for p in predictions]

        return {
            'total': len(signals),
            'buy_count': signals.count('BUY'),
            'sell_count': signals.count('SELL'),
            'hold_count': signals.count('HOLD'),
            'buy_pct': signals.count('BUY') / len(signals) *100,
            'sell_pct': signals.count('SELL') / len(signals) *100,
            'hold_pct': signals.count('HOLD') / len(signals) *100,
        }


# Quick prediction function
def quick_predict(df, model_path="models/rocket_trading_model.pkl", 
                  window_size=20, horizon=5):
    """
    One-liner prediction function.

    Usage:
        from ml.predictor import quick_predict
        results = quick_predict(df)
        print(result['signal'])  # 'BUY', 'SELL', 'HOLD'
    """
    from ml.feature_builder import FeatureBuilder

    feature_builder = FeatureBuilder(window_size=window_size, horizon=horizon)
    predictor = TradingPredictor(model_path, feature_builder)

    return predictor.predict(df)

if __name__ == "__main__":
    # Example usage
    print("TradingPredictor module")
    print("Usage:")
    print("  from ml.predictor import TradingPredictor, quick_predict")
    print("  result = quick_predict(df)")
    print("  print(result['signal'])  # 'BUY', 'SELL', 'HOLD'")