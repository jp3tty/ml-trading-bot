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
                        In None, loads the most recent champion.
            results_dir: Directory containing champion models.
        """
        self.results_dir = results_dir

        if model_path is None:
            model_path = self._find_latest_champion()

        self.model_path = model_path 