"""
Hyperparameter Search for 4-Hour Candlestick Data
Adjusted parameters for intraday timeframes.
"""

import numpy as np
import pandas as pd
import itertools
import logging
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from aeon.classification.convolution_based import RocketClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class HyperparameterSearch4H:
    """
    Grid search for 4-hour candlestick data.
    Adjusted window sizes and thresholds for intraday timeframes.
    """

    def __init__(self, data_dir="saved_data/historical_4h", results_dir="models/search_results_4h"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.results = []
        self.champion = None
        self.champion_score = 0

    def define_search_space(self):
        """
        Define hyperparameter grid for 4-hour data.
        
        Window sizes are in 4H bars:
        - 30 bars = ~5 trading days (6 bars/day)
        - 60 bars = ~10 trading days
        - 90 bars = ~15 trading days
        
        Horizons are in 4H bars:
        - 6 bars = ~1 trading day
        - 12 bars = ~2 trading days  
        - 18 bars = ~3 trading days
        
        Label thresholds are smaller for intraday moves:
        - 0.5%, 1%, 1.5% instead of 1%, 2%, 3%
        """
        return {
            # Feature Builder params (in 4H bars)
            'window_size': [30, 60, 90],
            'horizon': [6, 12, 18],

            # Label threshold (smaller for intraday)
            'label_threshold': [0.005, 0.01, 0.015],

            # Model params
            'n_kernels': [1000, 2000, 3000],

            # Data params
            'max_samples': [75000],  # More samples available with 4H data
        }

    def build_features_with_threshold(self, df, window_size, horizon, threshold):
        """Build features with custom label threshold"""
        try:
            from feature_builder import FeatureBuilder
        except:
            from ml.feature_builder import FeatureBuilder

        fb = FeatureBuilder(window_size=window_size, horizon=horizon)

        df = df.copy()
        df.columns = df.columns.str.lower()

        if len(df) < window_size + horizon + 50:
            return np.array([]), np.array([])

        # Add technical indicators
        df = fb.ta.momentum_trend(df)
        df['hammer'] = fb.ta.hammer_signal(df).astype(int)
        df['engulfing'] = fb.ta.engulfing_signal(df)

        doji_map = {'standard': 1, 'dragonfly': 2, 'gravestone': 3, 'long_legged': 4}
        df['doji_num'] = fb.ta.doji_signal(df).map(doji_map).fillna(0)
        
        for col in ['open', 'high', 'low', 'close']:
            df[f'{col}_norm'] = df[col].pct_change(window_size)

        # Custom threshold labels
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        df['label'] = pd.cut(
            df['future_return'],
            bins=[-np.inf, -threshold, threshold, np.inf],
            labels=[0, 1, 2]
        )

        # Build sliding windows
        X, y = [], []
        feature_cols = fb.feature_cols

        valid_mask = df[feature_cols + ['label']].notna().all(axis=1)
        valid_indices = df.index[valid_mask].tolist()

        for i in range(len(valid_indices) - window_size):
            window_indices = valid_indices[i:i+window_size]
            window = df.loc[window_indices, feature_cols].values
            label = df.loc[window_indices[-1], 'label']

            if not np.isnan(window).any() and pd.notna(label):
                X.append(window.T)
                y.append(int(label))

        return np.array(X), np.array(y)

    def load_data(self, window_size, horizon, threshold, max_samples):
        """Load and prepare 4H data with given parameters"""
        import glob

        all_X, all_y = [], []
        files = glob.glob(f"{self.data_dir}/*.parquet")

        if not files:
            logging.error(f"No parquet files found in {self.data_dir}")
            return None, None

        logging.info(f"Loading from {len(files)} parquet files...")

        for filepath in files[:200]:
            try:
                df = pd.read_parquet(filepath)
                # Need more data for 4H (at least 500 bars)
                if len(df) < 500:
                    continue

                X, y = self.build_features_with_threshold(df, window_size, horizon, threshold)

                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
            except:
                continue

        if not all_X:
            return None, None

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        logging.info(f"Total samples before subsampling: {len(X)}")

        # Subsample
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            indices.sort()
            X = X[indices]
            y = y[indices]

        return X, y

    def evaluate_params(self, params):
        """Train and evaluate a single parameter combination"""
        logging.info(f"Evaluating params: {params}")

        try:
            X, y = self.load_data(
                window_size=params['window_size'],
                horizon=params['horizon'],
                threshold=params['label_threshold'],
                max_samples=params['max_samples']
            )

            if X is None or len(X) < 1000:
                logging.warning("Insufficient data")
                return None

            # Split (no shuffle for time series)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            logging.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
            logging.info(f"Label distribution - Sell: {sum(y==0)}, Hold: {sum(y==1)}, Buy: {sum(y==2)}")

            # Train
            model = RocketClassifier(
                n_kernels=params['n_kernels'],
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            report = classification_report(y_test, y_pred, output_dict=True)

            result = {
                **params,
                'accuracy': accuracy,
                'f1_weighted': f1,
                'f1_sell': report['0']['f1-score'],
                'f1_hold': report['1']['f1-score'],
                'f1_buy': report['2']['f1-score'],
                'samples_train': len(X_train),
                'samples_test': len(X_test),
            }

            logging.info(f"    Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

            # Check if champion
            if f1 > self.champion_score:
                self.champion_score = f1
                self.champion = {'params': params, 'model': model, 'score': f1}
                logging.info(f"    *** NEW CHAMPION! F1: {f1:.4f} ***")

            return result

        except Exception as e:
            logging.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_search(self, quick=False):
        """
        Run grid search.
        
        Args:
            quick: If True, run reduced search space for faster iteration
        """
        search_space = self.define_search_space()

        if quick:
            # Reduced search for quick testing
            search_space = {
                'window_size': [30, 60],
                'horizon': [6, 12],
                'label_threshold': [0.01],
                'n_kernels': [2000],
                'max_samples': [50000],
            }
            logging.info("Running QUICK search with reduced parameters")

        # Generate all combinations
        keys = search_space.keys()
        combinations = list(itertools.product(*search_space.values()))

        logging.info(f"Running grid search with {len(combinations)} combinations")

        for i, values in enumerate(combinations):
            params = dict(zip(keys, values))
            logging.info(f"\n{'='*60}")
            logging.info(f"Combination {i+1}/{len(combinations)}")
            logging.info(f"{'='*60}")

            result = self.evaluate_params(params)
            if result:
                self.results.append(result)

        self.save_results()

        return self.champion

    def save_results(self):
        """Save search results and champion model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if self.results:
            df = pd.DataFrame(self.results)
            df = df.sort_values('f1_weighted', ascending=False)
            results_path = f"{self.results_dir}/search_results_4h_{timestamp}.csv"
            df.to_csv(results_path, index=False)

            logging.info(f"\nResults saved to {results_path}")
            logging.info(f"\nTop 5 configurations:")
            print(df.head()[['window_size', 'horizon', 'label_threshold', 'n_kernels', 'accuracy', 'f1_weighted']])

        if self.champion:
            import joblib

            champion_path = f"{self.results_dir}/champion_model_4h_{timestamp}.pkl"
            joblib.dump(self.champion['model'], champion_path)

            params_path = f"{self.results_dir}/champion_params_4h_{timestamp}.json"
            with open(params_path, 'w') as f:
                json.dump({
                    'params': self.champion['params'],
                    'score': self.champion['score'],
                    'timeframe': '4H'
                }, f, indent=2)

            logging.info(f"\nChampion model saved to {champion_path}")
            logging.info(f"Champion params: {self.champion['params']}")
            logging.info(f"Champion F1 score: {self.champion['score']:.4f}")


def run_grid_search_4h(quick=False):
    """Quick function to run 4H grid search"""
    search = HyperparameterSearch4H()
    champion = search.run_search(quick=quick)
    return champion


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="4H Hyperparameter Search")
    parser.add_argument('--quick', action='store_true', help='Run reduced search space')
    args = parser.parse_args()
    
    run_grid_search_4h(quick=args.quick)
