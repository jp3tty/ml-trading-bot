"""
Hyperparameter Search for ML Models
Finds the best combination of parameters and saves the champion model.
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

class HyperparameterSearch:
    """
    Grid search for trading model hyperparameters.
    Tracks champion model with best performance.
    """

    def __init__(self, data_dir="saved_data/historical", results_dir="models/search_results"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.results = []
        self.champion = None
        self.champion_score = 0

    def define_search_space(self):
        """Define hyperparameter grid"""
        return {
            # Feature Builder params
            'window_size': [10, 20, 30],
            'horizon': [3, 5, 7],

            # Label threshold (as percentages)
            'label_threshold': [0.01, 0.02, 0.03],

            # Model params
            'n_kernels': [1000, 2000, 3000],

            # Data params
            'max_samples': [50000], # Keeping fixed for now (for speed)
        }

    def build_features_with_threshold(self, df, window_size, horizon, threshold):
        """Build features with custom label threshold"""
        try:
            from feature_builder import FeatureBuilder
        except:
            from ml.feature_builder import FeatureBuilder

        # Create feature builder
        fb = FeatureBuilder(window_size=window_size, horizon=horizon)

        # Override the label creation with custom threshold
        df = df.copy()
        df.columns = df.columns.str.lower()

        if len(df) < window_size + horizon + 50:
            return np.array([]), np.array([])

        # Add technical indicators from TechnicalAnalysis
        df = fb.ta.momentum_trend(df)
        df['hammer'] = fb.ta.hammer_signal(df).astype(int)
        df['engulfing'] = fb.ta.engulfing_signal(df)

        # Covert doji to numeric (0=none, 1=standard, 2=dragonfly, 3=gravestone, 4=long_legged)
        doji_map = {'standard': 1, 'dragonfly': 2, 'gravestone': 3, 'long_legged': 4}
        df['doji_num'] = fb.ta.doji_signal(df).map(doji_map).fillna(0)
        
        for col in ['open', 'high', 'low', 'close']:
            df[f'{col}_norm'] = df[col].pct_change(window_size)

        # Custom threshold labels
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        df['label'] = pd.cut(
            df['future_return'],
            bins=[-np.inf, -threshold, threshold, np.inf],
            labels=[0, 1, 2] # 0=Sell, 1=Hold, 2=Buy
        )

        # Build sliding windows
        X, y = [], []
        feature_cols = fb.feature_cols

        valid_mask = df[feature_cols + ['label']].notna().all(axis=1)
        valid_indices = df.index[valid_mask].tolist()

        for i in range(len(valid_indices) - window_size):
            window_indicies = valid_indices[i:i+window_size]
            window = df.loc[window_indicies, feature_cols].values
            label = df.loc[window_indicies[-1], 'label']

            if not np.isnan(window).any() and pd.notna(label):
                X.append(window.T) # Transpose to (n_channels, window_size)
                y.append(int(label))

        return np.array(X), np.array(y)

    def load_data(self, window_size, horizon, threshold, max_samples):
        """Load and prepare data with given parameters"""
        import glob

        all_X, all_y = [], []
        files = glob.glob(f"{self.data_dir}/*.parquet")

        for filepath in files[:200]: # Limit to first 200 for speed
            try:
                df = pd.read_parquet(filepath)
                if len(df) < 100:
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

        # Subsample
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            indices.sort() # Keep temporal order
            X = X[indices]
            y = y[indices]

        return X, y

    def evaluate_params(self, params):
        """Train and evaluate a single parameter combination"""
        logging.info(f"Evaluating params: {params}")

        try:
            # Load data with these params
            X, y = self.load_data(
                window_size=params['window_size'],
                horizon=params['horizon'],
                threshold=params['label_threshold'],
                max_samples=params['max_samples']
            )

            if X is None or len(X) < 1000:
                logging.warning("Insufficient data")
                return None

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

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

            # Per-class metrics
            report = classification_report(y_test, y_pred, output_dict=True)

            result = {
                **params,
                'accuracy': accuracy,
                'f1_weighted': f1,
                'f1_sell': report['0']['f1-score'],
                'f1_hold': report['1']['f1-score'],
                'f1_buy': report['2']['f1-score'],
                'sample_train': len(X_train),
                'sample_test': len(X_test),
            }

            logging.info(f"    Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

            # Check if champion
            if f1 > self.champion_score:
                self.champion_score = f1
                self.champion = {'params': params, 'model': model, 'score': f1}
                logging.info(f"   *** NEW CHAMPION! F1: {f1:.4f} ***")

            return result

        except Exception as e:
            logging.error(f"Error: {e}")
            return None

    def run_search(self):
        """Run full grid search"""
        search_space = self.define_search_space()

        # Generate all combinations
        keys = search_space.keys()
        combinations = list(itertools.product(*search_space.values()))

        logging.info(f"Running grid search with {len(combinations)} combinations")

        for i, values in enumerate(combinations):
            params = dict(zip(keys, values))
            logging.info(f"\n=== Combination {i+1}/{len(combinations)} ===")

            result = self.evaluate_params(params)
            if result:
                self.results.append(result)

        # Save results
        self.save_results()

        return self.champion

    def save_results(self):
        """Save search results and champion model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save results DataFrame
        if self.results:
            df = pd.DataFrame(self.results)
            df = df.sort_values('f1_weighted', ascending=False)
            df.to_csv(f"{self.results_dir}/search_results_{timestamp}.csv", index=False)

            logging.info(f"\nTop 5 configurations:")
            print(df.head()[['window_size', 'horizon', 'label_threshold', 'n_kernels', 'accuracy', 'f1_weighted']])

        # Save champion model
        if self.champion:
            import joblib

            champion_path = f"{self.results_dir}/champion_model_{timestamp}.pkl"
            joblib.dump(self.champion['model'], champion_path)

            # Save champion params
            with open(f"{self.results_dir}/champion_params_{timestamp}.json", 'w') as f:
                json.dump({
                    'params': self.champion['params'],
                    'score': self.champion['score']
                }, f, indent=2)

            logging.info(f"\nChampion model saved to {champion_path}")
            logging.info(f"Champion params: {self.champion['params']}")
            logging.info(f"Champion F1 score: {self.champion['score']:.4f}")


def run_grid_search():
    """Quick function to run grid search"""
    search = HyperparameterSearch()
    champion = search.run_search()
    return champion


if __name__ == "__main__":
    run_grid_search()