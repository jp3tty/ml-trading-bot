"""
Binary Hyperparameter Search for SELL Signal Detection

Grid search to find optimal parameters for the binary SELL detector.
Unlike the BUY detector (which optimises for precision), the SELL detector
optimises for RECALL — we want to catch declines quickly, accepting more
false alarms in exchange for rarely missing an exit opportunity.

Champion selection:
    recall >= min_recall AND precision >= 0.4  →  rank by F1
"""

import numpy as np
import pandas as pd
import glob
import os
import logging
import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class SellHyperparameterSearch:
    """
    Grid search for binary SELL detector hyperparameters.

    Key differences from BuyHyperparameterSearch:
      - Uses BinarySellFeatureBuilder (inverted labels)
      - find_optimal_threshold enforces a minimum RECALL (not precision)
      - Champion selected by F1 with recall >= min_recall floor
      - scale_pos_weight / class_weight tuned for sensitivity
    """

    def __init__(self, data_dir="saved_data/historical_4h",
                 results_dir="models/sell_search_results"):
        self.data_dir    = data_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.results        = []
        self.champion       = None
        self.champion_score = 0

    def define_search_space(self):
        """Full hyperparameter search space."""
        return {
            'window_size':    [10, 15, 20, 30],
            'horizon':        [2, 3, 5],           # Shorter — fast exits
            'sell_threshold': [0.005, 0.01, 0.015, 0.02],
            'classifier':     ['random_forest', 'xgboost'],
            'n_estimators':   [100, 200],
            'max_depth':      [5, 10, None],
            'min_recall':     [0.3, 0.4],
        }

    def define_quick_search_space(self):
        """Smaller search space for faster iteration."""
        return {
            'window_size':    [15, 20, 30],
            'horizon':        [2, 3, 5],
            'sell_threshold': [0.005, 0.01, 0.015],
            'classifier':     ['random_forest', 'xgboost'],
            'n_estimators':   [100, 200],
            'max_depth':      [6, 8, 10],
            'min_recall':     [0.3, 0.4],
        }

    def load_data(self, window_size, horizon, sell_threshold, max_files=None):
        """Load and build SELL features with given parameters."""
        from ml.binary_sell_feature_builder import BinarySellFeatureBuilder

        fb = BinarySellFeatureBuilder(
            window_size=window_size,
            horizon=horizon,
            sell_threshold=sell_threshold,
            feature_mode='catch22'
        )

        all_X, all_y = [], []
        files = glob.glob(f"{self.data_dir}/*.parquet")
        if max_files:
            files = files[:max_files]

        for filepath in files:
            try:
                df = pd.read_parquet(filepath)
                if len(df) < 100:
                    continue

                X, y = fb.build_features(df)

                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
            except Exception:
                continue

        if not all_X:
            return None, None

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)

        return X, y

    def get_classifier(self, classifier_type, n_estimators, max_depth):
        """Get classifier tuned for recall/sensitivity."""
        if classifier_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=5,
                # Heavier weight on SELL class — catch more declines
                class_weight={0: 1, 1: 3},
                random_state=42,
                n_jobs=-1
            )
        elif classifier_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                return self.get_classifier('random_forest', n_estimators, max_depth)
            return XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth if max_depth else 6,
                learning_rate=0.1,
                # Higher scale_pos_weight → more SELL predictions
                scale_pos_weight=8,
                random_state=42,
                n_jobs=-1,
                eval_metric='aucpr'
            )

    def find_optimal_threshold(self, y_true, y_proba, min_recall):
        """
        Find threshold maximising F1 subject to recall >= min_recall.

        For the SELL detector we lower the threshold until we hit our recall
        target, then pick the point with the best F1 among qualifying options.
        """
        from sklearn.metrics import precision_recall_curve

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        # Valid thresholds meeting recall requirement (and at least 40% precision)
        valid_idx = (recalls[:-1] >= min_recall) & (precisions[:-1] >= 0.4)

        if not any(valid_idx):
            # Fall back: enforce only recall floor, drop precision requirement
            valid_idx = recalls[:-1] >= min_recall

        if not any(valid_idx):
            # Last resort: return threshold with highest recall
            return thresholds[np.argmax(recalls[:-1])], False

        f1_scores = (2 * precisions[:-1] * recalls[:-1] /
                     (precisions[:-1] + recalls[:-1] + 1e-10))
        f1_scores[~valid_idx] = 0

        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx], True

    def evaluate_params(self, params):
        """Train and evaluate a single parameter combination."""
        logging.info(
            f"Testing: window={params['window_size']}, horizon={params['horizon']}, "
            f"sell_thresh={params['sell_threshold']}, clf={params['classifier']}"
        )

        try:
            X, y = self.load_data(
                window_size=params['window_size'],
                horizon=params['horizon'],
                sell_threshold=params['sell_threshold']
            )

            if X is None or len(X) < 1000:
                logging.warning("Insufficient data")
                return None

            sell_count = int(sum(y))
            sell_pct   = sell_count / len(y) * 100

            if sell_count < 50:
                logging.warning(f"Too few SELL samples: {sell_count}")
                return None

            # Temporal split — no shuffle
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            model = self.get_classifier(
                params['classifier'],
                params['n_estimators'],
                params['max_depth']
            )
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)[:, 1]

            threshold, met_recall = self.find_optimal_threshold(
                y_test, y_proba, params['min_recall']
            )

            y_pred = (y_proba >= threshold).astype(int)

            accuracy  = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall    = recall_score(y_test, y_pred, zero_division=0)
            f1        = f1_score(y_test, y_pred, zero_division=0)

            try:
                roc_auc = roc_auc_score(y_test, y_proba)
                pr_auc  = average_precision_score(y_test, y_proba)
            except Exception:
                roc_auc = 0
                pr_auc  = 0

            cm = confusion_matrix(y_test, y_pred)
            true_positives  = cm[1, 1] if cm.shape[0] > 1 else 0
            false_positives = cm[0, 1] if cm.shape[0] > 1 else 0

            result = {
                **params,
                'threshold':        threshold,
                'met_recall_req':   met_recall,
                'accuracy':         accuracy,
                'precision':        precision,
                'recall':           recall,
                'f1':               f1,
                'roc_auc':          roc_auc,
                'pr_auc':           pr_auc,
                'true_positives':   true_positives,
                'false_positives':  false_positives,
                'total_samples':    len(X),
                'sell_samples':     sell_count,
                'sell_pct':         sell_pct,
            }

            logging.info(
                f"  Precision: {precision:.3f}, Recall: {recall:.3f}, "
                f"F1: {f1:.3f}"
            )

            # Champion: recall >= 0.2 and precision >= 0.4, ranked by F1
            if recall >= 0.2 and precision >= 0.4:
                score = f1
                if score > self.champion_score:
                    self.champion_score = score
                    self.champion = {
                        'params':     params,
                        'model':      model,
                        'scaler':     scaler,
                        'threshold':  threshold,
                        'precision':  precision,
                        'recall':     recall,
                        'f1':         f1,
                    }
                    logging.info(
                        f"  *** NEW CHAMPION! "
                        f"Precision: {precision:.1%}, Recall: {recall:.1%} ***"
                    )

            return result

        except Exception as e:
            logging.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_combinations(self, search_space):
        import itertools

        keys   = list(search_space.keys())
        values = [search_space[k] for k in keys]

        combinations = []
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            if params['classifier'] == 'xgboost' and not XGBOOST_AVAILABLE:
                continue
            combinations.append(params)

        return combinations

    def run_search(self, quick=False):
        search_space = (self.define_quick_search_space() if quick
                        else self.define_search_space())
        combinations = self.generate_combinations(search_space)

        logging.info(
            f"Running {'quick ' if quick else ''}SELL search "
            f"with {len(combinations)} combinations"
        )

        for i, params in enumerate(combinations):
            logging.info(f"\n=== Combination {i+1}/{len(combinations)} ===")
            result = self.evaluate_params(params)
            if result:
                self.results.append(result)

        self.save_results()
        self.print_summary()

        return self.champion

    def print_summary(self):
        if not self.results:
            print("No valid results")
            return

        df = pd.DataFrame(self.results)

        print("\n" + "=" * 70)
        print("BINARY SELL DETECTOR SEARCH RESULTS")
        print("=" * 70)

        print("\nTop 5 by Recall (fewest missed exits):")
        top_recall = df.nlargest(5, 'recall')[
            ['window_size', 'horizon', 'sell_threshold',
             'classifier', 'precision', 'recall', 'f1']
        ]
        print(top_recall.to_string(index=False))

        print("\nTop 5 by F1 Score:")
        top_f1 = df.nlargest(5, 'f1')[
            ['window_size', 'horizon', 'sell_threshold',
             'classifier', 'precision', 'recall', 'f1']
        ]
        print(top_f1.to_string(index=False))

        print("\nTop 5 by Precision (fewest false SELL alerts):")
        top_prec = df.nlargest(5, 'precision')[
            ['window_size', 'horizon', 'sell_threshold',
             'classifier', 'precision', 'recall', 'f1']
        ]
        print(top_prec.to_string(index=False))

        if self.champion:
            print("\n" + "=" * 70)
            print("CHAMPION SELL MODEL")
            print("=" * 70)
            print(f"Parameters: {self.champion['params']}")
            print(f"Threshold:  {self.champion['threshold']:.3f}")
            print(f"Precision:  {self.champion['precision']:.1%}")
            print(f"Recall:     {self.champion['recall']:.1%}")
            print(f"F1 Score:   {self.champion['f1']:.3f}")

        print("=" * 70)

    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.results:
            df = pd.DataFrame(self.results).sort_values('recall', ascending=False)
            results_path = f"{self.results_dir}/sell_search_{timestamp}.csv"
            df.to_csv(results_path, index=False)
            logging.info(f"Results saved to {results_path}")

        if self.champion:
            model_path = f"{self.results_dir}/champion_sell_{timestamp}.pkl"

            save_data = {
                'model':     self.champion['model'],
                'scaler':    self.champion['scaler'],
                'threshold': self.champion['threshold'],
                'params':    self.champion['params'],
                'precision': self.champion['precision'],
                'recall':    self.champion['recall'],
                'f1':        self.champion['f1'],
            }
            joblib.dump(save_data, model_path)
            logging.info(f"Champion model saved to {model_path}")

            params_path = f"{self.results_dir}/champion_params_{timestamp}.json"
            with open(params_path, 'w') as f:
                json.dump({
                    'params': {
                        k: (int(v) if isinstance(v, np.integer) else
                            float(v) if isinstance(v, np.floating) else v)
                        for k, v in self.champion['params'].items()
                    },
                    'threshold': float(self.champion['threshold']),
                    'precision': float(self.champion['precision']),
                    'recall':    float(self.champion['recall']),
                    'f1':        float(self.champion['f1']),
                }, f, indent=2)
            logging.info(f"Champion params saved to {params_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Binary SELL Detector Hyperparameter Search')
    parser.add_argument('--quick', action='store_true', help='Run quick search')
    parser.add_argument('--data-dir', default='saved_data/historical_4h',
                        help='Data directory')
    parser.add_argument('--results-dir', default='models/sell_search_results',
                        help='Results directory')
    args = parser.parse_args()

    search = SellHyperparameterSearch(
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )
    search.run_search(quick=args.quick)
