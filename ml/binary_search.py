"""
Binary Hyperparameter Search for BUY Signal Detection

Grid search to find optimal parameters for the binary BUY detector.
Optimizes for precision (correct BUY signals) and F1 score.
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

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class BinaryHyperparameterSearch:
    """
    Grid search for binary BUY detector hyperparameters.
    """
    
    def __init__(self, data_dir="saved_data/historical_4h", 
                 results_dir="models/binary_search_results"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.results = []
        self.champion = None
        self.champion_score = 0
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def define_search_space(self):
        """Full hyperparameter search space.
        Optimizes for F-beta (β=0.5), which weights precision 4× more than recall.
        Target: 55%+ precision to produce a live win rate above 50%.
        """
        return {
            'window_size':   [10, 15, 20, 30],
            'horizon':       [3, 6, 12],
            'take_profit':   [0.005, 0.010, 0.015],
            'stop_loss':     [0.003, 0.005, 0.008],
            'classifier':    ['random_forest', 'xgboost'],
            'n_estimators':  [100, 200],
            'max_depth':     [5, 10, None],
            'min_precision': [0.50, 0.55],
        }

    def define_quick_search_space(self):
        """Focused search space targeting tighter BUY labels to reduce ~25% BUY rate.
        Raises take_profit to 2-4% on 4h bars to cut label rate to ~5-10%.
        Optimizes for F-beta (β=0.5) — precision-weighted to target 55%+ live win rate.
        """
        return {
            'window_size':   [21, 30],
            'horizon':       [6, 9, 12],
            'take_profit':   [0.020, 0.030, 0.040],
            'stop_loss':     [0.010, 0.015],
            'classifier':    ['random_forest', 'xgboost'],
            'n_estimators':  [200],
            'max_depth':     [8, 12],
            'min_precision': [0.50, 0.55],
        }
    
    def load_data(self, window_size, horizon, take_profit, stop_loss, max_files=None):
        """Load and build features with given parameters"""
        try:
            from ml.binary_feature_builder import BinaryFeatureBuilder
        except:
            from binary_feature_builder import BinaryFeatureBuilder

        fb = BinaryFeatureBuilder(
            window_size=window_size,
            horizon=horizon,
            take_profit=take_profit,
            stop_loss=stop_loss,
            feature_mode='combined'
        )
        
        all_X, all_y = [], []
        files = glob.glob(f"{self.data_dir}/*.parquet")[:max_files]
        
        for filepath in files:
            try:
                df = pd.read_parquet(filepath)
                if len(df) < 100:
                    continue
                
                X, y = fb.build_features(df)
                
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
            except:
                continue
        
        if not all_X:
            return None, None
        
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        
        # Flatten if 3D
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        return X, y
    
    def get_classifier(self, classifier_type, n_estimators, max_depth):
        """Get classifier instance"""
        if classifier_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=10,
                class_weight='balanced',
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
                scale_pos_weight=5,
                random_state=42,
                n_jobs=-1,
                eval_metric='aucpr'
            )
    
    def find_optimal_threshold(self, y_true, y_proba, min_precision, min_recall=0.0,
                               fbeta=0.5):
        """Find threshold maximizing F-beta (β=0.5) subject to a minimum precision floor.

        β=0.5 weights precision 4× more than recall. This biases the threshold
        toward fewer but higher-quality signals — the change that was missing from
        the original recall-maximizing objective that produced a 40% live win rate.
        """
        from sklearn.metrics import precision_recall_curve

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        valid_idx = precisions[:-1] >= min_precision

        if not any(valid_idx):
            return thresholds[np.argmax(precisions[:-1])], False

        # Compute F-beta for each valid threshold
        beta2 = fbeta ** 2
        p = precisions[:-1]
        r = recalls[:-1]
        denom = beta2 * p + r
        fbeta_scores = np.where(denom > 0, (1 + beta2) * p * r / denom, 0.0)

        # Pick threshold with best F-beta among those meeting the precision floor
        valid_fbeta = np.where(valid_idx, fbeta_scores, 0.0)
        best_idx = np.argmax(valid_fbeta)
        return thresholds[best_idx], True
    
    def evaluate_params(self, params):
        """Train and evaluate a single parameter combination"""
        logging.info(f"Testing: window={params['window_size']}, horizon={params['horizon']}, "
                     f"take_profit={params['take_profit']}, stop_loss={params['stop_loss']}, clf={params['classifier']}")

        try:
            X, y = self.load_data(
                window_size=params['window_size'],
                horizon=params['horizon'],
                take_profit=params['take_profit'],
                stop_loss=params['stop_loss'],
            )
            
            if X is None or len(X) < 1000:
                logging.warning("Insufficient data")
                return None
            
            buy_count = sum(y)
            buy_pct = buy_count / len(y) * 100
            
            if buy_count < 50:
                logging.warning(f"Too few BUY samples: {buy_count}")
                return None
            
            # Split (no shuffle for time series)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # Scale
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Train
            model = self.get_classifier(
                params['classifier'],
                params['n_estimators'],
                params['max_depth']
            )
            model.fit(X_train, y_train)
            
            # Get probabilities
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Find optimal threshold
            threshold, met_precision = self.find_optimal_threshold(
                y_test, y_proba, params['min_precision']
            )
            
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_test, y_proba)
                pr_auc = average_precision_score(y_test, y_proba)
            except:
                roc_auc = 0
                pr_auc = 0
            
            # Confusion matrix for win rate
            cm = confusion_matrix(y_test, y_pred)
            true_positives = cm[1, 1] if cm.shape[0] > 1 else 0
            false_positives = cm[0, 1] if cm.shape[0] > 1 else 0
            
            if true_positives + false_positives > 0:
                win_rate = true_positives / (true_positives + false_positives)
            else:
                win_rate = 0
            
            result = {
                **params,
                'threshold': threshold,
                'met_precision_req': met_precision,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'win_rate': win_rate,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'total_samples': len(X),
                'buy_samples': buy_count,
                'buy_pct': buy_pct,
            }
            
            logging.info(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, "
                        f"F1: {f1:.3f}, Win Rate: {win_rate:.1%}")
            
            # Champion ranked by F-beta (β=0.5) — precision-weighted objective
            if precision >= params.get('min_precision', 0.50):
                beta2 = 0.25
                denom = beta2 * precision + recall
                score = (1 + beta2) * precision * recall / denom if denom > 0 else 0.0
                if score > self.champion_score:
                    self.champion_score = score
                    self.champion = {
                        'params': params,
                        'model': model,
                        'scaler': scaler,
                        'threshold': threshold,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'win_rate': win_rate
                    }
                    logging.info(f"  *** NEW CHAMPION! Precision: {precision:.1%}, Recall: {recall:.1%}, F0.5: {score:.3f} ***")

            return result
            
        except Exception as e:
            logging.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_combinations(self, search_space):
        """Generate all valid parameter combinations"""
        import itertools
        
        keys = list(search_space.keys())
        values = [search_space[k] for k in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            
            # Skip XGBoost if not available
            if params['classifier'] == 'xgboost' and not XGBOOST_AVAILABLE:
                continue

            # Require take_profit > stop_loss (at least 1:1 reward:risk)
            if 'take_profit' in params and 'stop_loss' in params:
                if params['take_profit'] <= params['stop_loss']:
                    continue

            combinations.append(params)

        return combinations
    
    def _save_incremental_results(self):
        if not self.results:
            return
        df = pd.DataFrame(self.results).sort_values('f1', ascending=False)
        df.to_csv(f"{self.results_dir}/binary_search_{self.timestamp}.csv", index=False)

    def _save_champion_json(self):
        if not self.champion:
            return
        params_path = f"{self.results_dir}/champion_params_{self.timestamp}.json"
        with open(params_path, 'w') as f:
            json.dump({
                'params': {k: (int(v) if isinstance(v, np.integer) else
                               float(v) if isinstance(v, np.floating) else v)
                           for k, v in self.champion['params'].items()},
                'threshold': float(self.champion['threshold']),
                'precision': float(self.champion['precision']),
                'recall':    float(self.champion['recall']),
                'f1':        float(self.champion['f1']),
                'win_rate':  float(self.champion['win_rate'])
            }, f, indent=2)

    def run_search(self, quick=False, max_files=None):
        """Run grid search"""
        search_space = self.define_quick_search_space() if quick else self.define_search_space()
        combinations = self.generate_combinations(search_space)

        logging.info(f"Running {'quick ' if quick else ''}search with {len(combinations)} combinations")

        # Group by data config so features are built once per unique (window, horizon, take_profit, stop_loss)
        from itertools import groupby
        data_keys = ('window_size', 'horizon', 'take_profit', 'stop_loss')
        sorted_combos = sorted(combinations, key=lambda c: tuple(c[k] for k in data_keys))

        total = len(sorted_combos)
        done = 0

        for data_cfg, group in groupby(sorted_combos, key=lambda c: tuple(c[k] for k in data_keys)):
            window_size, horizon, take_profit, stop_loss = data_cfg

            logging.info(f"\n--- Loading data: window={window_size}, horizon={horizon}, "
                         f"take_profit={take_profit}, stop_loss={stop_loss} ---")
            X, y = self.load_data(window_size=window_size, horizon=horizon,
                                  take_profit=take_profit, stop_loss=stop_loss, max_files=max_files)

            if X is None or len(X) < 1000:
                logging.warning("Insufficient data for this config — skipping")
                done += len(list(group))
                continue

            buy_count = int(sum(y))
            if buy_count < 50:
                logging.warning(f"Too few BUY samples ({buy_count}) — skipping")
                done += len(list(group))
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            for params in group:
                done += 1
                logging.info(f"\n=== Combination {done}/{total} ===")
                logging.info(
                    f"Testing: window={params['window_size']}, horizon={params['horizon']}, "
                    f"take_profit={params['take_profit']}, stop_loss={params['stop_loss']}, clf={params['classifier']}, "
                    f"n_est={params['n_estimators']}, depth={params['max_depth']}, "
                    f"min_prec={params['min_precision']}"
                )

                try:
                    model = self.get_classifier(params['classifier'], params['n_estimators'], params['max_depth'])
                    model.fit(X_train, y_train)
                    y_proba = model.predict_proba(X_test)[:, 1]

                    threshold, met_precision = self.find_optimal_threshold(
                        y_test, y_proba, params['min_precision'], params.get('min_recall', 0.0)
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
                    win_rate = (true_positives / (true_positives + false_positives)
                                if true_positives + false_positives > 0 else 0)

                    result = {
                        **params,
                        'threshold':         threshold,
                        'met_constraints':   met_precision,
                        'accuracy':          accuracy,
                        'precision':         precision,
                        'recall':            recall,
                        'f1':                f1,
                        'roc_auc':           roc_auc,
                        'pr_auc':            pr_auc,
                        'win_rate':          win_rate,
                        'true_positives':    true_positives,
                        'false_positives':   false_positives,
                        'total_samples':     len(X),
                        'buy_samples':       buy_count,
                        'buy_pct':           buy_count / len(y) * 100,
                    }

                    logging.info(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, "
                                 f"F1: {f1:.3f}, Win Rate: {win_rate:.1%}")

                    if precision >= params.get('min_precision', 0.50):
                        beta2 = 0.25
                        denom = beta2 * precision + recall
                        score = (1 + beta2) * precision * recall / denom if denom > 0 else 0.0
                        if score > self.champion_score:
                            self.champion_score = score
                            self.champion = {
                                'params':    params,
                                'model':     model,
                                'scaler':    scaler,
                                'threshold': threshold,
                                'precision': precision,
                                'recall':    recall,
                                'f1':        f1,
                                'win_rate':  win_rate,
                            }
                            logging.info(f"  *** NEW CHAMPION! Precision: {precision:.1%}, Recall: {recall:.1%}, F0.5: {score:.3f} ***")
                            self._save_champion_json()

                    self.results.append(result)
                    self._save_incremental_results()

                except Exception as e:
                    logging.error(f"Error evaluating params: {e}")
                    import traceback
                    traceback.print_exc()

        self.save_results()
        self.print_summary()

        return self.champion
    
    def print_summary(self):
        """Print search summary"""
        if not self.results:
            print("No valid results")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*70)
        print("BINARY BUY DETECTOR SEARCH RESULTS")
        print("="*70)
        
        cols = ['window_size', 'horizon', 'take_profit', 'stop_loss',
                'classifier', 'precision', 'recall', 'f1', 'win_rate']

        # Compute F-beta (β=0.5) for ranking
        beta2 = 0.25
        denom = beta2 * df['precision'] + df['recall']
        df['fbeta_05'] = np.where(denom > 0, (1 + beta2) * df['precision'] * df['recall'] / denom, 0.0)

        # Top by F-beta (β=0.5) — primary objective
        print("\nTop 5 by F-beta β=0.5 (primary objective — precision-weighted):")
        print(df.nlargest(5, 'fbeta_05')[cols + ['fbeta_05']].to_string(index=False))

        # Top by precision
        print("\nTop 5 by Precision:")
        print(df.nlargest(5, 'precision')[cols].to_string(index=False))

        # Top by F1 (for reference)
        print("\nTop 5 by F1 Score (for reference):")
        print(df.nlargest(5, 'f1')[cols].to_string(index=False))
        
        if self.champion:
            print("\n" + "="*70)
            print("CHAMPION MODEL")
            print("="*70)
            print(f"Parameters: {self.champion['params']}")
            print(f"Threshold:  {self.champion['threshold']:.3f}")
            print(f"Precision:  {self.champion['precision']:.1%}")
            print(f"Recall:     {self.champion['recall']:.1%}")
            print(f"F1 Score:   {self.champion['f1']:.3f}")
            print(f"Win Rate:   {self.champion['win_rate']:.1%}")
        
        print("="*70)
    
    def save_results(self):
        """Save final results — CSV and champion JSON are already current from incremental saves."""
        results_path = f"{self.results_dir}/binary_search_{self.timestamp}.csv"
        logging.info(f"Results saved to {results_path}")

        if self.champion:
            model_data = {
                'model':     self.champion['model'],
                'scaler':    self.champion['scaler'],
                'threshold': self.champion['threshold'],
                'params':    self.champion['params'],
                'precision': self.champion['precision'],
                'recall':    self.champion['recall'],
                'f1':        self.champion['f1'],
                'win_rate':  self.champion['win_rate']
            }
            model_path = f"{self.results_dir}/champion_buy.pkl"
            joblib.dump(model_data, model_path)
            logging.info(f"Champion model saved to {model_path}")
            self._save_champion_json()


def run_quick_search():
    """Run quick search"""
    search = BinaryHyperparameterSearch()
    return search.run_search(quick=True)


def run_full_search():
    """Run full search"""
    search = BinaryHyperparameterSearch()
    return search.run_search(quick=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Binary BUY Detector Hyperparameter Search')
    parser.add_argument('--quick', action='store_true', help='Run quick search')
    parser.add_argument('--data-dir', default='saved_data/historical_4h', help='Data directory')
    parser.add_argument('--max-files', type=int, default=None, help='Limit number of parquet files loaded per combination')
    args = parser.parse_args()

    search = BinaryHyperparameterSearch(data_dir=args.data_dir)
    search.run_search(quick=args.quick, max_files=args.max_files)