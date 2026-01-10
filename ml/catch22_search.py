"""
Catch22 Hyperparameter Search with ROCKET Benchmark

Compares three approaches:
1. ROCKET with technical indicators (baseline)
2. catch22 features only
3. Combined: indicators + catch22

Uses Random Forest and XGBoost as classifiers for catch22 features
since they're fixed-length vectors (not time series).
"""

import numpy as np
import pandas as pd
import itertools
import logging
import json
import os
import glob
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Optional imports
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from aeon.classification.convolution_based import RocketClassifier
    ROCKET_AVAILABLE = True
except ImportError:
    ROCKET_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Catch22Search:
    """
    Hyperparameter search comparing ROCKET vs catch22 approaches.
    """
    
    def __init__(self, data_dir="saved_data/historical", results_dir="models/catch22_results"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.results = []
        self.champion = None
        self.champion_score = 0
        
    def define_search_space(self):
        """
        Define hyperparameter grid for catch22 search.
        
        Hyperparameters:
        - window_size: How many days the model looks at
        - horizon: How far ahead to predict
        - label_threshold: % threshold for BUY/SELL signals
        - feature_mode: Which features to use
        - classifier: Which ML model to use
        - n_estimators: Number of trees (for RF/XGB)
        - max_depth: Tree depth (for RF/XGB)
        """
        search_space = {
            # Feature engineering params
            'window_size': [10, 20, 30],
            'horizon': [3, 5, 7],
            'label_threshold': [0.01, 0.02, 0.03],
            
            # Feature mode
            'feature_mode': ['indicators', 'catch22', 'combined'],
            
            # Classifier (only for catch22/combined - ROCKET uses its own)
            'classifier': ['random_forest', 'xgboost'],
            
            # Classifier hyperparameters
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
        }
        
        return search_space
    
    def define_quick_search_space(self):
        """Smaller search space for faster iteration"""
        return {
            'window_size': [15, 20],
            'horizon': [5],
            'label_threshold': [0.02],
            'feature_mode': ['indicators', 'catch22', 'combined'],
            'classifier': ['random_forest'],
            'n_estimators': [100],
            'max_depth': [10],
        }
    
    def load_data(self, feature_mode, window_size, horizon, label_threshold, max_files=200, max_samples=50000):
        """Load and prepare data with given parameters"""
        try:
            from feature_builder import FeatureBuilder
        except:
            from ml.feature_builder import FeatureBuilder
        
        fb = FeatureBuilder(
            window_size=window_size,
            horizon=horizon,
            feature_mode=feature_mode,
            label_threshold=label_threshold
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
            except Exception as e:
                continue
        
        if not all_X:
            return None, None
        
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        
        # Subsample if needed
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            indices.sort()
            X = X[indices]
            y = y[indices]
        
        return X, y, fb
    
    def get_classifier(self, classifier_name, n_estimators, max_depth):
        """Get classifier instance"""
        if classifier_name == 'random_forest':
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif classifier_name == 'xgboost':
            if not XGBOOST_AVAILABLE:
                logging.warning("XGBoost not available, using Random Forest")
                return self.get_classifier('random_forest', n_estimators, max_depth)
            return XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth if max_depth else 6,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        elif classifier_name == 'rocket':
            if not ROCKET_AVAILABLE:
                raise ImportError("aeon not available for ROCKET")
            return RocketClassifier(n_kernels=2000, random_state=42)
        else:
            raise ValueError(f"Unknown classifier: {classifier_name}")
    
    def evaluate_params(self, params):
        """Train and evaluate a single parameter combination"""
        logging.info(f"Testing: {params}")
        
        try:
            feature_mode = params['feature_mode']
            
            # Load data
            result = self.load_data(
                feature_mode=feature_mode,
                window_size=params['window_size'],
                horizon=params['horizon'],
                label_threshold=params['label_threshold']
            )
            
            if result[0] is None:
                logging.warning("Insufficient data")
                return None
            
            X, y, fb = result
            
            if len(X) < 1000:
                logging.warning(f"Only {len(X)} samples, skipping")
                return None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # For ROCKET with indicators mode, use ROCKET classifier
            if feature_mode == 'indicators' and params['classifier'] == 'rocket':
                if ROCKET_AVAILABLE:
                    model = RocketClassifier(n_kernels=2000, random_state=42)
                else:
                    logging.warning("ROCKET not available, skipping")
                    return None
            else:
                # For catch22/combined or non-ROCKET classifiers, reshape if needed
                if feature_mode == 'indicators' and len(X_train.shape) == 3:
                    # Flatten for non-ROCKET classifiers
                    X_train = X_train.reshape(X_train.shape[0], -1)
                    X_test = X_test.reshape(X_test.shape[0], -1)
                
                # Scale features for catch22/combined
                if feature_mode in ['catch22', 'combined']:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                
                model = self.get_classifier(
                    params['classifier'],
                    params['n_estimators'],
                    params['max_depth']
                )
            
            # Train
            logging.info(f"  Training on {len(X_train)} samples...")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            result = {
                **params,
                'accuracy': accuracy,
                'f1_weighted': f1,
                'f1_sell': report['0']['f1-score'] if '0' in report else 0,

                
                'f1_hold': report['1']['f1-score'] if '1' in report else 0,
                'f1_buy': report['2']['f1-score'] if '2' in report else 0,
                'samples_train': len(X_train),
                'samples_test': len(X_test),
                'n_features': X_train.shape[1] if len(X_train.shape) == 2 else X_train.shape[1] * X_train.shape[2],
            }
            
            logging.info(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Check if champion
            if f1 > self.champion_score:
                self.champion_score = f1
                self.champion = {'params': params, 'model': model, 'score': f1}
                logging.info(f"  *** NEW CHAMPION! F1: {f1:.4f} ***")
            
            return result
            
        except Exception as e:
            logging.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_valid_combinations(self, search_space):
        """Generate valid parameter combinations"""
        combinations = []
        
        for ws in search_space['window_size']:
            for hz in search_space['horizon']:
                for lt in search_space['label_threshold']:
                    for fm in search_space['feature_mode']:
                        for clf in search_space['classifier']:
                            for ne in search_space['n_estimators']:
                                for md in search_space['max_depth']:
                                    # Skip ROCKET for catch22/combined (ROCKET needs 3D input)
                                    if clf == 'rocket' and fm != 'indicators':
                                        continue
                                    
                                    # For ROCKET, n_estimators and max_depth don't apply
                                    if clf == 'rocket' and (ne != search_space['n_estimators'][0] or 
                                                           md != search_space['max_depth'][0]):
                                        continue
                                    
                                    combinations.append({
                                        'window_size': ws,
                                        'horizon': hz,
                                        'label_threshold': lt,
                                        'feature_mode': fm,
                                        'classifier': clf,
                                        'n_estimators': ne,
                                        'max_depth': md,
                                    })
        
        return combinations
    
    def run_search(self, quick=False):
        """Run full or quick grid search"""
        search_space = self.define_quick_search_space() if quick else self.define_search_space()
        
        combinations = self.generate_valid_combinations(search_space)
        
        logging.info(f"Running {'quick ' if quick else ''}grid search with {len(combinations)} combinations")
        
        for i, params in enumerate(combinations):
            logging.info(f"\n=== Combination {i+1}/{len(combinations)} ===")
            
            result = self.evaluate_params(params)
            if result:
                self.results.append(result)
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
        
        return self.champion
    
    def print_summary(self):
        """Print summary of results by feature mode"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*60)
        print("SUMMARY BY FEATURE MODE")
        print("="*60)
        
        for mode in ['indicators', 'catch22', 'combined']:
            mode_df = df[df['feature_mode'] == mode]
            if len(mode_df) > 0:
                best = mode_df.loc[mode_df['f1_weighted'].idxmax()]
                print(f"\n{mode.upper()}:")
                print(f"  Best F1: {best['f1_weighted']:.4f}")
                print(f"  Best Accuracy: {best['accuracy']:.4f}")
                print(f"  Params: window={best['window_size']}, horizon={best['horizon']}, "
                      f"threshold={best['label_threshold']}, classifier={best['classifier']}")
        
        print("\n" + "="*60)
        print(f"OVERALL CHAMPION: {self.champion['params']['feature_mode']} with F1={self.champion_score:.4f}")
        print("="*60)
    
    def save_results(self):
        """Save search results and champion model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results DataFrame
        if self.results:
            df = pd.DataFrame(self.results)
            df = df.sort_values('f1_weighted', ascending=False)
            results_path = f"{self.results_dir}/catch22_search_{timestamp}.csv"
            df.to_csv(results_path, index=False)
            logging.info(f"Results saved to {results_path}")
            
            print("\nTop 10 configurations:")
            print(df.head(10)[['feature_mode', 'classifier', 'window_size', 'horizon', 
                               'label_threshold', 'accuracy', 'f1_weighted']].to_string())
        
        # Save champion model
        if self.champion:
            import joblib
            
            champion_path = f"{self.results_dir}/champion_catch22_{timestamp}.pkl"
            joblib.dump(self.champion['model'], champion_path)
            
            # Save champion params
            params_path = f"{self.results_dir}/champion_params_{timestamp}.json"
            with open(params_path, 'w') as f:
                # Convert numpy types to Python types for JSON
                params_clean = {k: (int(v) if isinstance(v, np.integer) else 
                                   float(v) if isinstance(v, np.floating) else v)
                               for k, v in self.champion['params'].items()}
                json.dump({
                    'params': params_clean,
                    'score': float(self.champion['score'])
                }, f, indent=2)
            
            logging.info(f"Champion model saved to {champion_path}")


def run_quick_search():
    """Run quick search for fast iteration"""
    search = Catch22Search()
    return search.run_search(quick=True)


def run_full_search():
    """Run full grid search"""
    search = Catch22Search()
    return search.run_search(quick=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Catch22 Hyperparameter Search')
    parser.add_argument('--quick', action='store_true', help='Run quick search with fewer combinations')
    args = parser.parse_args()
    
    if args.quick:
        run_quick_search()
    else:
        run_full_search()

