import numpy as np
import pandas as pd
import glob
import os
import logging
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.liner_model import LogisticRegression
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


class BinaryModelTrainer:
    """
    Trains binary classification for BUY signal detection.
    Optimized for high precision trading signals.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_builder = None
        self.threshold = 0.5 # Tunable classification threshold

    def prepare_dataset(self, data_dir, feature_builder, max_files=None):
        """
        Load parquet files and build binary classification dataset.

        Args:
            data_dir: Directory with parquet files
            feature_builder: BinaryFeatureBuilder instance
            max_files: Limit number of files (None = all)
        
        Returns:
            X, y: Feature matrix and binary labels
        """
        self.feature_builder = feature_builder

        all_X, all_y = [], []
        files = glob.glob(f"{data_dir}/*.parquet")

        if max_files:
            files = files[:max_files]

        if not files:
            raise ValueError(f"No parquet files found in {data_dir}")

        logging.info(f"Processing {len(files)} parquet files...")

        successful = 0
        for i, filepath in enumerate(files):
            ticker = os.path.basename(filepath).replace('.parquet', '')

            try:
                df = pd.read_parquet(filepath)

                # Need sufficient data for features
                if len(df) < 100:
                    continue
                    
                X, y = feature_builder.build_features(df)

                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
                    successful += 1

            except Exception as e:
                logging.error(f"Error processing {ticker}: {e}")
                continue

            if (i + 1) % 100 == 0:
                logging.info(f"Processed {i + 1}/{len(files)} files...")

        if not all_X:
            raise ValueError("No valid data extracted")

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        # Flatten if 3D (for non-ROCKET classifiers)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)

        logging.info(f"Dataset prepared: {len(X)} samples from {successful} tickers")
        logging.info(f"Feature shape: {X.shape[1]}")
        logging.info(f"BUY signals: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        logging.info(f"NOT_BUY: {len(y) - sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")

        return X, y

    def get_classifier(self, classifier_type, class_weight='balanced'):
        f"""
        Get classifier instance
        
        Args:
            classifier_type: 'random_forest', 'xgboost', or 'logistic'
            class_weight: 'balanced' or dict like {0: 1, 1: 3}
        """
        if classifier_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=10,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1
            )
        elif classifier_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                logging.warning("XGBoost not available, using Random Forest")
                return self.get_classifier('random_forest', class_weight)

            # Calculate scale_pos_weight for imbalanced data
            return XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=5, # Weight BUY signals higher
                random_state=42,
                n_jobs=-1,
                eval_metric='aucpr' # Area under precision-recall curve
            )
        elif classifier_type == 'logistic':
            return LogisticRegression(
                class_weight=class_weight,
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown classifier: {classifier_type}")

    def train(self, X, y, classifier_type='random_forest', test_size=0.2,
              scale_features=True, optimize_threshold=True):
        """
        Train binary classifier.

        Args:
            X: Feature matrix
            y: Binary labels
            classifier_type: 'random_forest', 'xgboost', or 'logistic'
            test_size: Fraction for test set
            scale_features: Whether to standardize features
            optimize_threshold: Find optimal classification threshold

        Returns:
            Trained model
        """
        # Temporal split (no shuffling for time series!!!)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        logging.info(f"Train: {len()} samples ({sum(y_train)} BUY)")
        logging.info(f"Test: {len(X_test)} samples ({sum(y_test)} BUY)")

        # Scale features
        if scale_features:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        # Train model
        self.model = self.get_classifier(classifier_type)
        logging.info(f"Training {classifier_type} model...")
        self.model.fit(X_train, y_train)

        # Get probabilities for threshold optimization
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X_test)[:, 1]

            if optimize_threshold:
                self.threshold = self._find_optimal_threshold(y_test, y_proba)
                logging.info(f"Optimal threshold: {self.threshold:.3f}")

            y_pred = (y_proba > self.threshold).astype(int)
        else:
            y_pred = self.model.predict(X_test)

        # Evaluate
        self._print_evaluation(y_test, y_pred, y_proba if hasattr(self.model, 'predict_proba') else None)

        return self.model

    def _find_optimal_threshold(self, y_true, y_proba, min_precision=0.6):
        """
        Find threshold that maximizes F1 while maintaining minimum precision.

        For trading, we want HIGH PRECISION (few false signals).
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        # Find thresholds where precision >= min_precision
        valid_idx = precisions[:1] >= min_precision

        if not any(valid_idx):
            # No threshold meets precision requirement, use highest precision
            logging.warning(f"No threshold achieves {min_precision:.0%} precision")
            return thresholds[np.argmax(precisions[:1])]

        # Among valid thresholds, maximize F1
        f1_scores = 2 * (precisions[:1] * recalls[:1]) / (precisions[:1] + recalls[:1] + 1e-10)
        f1_scores[~valid_idx] = 0 # Zero out invalid thresholds

        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx]

    def _print_evaluation(self, y_true, y_pred, y_proba=None):
        """Print comprehensive evalutation metrics."""
        print("\n" + "="*60)
        print("BINARY CLASSIFICATION RESULTS (BUY vs NOT_BUY)")
        print("="*60)

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}  (When it says BUY, how often is it correct?)")
        print(f"Recall: {recall:.4f}  (What % of actual BUYs were found)")
        print(f"F1 Score: {f1:.4f}  (Harmonic mean of precision and recall)")

        if y_proba is not None:
            auc_roc = roc_auc_score(y_true, y_proba)
            auc_pr = average_precision_score(y_true, y_proba)
            print(f"ROC AUC: {auc_roc:.4f} (Area under ROC curve)")
            print(f"PR AUC: {auc_pr:.4f} (Best for imbalanced data)")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"              NOT_BUY   BUY")
        print(f"Actual NOT_BUY  {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"Actual BUY      {cm[1,0]:5d}  {cm[1,1]:5d}")
        
        # Trading interpretation
        print(f"\n--- Trading Interpretation ---")
        print(f"True Positives (correct BUY signals):  {cm[1,1]}")
        print(f"False Positives (wrong BUY signals):   {cm[0,1]}")
        print(f"Missed Opportunities (false negatives): {cm[1,0]}")

        if cm[1,1] + cm[0,1] > 0:
            win_rate = cm[1,1] / (cm[1,1] + cm[0,1])
            print(f"Win Rate (if you follow all BUY signals): {win_rate:.1%}")

        print("="*60)

    def predict(self, X):
        """Predict BUY/NOT_BUY with current threshold."""
        if self.model is None:
            raise ValueError("Model not trained")

        if self.scaler is not None:
            X = self.scaler.transform(X)

        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[:, 1]
            return (proba > self.threshold).astype(int), proba
        else:
            return self.model.predict(X), None

    def save_model(self, path="models/binary_buy_model.pkl"):
        """Save trained model, scaler, and threshold."""
        if self.model is None:
            raise ValueError("No model to save")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'timestamp': datetime.now().isoformat()
        }

        joblib.dump(save_data, path)
        logging.info(f"Model saved to {path}")

    def load_model(self, path="models/binary_buy_model.pkl"):
        """load model, scaler, and threshold."""
        saved_data = joblib.load(path)

        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.threshold = saved_data['threshold']

        logging.info(f"Model loaded from {path}")
        return self.model


def train_buy_detector(data_dir="saved_data/historical_4h",
                       model_path="models/binary_buy_model.pkl",
                       classifier_type='random_forest',
                       window_size=20,
                       horizon=6,
                       buy_threshold=0.02,
                       max_files=None):
    """
    Quick function to train a BUY detector.

    Usage:
        from ml.binary_trainer import train_buy_detector
        train_buy_detector()
    """
    try:
        from ml.binary_feature_builder import BinaryFeatureBuilder
    except:
        from binary_feature_builder import BinaryFeatureBuilder

    fb = BinaryFeatureBuilder(
        window_size=window_size,
        horizon=horizon,
        buy_threshold=buy_threshold,
        feature_mode='indicators'
    )

    trainer = BinaryModelTrainer()
    X, y = trainer.prepare_dataset(data_dir, fb, max_files)
    trainer.train(X, y, classifier_type=classifier_type)
    trainer.save_model(model_path)

    return trainer


if __name__ == "__main__":
    train_buy_detector()