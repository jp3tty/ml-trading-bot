import numpy as np
import pandas as pd
import glob
import os
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from aeon.classification.convolution_based import RocketClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class TradingModelTrainer:
    """
    Trains aeon time series classifiers on historical stock data.
    """

    def __init__(self):
        self.model = None
        self.feature_builder = None

    def prepare_dataset(self, data_dir, feature_builder):
        """
        Load all parquet files and build combined dataset.

        Args:
            data_dir: Directory containing parquet files (one per ticker)
            feature_builder: FeatureBuilder instance
    
        Returns:
            X: numpy array of shape (n_samples, n_channels, window_size)
            y: numpy array of labels (0=Sell, 1=Hold, 2=Buy)
        """

        self.feature_builder = feature_builder

        all_X, all_y = [], []
        files = glob.glob(f"{data_dir}/*.parquet")

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
        
            # Progress update every 100 files
            if (i + 1) % 100 == 0:
                logging.info(f"Processed {i + 1}/{len(files)} files...")

        if not all_X:
            raise ValueError("No valid data found after processing all files")

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        logging.info(f"Dataset prepared: {len(X)} samples from {successful} tickers")
        logging.info(f"Feature shape: {X.shape}")
        logging.info(f"Label distribution: Sell={sum(y==0)}, Hold={sum(y==1)}, Buy={sum(y==2)}")

        return X, y

    def train(self, X, y, model_type='rocket', test_size=0.2, max_samples=200000):
        """
        Train the classifier.
        """
        # Subsample if dataset is too large for memory
        if len(X) > max_samples:
            logging.info(f"Subsampling from {len(X)} to {max_samples} samples for memory")
            # Use stratified sampling to maintain class balance
            indices = np.random.choice(len(X), max_samples, replace=False)
            indices.sort()  # Keep temporal order
            X = X[indices]
            y = y[indices]
        
        # IMPORTANT: Don't shuffle time series data!
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        logging.info(f"Training set: {len(X_train)} samples")
        logging.info(f"Test set: {len(X_test)} samples")

        # Initialize model
        if model_type == 'rocket':
            self.model = RocketClassifier(
                n_kernels=2000,
                random_state=42
            )
        elif model_type == 'inception':
            # Import only if needed (requires tensorflow)
            from aeon.classification.deep_learning import InceptionTimeClassifier
            self.model = InceptionTimeClassifier(
                n_epochs=100,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logging.info(f"Training {model_type} model...")
        self.model.fit(X_train, y_train)

        # Evaluate
        logging.info("Evaluating model...")
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Test accuracy: {accuracy:.4f}")

        print("\nClassification report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['Sell', 'Hold', 'Buy'],
            digits=3
        ))

        return self.model
    
    def save_model(self, path="models/trading_model.pkl"):
        """Save trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        joblib.dump(self.model, path)
        logging.info(f"Model save to {path}")

    def load_model(self, path="models/trading_model.pkl"):
        """Load trained model"""
        self.model = joblib.load(path)
        logging.info(f"Model loaded from {path}")
        return self.model


# Convenience function for quick training
def train_model(data_dir="saved_data/historical",
                model_path="models/rocket_trading_model.pkl",
                window_size=20,
                horizon=5,
                model_type='rocket'):
    """
    Quick training function.

    Usage:
        from ml.trainer import train_model
        train_model()
    """
    # Handle imports based on how script is run
    try:
        from ml.feature_builder import FeatureBuilder
    except ModuleNotFoundError:
        from feature_builder import FeatureBuilder

    feature_builder = FeatureBuilder(window_size=window_size, horizon=horizon)
    trainer = TradingModelTrainer()

    X, y = trainer.prepare_dataset(data_dir, feature_builder)
    trainer.train(X, y, model_type=model_type)
    trainer.save_model(model_path)

    return trainer

if __name__ == "__main__":
    # Run training when executed directly
    train_model()