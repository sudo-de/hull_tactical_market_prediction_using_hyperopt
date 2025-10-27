"""Data loading and preprocessing module."""
import polars as pl
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import StandardScaler
import numpy as np
from dataclasses import dataclass

DATA_PATH = Path('/Users/sudip/hull_tactical_market_prediction_using_hyperopt/input/hull-tactical-market-prediction/')

TRAIN_FILE = DATA_PATH / 'train.csv'
TEST_FILE = DATA_PATH / 'test.csv'

@dataclass
class Dataset:
    """Container for dataset splits."""
    X_train: pl.DataFrame
    X_test: pl.DataFrame
    y_train: pl.Series
    y_test: pl.Series
    scaler: StandardScaler
    feature_names: list

class DataLoader:
    """Load and preprocess data for training."""
    
    def __init__(self, data_path: Path = DATA_PATH):
        self.data_path = data_path
    
    def load_train(self) -> pl.DataFrame:
        """Load training data."""
        df = pl.read_csv(TRAIN_FILE)
        df = df.rename({'market_forward_excess_returns': 'target'})
        df = df.with_columns(
            pl.exclude('date_id').cast(pl.Float64, strict=False)
        )
        df = df.head(-10)  # Remove last 10 rows for validation
        return df
    
    def load_test(self) -> pl.DataFrame:
        """Load test data."""
        df = pl.read_csv(TEST_FILE)
        if 'lagged_forward_returns' in df.columns:
            df = df.rename({'lagged_forward_returns': 'target'})
        df = df.with_columns(
            pl.exclude(['date_id', 'is_scored']).cast(pl.Float64, strict=False)
        )
        return df
    
    def prepare_dataset(
        self,
        train: pl.DataFrame,
        test: pl.DataFrame,
        features: list,
        scale: bool = True
    ) -> Dataset:
        """Prepare dataset for training."""
        X_train = train.drop(['date_id', 'target'])
        y_train = train['target']
        X_test = test.drop(['date_id', 'target'])
        y_test = test['target']
        
        scaler = None
        if scale:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train = pl.from_numpy(X_train_scaled, schema=features)
            X_test = pl.from_numpy(X_test_scaled, schema=features)
        
        return Dataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            feature_names=features
        )

