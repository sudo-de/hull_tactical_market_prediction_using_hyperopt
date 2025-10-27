import os
from pathlib import Path
import datetime
from typing import List

from tqdm import tqdm
from dataclasses import dataclass, asdict

import polars as pl 
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler

## Project Directory Structure
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/Users/sudip/hull_tactical_market_prediction_using_hyperopt/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, they won't be saved outside of the current session

## Configurations
# ============ PATHS ============
DATA_PATH: Path = Path('/Users/sudip/hull_tactical_market_prediction_using_hyperopt/input/hull-tactical-market-prediction/')

# ============ RETURNS TO SIGNAL CONFIGS ============
MIN_SIGNAL: float = 0.0                         # Minimum value for the daily signal 
MAX_SIGNAL: float = 2.0                         # Maximum value for the daily signal 
SIGNAL_MULTIPLIER: float = 400.0                # Multiplier of the OLS market forward excess returns predictions to signal 

# ============ MODEL CONFIGS ============
CV: int = 10                                    # Number of cross validation folds in the model fitting
L1_RATIO: float = 0.5                           # ElasticNet mixing parameter
ALPHAS: np.ndarray = np.logspace(-4, 2, 100)    # Constant that multiplies the penalty terms
MAX_ITER: int = 1000000                         # The maximum number of iterations

## Dataclasses Helpers
@dataclass
class DatasetOutput:
    X_train : pl.DataFrame 
    X_test: pl.DataFrame
    y_train: pl.Series
    y_test: pl.Series
    scaler: StandardScaler

@dataclass 
class ElasticNetParameters:
    l1_ratio : float 
    cv: int
    alphas: np.ndarray 
    max_iter: int 
    
    def __post_init__(self): 
        if self.l1_ratio < 0 or self.l1_ratio > 1: 
            raise ValueError("Wrong initializing value for ElasticNet l1_ratio")
        
@dataclass(frozen=True)
class RetToSignalParameters:
    signal_multiplier: float 
    min_signal : float = MIN_SIGNAL
    max_signal : float = MAX_SIGNAL

## Set the Parameters
ret_signal_params = RetToSignalParameters(
    signal_multiplier= SIGNAL_MULTIPLIER
)

enet_params = ElasticNetParameters(
    l1_ratio = L1_RATIO, 
    cv = CV, 
    alphas = ALPHAS, 
    max_iter = MAX_ITER
)

## Dataset Loading/Creating Helper Functions
def load_trainset() -> pl.DataFrame:
    """
    Loads and preprocesses the training dataset.

    Returns:
        pl.DataFrame: The preprocessed training DataFrame.
    """
    return (
        pl.read_csv(DATA_PATH / "train.csv")
        .rename({'market_forward_excess_returns':'target'})
        .with_columns(
            pl.exclude('date_id').cast(pl.Float64, strict=False)
        )
        .head(-10)
    )

def load_testset() -> pl.DataFrame:
    """
    Loads and preprocesses the testing dataset.

    Returns:
        pl.DataFrame: The preprocessed testing DataFrame.
    """
    return (
        pl.read_csv(DATA_PATH / "test.csv")
        .rename({'lagged_forward_returns':'target'})
        .with_columns(
            pl.exclude('date_id').cast(pl.Float64, strict=False)
        )
    )

def create_example_dataset(df: pl.DataFrame) -> pl.DataFrame:
    """
    Creates new features and cleans a DataFrame.

    Args:
        df (pl.DataFrame): The input Polars DataFrame.

    Returns:
        pl.DataFrame: The DataFrame with new features, selected columns, and no null values.
    """
    vars_to_keep: List[str] = [
        "S2", "E2", "E3", "P9", "S1", "S5", "I2", "P8",
        "P10", "P12", "P13", "U1", "U2"
    ]

    return (
        df.with_columns(
            (pl.col("I2") - pl.col("I1")).alias("U1"),
            (pl.col("M11") / ((pl.col("I2") + pl.col("I9") + pl.col("I7")) / 3)).alias("U2")
        )
        .select(["date_id", "target"] + vars_to_keep)
        .with_columns([
            pl.col(col).fill_null(pl.col(col).ewm_mean(com=0.5))
            for col in vars_to_keep
        ])
        .drop_nulls()
    )
    
def join_train_test_dataframes(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    """
    Joins two dataframes by common columns and concatenates them vertically.

    Args:
        train (pl.DataFrame): The training DataFrame.
        test (pl.DataFrame): The testing DataFrame.

    Returns:
        pl.DataFrame: A single DataFrame with vertically stacked data from common columns.
    """
    common_columns: list[str] = [col for col in train.columns if col in test.columns]
    
    return pl.concat([train.select(common_columns), test.select(common_columns)], how="vertical")

def split_dataset(train: pl.DataFrame, test: pl.DataFrame, features: list[str]) -> DatasetOutput: 
    """
    Splits the data into features (X) and target (y), and scales the features.

    Args:
        train (pl.DataFrame): The processed training DataFrame.
        test (pl.DataFrame): The processed testing DataFrame.
        features (list[str]): List of features to used in model. 

    Returns:
        DatasetOutput: A dataclass containing the scaled feature sets, target series, and the fitted scaler.
    """
    X_train = train.drop(['date_id','target']) 
    y_train = train.get_column('target')
    X_test = test.drop(['date_id','target']) 
    y_test = test.get_column('target')
    
    scaler = StandardScaler() 
    
    X_train_scaled_np = scaler.fit_transform(X_train)
    X_train = pl.from_numpy(X_train_scaled_np, schema=features)
    
    X_test_scaled_np = scaler.transform(X_test)
    X_test = pl.from_numpy(X_test_scaled_np, schema=features)
    
    
    return DatasetOutput(
        X_train = X_train,
        y_train = y_train, 
        X_test = X_test, 
        y_test = y_test,
        scaler = scaler
    )

## Converting Return Prediction to Signal
def convert_ret_to_signal(
    ret_arr: np.ndarray,
    params: RetToSignalParameters
) -> np.ndarray:
    """
    Converts raw model predictions (expected returns) into a trading signal.

    Args:
        ret_arr (np.ndarray): The array of predicted returns.
        params (RetToSignalParameters): Parameters for scaling and clipping the signal.

    Returns:
        np.ndarray: The resulting trading signal, clipped between min and max values.
    """
    return np.clip(
        ret_arr * params.signal_multiplier + 1, params.min_signal, params.max_signal
    )

def main():
    """Main function to run the Hull Tactical Market Prediction model."""
    print("Starting Hull Tactical Market Prediction...")
    
    # Looking at the Data
    train: pl.DataFrame = load_trainset()
    test: pl.DataFrame = load_testset() 
    print("Training data shape:", train.shape)
    print("Test data shape:", test.shape)
    print(train.tail(3)) 
    print(test.head(3))

    # Generating the Train and Test
    df: pl.DataFrame = join_train_test_dataframes(train, test)
    df = create_example_dataset(df=df) 
    train_date_ids = train.get_column('date_id')
    test_date_ids = test.get_column('date_id')
    train: pl.DataFrame = df.filter(pl.col('date_id').is_in(train_date_ids.to_list()))
    test: pl.DataFrame = df.filter(pl.col('date_id').is_in(test_date_ids.to_list()))

    FEATURES: list[str] = [col for col in test.columns if col not in ['date_id', 'target']]
    print(f"Features used: {FEATURES}")

    dataset: DatasetOutput = split_dataset(train=train, test=test, features=FEATURES) 

    X_train: pl.DataFrame = dataset.X_train
    X_test: pl.DataFrame = dataset.X_test
    y_train: pl.Series = dataset.y_train
    y_test: pl.Series = dataset.y_test
    scaler: StandardScaler = dataset.scaler 

    # Fitting the Model
    print("Fitting ElasticNet model with cross-validation...")
    model_cv: ElasticNetCV = ElasticNetCV(
        **asdict(enet_params)
    )
    model_cv.fit(X_train, y_train) 
            
    # Fit the final model using the best alpha found by cross-validation
    model: ElasticNet = ElasticNet(alpha=model_cv.alpha_, l1_ratio=enet_params.l1_ratio) 
    model.fit(X_train, y_train)
    
    print(f"Best alpha found: {model_cv.alpha_}")
    print(f"Model coefficients: {model.coef_}")
    
    # Make predictions
    predictions = model.predict(X_test)
    signals = convert_ret_to_signal(predictions, ret_signal_params)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Signals shape: {signals.shape}")
    print(f"Signal range: [{signals.min():.4f}, {signals.max():.4f}]")
    
    return model, predictions, signals, test

if __name__ == "__main__":
    model, predictions, signals, test_data = main()
