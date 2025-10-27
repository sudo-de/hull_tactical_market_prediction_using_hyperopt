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
        # D Columns
        "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9",
        # E Columns
        "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12",
        "E13", "E14", "E15", "E16", "E17", "E18", "E19", "E20",
        # I Columns
        "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9",
        # M Columns
        "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12",
        "M13", "M14", "M15", "M16", "M17", "M18",
        # P Columns
        "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13",
        # S Columns
        "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12",
        # V Columns
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13",
        # Derived features
        "U1", "U2"
    ]
    
    # Only keep columns that actually exist in the dataframe
    available_cols = df.columns
    vars_to_keep = [col for col in vars_to_keep if col in available_cols]

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
    X_train = train.select(features)
    y_train = train.get_column('target')
    X_test = test.select(features)
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

def calculate_adjusted_sharpe(
    position: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
    min_investment: float = 0.0,
    max_investment: float = 2.0,
    trading_days_per_yr: int = 252
) -> float:
    """
    Calculates a custom evaluation metric (volatility-adjusted Sharpe ratio).

    This metric penalizes strategies that take on significantly more volatility
    than the underlying market.

    Args:
        position: The predicted position/signal (0 to 2).
        forward_returns: Forward returns of the market.
        risk_free_rate: Risk-free rate.
        min_investment: Minimum allowed position (default 0.0).
        max_investment: Maximum allowed position (default 2.0).
        trading_days_per_yr: Trading days per year (default 252).

    Returns:
        float: The calculated adjusted Sharpe ratio.
    """
    # Validate position range
    if position.max() > max_investment:
        raise ValueError(f'Position of {position.max()} exceeds maximum of {max_investment}')
    if position.min() < min_investment:
        raise ValueError(f'Position of {position.min()} below minimum of {min_investment}')
    
    # Calculate strategy returns
    strategy_returns = risk_free_rate * (1 - position) + position * forward_returns
    
    # Calculate strategy's Sharpe ratio
    strategy_excess_returns = strategy_returns - risk_free_rate
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(forward_returns)) - 1
    strategy_std = strategy_returns.std()
    
    if strategy_std == 0:
        raise ValueError('Division by zero, strategy std is zero')
    
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)
    
    # Calculate market return and volatility
    market_excess_returns = forward_returns - risk_free_rate
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(forward_returns)) - 1
    market_std = forward_returns.std()
    
    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)
    
    if market_volatility == 0:
        raise ValueError('Division by zero, market std is zero')
    
    # Calculate the volatility penalty
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol
    
    # Calculate the return penalty
    return_gap = max(
        0,
        (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr,
    )
    return_penalty = 1 + (return_gap**2) / 100
    
    # Adjust the Sharpe ratio by the volatility and return penalty
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000)

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

    # Store financial columns from training data before joining
    train_financial = train.select(['date_id', 'forward_returns', 'risk_free_rate'])
    
    # Generating the Train and Test
    df: pl.DataFrame = join_train_test_dataframes(train, test)
    df = create_example_dataset(df=df) 
    train_date_ids = train.get_column('date_id')
    test_date_ids = test.get_column('date_id')
    train: pl.DataFrame = df.filter(pl.col('date_id').is_in(train_date_ids.to_list()))
    test: pl.DataFrame = df.filter(pl.col('date_id').is_in(test_date_ids.to_list()))
    
    # Join financial columns back to train
    train = train.join(train_financial, on='date_id', how='left')

    # Exclude financial columns from features to avoid data leakage
    excluded_cols = ['date_id', 'target', 'forward_returns', 'risk_free_rate']
    FEATURES: list[str] = [col for col in test.columns if col not in excluded_cols]
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
    
    # Calculate the adjusted Sharpe ratio score on training data (validation)
    if 'forward_returns' in train.columns and 'risk_free_rate' in train.columns:
        # Use the last portion of training data as validation
        val_size = min(1000, len(train))
        val_train = train.tail(val_size)
        val_X = val_train.select(FEATURES)
        val_X_scaled_np = scaler.transform(val_X)
        val_X_scaled = pl.from_numpy(val_X_scaled_np, schema=FEATURES)
        
        val_predictions = model.predict(val_X_scaled)
        val_signals = convert_ret_to_signal(val_predictions, ret_signal_params)
        
        forward_returns = val_train.get_column('forward_returns').to_numpy()
        risk_free_rate = val_train.get_column('risk_free_rate').to_numpy()
        
        try:
            adjusted_sharpe = calculate_adjusted_sharpe(
                position=val_signals,
                forward_returns=forward_returns,
                risk_free_rate=risk_free_rate
            )
            print(f"\nValidation Adjusted Sharpe Ratio: {adjusted_sharpe:.4f}")
        except Exception as e:
            print(f"\nError calculating validation score: {e}")
    
    return model, predictions, signals, test

if __name__ == "__main__":
    model, predictions, signals, test_data = main()
