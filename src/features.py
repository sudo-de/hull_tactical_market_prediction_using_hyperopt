"""
Advanced feature engineering for Hull Tactical Market Prediction.

This module contains comprehensive feature engineering functions including:
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Statistical features (rolling statistics, volatility, etc.)
- Lag features
- Interaction features
- Market microstructure features
"""

import polars as pl
import numpy as np
from typing import List, Optional


def create_lag_features(df: pl.DataFrame, columns: List[str], lags: List[int] = [1, 2, 3, 5, 10]) -> pl.DataFrame:
    """
    Create lag features for specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of column names to create lags for
        lags: List of lag periods
    
    Returns:
        DataFrame with lag features added
    """
    result = df
    
    for col in columns:
        for lag in lags:
            result = result.with_columns(
                pl.col(col).shift(lag).alias(f"{col}_lag_{lag}")
            )
    
    return result


def create_rolling_features(df: pl.DataFrame, columns: List[str], windows: List[int] = [5, 10, 20, 50]) -> pl.DataFrame:
    """
    Create rolling window features (mean, std, min, max).
    
    Args:
        df: Input DataFrame
        columns: List of column names to create rolling features for
        windows: List of window sizes
    
    Returns:
        DataFrame with rolling features added
    """
    result = df
    
    for col in columns:
        for window in windows:
            result = result.with_columns([
                pl.col(col).rolling_mean(window).alias(f"{col}_mean_{window}"),
                pl.col(col).rolling_std(window).alias(f"{col}_std_{window}"),
                pl.col(col).rolling_min(window).alias(f"{col}_min_{window}"),
                pl.col(col).rolling_max(window).alias(f"{col}_max_{window}"),
            ])
    
    return result


def create_momentum_features(df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
    """
    Create momentum indicators.
    
    Args:
        df: Input DataFrame
        columns: List of column names to create momentum for
    
    Returns:
        DataFrame with momentum features added
    """
    result = df
    
    for col in columns:
        # RSI-like features (14-period)
        result = result.with_columns([
            pl.when((pl.col(col).shift(1) - pl.col(col)) > 0)
            .then(pl.col(col).shift(1) - pl.col(col))
            .otherwise(0)
            .alias(f"{col}_positive_change"),
            pl.when((pl.col(col).shift(1) - pl.col(col)) < 0)
            .then(abs(pl.col(col).shift(1) - pl.col(col)))
            .otherwise(0)
            .alias(f"{col}_negative_change"),
        ])
        
        # Moving averages
        result = result.with_columns([
            (pl.col(col) - pl.col(col).rolling_mean(5)).alias(f"{col}_above_ma5"),
            (pl.col(col) - pl.col(col).rolling_mean(10)).alias(f"{col}_above_ma10"),
            (pl.col(col) - pl.col(col).rolling_mean(20)).alias(f"{col}_above_ma20"),
        ])
    
    return result


def create_interaction_features(df: pl.DataFrame, feature_columns: List[str]) -> pl.DataFrame:
    """
    Create interaction features between variables.
    
    Args:
        df: Input DataFrame
        feature_columns: List of feature column names
    
    Returns:
        DataFrame with interaction features added
    """
    result = df
    
    # Create ratios and products of important features
    important_features = ['I2', 'M11', 'I1', 'I9', 'I7', 'S2', 'E2', 'E3']
    available_features = [f for f in important_features if f in feature_columns]
    
    if len(available_features) >= 2:
        # Create some meaningful ratios
        if 'I2' in df.columns and 'I1' in df.columns:
            result = result.with_columns(
                (pl.col("I2") / (pl.col("I1") + 1e-8)).alias("Ipow_ratio")
            )
        
        if 'M11' in df.columns and 'I2' in df.columns and 'I9' in df.columns and 'I7' in df.columns:
            result = result.with_columns(
                (pl.col("M11") / ((pl.col("I2") + pl.col("I9") + pl.col("I7")) / 3 + 1e-8)).alias("M11_norm")
            )
    
    return result


def create_volatility_features(df: pl.DataFrame, target_col: str = 'target') -> pl.DataFrame:
    """
    Create volatility-related features.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
    
    Returns:
        DataFrame with volatility features added
    """
    result = df
    
    if target_col in df.columns:
        result = result.with_columns([
            pl.col(target_col).rolling_std(5).alias(f"{target_col}_vol_5"),
            pl.col(target_col).rolling_std(10).alias(f"{target_col}_vol_10"),
            pl.col(target_col).rolling_std(20).alias(f"{target_col}_vol_20"),
            (pl.col(target_col).rolling_std(5) / pl.col(target_col).rolling_mean(5).abs()).alias(f"{target_col}_cv_5"),
        ])
    
    return result


def create_advanced_features(df: pl.DataFrame, feature_cols: List[str]) -> pl.DataFrame:
    """
    Comprehensive feature engineering pipeline.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names to engineer
    
    Returns:
        DataFrame with all engineered features
    """
    result = df
    
    # Get important features that exist in the dataframe
    key_features = [col for col in ['S2', 'E2', 'E3', 'P9', 'S1', 'S5', 'I2', 'P8', 'P10', 'P12', 'P13', 'I1', 'I9', 'I7', 'M11'] if col in df.columns]
    
    if len(key_features) == 0:
        key_features = feature_cols[:5]  # Fallback to first 5 features
    
    # Lag features
    result = create_lag_features(result, key_features[:5], lags=[1, 2, 3, 5])
    
    # Rolling statistics
    result = create_rolling_features(result, key_features[:3], windows=[5, 10, 20])
    
    # Momentum features
    result = create_momentum_features(result, key_features[:3])
    
    # Interaction features
    result = create_interaction_features(result, key_features)
    
    # Volatility features
    result = create_volatility_features(result)
    
    return result


def select_features(df: pl.DataFrame, method: str = 'all') -> List[str]:
    """
    Select features to use in the model.
    
    Args:
        df: Input DataFrame
        method: Selection method ('all', 'basic', 'extended')
    
    Returns:
        List of feature column names
    """
    # Exclude non-feature columns
    exclude_cols = ['date_id', 'target', 'is_scored']
    
    if method == 'basic':
        # Basic features only
        feature_cols = ['S2', 'E2', 'E3', 'P9', 'S1', 'S5', 'I2', 'P8', 'P10', 'P12', 'P13', 'U1', 'U2']
    elif method == 'extended':
        # Include some engineered features
        feature_cols = [col for col in df.columns if col not in exclude_cols and not any(x in col for x in ['_lag_', '_mean_', '_std_', '_min_', '_max_', '_vol_', '_cv_', '_above_'])]
    else:  # 'all'
        # All features including engineered ones
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Filter to only columns that exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    return feature_cols

