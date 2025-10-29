#!/usr/bin/env python3
"""
Example script showing how to use CatBoost model in the Hull Tactical Market Prediction pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.models.catboost_model import CatBoostModel
from src.models.ensemble import create_default_ensemble

def load_sample_data():
    """Load and prepare sample data."""
    DATA_PATH = Path('/Users/sudip/hull_tactical_market_prediction_using_hyperopt/input/hull-tactical-market-prediction/')
    
    # Load training data
    train = (
        pl.read_csv(DATA_PATH / "train.csv")
        .rename({'market_forward_excess_returns':'target'})
        .with_columns(pl.exclude('date_id').cast(pl.Float64, strict=False))
        .head(2000)  # Use larger subset for demo
    )
    
    # Select features (excluding financial columns for demo)
    feature_cols = [col for col in train.columns if col not in ['date_id', 'target', 'forward_returns', 'risk_free_rate']]
    
    # Handle NaN values more aggressively
    print(f"Original data shape: {train.shape}")
    print(f"NaN count per column:")
    for col in feature_cols + ['target']:
        nan_count = train.select(pl.col(col).is_null().sum()).item()
        print(f"  {col}: {nan_count}")
    
    # Fill NaN values with median for numeric columns
    train_clean = train.select(feature_cols + ['target']).with_columns([
        pl.col(col).fill_null(pl.col(col).median()) for col in feature_cols + ['target']
    ])
    
    # If still NaN, fill with 0
    train_clean = train_clean.fill_null(0)
    
    X = train_clean.select(feature_cols).to_numpy()
    y = train_clean.get_column('target').to_numpy()
    
    # Final check for NaN values
    if np.isnan(X).any() or np.isnan(y).any():
        print("Warning: NaN values still present, dropping rows with NaN")
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        print(f"After NaN removal: {X.shape}")
    
    if X.shape[0] == 0:
        raise ValueError("No valid data remaining after NaN removal!")
    
    print(f"Final data shape: {X.shape}")
    return X, y, feature_cols

def demo_catboost():
    """Demonstrate CatBoost model usage."""
    print("=== CatBoost Model Demo ===")
    
    # Load data
    X, y, feature_names = load_sample_data()
    print(f"Data shape: {X.shape}")
    print(f"Features: {len(feature_names)}")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Initialize CatBoost model
    model = CatBoostModel(n_trials=20, cv_folds=3, random_state=42)
    
    print("\n1. Optimizing hyperparameters...")
    best_params = model.optimize(X_train, y_train)
    print(f"Best parameters: {best_params}")
    print(f"Best CV score: {model.get_best_score():.4f}")
    
    print("\n2. Fitting model with best parameters...")
    model.fit(X_train, y_train)
    
    print("\n3. Making predictions...")
    predictions = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
    mae = np.mean(np.abs(y_test - predictions))
    
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    print("\n4. Feature importance (top 10):")
    importance = model.get_feature_importance()
    top_features = np.argsort(importance)[-10:][::-1]
    
    for i, idx in enumerate(top_features):
        print(f"{i+1:2d}. {feature_names[idx]:15s}: {importance[idx]:.4f}")
    
    print("\nCatBoost demo completed successfully!")

def demo_ensemble():
    """Demonstrate ensemble with CatBoost."""
    print("\n=== Ensemble Model Demo ===")
    
    # Load data
    X, y, feature_names = load_sample_data()
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create ensemble with all models including CatBoost
    ensemble = create_default_ensemble(n_trials=10, cv_folds=3, random_state=42)
    
    print("Training ensemble models...")
    ensemble.fit(X_train, y_train, optimize=True)
    
    print("Making ensemble predictions...")
    ensemble_pred = ensemble.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((y_test - ensemble_pred) ** 2))
    mae = np.mean(np.abs(y_test - ensemble_pred))
    
    print(f"Ensemble Test RMSE: {rmse:.4f}")
    print(f"Ensemble Test MAE: {mae:.4f}")
    
    print("\nIndividual model predictions:")
    individual_preds = ensemble.get_model_predictions(X_test)
    for model_name, pred in individual_preds.items():
        model_rmse = np.sqrt(np.mean((y_test - pred) ** 2))
        print(f"{model_name:15s}: RMSE = {model_rmse:.4f}")
    
    print("\nEnsemble demo completed successfully!")

if __name__ == "__main__":
    demo_catboost()
    demo_ensemble()
