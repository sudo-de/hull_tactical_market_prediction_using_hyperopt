"""
Advanced training script with multiple algorithms, hyperparameter tuning, and evaluation.
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.features import create_advanced_features, select_features
from src.models.elastic_net import ElasticNetModel
from src.models.lightgbm_model import LightGBMModel
from src.models.xgboost_model import XGBoostModel
from src.models.catboost_model import CatBoostModel
from src.models.ensemble import create_default_ensemble
from evaluation import score


def load_data(data_path: Path):
    """Load and preprocess data."""
    print("Loading data...")
    
    train = (
        pl.read_csv(data_path / "train.csv")
        .rename({'market_forward_excess_returns': 'target'})
        .with_columns(pl.exclude('date_id').cast(pl.Float64, strict=False))
        .head(-10)
    )
    
    test = (
        pl.read_csv(data_path / "test.csv")
        .rename({'lagged_forward_returns': 'target'})
        .with_columns(pl.exclude('date_id').cast(pl.Float64, strict=False))
    )
    
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    return train, test


def create_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create features from raw data."""
    print("Creating features...")
    
    vars_to_keep = ["S2", "E2", "E3", "P9", "S1", "S5", "I2", "P8", "P10", "P12", "P13"]
    
    # Create basic features
    df = df.with_columns([
        (pl.col("I2") - pl.col("I1")).alias("U1") if all(c in df.columns for c in ["I2", "I1"]) else pl.lit(None),
        (pl.col("M11") / ((pl.col("I2") + pl.col("I9") + pl.col("I7")) / 3)).alias("U2") if all(c in df.columns for c in ["M11", "I2", "I9", "I7"]) else pl.lit(None)
    ])
    
    df = df.select(["date_id", "target"] + vars_to_keep + ["U1", "U2"])
    
    # Fill nulls
    feature_cols = [col for col in df.columns if col not in ['date_id', 'target']]
    df = df.with_columns([
        pl.col(col).fill_null(pl.col(col).ewm_mean(com=0.5))
        for col in feature_cols
    ])
    
    # Create advanced features
    df = create_advanced_features(df, feature_cols)
    
    return df.drop_nulls()


def prepare_data(train: pl.DataFrame, test: pl.DataFrame):
    """Prepare train and test datasets."""
    print("Preparing data...")
    
    # Join and create features
    common_columns = [col for col in train.columns if col in test.columns]
    df = pl.concat([train.select(common_columns), test.select(common_columns)], how="vertical")
    
    df = create_features(df)
    
    # Split back
    train_date_ids = set(train.get_column('date_id'))
    test_date_ids = set(test.get_column('date_id'))
    
    train_df = df.filter(pl.col('date_id').is_in(list(train_date_ids)))
    test_df = df.filter(pl.col('date_id').is_in(list(test_date_ids)))
    
    # Get features
    feature_cols = select_features(train_df, method='all')
    print(f"Number of features: {len(feature_cols)}")
    
    # NaN imputation: fill feature columns with median, then 0 for any remaining
    # Compute medians on the training set to avoid leakage
    medians_exprs = {col: train_df.select(pl.col(col).median()).item() for col in feature_cols}
    train_df = train_df.with_columns([
        pl.col(col).fill_null(medians_exprs[col]) for col in feature_cols
    ]).fill_null(0)
    test_df = test_df.with_columns([
        pl.col(col).fill_null(medians_exprs[col]) for col in feature_cols
    ]).fill_null(0)
    
    # Prepare X and y
    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df.get_column('target').to_numpy()
    
    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df.get_column('target').to_numpy()
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test, scaler, feature_cols, test_df, train_df


def evaluate_model(name: str, predictions: np.ndarray, test_data: pl.DataFrame):
    """Evaluate model performance."""
    print(f"\nEvaluating {name}...")
    
    # Clip predictions to [0, 2] range
    signals = np.clip(predictions * 400 + 1, 0.0, 2.0)
    
    print(f"Signal range: [{signals.min():.4f}, {signals.max():.4f}]")
    print(f"Signal mean: {signals.mean():.4f}")
    
    return signals


def train_all_models(X_train, y_train, X_test, optimize=True):
    """Train all models and return predictions."""
    results = {}
    
    # ElasticNet
    print("\n" + "="*50)
    print("Training ElasticNet...")
    print("="*50)
    enet = ElasticNetModel(n_trials=10 if optimize else 0, random_state=42)
    enet.fit(X_train, y_train, optimize=optimize)
    results['ElasticNet'] = (enet, enet.predict(X_test))
    
    # LightGBM
    print("\n" + "="*50)
    print("Training LightGBM...")
    print("="*50)
    lgbm = LightGBMModel(n_trials=10 if optimize else 0, random_state=42)
    lgbm.fit(X_train, y_train, optimize=optimize)
    results['LightGBM'] = (lgbm, lgbm.predict(X_test))
    
    # XGBoost
    print("\n" + "="*50)
    print("Training XGBoost...")
    print("="*50)
    xgb_model = XGBoostModel(n_trials=10 if optimize else 0, random_state=42)
    xgb_model.fit(X_train, y_train, optimize=optimize)
    results['XGBoost'] = (xgb_model, xgb_model.predict(X_test))
    
    # CatBoost
    print("\n" + "="*50)
    print("Training CatBoost...")
    print("="*50)
    catboost = CatBoostModel(n_trials=10 if optimize else 0, random_state=42)
    catboost.fit(X_train, y_train, optimize=optimize)
    results['CatBoost'] = (catboost, catboost.predict(X_test))
    
    # Ensemble (all models)
    print("\n" + "="*50)
    print("Training Ensemble...")
    print("="*50)
    ensemble = create_default_ensemble(n_trials=10 if optimize else 0, random_state=42)
    ensemble.fit(X_train, y_train, optimize=optimize)
    results['Ensemble'] = (ensemble, ensemble.predict(X_test))
    
    return results


def calculate_adjusted_sharpe(position: np.ndarray,
                              forward_returns: np.ndarray,
                              risk_free_rate: np.ndarray,
                              trading_days_per_yr: int = 252) -> float:
    """
    Compute the volatility-adjusted Sharpe-like metric used in the competition.
    """
    # strategy returns
    strategy_returns = risk_free_rate * (1 - position) + position * forward_returns
    strategy_excess = strategy_returns - risk_free_rate
    # annualized mean excess via geometric mean
    cumulative = np.prod(1 + strategy_excess) if len(strategy_excess) > 0 else 1.0
    mean_excess = cumulative ** (1.0 / max(len(strategy_excess), 1)) - 1.0
    std = strategy_returns.std()
    if std == 0:
        return 0.0
    sharpe = mean_excess / std * np.sqrt(trading_days_per_yr)
    # market proxy from forward_returns vs risk_free_rate
    market_excess = forward_returns - risk_free_rate
    m_cum = np.prod(1 + market_excess) if len(market_excess) > 0 else 1.0
    m_mean_excess = m_cum ** (1.0 / max(len(market_excess), 1)) - 1.0
    m_std = forward_returns.std()
    if m_std == 0:
        return 0.0
    strategy_vol = float(std * np.sqrt(trading_days_per_yr) * 100)
    market_vol = float(m_std * np.sqrt(trading_days_per_yr) * 100)
    excess_vol = max(0.0, strategy_vol / market_vol - 1.2) if market_vol > 0 else 0.0
    vol_penalty = 1.0 + excess_vol
    return_gap = max(0.0, (m_mean_excess - mean_excess) * 100 * trading_days_per_yr)
    return_penalty = 1.0 + (return_gap ** 2) / 100.0
    adjusted = sharpe / (vol_penalty * return_penalty)
    return float(min(adjusted, 1_000_000))


def main():
    """Main training function."""
    print("="*70)
    print("Hull Tactical Market Prediction - Advanced Training")
    print("="*70)
    
    # Paths
    data_path = Path('/Users/sudip/hull_tactical_market_prediction_using_hyperopt/input/hull-tactical-market-prediction/')
    
    # Load data
    train, test = load_data(data_path)
    
    # Prepare data
    X_train, y_train, X_test, y_test, scaler, feature_cols, test_df, train_df = prepare_data(train, test)
    
    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Train models
    results = train_all_models(X_train, y_train, X_test, optimize=True)
    
    # Evaluate models
    print("\n" + "="*70)
    print("Model Evaluation Results")
    print("="*70)
    
    all_signals = {}
    for name, (model, predictions) in results.items():
        signals = evaluate_model(name, predictions, test_df)
        all_signals[name] = signals
    
    # Feature importance
    print("\n" + "="*70)
    print("Feature Importance Analysis")
    print("="*70)
    
    # LightGBM Feature Importance
    print("\nLightGBM Top Features:")
    try:
        lgbm_model = results['LightGBM'][0]
        feature_importance = lgbm_model.get_feature_importance()
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            print(f"{i+1:2d}. {feature}: {importance:.4f}")
    except Exception as e:
        print(f"Could not get LightGBM feature importance: {e}")
    
    # CatBoost Feature Importance
    print("\nCatBoost Top Features:")
    try:
        catboost_model = results['CatBoost'][0]
        feature_importance = catboost_model.get_feature_importance()
        sorted_features = sorted(enumerate(feature_importance), key=lambda x: x[1], reverse=True)
        for i, (feature_idx, importance) in enumerate(sorted_features[:10]):
            feature_name = feature_cols[feature_idx] if feature_idx < len(feature_cols) else f"Feature_{feature_idx}"
            print(f"{i+1:2d}. {feature_name}: {importance:.4f}")
    except Exception as e:
        print(f"Could not get CatBoost feature importance: {e}")
    
    # Display Training Summary
    print("\n" + "="*70)
    print("Training Summary - Loss Functions, Epochs, and Metrics")
    print("="*70)
    
    for name, (model, _) in results.items():
        if hasattr(model, 'training_metrics') and model.training_metrics:
            metrics = model.training_metrics
            print(f"\n{name}:")
            print(f"  Loss Function: {metrics.get('loss_function', 'N/A')}")
            print(f"  Iterations/Epochs: {metrics.get('iterations', 'N/A')}")
            
            if 'train_history' in metrics and 'val_history' in metrics:
                train_hist = metrics['train_history']
                val_hist = metrics['val_history']
                if train_hist and val_hist:
                    print(f"  Final Training RMSE: {train_hist[-1]:.6f}")
                    print(f"  Final Validation RMSE: {val_hist[-1]:.6f}")
                    print(f"  Best Training RMSE: {min(train_hist):.6f}")
                    print(f"  Best Validation RMSE: {min(val_hist):.6f}")
                    print(f"  Worst Training RMSE: {max(train_hist):.6f}")
                    print(f"  Worst Validation RMSE: {max(val_hist):.6f}")
            elif 'train_rmse' in metrics and 'val_rmse' in metrics:
                print(f"  Training RMSE: {metrics['train_rmse']:.6f}")
                print(f"  Validation RMSE: {metrics['val_rmse']:.6f}")
        elif hasattr(model, 'models'):  # Ensemble model
            print(f"\n{name} (Ensemble):")
            print(f"  Type: Weighted average of base models")
            print(f"  Base Models: {len(model.models)}")
            # Show metrics from first base model as example
            if model.models:
                base_model = model.models[0]
                if hasattr(base_model, 'training_metrics') and base_model.training_metrics:
                    metrics = base_model.training_metrics
                    print(f"  Loss Functions: Similar to base models ({metrics.get('loss_function', 'N/A')})")
    
    # Validation Adjusted Sharpe on train tail
    print("\n" + "="*70)
    print("Validation Metrics")
    print("="*70)
    try:
        val_size = min(1000, train_df.shape[0])
        if val_size > 0:
            val_slice = train_df.tail(val_size)
            val_X = val_slice.select(feature_cols).to_numpy()
            val_X = scaler.transform(val_X)
            
            # Use ensemble for validation
            ensemble_model = results['Ensemble'][0]
            val_pred = ensemble_model.predict(val_X)
            val_signals = np.clip(val_pred * 400 + 1, 0.0, 2.0)
            
            # Get the actual returns for validation
            # The 'target' column should contain forward excess returns
            forward_returns = val_slice.get_column('target').to_numpy()
            risk_free_rate = np.zeros_like(forward_returns)  # Assume zero risk-free rate
            
            adj_sharpe = calculate_adjusted_sharpe(val_signals, forward_returns, risk_free_rate)
            print(f"Validation Adjusted Sharpe Ratio: {adj_sharpe:.4f}")
    except Exception as e:
        print(f"Could not compute validation Adjusted Sharpe: {e}")
        import traceback
        traceback.print_exc()

    return results, all_signals, test_df


def plot_training_results(all_signals: dict, test_df: pl.DataFrame, save_dir: Path = Path('artifacts')):
    """Plot training results using matplotlib."""
    save_dir.mkdir(exist_ok=True)
    
    # Plot 1: Prediction comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (model_name, signals) in enumerate(all_signals.items()):
        if idx < 6:
            ax = axes[idx]
            ax.plot(signals, marker='o', linestyle='-', markersize=6, linewidth=2)
            ax.set_title(f'{model_name} Predictions', fontsize=14, fontweight='bold')
            ax.set_xlabel('Sample Index', fontsize=12)
            ax.set_ylabel('Signal Value', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Neutral')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'predictions_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved predictions plot to {save_dir / 'predictions_comparison.png'}")
    plt.close()
    
    # Plot 2: Signal distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(all_signals))
    widths = 0.8 / len(all_signals)
    
    for idx, (model_name, signals) in enumerate(all_signals.items()):
        ax.hist(signals, bins=20, alpha=0.6, label=model_name, 
                density=True, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Signal Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Signal Distribution by Model', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'signal_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved distribution plot to {save_dir / 'signal_distribution.png'}")
    plt.close()
    
    # Plot 3: Summary statistics bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(all_signals.keys())
    means = [np.mean(all_signals[m]) for m in models]
    stds = [np.std(all_signals[m]) for m in models]
    mins = [np.min(all_signals[m]) for m in models]
    maxs = [np.max(all_signals[m]) for m in models]
    
    x = np.arange(len(models))
    width = 0.2
    
    ax.bar(x - 1.5*width, means, width, label='Mean', alpha=0.8)
    ax.bar(x - 0.5*width, stds, width, label='Std', alpha=0.8)
    ax.bar(x + 0.5*width, mins, width, label='Min', alpha=0.8)
    ax.bar(x + 1.5*width, maxs, width, label='Max', alpha=0.8)
    
    ax.set_ylabel('Signal Value', fontsize=12)
    ax.set_title('Model Signal Statistics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'signal_statistics.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved statistics plot to {save_dir / 'signal_statistics.png'}")
    plt.close()


if __name__ == "__main__":
    results, signals, test_data = main()
    
    # Plot results
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)
    plot_training_results(signals, test_data)
    
    print("\nTraining completed successfully!")

