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
from src.models.ensemble import EnsembleModel
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
    
    # Prepare X and y
    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df.get_column('target').to_numpy()
    
    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df.get_column('target').to_numpy()
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test, scaler, feature_cols, test_df


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
    enet = ElasticNetModel(n_trials=20 if optimize else 0)
    enet.fit(X_train, y_train, optimize=optimize)
    results['ElasticNet'] = (enet, enet.predict(X_test))
    
    # LightGBM
    print("\n" + "="*50)
    print("Training LightGBM...")
    print("="*50)
    lgbm = LightGBMModel(n_trials=20 if optimize else 0)
    lgbm.fit(X_train, y_train, optimize=optimize)
    results['LightGBM'] = (lgbm, lgbm.predict(X_test))
    
    # XGBoost
    print("\n" + "="*50)
    print("Training XGBoost...")
    print("="*50)
    xgb_model = XGBoostModel(n_trials=20 if optimize else 0)
    xgb_model.fit(X_train, y_train, optimize=optimize)
    results['XGBoost'] = (xgb_model, xgb_model.predict(X_test))
    
    # Ensemble (simple average)
    print("\n" + "="*50)
    print("Training Ensemble...")
    print("="*50)
    ensemble_models = [enet, lgbm, xgb_model]
    ensemble = EnsembleModel(ensemble_models, weights=[0.3, 0.35, 0.35])
    ensemble.fit(X_train, y_train, optimize=False)
    results['Ensemble'] = (ensemble, ensemble.predict(X_test))
    
    return results


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
    X_train, y_train, X_test, y_test, scaler, feature_cols, test_df = prepare_data(train, test)
    
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
    print("Feature Importance (LightGBM)")
    print("="*70)
    try:
        lgbm_model = results['LightGBM'][0]
        feature_importance = lgbm_model.get_feature_importance()
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            print(f"{i+1}. {feature}: {importance:.4f}")
    except Exception as e:
        print(f"Could not get feature importance: {e}")
    
    return results, all_signals, test_df


def plot_training_results(all_signals: dict, test_df: pl.DataFrame, save_dir: Path = Path('artifacts')):
    """Plot training results using matplotlib."""
    save_dir.mkdir(exist_ok=True)
    
    # Plot 1: Prediction comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (model_name, signals) in enumerate(all_signals.items()):
        if idx < 4:
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

