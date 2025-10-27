"""
Model comparison and evaluation framework.
"""

import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
    }


def compare_predictions(results: Dict[str, np.ndarray], model_names: List[str]):
    """Compare predictions from different models."""
    print("\n" + "="*70)
    print("Model Comparison")
    print("="*70)
    
    # Create comparison dataframe
    comparison_data = []
    for name in model_names:
        if name in results:
            preds = results[name]
            comparison_data.append({
                'Model': name,
                'Mean': np.mean(preds),
                'Std': np.std(preds),
                'Min': np.min(preds),
                'Max': np.max(preds),
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nPrediction Statistics:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def plot_predictions(results: Dict[str, np.ndarray], model_names: List[str], save_path: str = "artifacts/predictions_comparison.png"):
    """Plot predictions from different models."""
    fig, axes = plt.subplots(len(model_names), 1, figsize=(12, 4*len(model_names)))
    
    if len(model_names) == 1:
        axes = [axes]
    
    for i, name in enumerate(model_names):
        if name in results:
            axes[i].plot(results[name], label=name, linewidth=2)
            axes[i].set_title(f'{name} Predictions')
            axes[i].set_xlabel('Sample')
            axes[i].set_ylabel('Prediction')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    plt.close()


def plot_feature_importance(feature_importance: Dict[str, float], top_n: int = 20, save_path: str = "artifacts/feature_importance.png"):
    """Plot feature importance."""
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importances = zip(*sorted_features)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(features)), importances)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to {save_path}")
    plt.close()


def create_evaluation_report(results: Dict[str, np.ndarray], y_test: np.ndarray, save_path: str = "artifacts/evaluation_report.txt"):
    """Create comprehensive evaluation report."""
    Path(save_path).parent.mkdir(exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Model Evaluation Report\n")
        f.write("="*70 + "\n\n")
        
        for name, predictions in results.items():
            f.write(f"\n{name}:\n")
            f.write("-" * 70 + "\n")
            
            metrics = calculate_metrics(predictions, y_test)
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.6f}\n")
            
            f.write(f"\nPrediction Statistics:\n")
            f.write(f"  Mean: {np.mean(predictions):.6f}\n")
            f.write(f"  Std: {np.std(predictions):.6f}\n")
            f.write(f"  Min: {np.min(predictions):.6f}\n")
            f.write(f"  Max: {np.max(predictions):.6f}\n")
    
    print(f"\nEvaluation report saved to {save_path}")


def main():
    """Main comparison function."""
    import train
    
    print("Running training to get model results...")
    results, signals, test_data = train.main()
    
    # Convert signals to numpy arrays
    signal_arrays = {name: signal for name, signal in signals.items()}
    
    # Compare predictions
    comparison_df = compare_predictions(signal_arrays, list(signal_arrays.keys()))
    
    # Plot predictions
    plot_predictions(signal_arrays, list(signal_arrays.keys()))
    
    # Get feature importance and plot
    try:
        lgbm_model = results['LightGBM'][0]
        feature_importance = lgbm_model.get_feature_importance()
        plot_feature_importance(feature_importance)
    except Exception as e:
        print(f"Could not plot feature importance: {e}")
    
    # Save evaluation report
    # Note: We'd need actual y_test values for proper evaluation
    print("\nComparison completed!")


if __name__ == "__main__":
    main()

