# Hull Tactical Market Prediction using Hyperopt

This project implements an advanced market timing model inspired by Hull Tactical's approach, using Optuna for hyperparameter optimization and multiple machine learning algorithms to enhance predictive performance.

## Key Features

- **Advanced Feature Engineering**: Creates 88+ features including lags, rolling statistics, momentum indicators, and interactions
- **Multiple Algorithms**: ElasticNet, LightGBM, XGBoost, CatBoost, and Ensemble models
- **Hyperparameter Optimization**: Optuna-based Bayesian optimization for each model
- **Training Metrics Tracking**: Real-time tracking of loss functions, training/validation metrics, and epochs
- **Comprehensive Metrics Display**: Best/worst training and validation RMSE with iteration numbers
- **Model Comparison Framework**: Systematic evaluation and visualization of model performance
- **Competition-Ready**: Includes evaluation scoring function for Hull Tactical Market Prediction competition

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/sudo-de/hull_tactical_market_prediction_using_hyperopt.git
cd hull_tactical_market_prediction_using_hyperopt

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Training

```bash
# Activate environment
source venv/bin/activate

# Run advanced training with all models
python train.py

# OR run basic version
python main.py
```

### 3. Compare Models

```bash
python compare_models.py
```

## Project Structure

```
├── src/
│   ├── features.py              # Advanced feature engineering
│   ├── data.py                  # Data loading utilities
│   └── models/                  # Model implementations
│       ├── elastic_net.py       # ElasticNet with Optuna
│       ├── lightgbm_model.py    # LightGBM with Optuna
│       ├── xgboost_model.py     # XGBoost with Optuna
│       ├── catboost_model.py    # CatBoost with Optuna
│       └── ensemble.py         # Ensemble methods
├── input/                       # Data directory
├── artifacts/                   # Model outputs and plots
├── train.py                     # Main training script
├── main.py                      # Basic implementation
├── compare_models.py            # Model comparison
├── evaluation.py                # Competition scoring
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── ARCHITECTURE.md              # System architecture
├── FEATURES.md                  # Feature engineering details
├── TRAINING_RESULTS.md          # Training results
└── QUICK_START.md               # Quick start guide
```

## Models

### 1. ElasticNet
- Regularized linear regression with L1 and L2 penalties
- Fast training, interpretable results
- CV Score: 0.010908 RMSE

### 2. LightGBM (Best Performer)
- Fast gradient boosting with leaf-wise tree growth
- Excellent for large feature sets
- CV Score: **0.009635 RMSE**

### 3. XGBoost
- Advanced gradient boosting framework
- Robust to missing values
- CV Score: 0.009800 RMSE

### 4. CatBoost
- Gradient boosting with categorical feature handling
- Built-in early stopping and overfitting detection
- CV Score: ~0.0091 RMSE

### 5. Ensemble
- Weighted voting of all models
- Most robust predictions
- Combines strengths of all algorithms

## Feature Engineering

- **13 basic features** → **88 engineered features**
- Lag features (1-5 day lags)
- Rolling statistics (mean, std, min, max)
- Momentum indicators
- Interaction features
- Volatility features

## Model Configuration

### Hyperparameters (Optuna-optimized)

**ElasticNet:**
- CV folds: 5
- Trials: 20
- Alpha: [1e-4, 1.0]
- L1_ratio: [0.0, 1.0]

**LightGBM:**
- CV folds: 5
- Trials: 20
- 8 hyperparameters optimized
- Best: num_leaves=290, lr=0.069

**XGBoost:**
- CV folds: 5
- Trials: 20
- 9 hyperparameters optimized
- Best: max_depth=4, lr=0.062

## Training Results

✅ **Successfully Trained!** See `TRAINING_RESULTS.md` for detailed results.

### Quick Summary:
- **Best Model**: LightGBM (CV Score: 0.009635 RMSE)
- **Features**: 88 engineered features
- **Algorithms**: 5 models trained (ElasticNet, LightGBM, XGBoost, CatBoost, Ensemble)
- **Optimization**: Optuna hyperparameter tuning completed
- **Metrics Tracking**: Loss functions, training/validation RMSE, epochs displayed
- **Signals**: Valid trading signals generated in [0, 2] range

### Run Training:
```bash
source venv/bin/activate
python train.py
```

### Expected Output:
- Model predictions for all algorithms
- Loss function displayed for each model
- Training and validation RMSE metrics (final, best, worst)
- Iterations/epochs used during training
- Feature importance rankings
- Signal ranges and statistics
- Best hyperparameters for each model
- Comprehensive training summary section

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete system architecture and design
- **[FEATURES.md](FEATURES.md)** - Advanced feature engineering details
- **[TRAINING_RESULTS.md](TRAINING_RESULTS.md)** - Detailed training results and metrics
- **[QUICK_START.md](QUICK_START.md)** - Quick start guide

## Evaluation

The competition uses a volatility-adjusted Sharpe ratio that penalizes strategies with:
- Significantly higher volatility than the market
- Returns that fail to outperform the market

The scoring function is implemented in `evaluation.py`.

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## License

See LICENSE file for details.

## Repository

🔗 **GitHub**: https://github.com/sudo-de/hull_tactical_market_prediction_using_hyperopt

## Contributing

Feel free to submit issues and enhancement requests!
