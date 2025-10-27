# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HULL TACTICAL MARKET PREDICTION             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│   Data Input    │
│   train.csv     │
│   test.csv      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPROCESSING LAYER                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  load_trainset() / load_testset()                      │   │
│  │  - Load CSV with Polars                                │   │
│  │  - Rename target columns                                │   │
│  │  - Type casting (Float64)                              │   │
│  │  - Basic cleaning                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING LAYER                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  src/features.py                                        │   │
│  │                                                           │   │
│  │  • create_lag_features()                                │   │
│  │    - Lags: [1, 2, 3, 5, 10 days]                        │   │
│  │                                                           │   │
│  │  • create_rolling_features()                            │   │
│  │    - Mean/Std/Min/Max                                   │   │
│  │    - Windows: [5, 10, 20, 50]                          │   │
│  │                                                           │   │
│  │  • create_momentum_features()                            │   │
│  │    - RSI-like indicators                                │   │
│  │    - Moving average deviations                          │   │
│  │                                                           │   │
│  │  • create_interaction_features()                        │   │
│  │    - Ratios, products                                   │   │
│  │                                                           │   │
│  │  • create_volatility_features()                         │   │
│  │    - Rolling volatility                                 │   │
│  │    - Coefficient of variation                           │   │
│  │                                                           │   │
│  │  Result: 88 features from 13 basic features             │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE SCALING LAYER                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  StandardScaler from sklearn                           │   │
│  │  - fit() on training data                               │   │
│  │  - transform() on train and test                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING LAYER                         │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  ElasticNet  │  │  LightGBM    │  │  XGBoost     │         │
│  │  ──────────  │  │  ──────────  │  │  ──────────  │         │
│  │  src/models/ │  │  src/models/ │  │  src/models/ │         │
│  │  elastic_net.│  │  lightgbm_   │  │  xgboost_    │         │
│  │  py          │  │  model.py    │  │  model.py    │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                │
│         └──────────────────┼──────────────────┘                │
│                            ▼                                     │
│                  ┌─────────────────────┐                          │
│                  │   Ensemble Model    │                          │
│                  │   src/models/       │                          │
│                  │   ensemble.py       │                          │
│                  │                     │                          │
│                  │  • Voting Ensemble │                          │
│                  │  • Weighted Average │                          │
│                  │  • Stacking         │                          │
│                  └─────────────────────┘                          │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   HYPERPARAMETER OPTIMIZATION                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Optuna (Bayesian Optimization)                        │   │
│  │  ────────────────────────────────────────               │   │
│  │                                                           │   │
│  │  • Time Series Cross-Validation (5 folds)                │   │
│  │  • N trials per model (default: 20)                     │   │
│  │  • Parallel execution support                            │   │
│  │  • Early stopping                                        │   │
│  │  • Best parameter selection                              │   │
│  │                                                           │   │
│  │  Search Spaces:                                          │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐     │   │
│  │  │ ElasticNet   │  │  LightGBM    │  │  XGBoost    │     │   │
│  │  │ • alpha      │  │ • leaves     │  │ • depth     │     │   │
│  │  │ • l1_ratio   │  │ • lr         │  │ • lr        │     │   │
│  │  └─────────────┘  │ • features   │  │ • subsample │     │   │
│  │                    │ • bagging    │  │ • gamma     │     │   │
│  │                    │ • lambda     │  │ • lambda    │     │   │
│  │                    └──────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL EVALUATION LAYER                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  compare_models.py                                     │   │
│  │  ─────────────────────────────────────                 │   │
│  │                                                          │   │
│  │  • calculate_metrics()                                 │   │
│  │    - RMSE (Root Mean Squared Error)                    │   │
│  │    - MAE (Mean Absolute Error)                         │   │
│  │    - R² Score                                           │   │
│  │                                                          │   │
│  │  • plot_predictions()                                   │   │
│  │    - Visualization of predictions                      │   │
│  │                                                          │   │
│  │  • plot_feature_importance()                            │   │
│  │    - Top N features visualization                      │   │
│  │                                                          │   │
│  │  • evaluate_model()                                    │   │
│  │    - Signal conversion                                 │   │
│  │    - Range validation                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PREDICTION & OUTPUT                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  convert_ret_to_signal()                               │   │
│  │  ─────────────────────────────────────                │   │
│  │                                                          │   │
│  │  • Input: Raw predictions (returns)                   │   │
│  │  • Process:                                             │   │
│  │    - Multiply by SIGNAL_MULTIPLIER (400)               │   │
│  │    - Add 1.0                                            │   │
│  │    - Clip to [0.0, 2.0] range                          │   │
│  │  • Output: Trading signals                             │   │
│  │                                                          │   │
│  │  • Competition scoring:                                 │   │
│  │    - evaluation.py                                      │   │
│  │    - Volatility-adjusted Sharpe ratio                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌──────────────┐
│  Raw Data    │
│  CSV Files   │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Preprocessing                          │
│  • Load CSV                              │
│  • Type conversion                       │
│  • Basic cleaning                        │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Feature Engineering                    │
│  • 13 → 88 features                      │
│  • Lags, rolling stats                  │
│  • Momentum, interactions              │
│  • Volatility metrics                   │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Scaling                                │
│  • StandardScaler                       │
│  • Fit on train, transform both         │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Optuna Optimization                    │
│  • Bayesian search                       │
│  • Cross-validation                      │
│  • Best params selected                 │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Model Training                          │
│  • ElasticNet                            │
│  • LightGBM ✅ Best                      │
│  • XGBoost                               │
│  • Ensemble                              │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Predictions                            │
│  • Raw returns                           │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Signal Conversion                      │
│  • ret * 400 + 1                         │
│  • Clip [0, 2]                           │
│  • Trading signals                       │
└─────────────────────────────────────────┘
```

## Model Architecture Details

### 1. ElasticNet Model
```
┌─────────────────────────────────────────┐
│         ElasticNet Architecture         │
├─────────────────────────────────────────┤
│  Algorithm: Regularized Regression      │
│  ─────────────────────────────────────  │
│                                         │
│  Input: X (88 features)                 │
│         └─► StandardScaler             │
│         └─► Linear Model               │
│                                         │
│  Regularization:                        │
│  • L1 (Lasso): Alpha × L1_ratio        │
│  • L2 (Ridge): Alpha × (1 - L1_ratio)  │
│                                         │
│  Hyperparameters (Optuna):             │
│  • alpha: [1e-4, 1.0] (log scale)     │
│  • l1_ratio: [0.0, 1.0]                │
│                                         │
│  Best Score: 0.010908 RMSE             │
└─────────────────────────────────────────┘
```

### 2. LightGBM Model (Best Performer)
```
┌─────────────────────────────────────────┐
│       LightGBM Architecture             │
├─────────────────────────────────────────┤
│  Algorithm: Gradient Boosting (Tree)   │
│  ─────────────────────────────────────  │
│                                         │
│  Input: X (88 features)                 │
│         └─► Decision Trees              │
│         └─► Leaf-wise growth            │
│         └─► Ensemble of trees          │
│                                         │
│  Hyperparameters (Optuna):             │
│  • num_leaves: [10, 300]               │
│  • learning_rate: [0.01, 0.3]          │
│  • feature_fraction: [0.4, 1.0]        │
│  • bagging_fraction: [0.4, 1.0]        │
│  • min_child_samples: [5, 100]         │
│  • reg_alpha: [1e-8, 10]               │
│  • reg_lambda: [1e-8, 10]              │
│                                         │
│  Best Score: 0.009635 RMSE 🏆          │
│  Best Params:                          │
│    • num_leaves: 290                   │
│    • lr: 0.069                         │
│    • feature_fraction: 0.695            │
└─────────────────────────────────────────┘
```

### 3. XGBoost Model
```
┌─────────────────────────────────────────┐
│         XGBoost Architecture            │
├─────────────────────────────────────────┤
│  Algorithm: Extreme Gradient Boosting  │
│  ─────────────────────────────────────  │
│                                         │
│  Input: X (88 features)                 │
│         └─► Level-wise trees            │
│         └─► Regularization              │
│         └─► Handling missing values     │
│                                         │
│  Hyperparameters (Optuna):             │
│  • max_depth: [3, 10]                   │
│  • learning_rate: [0.01, 0.3]           │
│  • subsample: [0.4, 1.0]                │
│  • colsample_bytree: [0.4, 1.0]        │
│  • min_child_weight: [1, 10]           │
│  • gamma: [1e-8, 10]                   │
│  • reg_alpha: [1e-8, 10]               │
│  • reg_lambda: [1e-8, 10]              │
│                                         │
│  Best Score: 0.009800 RMSE             │
└─────────────────────────────────────────┘
```

### 4. Ensemble Model
```
┌─────────────────────────────────────────┐
│       Ensemble Architecture             │
├─────────────────────────────────────────┤
│  Type: Weighted Voting                 │
│  ─────────────────────────────────────  │
│                                         │
│  Input: Individual model predictions   │
│                                         │
│  Models & Weights:                     │
│  • ElasticNet: 30% weight              │
│  • LightGBM: 35% weight                │
│  • XGBoost: 35% weight                 │
│                                         │
│  Output:                               │
│  Σ (prediction_i × weight_i)            │
│                                         │
│  Benefits:                             │
│  • Reduces overfitting                  │
│  • Better generalization                │
│  • Robust to individual model failures  │
│                                         │
│  Signal Range: [0.55, 2.00]            │
└─────────────────────────────────────────┘
```

## Feature Engineering Pipeline

```
Input: 13 Base Features
│
├─► Lag Features (×5 lags each)
│   • S2_lag_1, S2_lag_2, ..., I2_lag_5
│   ≈ +25 features
│
├─► Rolling Statistics (×4 stats ×3 windows)
│   • mean_5, std_5, min_5, max_5
│   • mean_10, std_10, min_10, max_10
│   • mean_20, std_20, min_20, max_20
│   ≈ +48 features
│
├─► Momentum Indicators
│   • RSI-like positive/negative changes
│   • Moving average deviations
│   ≈ +15 features
│
└─► Interaction Features
    • Ratios (I2/I1, M11/avg)
    • Volatility features
    ≈ +10 features
    │
    ▼
Output: 88 Engineered Features
```

## Training Pipeline Flow

```
┌──────────────────────────────────────────────┐
│  Training Script (train.py)                 │
├──────────────────────────────────────────────┤
│                                              │
│  1. Load Data                                │
│     ├─► train.csv                           │
│     └─► test.csv                            │
│                                              │
│  2. Feature Engineering                      │
│     ├─► create_advanced_features()           │
│     └─► select_features()                   │
│                                              │
│  3. Scale Features                           │
│     └─► StandardScaler.fit()               │
│                                              │
│  4. Train Models (with Optuna)               │
│     ├─► ElasticNet.fit()                    │
│     ├─► LightGBM.fit()                      │
│     ├─► XGBoost.fit()                       │
│     └─► Ensemble.fit()                      │
│                                              │
│  5. Generate Predictions                     │
│     ├─► model.predict(X_test)               │
│     └─► convert_ret_to_signal()            │
│                                              │
│  6. Evaluation                               │
│     ├─► Calculate metrics                   │
│     ├─► Feature importance                  │
│     └─► Signal validation                   │
└──────────────────────────────────────────────┘
```

## File Organization

```
hull_tactical_market_prediction_using_hyperopt/
│
├── src/
│   ├── features.py              # Feature engineering
│   ├── models/
│   │   ├── elastic_net.py       # ElasticNet model
│   │   ├── lightgbm_model.py   # LightGBM model
│   │   ├── xgboost_model.py    # XGBoost model
│   │   └── ensemble.py          # Ensemble methods
│   ├── config.py                # Configuration
│   └── data.py                  # Data loading
│
├── input/
│   └── hull-tactical-market-prediction/
│       ├── train.csv            # Training data
│       ├── test.csv             # Test data
│       └── kaggle_evaluation/   # Eval framework
│
├── train.py                     # Main training script
├── main.py                       # Basic implementation
├── compare_models.py            # Model comparison
├── evaluation.py                # Scoring function
├── requirements.txt             # Dependencies
│
├── Documentation/
│   ├── README.md                # Project overview
│   ├── ARCHITECTURE.md          # This file
│   ├── TRAINING_RESULTS.md      # Results summary
│   ├── QUICK_START.md          # Quick guide
│   ├── ADVANCED_FEATURES.md    # Feature docs
│   └── IMPLEMENTATION_SUMMARY.md
│
└── artifacts/                   # Outputs
    ├── predictions_comparison.png
    ├── feature_importance.png
    └── evaluation_report.txt
```

## Performance Characteristics

### Model Comparison Table

| Model | CV Score | Training Time | Parameters | Strengths |
|-------|----------|---------------|------------|-----------|
| **LightGBM** | 0.009635 | ~30s | 8 | Fast, accurate, great for many features |
| XGBoost | 0.009800 | ~45s | 9 | Robust, handles missing values well |
| Ensemble | Combined | ~120s | All | Most stable, best generalization |
| ElasticNet | 0.010908 | ~5s | 2 | Interpretable, fast, linear baseline |

### Feature Engineering Impact

```
Before: 13 basic features
         ↓
After:  88 engineered features
         ↑
    6.8x more features

Improvement:
• Captures temporal patterns (lags)
• Identifies trends (rolling stats)
• Detects momentum (RSI-like)
• Models interactions (ratios)
• Measures volatility (std, CV)
```

## Key Design Decisions

1. **Feature Engineering**: Aggressive feature creation to capture market dynamics
2. **Multiple Algorithms**: Different models capture different patterns
3. **Optuna Optimization**: Bayesian optimization vs grid search for efficiency
4. **Time Series CV**: Respects temporal order for realistic evaluation
5. **Ensemble Approach**: Combines strengths of all models for robustness
6. **Signal Conversion**: Clips to [0, 2] range for competition compliance

## Technology Stack

- **Data Processing**: Polars (fast dataframe operations)
- **ML Models**: scikit-learn, LightGBM, XGBoost
- **Optimization**: Optuna (Bayesian optimization)
- **Evaluation**: scikit-learn metrics
- **Visualization**: Matplotlib, Seaborn
- **Language**: Python 3.13

## Scalability Considerations

- **Feature Engineering**: O(n) where n = number of features
- **Model Training**: O(n × m × trials) where n=samples, m=features
- **Predictions**: O(1) per prediction
- **Memory**: ~2GB RAM for full training
- **Parallel**: Optuna can use multiple cores

## Extensibility

The architecture supports:
- Adding new models via `src/models/`
- New features via `src/features.py`
- Custom evaluation metrics
- Additional ensemble strategies
- Different optimization algorithms

