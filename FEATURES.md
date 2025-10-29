# Advanced Features Implementation

## Overview

This document describes the comprehensive set of advanced features implemented for the Hull Tactical Market Prediction project.

## 1. Advanced Feature Engineering

### Location: `src/features.py`

### Features Implemented:

#### 1.1 Lag Features
- Creates lag features for specified columns with customizable lag periods (default: 1, 2, 3, 5, 10 days)
- Captures temporal dependencies and market momentum

#### 1.2 Rolling Statistics
- Rolling mean, standard deviation, minimum, and maximum
- Window sizes: 5, 10, 20, 50 days
- Helps capture trends and volatility patterns

#### 1.3 Momentum Indicators
- RSI-like features using positive/negative changes
- Moving average deviations (above MA5, MA10, MA20)
- Captures overbought/oversold conditions

#### 1.4 Interaction Features
- Ratio features (e.g., I2/I1, M11 normalization)
- Mathematical combinations of important features
- Captures non-linear relationships

#### 1.5 Volatility Features
- Rolling volatility (5, 10, 20 day windows)
- Coefficient of variation
- Helps identify market regime changes

### Usage:
```python
from src.features import create_advanced_features, select_features

# Create features
df = create_advanced_features(df, feature_cols)

# Select features by method
features = select_features(df, method='all')  # 'all', 'basic', or 'extended'
```

## 2. Multiple Algorithm Implementations

### 2.1 ElasticNet Model
**Location:** `src/models/elastic_net.py`
- Regularized linear regression with L1 and L2 penalties
- Optuna hyperparameter optimization
- Time series cross-validation
- Best for capturing linear relationships with regularization

### 2.2 LightGBM Model
**Location:** `src/models/lightgbm_model.py`
- Gradient boosting with leaf-wise tree growth
- Fast training and efficient memory usage
- Feature importance analysis
- Best for non-linear relationships and large feature sets

**Hyperparameters tuned:**
- num_leaves, learning_rate, feature_fraction
- bagging_fraction, bagging_freq, min_child_samples
- reg_alpha, reg_lambda

### 2.3 XGBoost Model
**Location:** `src/models/xgboost_model.py`
- Advanced gradient boosting framework
- Handles missing values and regularization
- Excellent generalization capability
- Best for complex interactions

**Hyperparameters tuned:**
- max_depth, learning_rate, subsample
- colsample_bytree, colsample_bylevel
- min_child_weight, gamma, reg_alpha, reg_lambda

### 2.4 CatBoost Model
**Location:** `src/models/catboost_model.py`
- Gradient boosting optimized for categorical features
- Built-in early stopping and overfitting detection
- Robust handling of categorical variables
- Automatic handling of missing values

**Loss Function:** RMSE (Root Mean Squared Error)

**Hyperparameters tuned:**
- iterations: [100, 2000]
- learning_rate: [0.01, 0.3]
- depth: [4, 10]
- l2_leaf_reg: [1e-8, 10.0]
- bootstrap_type: ['Bayesian', 'Bernoulli', 'MVS']
- random_strength: [1e-8, 10.0]
- od_type: ['IncToDec', 'Iter']
- od_wait: [10, 50]

**Training Metrics Tracking:**
- Real-time training and validation RMSE
- Best/worst metrics with iteration numbers
- Early stopping with best iteration selection

### 2.5 Ensemble Model
**Location:** `src/models/ensemble.py`

#### Simple Voting Ensemble:
- Combines predictions from multiple models
- Weighted or unweighted averaging
- Reduces overfitting and improves robustness

#### Stacking Ensemble:
- Uses meta-learner to combine base model predictions
- Cross-validation for meta-feature generation
- Typically achieves best performance

## 3. Hyperparameter Tuning with Optuna

### Features:
- Bayesian optimization for efficient search
- Time series cross-validation (respects temporal order)
- Configurable number of trials
- Progress tracking and logging
- Best parameter selection

### Diagram of Tuning Process:

```
Training Data
    ↓
Time Series Split (5 folds)
    ↓
Optuna Trial:
  - Generate hyperparameters
  - Train on fold
  - Evaluate on validation set
  - Calculate RMSE
    ↓
Select Best Parameters
    ↓
Retrain on Full Dataset
```

### Example Usage:
```python
# Automatic optimization
model = LightGBMModel(n_trials=50)
model.fit(X_train, y_train, optimize=True)

# Without optimization
model.fit(X_train, y_train, optimize=False)
```

## 4. Model Comparison Framework

### Location: `compare_models.py`

### Features:

#### 4.1 Evaluation Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R2 (Coefficient of Determination)

#### 4.2 Visualization
- Prediction comparison plots
- Feature importance charts
- Model performance comparison tables

#### 4.3 Reporting
- Comprehensive evaluation reports
- Save plots and reports to `artifacts/` directory
- Feature importance rankings

### Usage:
```bash
python compare_models.py
```

## 5. Training Pipeline

### Location: `train.py`

### Features:
1. **Data Loading**: Loads and preprocesses train/test data
2. **Feature Engineering**: Creates advanced features automatically
3. **Model Training**: Trains ElasticNet, LightGBM, XGBoost, CatBoost, and Ensemble
4. **Hyperparameter Optimization**: Uses Optuna for each model
5. **Metrics Tracking**: 
   - Loss function display for each model
   - Training and validation RMSE tracking
   - Best/worst metrics with iteration numbers
   - Epochs/iterations display
6. **Evaluation**: Calculates metrics and signal ranges
7. **Feature Importance**: Shows top important features
8. **Training Summary**: Comprehensive summary of all metrics

### Workflow:

```
Load Data
    ↓
Feature Engineering
    ↓
Scale Features
    ↓
Train Models (with Optuna):
  - ElasticNet
  - LightGBM
  - XGBoost
  - CatBoost
  - Ensemble
    ↓
Generate Predictions
    ↓
Evaluate & Show Results
```

### Usage:
```bash
# Full training with optimization
python train.py

# Quick test (without optimization)
# Modify train.py to set optimize=False
```

## 6. File Structure

```
├── src/
│   ├── features.py           # Advanced feature engineering
│   └── models/
│       ├── elastic_net.py    # ElasticNet implementation
│       ├── lightgbm_model.py # LightGBM implementation
│       ├── xgboost_model.py  # XGBoost implementation
│       ├── catboost_model.py # CatBoost implementation
│       └── ensemble.py       # Ensemble methods
├── train.py                  # Main training script
├── compare_models.py         # Model comparison
├── evaluation.py             # Competition scoring
└── artifacts/                # Saved results and plots
```

## 7. Benefits of Implementation

### 7.1 Advanced Feature Engineering
- **88 features** vs 13 basic features
- Captures complex market dynamics
- Improves model predictive power

### 7.2 Multiple Algorithms
- Different models capture different patterns
- 5 algorithms (ElasticNet, LightGBM, XGBoost, CatBoost, Ensemble)
- Ensemble reduces risk of poor performance
- Opportunity to compare and select best model

### 7.5 Training Metrics & Monitoring
- Real-time loss function tracking
- Training and validation metrics displayed
- Best/worst performance metrics identified
- Iteration/epoch tracking with early stopping
- Comprehensive training summary for all models

### 7.3 Hyperparameter Tuning
- Optimizes model performance
- Reduces overfitting
- Finds optimal complexity

### 7.4 Evaluation Framework
- Systematic model comparison
- Visual understanding of performance
- Feature importance insights

## 8. Performance Expectations

With these implementations:
- **Feature Engineering**: Increases features from 13 to 88+
- **Model Variety**: 5 different modeling approaches (including CatBoost)
- **Optimization**: Each model tuned individually
- **Metrics Tracking**: Comprehensive training/validation metrics for all models
- **Ensemble**: Combines best aspects of all models

Expected improvements:
- Lower RMSE on validation set
- Better generalization to test data
- More stable predictions
- Improved Sharpe ratio on competition metric

## 9. Next Steps

1. **Experiment with more trials**: Increase `n_trials` for better optimization
2. **Feature selection**: Use feature importance to select most important features
3. **Stacking refinement**: Optimize meta-learner hyperparameters
4. **Additional features**: Consider adding more domain-specific features
5. **Cross-validation**: Experiment with different CV strategies

## 10. Configuration

Key parameters in `train.py`:
- `optimize=True/False`: Enable/disable Optuna optimization
- `n_trials`: Number of optimization trials (20 default)
- `cv_folds`: Number of cross-validation folds (5 default)

Adjust these based on:
- Available computational resources
- Time constraints
- Desired optimization depth
