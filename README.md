# Hull Tactical Market Prediction using Hyperopt

This project implements an advanced market timing model inspired by Hull Tactical's approach, using Optuna for hyperparameter optimization and multiple machine learning algorithms to enhance predictive performance.

## Key Features

- **Advanced Feature Engineering**: Creates 88+ features including lags, rolling statistics, momentum indicators, and interactions
- **Multiple Algorithms**: ElasticNet, LightGBM, XGBoost, and Ensemble models
- **Hyperparameter Optimization**: Optuna-based Bayesian optimization for each model
- **Model Comparison Framework**: Systematic evaluation and visualization of model performance
- **Competition-Ready**: Includes evaluation scoring function for Hull Tactical Market Prediction competition

## Setup

### 1. Create a Virtual Environment

```bash
python3 -m venv venv
```

### 2. Activate the Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Running the Main Script

```bash
# Make sure venv is activated
source venv/bin/activate

# Run the main script
python main.py
```

### Training the Model

```bash
python train.py
```

### Comparing Models

```bash
python compare_models.py
```

### Creating Submissions

```bash
python create_submission.py
```

## Project Structure

```
├── main.py                    # Main ElasticNet model implementation
├── evaluation.py              # Competition scoring function
├── train.py                   # Training script
├── compare_models.py          # Model comparison
├── create_submission.py       # Submission file generation
├── src/
│   ├── config.py             # Configuration settings
│   ├── data.py               # Data loading utilities
│   ├── features.py            # Feature engineering
│   └── models/                # Model implementations
│       ├── elastic_net.py     # ElasticNet model
│       ├── xgboost_model.py   # XGBoost model
│       ├── lightgbm_model.py # LightGBM model
│       └── ensemble.py        # Ensemble models
├── input/                     # Input data directory
├── artifacts/                 # Saved models and results
├── experiments/               # Experiment tracking
└── notebooks/                 # Jupyter notebooks

```

## Features

- **ElasticNet Model**: Regularized regression with L1 and L2 penalties
- **XGBoost Model**: Gradient boosting for improved predictions
- **LightGBM Model**: Fast gradient boosting framework
- **Ensemble Methods**: Combining multiple models for better performance
- **Hyperparameter Optimization**: Using Hyperopt for optimal parameter tuning
- **Evaluation**: Custom Sharpe ratio-based scoring function

## Model Configuration

### ElasticNet Parameters
- `CV`: Number of cross-validation folds (default: 10)
- `L1_RATIO`: ElasticNet mixing parameter (default: 0.5)
- `ALPHAS`: Regularization constants (logspace from -4 to 2)
- `MAX_ITER`: Maximum iterations (default: 1,000,000)

### Signal Parameters
- `MIN_SIGNAL`: Minimum signal value (default: 0.0)
- `MAX_SIGNAL`: Maximum signal value (default: 2.0)
- `SIGNAL_MULTIPLIER`: Multiplier for predictions (default: 400.0)

## Evaluation

The competition uses a volatility-adjusted Sharpe ratio that penalizes strategies with:
- Significantly higher volatility than the market
- Returns that fail to outperform the market

The scoring function is implemented in `evaluation.py`.

## Training Results

✅ **Successfully Trained!** See `TRAINING_RESULTS.md` for detailed results.

### Quick Summary:
- **Best Model**: LightGBM (CV Score: 0.009635 RMSE)
- **Features**: 88 engineered features
- **Algorithms**: 4 models trained (ElasticNet, LightGBM, XGBoost, Ensemble)
- **Optimization**: Optuna hyperparameter tuning completed
- **Signals**: Valid trading signals generated in [0, 2] range

### Run Training:
```bash
source venv/bin/activate
python train.py
```

### Expected Output:
- Model predictions for all algorithms
- Feature importance rankings
- Signal ranges and statistics
- Best hyperparameters for each model

## License

See LICENSE file for details.