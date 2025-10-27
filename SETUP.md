# Setup Guide

## Quick Start

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # OR
   venv\Scripts\activate     # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Run the main script:**
   ```bash
   python main.py
   ```

## Files Created/Modified

### New Files:
- `main.py` - Main ElasticNet model implementation with data loading, preprocessing, and training
- `evaluation.py` - Competition scoring function (volatility-adjusted Sharpe ratio)
- `.gitignore` - Git ignore patterns for Python projects
- `SETUP.md` - This setup guide
- `README.md` - Updated with comprehensive project documentation

### Key Features in main.py:
- **Data Loading**: Loads train and test datasets from Polars CSV files
- **Feature Engineering**: Creates additional features (U1, U2) and selects relevant variables
- **Data Preprocessing**: Standardizes features using StandardScaler
- **Model Training**: ElasticNet with cross-validation for hyperparameter tuning
- **Signal Conversion**: Converts return predictions to trading signals (0-2 range)

## Model Configuration

The model uses the following key parameters:

### ElasticNet Parameters:
- CV folds: 10
- L1 ratio: 0.5
- Alpha range: logspace(-4, 2, 100)
- Max iterations: 1,000,000

### Signal Parameters:
- Min signal: 0.0
- Max signal: 2.0
- Signal multiplier: 400.0

## Output

When running `main.py`, you'll see:
- Training and test data shapes
- Sample data preview
- Features used in the model
- Best alpha found during cross-validation
- Model coefficients
- Prediction and signal ranges

## Next Steps

- Run `python train.py` for additional model training
- Use `python compare_models.py` to compare different models
- Generate submissions with `python create_submission.py`

## Troubleshooting

### Module not found errors:
Make sure the virtual environment is activated and dependencies are installed.

### Data path issues:
Check that the data files are in `input/hull-tactical-market-prediction/`

### Polars version issues:
The code has been updated to fix the deprecated `is_in` method by converting Series to lists.
