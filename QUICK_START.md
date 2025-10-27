# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Activate Virtual Environment
```bash
cd /Users/sudip/hull_tactical_market_prediction_using_hyperopt
source venv/bin/activate
```

### 2. Run Training
```bash
# Full training with hyperparameter optimization (takes 2-5 minutes)
python train.py

# OR run original basic version
python main.py
```

### 3. Compare Models
```bash
python compare_models.py
```

## 📊 What You'll Get

Running `train.py` will:
- ✅ Create 88 advanced features from raw data
- ✅ Train 4 models: ElasticNet, LightGBM, XGBoost, Ensemble
- ✅ Optimize hyperparameters with Optuna
- ✅ Show feature importance
- ✅ Generate predictions for test set

## 🎯 Quick Comparison

| Feature | Original (`main.py`) | Advanced (`train.py`) |
|---------|---------------------|----------------------|
| Features | 13 basic | 88 engineered |
| Algorithms | 1 (ElasticNet) | 4 (ElasticNet, LightGBM, XGBoost, Ensemble) |
| Hyperparameter Tuning | Manual | Optuna (automatic) |
| Feature Engineering | Minimal | Advanced (lags, rolling, momentum) |
| Ensemble | ❌ | ✅ |
| Evaluation Framework | ❌ | ✅ |

## 📁 Output Files

After running, check:
- `artifacts/` - Model outputs, plots, reports
- Console output - Training progress and results

## ⚡ Configuration Tips

### For Quick Testing:
Edit `train.py` line ~208:
```python
results = train_all_DF(X_train, y_train, X_test, optimize=False)  # Disable Optuna
```

### For Production:
Edit `train.py` to set:
```python
enet = ElasticNetModel(n_trials=100)      # More trials
lgbm = LightGBMModel(n_trials=100)       # Better optimization
xgb_model = XGBoostModel(n_trials=100)
```

### For Faster Training:
Reduce number of trials:
```python
enet = ElasticNetModel(n_trials=10)      # Quick optimization
```

## 🔍 Understanding Results

### Signal Range:
- **0.0 - 2.0**: Valid signal range for competition
- **Higher = More bullish** (invest more)
- **Lower = More bearish** (invest less)

### Feature Importance:
Top features show which variables matter most for predictions

### Model Performance:
- Lower RMSE = Better predictions
- Compare models to see which performs best

## 🐛 Troubleshooting

### Import Errors:
```bash
pip install -r requirements.txt
```

### Memory Issues:
Reduce `n_trials` in model initialization

### Slow Training:
Set `optimize=False` or reduce `n_trials`

## 📚 Learn More

- `ADVANCED_FEATURES.md` - Detailed feature documentation
- `IMPLEMENTATION_SUMMARY.md` - Complete implementation overview
- `README.md` - Full project documentation

## 💡 Tips

1. Start with `main.py` to understand basics
2. Then run `train.py` for advanced features
3. Use `compare_models.py` to analyze differences
4. Adjust hyperparameters based on results
5. Increase trials for better performance

Happy Training! 🎉

