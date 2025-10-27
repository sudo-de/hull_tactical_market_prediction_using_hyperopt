"""
XGBoost model implementation with Optuna hyperparameter optimization.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, Optional
import xgboost as xgb
import optuna
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error


class XGBoostModel:
    """XGBoost model with Optuna hyperparameter tuning."""
    
    def __init__(self,
                 n_trials: int = 50,
                 cv_folds: int = 5,
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Initialize XGBoost model.
        
        Args:
            n_trials: Number of Optuna trials
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            n_jobs: Number of jobs to run in parallel
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.best_params = None
        self.best_score = None
        
    def objective(self, trial, X: np.ndarray, y: np.ndarray) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            X: Feature matrix
            y: Target values
        
        Returns:
            CV score
        """
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': self.random_state,
            'verbosity': 0,
        }
        
        # Time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dval, 'eval')],
                early_stopping_rounds=10,
                verbose_eval=False
            )
            
            y_pred = model.predict(dval)
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            scores.append(score)
        
        return np.mean(scores)
    
    def fit(self, X: np.ndarray, y: np.ndarray, optimize: bool = True):
        """
        Fit the XGBoost model.
        
        Args:
            X: Feature matrix
            y: Target values
            optimize: Whether to optimize hyperparameters with Optuna
        """
        if optimize:
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda trial: self.objective(trial, X, y),
                n_trials=self.n_trials,
                show_progress_bar=True
            )
            
            self.best_params = study.best_params
            self.best_score = study.best_value
            print(f"Best XGBoost score: {self.best_score:.6f}")
            print(f"Best XGBoost params: {self.best_params}")
        else:
            self.best_params = {
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 0.8,
                'min_child_weight': 3,
                'gamma': 0,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
            }
        
        # Add fixed parameters
        self.best_params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'random_state': self.random_state,
            'verbosity': 0,
        })
        
        # Train final model
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(
            self.best_params,
            dtrain,
            num_boost_round=1000,
            callbacks=[]  # No callbacks for final training
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance.
        
        Returns:
            Dictionary of feature importances
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")
        return dict(self.model.get_score(importance_type='gain'))
