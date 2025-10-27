"""
LightGBM model implementation with Optuna hyperparameter optimization.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, Optional
import lightgbm as lgb
import optuna
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error


class LightGBMModel:
    """LightGBM model with Optuna hyperparameter tuning."""
    
    def __init__(self,
                 n_trials: int = 50,
                 cv_folds: int = 5,
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Initialize LightGBM model.
        
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
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': self.random_state,
            'verbose': -1,
        }
        
        # Time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
            )
            
            y_pred = model.predict(X_val)
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            scores.append(score)
        
        return np.mean(scores)
    
    def fit(self, X: np.ndarray, y: np.ndarray, optimize: bool = True):
        """
        Fit the LightGBM model.
        
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
            print(f"Best LightGBM score: {self.best_score:.6f}")
            print(f"Best LightGBM params: {self.best_params}")
        else:
            self.best_params = {
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
            }
        
        # Add fixed parameters
        self.best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'random_state': self.random_state,
            'verbose': -1,
        })
        
        # Train final model
        train_data = lgb.Dataset(X, label=y, params=self.best_params)
        self.model = lgb.train(
            self.best_params,
            train_data,
            num_boost_round=1000,
            callbacks=[]  # No callbacks needed for final training
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
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance.
        
        Returns:
            Dictionary of feature importances
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")
        return dict(zip(self.model.feature_name(), self.model.feature_importance(importance_type='gain')))

