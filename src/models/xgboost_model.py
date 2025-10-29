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
        self.training_metrics = None
        
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
        
        # Split data for train/val tracking
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, shuffle=False
        )
        
        print(f"Loss function: reg:squarederror (RMSE)")
        print(f"Training with up to 1000 boosting rounds...")
        
        # Train with validation set to track metrics
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Track metrics
        evals_result = {}
        self.model = xgb.train(
            self.best_params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'eval')],
            evals_result=evals_result,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        # Extract metrics
        train_losses = evals_result.get('train', {}).get('rmse', [])
        val_losses = evals_result.get('eval', {}).get('rmse', [])
        
        if train_losses and val_losses:
            print(f"\n📊 XGBoost Training Metrics:")
            print(f"   Final Training RMSE: {train_losses[-1]:.6f}")
            print(f"   Final Validation RMSE: {val_losses[-1]:.6f}")
            print(f"   Best Training RMSE: {min(train_losses):.6f} (round {train_losses.index(min(train_losses))})")
            print(f"   Best Validation RMSE: {min(val_losses):.6f} (round {val_losses.index(min(val_losses))})")
            print(f"   Worst Training RMSE: {max(train_losses):.6f} (round {train_losses.index(max(train_losses))})")
            print(f"   Worst Validation RMSE: {max(val_losses):.6f} (round {val_losses.index(max(val_losses))})")
            print(f"   Actual boosting rounds used: {len(train_losses)}")
        
        # Retrain on full dataset with best iteration
        # Use best iteration from model if available, otherwise use length of training history
        best_iterations = len(train_losses) if train_losses else 1000
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
            best_iterations = self.model.best_iteration
        
        dtrain_full = xgb.DMatrix(X, label=y)
        self.model = xgb.train(
            self.best_params,
            dtrain_full,
            num_boost_round=best_iterations,
            verbose_eval=0
        )
        
        self.training_metrics = {
            'loss_function': 'reg:squarederror (RMSE)',
            'iterations': best_iterations,
            'train_history': train_losses,
            'val_history': val_losses
        }
    
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
