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
        
        # Split data for train/val tracking
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, shuffle=False
        )
        
        print(f"Loss function: Regression (RMSE)")
        print(f"Training with up to 1000 boosting rounds...")
        
        # Train with validation set to track metrics
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Track training history
        train_losses = []
        val_losses = []
        
        def callback(env):
            # Extract RMSE from evaluation results
            # evaluation_result_list format: [('train', 'rmse', value), ('eval', 'rmse', value)]
            if env.evaluation_result_list:
                # Training metric
                if len(env.evaluation_result_list) > 0:
                    train_losses.append(env.evaluation_result_list[0][2])
                # Validation metric
                if len(env.evaluation_result_list) > 1:
                    val_losses.append(env.evaluation_result_list[1][2])
        
        self.model = lgb.train(
            self.best_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'eval'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(100),
                callback
            ]
        )
        
        # Calculate metrics
        if train_losses and val_losses:
            print(f"\n📊 LightGBM Training Metrics:")
            print(f"   Final Training RMSE: {train_losses[-1]:.6f}")
            print(f"   Final Validation RMSE: {val_losses[-1]:.6f}")
            print(f"   Best Training RMSE: {min(train_losses):.6f} (round {train_losses.index(min(train_losses))})")
            print(f"   Best Validation RMSE: {min(val_losses):.6f} (round {val_losses.index(min(val_losses))})")
            print(f"   Worst Training RMSE: {max(train_losses):.6f} (round {train_losses.index(max(train_losses))})")
            print(f"   Worst Validation RMSE: {max(val_losses):.6f} (round {val_losses.index(max(val_losses))})")
            print(f"   Actual boosting rounds used: {len(train_losses)}")
        elif train_losses:
            print(f"\n📊 LightGBM Training Metrics:")
            print(f"   Final Training RMSE: {train_losses[-1]:.6f}")
            print(f"   Actual boosting rounds used: {len(train_losses)}")
        
        # Retrain on full dataset
        # Use best iteration from model if available, otherwise use length of training history
        best_iterations = len(train_losses) if train_losses else 1000
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
            best_iterations = self.model.best_iteration
        
        train_data_full = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            self.best_params,
            train_data_full,
            num_boost_round=best_iterations,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        self.training_metrics = {
            'loss_function': 'Regression (RMSE)',
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

