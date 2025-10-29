"""
ElasticNet model implementation with Optuna hyperparameter optimization.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import optuna


class ElasticNetModel:
    """ElasticNet model with Optuna hyperparameter tuning."""
    
    def __init__(self,
                 n_trials: int = 50,
                 cv_folds: int = 5,
                 random_state: int = 42,
                 max_iter: int = 1000000):
        """
        Initialize ElasticNet model.
        
        Args:
            n_trials: Number of Optuna trials
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            max_iter: Maximum number of iterations
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.max_iter = max_iter
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
        alpha = trial.suggest_float('alpha', 1e-4, 1.0, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        
        # Time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=self.max_iter, random_state=self.random_state)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            scores.append(score)
        
        return np.mean(scores)
    
    def fit(self, X: np.ndarray, y: np.ndarray, optimize: bool = True):
        """
        Fit the ElasticNet model.
        
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
            print(f"Best ElasticNet score: {self.best_score:.6f}")
            print(f"Best ElasticNet params: {self.best_params}")
        else:
            self.best_params = {'alpha': 0.1, 'l1_ratio': 0.5}
        
        # Split data for train/val tracking
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, shuffle=False
        )
        
        print(f"Loss function: Squared Error (MSE)")
        print(f"Training ElasticNet model...")
        
        # Train model
        self.model = ElasticNet(
            alpha=self.best_params['alpha'],
            l1_ratio=self.best_params['l1_ratio'],
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        print(f"\n📊 ElasticNet Training Metrics:")
        print(f"   Training RMSE: {train_rmse:.6f}")
        print(f"   Validation RMSE: {val_rmse:.6f}")
        print(f"   Iterations: {self.model.n_iter_} (max allowed: {self.max_iter})")
        
        # Retrain on full dataset
        self.model.fit(X, y)
        
        self.training_metrics = {
            'loss_function': 'Squared Error (MSE)',
            'iterations': self.model.n_iter_,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse
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
    
    def get_coefficients(self) -> np.ndarray:
        """
        Get model coefficients.
        
        Returns:
            Model coefficients
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting coefficients")
        return self.model.coef_

