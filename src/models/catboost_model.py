"""
CatBoost model implementation with Optuna hyperparameter optimization.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, Optional
import catboost as cb
import optuna
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error


class CatBoostModel:
    """CatBoost model with Optuna hyperparameter tuning."""
    
    def __init__(self,
                 n_trials: int = 50,
                 cv_folds: int = 5,
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Initialize CatBoost model.
        
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
            'iterations': trial.suggest_int('iterations', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
            'od_wait': trial.suggest_int('od_wait', 10, 50),
            'random_seed': self.random_state,
            'verbose': False,
            'thread_count': self.n_jobs,
        }
        
        # Add bagging_temperature only for Bayesian bootstrap
        if params['bootstrap_type'] == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 1.0)
        
        # Time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = cb.CatBoostRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            scores.append(score)
        
        return np.mean(scores)
    
    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X: Feature matrix
            y: Target values
        
        Returns:
            Best parameters
        """
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=self.n_trials,
            n_jobs=self.n_jobs
        )
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        return self.best_params
    
    def fit(self, X: np.ndarray, y: np.ndarray, params: Optional[Dict[str, Any]] = None, optimize: bool = True) -> 'CatBoostModel':
        """
        Fit the CatBoost model.
        
        Args:
            X: Feature matrix
            y: Target values
            params: Optional parameters (if None, uses best_params from optimization)
            optimize: Whether to optimize hyperparameters first
        
        Returns:
            Self
        """
        try:
            if optimize and self.best_params is None:
                # Optimize hyperparameters if not already done
                print("Optimizing CatBoost hyperparameters...")
                self.optimize(X, y)
                print(f"Best CatBoost score: {self.get_best_score():.6f}")
                print(f"Best CatBoost params: {self.best_params}")
            
            if params is None:
                if self.best_params is None:
                    # Use default parameters if no optimization was done
                    print("Using default CatBoost parameters...")
                    params = {
                        'iterations': 1000,
                        'learning_rate': 0.1,
                        'depth': 6,
                        'l2_leaf_reg': 3.0,
                        'random_seed': self.random_state,
                        'verbose': False,
                        'thread_count': self.n_jobs,
                    }
                else:
                    params = self.best_params.copy()
            
            # Split data for train/val tracking
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, shuffle=False
            )
            
            print(f"Loss function: RMSE (Root Mean Squared Error)")
            print(f"Fitting CatBoost with params: {params}")
            print(f"Training with up to {params.get('iterations', 1000)} iterations...")
            
            # Fit with validation set to track metrics
            self.model = cb.CatBoostRegressor(**params, loss_function='RMSE')
            
            # Track training history
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=100,  # Print every 100 iterations
                use_best_model=True
            )
            
            # Get training history
            train_loss = []
            val_loss = []
            history = self.model.get_evals_result()
            if history:
                train_loss = history['learn']['RMSE'] if 'learn' in history and 'RMSE' in history['learn'] else []
                val_loss = history['validation']['RMSE'] if 'validation' in history and 'RMSE' in history['validation'] else []
                
                if train_loss and val_loss:
                    print(f"\n📊 CatBoost Training Metrics:")
                    print(f"   Final Training RMSE: {train_loss[-1]:.6f}")
                    print(f"   Final Validation RMSE: {val_loss[-1]:.6f}")
                    print(f"   Best Training RMSE: {min(train_loss):.6f} (iteration {train_loss.index(min(train_loss))})")
                    print(f"   Best Validation RMSE: {min(val_loss):.6f} (iteration {val_loss.index(min(val_loss))})")
                    print(f"   Worst Training RMSE: {max(train_loss):.6f} (iteration {train_loss.index(max(train_loss))})")
                    print(f"   Worst Validation RMSE: {max(val_loss):.6f} (iteration {val_loss.index(max(val_loss))})")
                    print(f"   Actual iterations used: {len(train_loss)}")
            
            # Retrain on full dataset with best params
            # Get best iteration from model or use length of training history
            if train_loss and val_loss:
                best_iterations = len(train_loss)
            elif hasattr(self.model, 'get_best_iteration'):
                try:
                    best_iterations = self.model.get_best_iteration()
                    if best_iterations is None or best_iterations == 0:
                        best_iterations = params.get('iterations', 1000)
                except:
                    best_iterations = params.get('iterations', 1000)
            elif hasattr(self.model, 'best_iteration_') and self.model.best_iteration_ is not None:
                best_iterations = self.model.best_iteration_
            else:
                best_iterations = params.get('iterations', 1000)
            
            self.model = cb.CatBoostRegressor(**{**params, 'iterations': best_iterations}, loss_function='RMSE')
            self.model.fit(X, y, verbose=False)
            
            self.training_metrics = {
                'loss_function': 'RMSE',
                'iterations': best_iterations,
                'train_history': train_loss,
                'val_history': val_loss
            }
            
            print("CatBoost training completed successfully!")
            
        except Exception as e:
            print(f"Error in CatBoost training: {e}")
            raise e
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def get_feature_importance(self, importance_type: str = 'PredictionValuesChange') -> np.ndarray:
        """
        Get feature importance.
        
        Args:
            importance_type: Type of importance ('PredictionValuesChange', 'LossFunctionChange', 'FeatureImportance')
        
        Returns:
            Feature importance values
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return self.model.get_feature_importance(type=importance_type)
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get best parameters from optimization."""
        return self.best_params
    
    def get_best_score(self) -> Optional[float]:
        """Get best score from optimization."""
        return self.best_score
