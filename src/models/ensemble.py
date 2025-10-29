"""
Ensemble model implementation combining multiple algorithms.
"""

import numpy as np
from typing import List, Dict, Optional
from sklearn.ensemble import VotingRegressor
from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel
from .elastic_net import ElasticNetModel
from .catboost_model import CatBoostModel


class EnsembleModel:
    """Ensemble model combining multiple base models."""
    
    def __init__(self, models: List, weights: Optional[List[float]] = None):
        """
        Initialize ensemble model.
        
        Args:
            models: List of model instances
            weights: Optional weights for each model
        """
        self.models = models
        self.weights = weights if weights is not None else [1.0 / len(models)] * len(models)
        self.ensemble = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, optimize: bool = True):
        """
        Fit all models in the ensemble.
        
        Args:
            X: Feature matrix
            y: Target values
            optimize: Whether to optimize hyperparameters (for models that support it)
        """
        for model in self.models:
            model.fit(X, y, optimize=optimize)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using weighted ensemble.
        
        Args:
            X: Feature matrix
        
        Returns:
            Ensemble predictions
        """
        predictions = []
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            predictions.append(pred * self.weights[i])
        
        return np.sum(predictions, axis=0)
    
    def get_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get individual model predictions.
        
        Args:
            X: Feature matrix
        
        Returns:
            Dictionary mapping model names to predictions
        """
        predictions = {}
        for i, model in enumerate(self.models):
            model_name = type(model).__name__
            predictions[model_name] = model.predict(X)
        return predictions
    
    def get_feature_importance(self) -> Dict[str, Dict]:
        """
        Get feature importance from all models.
        
        Returns:
            Dictionary mapping model names to their feature importances
        """
        importances = {}
        for model in self.models:
            model_name = type(model).__name__
            try:
                if hasattr(model, 'get_feature_importance'):
                    importances[model_name] = model.get_feature_importance()
                elif hasattr(model, 'feature_importances_'):
                    importances[model_name] = dict(enumerate(model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    importances[model_name] = dict(enumerate(model.coef_))
            except:
                importances[model_name] = {}
        
        return importances


def create_default_ensemble(n_trials: int = 50, cv_folds: int = 5, random_state: int = 42) -> EnsembleModel:
    """
    Create a default ensemble with all available models.
    
    Args:
        n_trials: Number of Optuna trials for hyperparameter optimization
        cv_folds: Number of cross-validation folds
        random_state: Random state for reproducibility
    
    Returns:
        EnsembleModel instance
    """
    models = [
        LightGBMModel(n_trials=n_trials, cv_folds=cv_folds, random_state=random_state),
        XGBoostModel(n_trials=n_trials, cv_folds=cv_folds, random_state=random_state),
        CatBoostModel(n_trials=n_trials, cv_folds=cv_folds, random_state=random_state),
        ElasticNetModel(cv_folds=cv_folds, random_state=random_state)
    ]
    
    # Equal weights for all models
    weights = [0.25] * len(models)
    
    return EnsembleModel(models=models, weights=weights)


class StackingEnsemble:
    """Stacking ensemble with meta-learner."""
    
    def __init__(self, base_models: List, meta_model):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: List of base models
            meta_model: Meta-learner model
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.meta_features = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, optimize: bool = True):
        """
        Fit base models and meta-learner.
        
        Args:
            X: Feature matrix
            y: Target values
            optimize: Whether to optimize hyperparameters
        """
        # Fit base models
        for model in self.base_models:
            model.fit(X, y, optimize=optimize)
        
        # Generate meta-features using cross-validation approach
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        
        meta_features_list = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            fold_meta_features = []
            
            for model in self.base_models:
                # Train model on fold
                model.fit(X_train, y_train, optimize=False)
                # Predict on validation set
                pred = model.predict(X_val)
                fold_meta_features.append(pred)
            
            meta_features_list.append(
                (X_val, np.column_stack(fold_meta_features), y_val)
            )
        
        # Combine all meta-features
        meta_X = np.vstack([mf[1] for mf in meta_features_list])
        meta_y = np.concatenate([mf[2] for mf in meta_features_list])
        
        # Fit meta-model
        self.meta_model.fit(meta_X, meta_y)
        
        # Retrain base models on full dataset
        for model in self.base_models:
            model.fit(X, y, optimize=optimize)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using stacking ensemble.
        
        Args:
            X: Feature matrix
        
        Returns:
            Ensemble predictions
        """
        # Get base model predictions
        base_predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            base_predictions.append(pred)
        
        # Stack predictions
        meta_X = np.column_stack(base_predictions)
        
        # Get meta-model prediction
        return self.meta_model.predict(meta_X)

