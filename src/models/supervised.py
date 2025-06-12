"""
Supervised learning models for fuel theft detection.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from ..config.config import Config
from ..utils.logger import get_logger
from ..utils.helpers import save_model

logger = get_logger(__name__)


class SupervisedModel:
    """Handles supervised learning for fuel theft classification."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize SupervisedModel.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = None
        
    def train_evaluate(self, 
                      data: pd.DataFrame,
                      features: List[str],
                      target: str = 'Stationary_drain') -> Dict[str, Any]:
        """
        Train and evaluate supervised models.
        
        Args:
            data: Full dataframe with features and target
            features: List of feature column names
            target: Target column name
            
        Returns:
            Dictionary with results
        """
        logger.info("Starting supervised learning...")
        
        # Prepare data
        X, y = self._prepare_data(data, features, target)
        
        # Split data
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self._handle_imbalance(X_train, y_train)
        
        # Train models
        model_results = self._train_models(X_train_balanced, y_train_balanced, X_test, y_test)
        
        # Select best model
        best_model_name = self._select_best_model(model_results)
        
        # Cross-validation on best model
        cv_results = self._cross_validate(X_train, y_train, best_model_name)
        
        # Feature importance
        feature_importance = self._get_feature_importance(features)
        
        # Create final predictions on full data
        final_predictions = self._predict_full_data(data, features)
        
        # Compile results
        results = {
            'model_results': model_results,
            'best_model': best_model_name,
            'best_score': model_results[best_model_name]['f1'],
            'cv_results': cv_results,
            'feature_importance': feature_importance,
            'predictions': final_predictions,
            'test_size': len(X_test),
            'train_size': len(X_train)
        }
        
        self.results = results
        return results
    
    def _prepare_data(self, data: pd.DataFrame, features: List[str], target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training."""
        # Select features and target
        available_features = [f for f in features if f in data.columns]
        X = data[available_features].copy()
        y = data[target].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        
        logger.info(f"Prepared data: {len(X)} samples, {len(available_features)} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split data into train and test sets."""
        return train_test_split(
            X, y,
            test_size=self.config.model.test_size,
            random_state=self.config.model.random_state,
            stratify=y
        )
    
    def _handle_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple:
        """Handle class imbalance."""
        method = self.config.model.resampling_method
        
        # Scale features first
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if method == 'none':
            return X_train_scaled, y_train
        
        logger.info(f"Handling class imbalance with {method}")
        
        if method == 'smote':
            # Ensure k_neighbors is valid
            n_neighbors = min(5, y_train.value_counts().min() - 1)
            sampler = SMOTE(random_state=self.config.model.random_state, k_neighbors=n_neighbors)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=self.config.model.random_state)
        else:
            logger.warning(f"Unknown resampling method: {method}. Using SMOTE.")
            n_neighbors = min(5, y_train.value_counts().min() - 1)
            sampler = SMOTE(random_state=self.config.model.random_state, k_neighbors=n_neighbors)
        
        X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)
        
        logger.info(f"Resampled data: {len(y_resampled)} samples")
        logger.info(f"New distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        return X_resampled, y_resampled
    
    def _train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """Train multiple models."""
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                random_state=self.config.model.random_state,
                max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                **self.config.model.rf_params
            ),
            'XGBoost': xgb.XGBClassifier(
                **self.config.model.xgb_params,
                scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=self.config.model.random_state,
                verbosity=-1
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                random_state=self.config.model.random_state,
                max_iter=500
            )
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
                }
                
                # Store results
                self.models[name] = model
                results[name] = metrics
                
                logger.info(f"  {name} - F1: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                results[name] = {'f1': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'roc_auc': 0}
        
        return results
    
    def _select_best_model(self, model_results: Dict[str, Dict]) -> str:
        """Select best model based on F1 score."""
        best_model = max(model_results.items(), key=lambda x: x[1]['f1'])
        self.best_model_name = best_model[0]
        self.best_model = self.models[self.best_model_name]
        
        logger.info(f"Best model: {self.best_model_name} (F1: {best_model[1]['f1']:.4f})")
        
        return self.best_model_name
    
    def _cross_validate(self, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict[str, float]:
        """Perform cross-validation on the best model."""
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        X_scaled = self.scaler.fit_transform(X)
        
        skf = StratifiedKFold(n_splits=self.config.model.cv_folds, shuffle=True, 
                             random_state=self.config.model.random_state)
        
        cv_scores = {
            'accuracy': cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy').mean(),
            'precision': cross_val_score(model, X_scaled, y, cv=skf, scoring='precision').mean(),
            'recall': cross_val_score(model, X_scaled, y, cv=skf, scoring='recall').mean(),
            'f1': cross_val_score(model, X_scaled, y, cv=skf, scoring='f1').mean()
        }
        
        logger.info(f"Cross-validation F1: {cv_scores['f1']:.4f}")
        
        return cv_scores
    
    def _get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from tree-based models."""
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = model.feature_importances_
        
        if not importance_dict:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame(importance_dict, index=feature_names)
        importance_df['Mean_Importance'] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values('Mean_Importance', ascending=False)
        
        return importance_df
    
    def _predict_full_data(self, data: pd.DataFrame, features: List[str]) -> np.ndarray:
        """Make predictions on full dataset."""
        if self.best_model is None:
            return np.array([])
        
        # Prepare features
        available_features = [f for f in features if f in data.columns]
        X = data[available_features].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.best_model.predict(X_scaled)
        
        # Add to dataframe
        data['Supervised_Prediction'] = predictions
        
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(X_scaled)[:, 1]
            data['Theft_Probability'] = probabilities
        
        return predictions
    
    def save_best_model(self, filepath: str) -> None:
        """Save the best model to file."""
        if self.best_model is None:
            logger.warning("No model to save. Train models first.")
            return
        
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'config': self.config
        }
        
        save_model(model_data, filepath)
        logger.info(f"Saved best model to {filepath}")