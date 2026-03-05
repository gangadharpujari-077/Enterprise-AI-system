"""
Machine Learning Models Module
Train and evaluate models for delay risk and burnout prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DelayRiskPredictor:
    """Predict project delivery delay risk"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = None
    
    def _get_model(self):
        """Initialize model based on type"""
        if self.model_type == 'logistic':
            return LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """
        Train delay risk prediction model
        
        Args:
            X: Feature matrix
            y: Binary target (0: no delay, 1: delay)
            test_size: Train/test split ratio
        
        Returns:
            Dictionary of evaluation metrics
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.model = self._get_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info(f"Delay Risk Model - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict delay risk
        
        Args:
            X: Feature matrix
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        return predictions, probabilities
    
    def save(self, path: str):
        """Save model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")


class BurnoutRiskPredictor:
    """Predict employee burnout probability"""
    
    def __init__(self, model_type: str = 'gradient_boosting'):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
    
    def _get_model(self):
        """Initialize model based on type"""
        if self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """
        Train burnout risk prediction model
        
        Args:
            X: Feature matrix
            y: Binary target (0: no burnout, 1: burnout risk)
            test_size: Train/test split ratio
        
        Returns:
            Dictionary of evaluation metrics
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.model = self._get_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info(f"Burnout Risk Model - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict burnout risk
        
        Args:
            X: Feature matrix
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        return predictions, probabilities
    
    def save(self, path: str):
        """Save model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None


# Prediction utilities
class PredictionEngine:
    """Combined prediction engine for all risk models"""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        self.delay_predictor = DelayRiskPredictor()
        self.burnout_predictor = BurnoutRiskPredictor()
    
    def predict_all(self, X: np.ndarray) -> Dict:
        """
        Generate all predictions for given features
        
        Args:
            X: Feature matrix
        
        Returns:
            Dictionary with all predictions
        """
        try:
            delay_pred, delay_prob = self.delay_predictor.predict(X)
            burnout_pred, burnout_prob = self.burnout_predictor.predict(X)
            
            return {
                'delay_risk': delay_prob,
                'burnout_risk': burnout_prob,
                'performance_score': 100 - (delay_prob * 50 + burnout_prob * 50)
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
