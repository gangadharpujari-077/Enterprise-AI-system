"""
ML Models - Prediction Module
Load and use trained models for predictions
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Manage trained ML models"""
    
    MODELS_DIR = Path(__file__).parent.parent / "models"
    
    DEFAULT_MODELS = {
        'delay_risk': 'delay_risk_model.pkl',
        'burnout_risk': 'burnout_risk_model.pkl',
        'clustering': 'clustering_model.pkl'
    }
    
    def __init__(self):
        self.models = {}
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_model(self, model_name: str) -> Optional[object]:
        """
        Load a trained model
        
        Args:
            model_name: Name of model to load (delay_risk, burnout_risk, clustering)
        
        Returns:
            Loaded model or None
        """
        if model_name in self.models:
            return self.models[model_name]
        
        if model_name not in self.DEFAULT_MODELS:
            logger.error(f"Unknown model: {model_name}")
            return None
        
        model_path = self.MODELS_DIR / self.DEFAULT_MODELS[model_name]
        
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return None
        
        try:
            model = joblib.load(model_path)
            self.models[model_name] = model
            logger.info(f"Model loaded: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def predict_delay_risk(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict delay risk
        
        Args:
            X: Feature matrix
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        model = self.load_model('delay_risk')
        
        if model is None:
            logger.warning("Using dummy predictions")
            return np.zeros(len(X)), np.random.rand(len(X))
        
        try:
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else np.zeros(len(X))
            return predictions, probabilities
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return np.zeros(len(X)), np.zeros(len(X))
    
    def predict_burnout_risk(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict burnout risk
        
        Args:
            X: Feature matrix
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        model = self.load_model('burnout_risk')
        
        if model is None:
            logger.warning("Using dummy predictions")
            return np.zeros(len(X)), np.random.rand(len(X))
        
        try:
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else np.zeros(len(X))
            return predictions, probabilities
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return np.zeros(len(X)), np.zeros(len(X))
    
    def get_feature_importance(self, model_name: str) -> Optional[np.ndarray]:
        """
        Get feature importance from model
        
        Args:
            model_name: Name of model
        
        Returns:
            Feature importance array or None
        """
        model = self.load_model(model_name)
        
        if model is None:
            return None
        
        try:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            elif hasattr(model, 'coef_'):
                return model.coef_[0]
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
        
        return None

# Global model manager instance
model_manager = ModelManager()
