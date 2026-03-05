"""
Model Training Script
Train and save ML models for the Enterprise AI System
Run this to train models on your data
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ml_models.train_models import DelayRiskPredictor, BurnoutRiskPredictor
from data_pipeline.preprocessing import DataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples: int = 200) -> tuple:
    """
    Generate synthetic training data
    
    Args:
        n_samples: Number of samples to generate
    
    Returns:
        Tuple of (X, y_delay, y_burnout)
    """
    logger.info(f"Generating {n_samples} synthetic samples...")
    
    # Features: [tasks, avg_time, hours, overtime, meetings, bugs, focus, deadline, task_per_hour, overtime_ratio]
    X = np.random.randn(n_samples, 10)
    
    # Scale features to reasonable ranges
    X[:, 0] = np.clip(X[:, 0] * 5 + 15, 5, 30)  # tasks_completed (5-30)
    X[:, 1] = np.clip(X[:, 1] + 3, 0.5, 8)  # avg_task_time (0.5-8 hours)
    X[:, 2] = np.clip(X[:, 2] * 5 + 40, 30, 60)  # working_hours (30-60)
    X[:, 3] = np.clip(X[:, 3] * 2, 0, 15)  # overtime_hours (0-15)
    X[:, 4] = np.clip(X[:, 4] * 3 + 5, 0, 20)  # meeting_hours (0-20)
    X[:, 5] = np.abs(np.random.randint(-10, 10, n_samples))  # bug_count (0-10)
    X[:, 6] = np.clip(X[:, 6] * 20 + 75, 40, 100)  # focus_score (40-100)
    X[:, 7] = np.clip(X[:, 7] + 5, 1, 20)  # deadline_gap (1-20 days)
    X[:, 8] = X[:, 0] / (X[:, 2] + 1)  # task_per_hour
    X[:, 9] = X[:, 3] / (X[:, 2] + 1)  # overtime_ratio
    
    # Target: delay_risk (0 or 1)
    # Higher overtime, lower focus, more bugs → higher delay risk
    delay_prob = (X[:, 3] / 15 * 0.3 + (100 - X[:, 6]) / 100 * 0.4 + X[:, 5] / 10 * 0.3)
    y_delay = (delay_prob > 0.5).astype(int)
    
    # Target: burnout_risk (0 or 1)
    # Higher overtime, lower focus, high hours → higher burnout risk
    burnout_prob = (X[:, 3] / 15 * 0.3 + (100 - X[:, 6]) / 100 * 0.3 + X[:, 2] / 60 * 0.4)
    y_burnout = (burnout_prob > 0.5).astype(int)
    
    logger.info(f"Data generated: X shape {X.shape}, y_delay balance {np.mean(y_delay):.1%}")
    
    return X, y_delay, y_burnout

def train_models(X: np.ndarray, y_delay: np.ndarray, y_burnout: np.ndarray):
    """
    Train all ML models
    
    Args:
        X: Feature matrix
        y_delay: Delay risk labels
        y_burnout: Burnout risk labels
    """
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Train Delay Risk Model
    logger.info("\n" + "="*60)
    logger.info("Training Delay Risk Prediction Model")
    logger.info("="*60)
    
    delay_predictor = DelayRiskPredictor(model_type='random_forest')
    delay_metrics = delay_predictor.train(X, y_delay)
    
    logger.info(f"Delay Risk Model Results:")
    logger.info(f"  Accuracy:  {delay_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {delay_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {delay_metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {delay_metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC:   {delay_metrics['auc_roc']:.4f}")
    
    delay_predictor.save(str(models_dir / "delay_risk_model.pkl"))
    
    # Train Burnout Risk Model
    logger.info("\n" + "="*60)
    logger.info("Training Burnout Risk Prediction Model")
    logger.info("="*60)
    
    burnout_predictor = BurnoutRiskPredictor(model_type='gradient_boosting')
    burnout_metrics = burnout_predictor.train(X, y_burnout)
    
    logger.info(f"Burnout Risk Model Results:")
    logger.info(f"  Accuracy:  {burnout_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {burnout_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {burnout_metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {burnout_metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC:   {burnout_metrics['auc_roc']:.4f}")
    
    burnout_predictor.save(str(models_dir / "burnout_risk_model.pkl"))
    
    return delay_metrics, burnout_metrics

def test_models(X: np.ndarray):
    """
    Test trained models
    
    Args:
        X: Test data
    """
    from ml_models.predict import model_manager
    
    logger.info("\n" + "="*60)
    logger.info("Testing Models")
    logger.info("="*60)
    
    # Test with first 5 samples
    test_X = X[:5]
    
    delay_pred, delay_prob = model_manager.predict_delay_risk(test_X)
    logger.info(f"\nDelay Risk Predictions (first 5 samples):")
    for i, (pred, prob) in enumerate(zip(delay_pred, delay_prob)):
        logger.info(f"  Sample {i+1}: Prediction={pred}, Probability={prob:.2%}")
    
    burnout_pred, burnout_prob = model_manager.predict_burnout_risk(test_X)
    logger.info(f"\nBurnout Risk Predictions (first 5 samples):")
    for i, (pred, prob) in enumerate(zip(burnout_pred, burnout_prob)):
        logger.info(f"  Sample {i+1}: Prediction={pred}, Probability={prob:.2%}")

def main():
    """Main training script"""
    logger.info("="*60)
    logger.info("Enterprise AI System - Model Training")
    logger.info("="*60)
    
    # Generate synthetic data
    X, y_delay, y_burnout = generate_synthetic_data(n_samples=200)
    
    # Train models
    delay_metrics, burnout_metrics = train_models(X, y_delay, y_burnout)
    
    # Test models
    test_models(X)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Training Complete!")
    logger.info("="*60)
    logger.info("\nModels saved to: models/")
    logger.info("\nNext steps:")
    logger.info("1. Run API: python api/main.py")
    logger.info("2. Run Dashboard: streamlit run dashboard/app.py")
    logger.info("3. Test predictions at: http://localhost:8000/docs")
    logger.info("\nTo train with your own data:")
    logger.info("1. Load your data: df = pd.read_csv('your_data.csv')")
    logger.info("2. Preprocess: preprocessor = DataPreprocessor()")
    logger.info("3. Train: model.train(X, y)")
    logger.info("4. Save: model.save('models/your_model.pkl')")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
