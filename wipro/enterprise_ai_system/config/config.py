"""
System configuration file
Centralized configuration for the Enterprise AI System
"""

import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

class Config:
    """Base configuration"""
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_TITLE = "Enterprise AI Risk & Performance Intelligence System"
    API_VERSION = "1.0.0"
    
    # Database Configuration
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # ML Configuration
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_STATE = 42
    MODEL_TYPE_DELAY = "random_forest"
    MODEL_TYPE_BURNOUT = "gradient_boosting"
    
    # Forecasting Configuration
    FORECAST_PERIODS = 7
    FORECAST_METHOD = "arima"
    
    # Clustering Configuration
    N_CLUSTERS = 3
    CLUSTER_METHOD = "kmeans"
    
    # RL Configuration
    RL_LEARNING_RATE = 0.1
    RL_DISCOUNT_FACTOR = 0.99
    RL_EXPLORATION_RATE = 0.1
    RL_EPISODES = 100
    
    # Computer Vision Configuration
    CV_MIN_CONFIDENCE = 0.5
    CV_VIDEO_SOURCE = 0  # Default webcam
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    ENVIRONMENT = "development"


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    ENVIRONMENT = "production"


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SUPABASE_URL = "mock://supabase"
    SUPABASE_KEY = "mock-key"


# Get configuration based on environment
def get_config() -> Config:
    """Get appropriate configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()


# Export active configuration
config = get_config()
