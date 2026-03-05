"""
Data Preprocessing Module
Handles data cleaning, normalization, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data cleaning and preprocessing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.is_fitted = False
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean missing values and outliers
        
        Args:
            df: Input dataframe
        
        Returns:
            Cleaned dataframe
        """
        logger.info(f"Original data shape: {df.shape}")
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        logger.info(f"Cleaned data shape: {df.shape}")
        return df
    
    def normalize_features(self, df: pd.DataFrame, 
                          numeric_cols: Optional[list] = None,
                          fit: bool = False) -> pd.DataFrame:
        """
        Normalize numeric features using StandardScaler
        
        Args:
            df: Input dataframe
            numeric_cols: List of numeric columns to normalize
            fit: Whether to fit the scaler
        
        Returns:
            DataFrame with normalized features
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_copy = df.copy()
        
        if fit:
            df_copy[numeric_cols] = self.scaler.fit_transform(df_copy[numeric_cols])
            self.is_fitted = True
        else:
            if not self.is_fitted:
                df_copy[numeric_cols] = self.scaler.fit_transform(df_copy[numeric_cols])
                self.is_fitted = True
            else:
                df_copy[numeric_cols] = self.scaler.transform(df_copy[numeric_cols])
        
        return df_copy
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw data
        
        Args:
            df: Input dataframe with work metrics
        
        Returns:
            DataFrame with additional engineered features
        """
        df = df.copy()
        
        # Calculate productivity metrics
        if 'tasks_completed' in df.columns and 'working_hours' in df.columns:
            df['task_per_hour'] = df['tasks_completed'] / (df['working_hours'] + 1)
        
        # Calculate overtime ratio
        if 'overtime_hours' in df.columns and 'working_hours' in df.columns:
            df['overtime_ratio'] = df['overtime_hours'] / (df['working_hours'] + 1)
        
        # Calculate workload intensity
        if 'meeting_hours' in df.columns and 'working_hours' in df.columns:
            df['meeting_ratio'] = df['meeting_hours'] / (df['working_hours'] + 1)
        
        # Calculate quality metric
        if 'bug_count' in df.columns and 'tasks_completed' in df.columns:
            df['bug_per_task'] = df['bug_count'] / (df['tasks_completed'] + 1)
        
        # Workload stress indicator
        if 'deadline_gap' in df.columns:
            df['deadline_pressure'] = 1 / (df['deadline_gap'] + 0.1)
        
        # Overall workload score
        if 'working_hours' in df.columns and 'meeting_hours' in df.columns:
            df['total_workload'] = df['working_hours'] + df['meeting_hours']
        
        logger.info(f"Engineered features added. New shape: {df.shape}")
        return df
    
    def prepare_for_modeling(self, df: pd.DataFrame, 
                           target_col: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare data for ML modeling
        
        Args:
            df: Clean input dataframe
            target_col: Name of target column if available
        
        Returns:
            Tuple of (features, target) or (features, None)
        """
        df = self.clean_data(df)
        df = self.normalize_features(df)
        df = self.feature_engineering(df)
        
        # Separate features and target
        if target_col and target_col in df.columns:
            X = df.drop(columns=[target_col, 'employee_id', 'date'], errors='ignore').values
            y = df[target_col].values
            return X, y
        else:
            X = df.drop(columns=['employee_id', 'date'], errors='ignore').values
            return X, None
    
    def get_feature_names(self, df: pd.DataFrame) -> list:
        """Get feature column names"""
        return df.drop(columns=['employee_id', 'date'], errors='ignore').columns.tolist()


# Utility function
def load_and_preprocess(data_path: str, target_col: Optional[str] = None) -> Tuple:
    """
    Load CSV data and preprocess it
    
    Args:
        data_path: Path to CSV file
        target_col: Target column name
    
    Returns:
        Preprocessed features and target
    """
    df = pd.read_csv(data_path)
    preprocessor = DataPreprocessor()
    return preprocessor.prepare_for_modeling(df, target_col)
