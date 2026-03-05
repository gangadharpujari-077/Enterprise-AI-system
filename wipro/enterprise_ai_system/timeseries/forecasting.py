"""
Time Series Forecasting Module
Forecast productivity trends and performance using ARIMA and Moving Averages
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesForecaster:
    """Forecast future productivity and performance metrics"""
    
    def __init__(self, method: str = 'arima'):
        self.method = method
        self.model = None
        self.fcast_results = None
        self.history = None
    
    def prepare_timeseries(self, df: pd.DataFrame, 
                          value_col: str, 
                          date_col: str = 'date') -> pd.Series:
        """
        Prepare time series data
        
        Args:
            df: Input dataframe
            value_col: Column name for values to forecast
            date_col: Column name for dates
        
        Returns:
            Time series as pandas Series
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        ts = df.set_index(date_col)[value_col]
        ts = ts.fillna(ts.mean())  # Fill missing values with mean
        
        self.history = ts
        return ts
    
    def fit_arima(self, ts: pd.Series, order: Tuple = (1, 1, 1)) -> Dict:
        """
        Fit ARIMA model
        
        Args:
            ts: Time series data
            order: ARIMA order (p, d, q)
        
        Returns:
            Model summary stats
        """
        try:
            self.model = ARIMA(ts, order=order)
            self.model = self.model.fit()
            
            logger.info(f"ARIMA{order} model fitted successfully")
            logger.info(f"AIC: {self.model.aic:.2f}, BIC: {self.model.bic:.2f}")
            
            return {
                'aic': self.model.aic,
                'bic': self.model.bic,
                'rmse': np.sqrt(np.mean(self.model.resid ** 2))
            }
        except Exception as e:
            logger.error(f"ARIMA fitting error: {e}")
            return None
    
    def fit_sarima(self, ts: pd.Series, 
                   order: Tuple = (1, 1, 1),
                   seasonal_order: Tuple = (1, 1, 1, 7)) -> Dict:
        """
        Fit SARIMA model (with seasonality)
        
        Args:
            ts: Time series data
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
        
        Returns:
            Model summary stats
        """
        try:
            self.model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
            self.model = self.model.fit(disp=False)
            
            logger.info(f"SARIMA model fitted successfully")
            logger.info(f"AIC: {self.model.aic:.2f}")
            
            return {
                'aic': self.model.aic,
                'bic': self.model.bic,
                'rmse': np.sqrt(np.mean(self.model.resid ** 2))
            }
        except Exception as e:
            logger.error(f"SARIMA fitting error: {e}")
            return None
    
    def forecast(self, steps: int = 7) -> np.ndarray:
        """
        Generate forecast
        
        Args:
            steps: Number of periods to forecast
        
        Returns:
            Array of forecasted values
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        try:
            forecast_result = self.model.get_forecast(steps=steps)
            forecast_values = forecast_result.predicted_mean.values
            
            logger.info(f"Forecast generated for {steps} periods")
            return forecast_values
        except Exception as e:
            logger.error(f"Forecast error: {e}")
            return None
    
    def forecast_with_ci(self, steps: int = 7, 
                         alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecast with confidence intervals
        
        Args:
            steps: Number of periods to forecast
            alpha: Confidence level (0.05 for 95% CI)
        
        Returns:
            Tuple of (forecast, lower_ci, upper_ci)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        try:
            forecast_result = self.model.get_forecast(steps=steps)
            forecast_df = forecast_result.conf_int(alpha=alpha)
            
            forecast_values = forecast_result.predicted_mean.values
            lower_ci = forecast_df.iloc[:, 0].values
            upper_ci = forecast_df.iloc[:, 1].values
            
            return forecast_values, lower_ci, upper_ci
        except Exception as e:
            logger.error(f"Forecast with CI error: {e}")
            return None, None, None
    
    def moving_average_forecast(self, ts: pd.Series, 
                               window: int = 7, 
                               steps: int = 7) -> np.ndarray:
        """
        Simple moving average forecast
        
        Args:
            ts: Time series data
            window: Window size for moving average
            steps: Number of periods to forecast
        
        Returns:
            Array of forecasted values
        """
        ma = ts.rolling(window=window).mean()
        last_ma = ma.iloc[-1]
        
        # For simple MA, use last value as constant forecast
        forecast = np.full(steps, last_ma)
        
        logger.info(f"Moving average forecast ({window}-period window) generated")
        return forecast
    
    def exponential_smoothing_forecast(self, ts: pd.Series, 
                                      alpha: float = 0.3,
                                      steps: int = 7) -> np.ndarray:
        """
        Simple exponential smoothing forecast
        
        Args:
            ts: Time series data
            alpha: Smoothing parameter (0 < alpha < 1)
            steps: Number of periods to forecast
        
        Returns:
            Array of forecasted values
        """
        forecast_list = []
        s_t = ts.iloc[0]
        
        for t in range(len(ts)):
            forecast_list.append(s_t)
            s_t = alpha * ts.iloc[t] + (1 - alpha) * s_t
        
        # Use last smoothed value for future forecast
        future_forecast = np.full(steps, s_t)
        
        logger.info(f"Exponential smoothing forecast generated (alpha={alpha})")
        return future_forecast


class PerformanceTrendAnalyzer:
    """Analyze and forecast performance trends"""
    
    @staticmethod
    def detect_trend(ts: pd.Series, window: int = 7) -> str:
        """
        Detect trend direction
        
        Args:
            ts: Time series data
            window: Window size for trend calculation
        
        Returns:
            'uptrend', 'downtrend', or 'stable'
        """
        if len(ts) < window:
            return 'stable'
        
        recent_values = ts.iloc[-window:]
        trend_value = np.polyfit(range(len(recent_values)), recent_values.values, 1)[0]
        
        if trend_value > 0.5:
            return 'uptrend'
        elif trend_value < -0.5:
            return 'downtrend'
        else:
            return 'stable'
    
    @staticmethod
    def calculate_volatility(ts: pd.Series, window: int = 7) -> float:
        """
        Calculate rolling volatility
        
        Args:
            ts: Time series data
            window: Window size
        
        Returns:
            Volatility score (0-1)
        """
        returns = ts.pct_change().dropna()
        volatility = returns.rolling(window=window).std().iloc[-1]
        
        # Normalize to 0-1 range
        return min(1.0, volatility * 10)
    
    @staticmethod
    def forecast_performance_score(productivity_forecast: np.ndarray,
                                  quality_forecast: np.ndarray,
                                  workload_forecast: np.ndarray) -> np.ndarray:
        """
        Generate composite performance score forecast
        
        Args:
            productivity_forecast: Forecasted productivity
            quality_forecast: Forecasted quality
            workload_forecast: Forecasted workload
        
        Returns:
            Array of performance scores (0-100)
        """
        # Normalize inputs to 0-1 range
        productivity_norm = (productivity_forecast - productivity_forecast.min()) / \
                           (productivity_forecast.max() - productivity_forecast.min() + 1e-8)
        quality_norm = (quality_forecast - quality_forecast.min()) / \
                      (quality_forecast.max() - quality_forecast.min() + 1e-8)
        
        # Workload should be balanced (not too high, not too low)
        workload_norm = np.abs(workload_forecast - workload_forecast.mean()) / \
                       (workload_forecast.std() + 1e-8)
        workload_norm = 1 - np.clip(workload_norm, 0, 1)
        
        # Composite score: 50% productivity, 30% quality, 20% workload balance
        performance_score = (productivity_norm * 0.5 + 
                            quality_norm * 0.3 + 
                            workload_norm * 0.2) * 100
        
        return performance_score
