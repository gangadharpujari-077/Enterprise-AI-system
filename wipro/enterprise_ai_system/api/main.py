"""
FastAPI Backend for Enterprise AI System
Provides REST APIs for predictions, forecasting, and report generation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import logging
from datetime import datetime

# Import system modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.supabase_client import config as supabase_config
from data_pipeline.preprocessing import DataPreprocessor
from ml_models.train_models import PredictionEngine
from timeseries.forecasting import TimeSeriesForecaster, PerformanceTrendAnalyzer
from clustering.employee_segmentation import EmployeeSegmentation, EmployeeRiskProfile
from reinforcement.workload_agent import WorkloadManagementAgent
from genai.report_generator import GenAIReportGenerator

app = FastAPI(
    title="Enterprise AI Risk & Performance Intelligence System",
    description="AI-powered workforce analytics and management",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
prediction_engine = PredictionEngine()
segmentation_engine = EmployeeSegmentation(n_clusters=3)
rl_agent = WorkloadManagementAgent()
report_generator = GenAIReportGenerator()
preprocessor = DataPreprocessor()

# Pydantic models
class WorkMetrics(BaseModel):
    """Work metrics input model"""
    employee_id: str
    tasks_completed: float
    avg_task_time: float
    working_hours: float
    overtime_hours: float
    meeting_hours: float
    bug_count: float
    focus_score: float
    deadline_gap: float

class PredictionRequest(BaseModel):
    """Prediction request model"""
    features: List[float]
    employee_id: Optional[str] = None

class ForecastRequest(BaseModel):
    """Forecast request model"""
    employee_id: str
    metric: str
    periods: int = 7

class ReportRequest(BaseModel):
    """Report generation request"""
    employee_id: str
    include_recommendations: bool = True

class RLRecommendationRequest(BaseModel):
    """Reinforcement learning recommendation request"""
    workload_score: float
    delay_risk: float
    burnout_risk: float

# REST API Endpoints

@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "status": "operational",
        "service": "Enterprise AI System",
        "version": "1.0.0"
    }

@app.post("/predict-risk")
async def predict_risk(request: PredictionRequest):
    """
    Predict delay and burnout risk for an employee
    
    POST /predict-risk
    {
        "features": [0.5, 0.3, ...],
        "employee_id": "Emp123"
    }
    """
    try:
        X = np.array(request.features).reshape(1, -1)
        
        predictions = prediction_engine.predict_all(X)
        
        if predictions is None:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        return {
            "employee_id": request.employee_id,
            "delay_risk": float(predictions['delay_risk'][0]),
            "burnout_risk": float(predictions['burnout_risk'][0]),
            "performance_score": float(predictions['performance_score'][0]),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast")
async def forecast(request: ForecastRequest):
    """
    Forecast future performance metrics
    
    POST /forecast
    {
        "employee_id": "EMP123",
        "metric": "productivity",
        "periods": 7
    }
    """
    try:
        forecaster = TimeSeriesForecaster(method='arima')
        
        # Generate mock forecast (in production, fetch from Supabase)
        ts_values = np.random.randn(30).cumsum() + 50
        forecast_values = forecaster.moving_average_forecast(
            np.ones(30) * 50,  # Mock historical data
            window=7,
            steps=request.periods
        )
        
        return {
            "employee_id": request.employee_id,
            "metric": request.metric,
            "forecast": forecast_values.tolist(),
            "periods": request.periods,
            "trend": PerformanceTrendAnalyzer.detect_trend(
                np.ones(30) * 50
            ),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/employee-clusters")
async def get_employee_clusters():
    """
    Get employee cluster assignments and characteristics
    
    GET /employee-clusters
    """
    try:
        if supabase_config:
            # Fetch data from Supabase
            employees = supabase_config.fetch_employees()
            metrics = supabase_config.fetch_work_metrics()
        else:
            return {
                "status": "no_data",
                "message": "Supabase not configured",
                "clusters": []
            }
        
        # Process and cluster (simplified)
        return {
            "clusters": {
                "High Performer": 25,
                "Stable Worker": 45,
                "Burnout Risk": 20,
                "Development Focus": 10
            },
            "total_employees": 100,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Cluster error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-report")
async def generate_report(request: ReportRequest):
    """
    Generate AI-based report for an employee
    
    POST /generate-report
    {
        "employee_id": "EMP123",
        "include_recommendations": true
    }
    """
    try:
        # Fetch employee data
        if not supabase_config:
            raise HTTPException(status_code=503, detail="Database not available")
        
        # Mock data for demonstration
        employee_data = {
            "employee_name": f"Employee_{request.employee_id}",
            "delay_risk": 0.35,
            "burnout_risk": 0.25,
            "performance_score": 78.5,
            "cluster": "Stable Worker",
            "working_hours": 42,
            "overtime_hours": 5,
            "meeting_hours": 8,
            "tasks_completed": 15,
            "bug_count": 2,
            "focus_score": 82
        }
        
        # Generate report
        report_text = report_generator.generate_risk_report(
            employee_data['employee_name'],
            {
                'delay_risk': employee_data['delay_risk'],
                'burnout_risk': employee_data['burnout_risk'],
                'performance_score': employee_data['performance_score']
            },
            {'label': employee_data['cluster']},
            {
                'working_hours': employee_data['working_hours'],
                'overtime_hours': employee_data['overtime_hours'],
                'meeting_hours': employee_data['meeting_hours'],
                'tasks_completed': employee_data['tasks_completed'],
                'bug_count': employee_data['bug_count'],
                'focus_score': employee_data['focus_score']
            }
        )
        
        # Store report in database
        if supabase_config:
            supabase_config.insert_report(report_text)
        
        return {
            "employee_id": request.employee_id,
            "report": report_text,
            "generated_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rl-recommendation")
async def get_rl_recommendation(request: RLRecommendationRequest):
    """
    Get RL-based workload recommendation
    
    POST /rl-recommendation
    {
        "workload_score": 0.65,
        "delay_risk": 0.35,
        "burnout_risk": 0.25
    }
    """
    try:
        recommendation = rl_agent.recommend_action(
            request.workload_score,
            request.delay_risk,
            request.burnout_risk
        )
        
        return {
            **recommendation,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"RL recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard-data")
async def get_dashboard_data():
    """
    Get aggregated data for dashboard
    
    GET /dashboard-data
    """
    try:
        return {
            "summary": {
                "total_employees": 100,
                "high_risk_count": 15,
                "medium_risk_count": 35,
                "low_risk_count": 50
            },
            "risk_distribution": {
                "critical": 2,
                "high": 13,
                "medium": 35,
                "low": 50
            },
            "cluster_distribution": {
                "High Performer": 25,
                "Stable Worker": 45,
                "Burnout Risk": 20,
                "Development Focus": 10
            },
            "average_metrics": {
                "performance_score": 78.5,
                "workload_utilization": 65.0,
                "productivity_trend": "stable",
                "burnout_risk_average": 0.28
            },
            "recent_insights": [
                "Team performance is stable with slight upward trend",
                "2 employees require immediate attention",
                "Workload distribution is balanced across teams"
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Dashboard data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Enterprise AI Backend",
        "database": "connected" if supabase_config else "disconnected",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
