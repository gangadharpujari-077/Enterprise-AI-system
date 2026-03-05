"""
Streamlit Dashboard for Enterprise AI System
Interactive visualization and monitoring dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Enterprise AI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high { color: #d62728; font-weight: bold; }
    .risk-medium { color: #ff7f0e; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

@st.cache_data
def fetch_dashboard_data():
    """Fetch aggregated dashboard data"""
    try:
        response = requests.get(f"{API_BASE_URL}/dashboard-data", timeout=5)
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching dashboard data: {e}")
        return None

@st.cache_data
def fetch_health_status():
    """Check API health"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json()
    except Exception as e:
        return {"status": "disconnected"}

def display_header():
    """Display dashboard header"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 class="main-header">📊 Enterprise AI Dashboard</h1>', 
                   unsafe_allow_html=True)
    with col2:
        health = fetch_health_status()
        status_color = "🟢" if health.get("status") == "healthy" else "🔴"
        st.metric("System Status", status_color)

def display_key_metrics(dashboard_data):
    """Display key performance metrics"""
    st.subheader("📈 Key Metrics")
    
    if not dashboard_data:
        st.warning("No data available")
        return
    
    summary = dashboard_data.get('summary', {})
    avg_metrics = dashboard_data.get('average_metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Employees",
            summary.get('total_employees', 0),
            delta="+5 this month"
        )
    
    with col2:
        high_risk = summary.get('high_risk_count', 0)
        st.metric(
            "At-Risk Employees",
            high_risk,
            delta="-2 from last week",
            delta_color="inverse"
        )
    
    with col3:
        perf_score = avg_metrics.get('performance_score', 0)
        st.metric(
            "Avg Performance",
            f"{perf_score:.1f}/100",
            delta="+2.5"
        )
    
    with col4:
        workload = avg_metrics.get('workload_utilization', 0)
        st.metric(
            "Workload Utilization",
            f"{workload:.0f}%",
            delta="+5%"
        )

def display_risk_distribution(dashboard_data):
    """Display risk distribution charts"""
    st.subheader("⚠️ Risk Distribution")
    
    if not dashboard_data:
        return
    
    col1, col2 = st.columns(2)
    
    # Risk level distribution
    with col1:
        risk_dist = dashboard_data.get('risk_distribution', {})
        fig = go.Figure(data=[
            go.Pie(
                labels=list(risk_dist.keys()),
                values=list(risk_dist.values()),
                marker=dict(
                    colors=['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c'],
                    line=dict(color='white', width=2)
                )
            )
        ])
        fig.update_layout(
            title="Risk Level Distribution",
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster distribution
    with col2:
        cluster_dist = dashboard_data.get('cluster_distribution', {})
        fig = go.Figure(data=[
            go.Bar(
                x=list(cluster_dist.keys()),
                y=list(cluster_dist.values()),
                marker=dict(
                    color=['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e'],
                    line=dict(color='white', width=1)
                )
            )
        ])
        fig.update_layout(
            title="Employee Cluster Distribution",
            xaxis_title="Cluster",
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def display_risk_breakdown(dashboard_data):
    """Display detailed risk breakdown"""
    st.subheader("📋 Risk Breakdown")
    
    if not dashboard_data:
        return
    
    summary = dashboard_data.get('summary', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "High Risk",
            summary.get('high_risk_count', 0),
            delta="Requires attention"
        )
    
    with col2:
        st.metric(
            "Medium Risk",
            summary.get('medium_risk_count', 0),
            delta="Monitor closely"
        )
    
    with col3:
        st.metric(
            "Low Risk",
            summary.get('low_risk_count', 0),
            delta="Stable"
        )

def display_productivity_trend():
    """Display productivity trend chart"""
    st.subheader("📈 Productivity Trend")
    
    # Generate mock data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    productivity = np.sin(np.linspace(0, 2*np.pi, 30)) * 10 + 75 + np.random.normal(0, 2, 30)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=productivity,
        mode='lines+markers',
        name='Productivity Score',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))
    
    fig.add_hline(y=75, line_dash="dash", line_color="gray", 
                  annotation_text="Target: 75")
    
    fig.update_layout(
        title="30-Day Productivity Trend",
        xaxis_title="Date",
        yaxis_title="Productivity Score",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_forecast():
    """Display forecasting section"""
    st.subheader("🔮 Performance Forecast")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        forecast_days = st.slider("Forecast Period (days)", 3, 30, 7)
    
    with col1:
        # Generate mock forecast
        dates = pd.date_range(start='2024-01-31', periods=forecast_days, freq='D')
        forecast_values = 75 + np.cumsum(np.random.normal(0, 1, forecast_days))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=forecast_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#2ca02c', width=3, dash='dash'),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"{forecast_days}-Day Performance Forecast",
            xaxis_title="Date",
            yaxis_title="Predicted Score",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_insights(dashboard_data):
    """Display key insights"""
    st.subheader("💡 Key Insights")
    
    if not dashboard_data:
        return
    
    insights = dashboard_data.get('recent_insights', [])
    
    for i, insight in enumerate(insights, 1):
        col1, col2 = st.columns([0.5, 9.5])
        with col1:
            st.write(f"**{i}.**")
        with col2:
            st.write(insight)

def display_employee_lookup():
    """Display employee lookup and analysis"""
    st.subheader("🔍 Employee Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        employee_id = st.text_input("Enter Employee ID", placeholder="EMP123")
    
    with col2:
        if st.button("Analyze"):
            if employee_id:
                try:
                    # Mock prediction request
                    features = np.random.rand(10)
                    
                    response = requests.post(
                        f"{API_BASE_URL}/predict-risk",
                        json={
                            "features": features.tolist(),
                            "employee_id": employee_id
                        },
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            delay_risk = data.get('delay_risk', 0)
                            risk_class = "risk-high" if delay_risk > 0.5 else \
                                       "risk-medium" if delay_risk > 0.3 else "risk-low"
                            st.metric(
                                "Delay Risk",
                                f"{delay_risk:.1%}",
                                delta=None
                            )
                        
                        with col2:
                            burnout_risk = data.get('burnout_risk', 0)
                            st.metric(
                                "Burnout Risk",
                                f"{burnout_risk:.1%}"
                            )
                        
                        with col3:
                            perf_score = data.get('performance_score', 0)
                            st.metric(
                                "Performance Score",
                                f"{perf_score:.1f}/100"
                            )
                
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter an Employee ID")

def main():
    """Main dashboard"""
    # Display header
    display_header()
    
    # Fetch data
    dashboard_data = fetch_dashboard_data()
    
    if dashboard_data:
        # Display key metrics
        display_key_metrics(dashboard_data)
        
        st.divider()
        
        # Display charts
        display_risk_distribution(dashboard_data)
        
        st.divider()
        
        display_risk_breakdown(dashboard_data)
        
        st.divider()
        
        display_productivity_trend()
        
        st.divider()
        
        display_forecast()
        
        st.divider()
        
        display_insights(dashboard_data)
        
        st.divider()
        
        display_employee_lookup()
    
    else:
        st.error("Unable to connect to backend. Please ensure the API is running.")
        st.info("Run the API with: `python api/main.py`")

if __name__ == "__main__":
    main()
