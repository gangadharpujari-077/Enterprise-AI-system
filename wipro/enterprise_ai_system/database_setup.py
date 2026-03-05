"""
Database Setup Script
Initialize Supabase tables and schema for Enterprise AI System
Run this script once to set up the database
"""

import os
from dotenv import load_dotenv

load_dotenv()

# SQL statements to create tables
SETUP_SQL = """
-- Create employees table
CREATE TABLE IF NOT EXISTS public.employees (
    employee_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(100),
    team VARCHAR(100),
    joining_date DATE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create work metrics table
CREATE TABLE IF NOT EXISTS public.work_metrics (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    employee_id UUID NOT NULL REFERENCES public.employees(employee_id),
    tasks_completed INTEGER,
    avg_task_time DECIMAL(10, 2),
    working_hours DECIMAL(10, 2),
    overtime_hours DECIMAL(10, 2),
    meeting_hours DECIMAL(10, 2),
    bug_count INTEGER,
    focus_score DECIMAL(5, 2),
    deadline_gap DECIMAL(10, 2),
    date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_employee_date UNIQUE(employee_id, date)
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS public.predictions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    employee_id UUID NOT NULL REFERENCES public.employees(employee_id),
    delay_risk DECIMAL(5, 4),
    burnout_risk DECIMAL(5, 4),
    performance_score DECIMAL(5, 2),
    cluster_label VARCHAR(100),
    prediction_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create AI reports table
CREATE TABLE IF NOT EXISTS public.ai_reports (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    employee_id UUID REFERENCES public.employees(employee_id),
    report_text TEXT NOT NULL,
    generated_at TIMESTAMP DEFAULT NOW(),
    report_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create forecasts table
CREATE TABLE IF NOT EXISTS public.forecasts (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    employee_id UUID NOT NULL REFERENCES public.employees(employee_id),
    metric_type VARCHAR(100),
    forecast_values DECIMAL(10, 2)[],
    forecast_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create employee clusters table
CREATE TABLE IF NOT EXISTS public.employee_clusters (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    employee_id UUID NOT NULL REFERENCES public.employees(employee_id),
    cluster_label VARCHAR(100),
    cluster_score DECIMAL(5, 4),
    assigned_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_employee_cluster UNIQUE(employee_id, assigned_date)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_work_metrics_employee ON public.work_metrics(employee_id);
CREATE INDEX IF NOT EXISTS idx_work_metrics_date ON public.work_metrics(date);
CREATE INDEX IF NOT EXISTS idx_predictions_employee ON public.predictions(employee_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON public.predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_ai_reports_employee ON public.ai_reports(employee_id);
CREATE INDEX IF NOT EXISTS idx_employee_clusters_employee ON public.employee_clusters(employee_id);

-- Create views for analytics
CREATE OR REPLACE VIEW public.employee_summary AS
SELECT 
    e.employee_id,
    e.name,
    e.role,
    e.team,
    COUNT(DISTINCT wm.date) as days_tracked,
    AVG(wm.focus_score) as avg_focus,
    AVG(wm.working_hours) as avg_hours,
    AVG(wm.overtime_hours) as avg_overtime,
    MAX(p.delay_risk) as latest_delay_risk,
    MAX(p.burnout_risk) as latest_burnout_risk,
    MAX(p.performance_score) as latest_performance
FROM public.employees e
LEFT JOIN public.work_metrics wm ON e.employee_id = wm.employee_id
LEFT JOIN public.predictions p ON e.employee_id = p.employee_id
GROUP BY e.employee_id, e.name, e.role, e.team;

-- Enable RLS (Row Level Security) - optional but recommended
ALTER TABLE public.employees ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.work_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_reports ENABLE ROW LEVEL SECURITY;
"""

def setup_database():
    """Setup database tables"""
    try:
        from config.supabase_client import SupabaseConfig
        
        config = SupabaseConfig()
        client = config.get_client()
        
        # Note: Supabase client doesn't execute raw SQL directly
        # Use Supabase web UI or pgAdmin to execute the SQL
        print("Database setup SQL generated. Please execute the following in your Supabase SQL editor:")
        print("\n" + "="*80)
        print(SETUP_SQL)
        print("="*80)
        print("\nSteps:")
        print("1. Go to your Supabase project")
        print("2. Navigate to SQL Editor")
        print("3. Create a new query and paste the SQL above")
        print("4. Execute the query")
        
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False

def insert_sample_data():
    """Insert sample data for testing"""
    sample_sql = """
-- Insert sample employees
INSERT INTO public.employees (name, role, team, joining_date) VALUES
('Alice Johnson', 'Senior Developer', 'Engineering', '2023-01-15'),
('Bob Smith', 'Product Manager', 'Product', '2023-02-20'),
('Carol Davis', 'Data Scientist', 'Analytics', '2023-03-10'),
('David Wilson', 'DevOps Engineer', 'Infrastructure', '2023-04-05'),
('Eve Martinez', 'UI Designer', 'Design', '2023-05-12')
ON CONFLICT DO NOTHING;

-- Insert sample work metrics
INSERT INTO public.work_metrics (employee_id, tasks_completed, avg_task_time, working_hours, 
                                 overtime_hours, meeting_hours, bug_count, focus_score, deadline_gap)
SELECT 
    employee_id,
    FLOOR(RANDOM() * 20 + 5),
    RANDOM() * 4 + 1,
    FLOOR(RANDOM() * 10 + 35),
    FLOOR(RANDOM() * 5),
    FLOOR(RANDOM() * 8),
    FLOOR(RANDOM() * 5),
    RANDOM() * 30 + 70,
    RANDOM() * 10 + 3
FROM public.employees
ON CONFLICT (employee_id, date) DO NOTHING;
"""
    
    print("\nSample data SQL:")
    print("\n" + "="*80)
    print(sample_sql)
    print("="*80)

if __name__ == "__main__":
    print("Enterprise AI System - Database Setup")
    print("="*80)
    
    setup_database()
    insert_sample_data()
    
    print("\n✓ Setup script complete!")
    print("\nNext steps:")
    print("1. Execute the SQL in your Supabase SQL editor")
    print("2. Create a .env file with your Supabase credentials")
    print("3. Install required packages: pip install -r requirements.txt")
    print("4. Run the API: python api/main.py")
    print("5. Run the dashboard: streamlit run dashboard/app.py")
