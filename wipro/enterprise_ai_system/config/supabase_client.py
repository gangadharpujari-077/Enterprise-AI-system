"""
Supabase Database Configuration Module
Handles connection and client initialization for Supabase PostgreSQL database
"""

# Mock supabase client for development
class MockClient:
    def table(self, name):
        return MockTable(name)

class MockTable:
    def __init__(self, name):
        self.name = name
    
    def select(self, columns):
        return self
    
    def execute(self):
        return MockResponse([])
    
    def order(self, column, desc=False):
        return self
    
    def limit(self, n):
        return self
    
    def insert(self, data):
        return self
    
    def eq(self, column, value):
        return self

class MockResponse:
    def __init__(self, data):
        self.data = data

# Mock the supabase imports
class Client:
    pass

def create_client(url, key):
    return MockClient()

from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SupabaseConfig:
    """Supabase configuration and client initialization"""
    
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        self.client: Optional[Client] = None
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
    
    def get_client(self) -> Client:
        """Get or create Supabase client"""
        if self.client is None:
            self.client = create_client(self.url, self.key)
        return self.client
    
    def fetch_employees(self):
        """Fetch all employees"""
        try:
            response = self.get_client().table("employees").select("*").execute()
            return response.data
        except Exception as e:
            print(f"Error fetching employees: {e}")
            return []
    
    def fetch_work_metrics(self, days: int = 30):
        """Fetch recent work metrics"""
        try:
            response = self.get_client().table("work_metrics").select("*").order(
                "date", desc=True
            ).limit(1000).execute()
            return response.data
        except Exception as e:
            print(f"Error fetching work metrics: {e}")
            return []
    
    def insert_prediction(self, prediction_data: dict):
        """Insert prediction results"""
        try:
            response = self.get_client().table("predictions").insert(
                prediction_data
            ).execute()
            return response.data
        except Exception as e:
            print(f"Error inserting prediction: {e}")
            return None
    
    def insert_report(self, report_text: str):
        """Insert AI-generated report"""
        try:
            response = self.get_client().table("ai_reports").insert({
                "report_text": report_text,
                "generated_at": "now()"
            }).execute()
            return response.data
        except Exception as e:
            print(f"Error inserting report: {e}")
            return None
    
    def fetch_predictions(self, employee_id: Optional[str] = None):
        """Fetch predictions"""
        try:
            query = self.get_client().table("predictions").select("*")
            if employee_id:
                query = query.eq("employee_id", employee_id)
            response = query.execute()
            return response.data
        except Exception as e:
            print(f"Error fetching predictions: {e}")
            return []
    
    def fetch_reports(self, limit: int = 10):
        """Fetch recent reports"""
        try:
            response = self.get_client().table("ai_reports").select("*").order(
                "generated_at", desc=True
            ).limit(limit).execute()
            return response.data
        except Exception as e:
            print(f"Error fetching reports: {e}")
            return []


# Initialize global client
try:
    config = SupabaseConfig()
except Exception as e:
    print(f"Warning: Supabase configuration not available: {e}")
    config = None
