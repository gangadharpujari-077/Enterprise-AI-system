"""
Quick Start Guide - Running the Enterprise AI System
Execute this script to set up and run the system components
"""

import os
import sys
import subprocess
from pathlib import Path

class QuickStart:
    """Quick start setup and runner"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.log("=" * 80)
        self.log("Enterprise AI System - Quick Start")
        self.log("=" * 80)
    
    def log(self, message):
        """Print formatted log message"""
        print(f"\n📌 {message}")
    
    def check_python_version(self):
        """Check Python version"""
        self.log("Checking Python version...")
        version = sys.version_info
        if version.major < 3 or version.minor < 10:
            self.log("❌ Python 3.10+ required")
            return False
        self.log(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_env_file(self):
        """Check if .env file exists"""
        self.log("Checking environment configuration...")
        env_file = self.root_dir / ".env"
        env_example = self.root_dir / ".env.example"
        
        if not env_file.exists():
            self.log("⚠️  .env file not found. Creating from template...")
            if env_example.exists():
                with open(env_example, 'r') as f:
                    content = f.read()
                with open(env_file, 'w') as f:
                    f.write(content)
                self.log("✓ .env file created. Please edit it with your credentials.")
                return False
        return True
    
    def check_dependencies(self):
        """Check if required packages are installed"""
        self.log("Checking dependencies...")
        
        required = ['fastapi', 'streamlit', 'pandas', 'sklearn', 'supabase']
        missing = []
        
        for package in required:
            try:
                __import__(package)
                self.log(f"✓ {package}")
            except ImportError:
                missing.append(package)
        
        if missing:
            self.log("❌ Missing packages: " + ", ".join(missing))
            self.log("Run: pip install -r requirements.txt")
            return False
        
        return True
    
    def setup_database(self):
        """Run database setup"""
        self.log("Setting up database...")
        setup_file = self.root_dir / "database_setup.py"
        
        if setup_file.exists():
            try:
                exec(open(setup_file).read())
                return True
            except Exception as e:
                self.log(f"⚠️  Database setup: {e}")
                return False
        return False
    
    def start_api(self):
        """Start FastAPI server"""
        self.log("Starting FastAPI backend...")
        api_file = self.root_dir / "api" / "main.py"
        
        if api_file.exists():
            cmd = [sys.executable, str(api_file)]
            self.log(f"Command: {' '.join(cmd)}")
            self.log("✓ API starting at http://localhost:8000")
            self.log("  Docs available at http://localhost:8000/docs")
            subprocess.Popen(cmd)
        else:
            self.log("❌ API file not found")
    
    def start_dashboard(self):
        """Start Streamlit dashboard"""
        self.log("Starting Streamlit dashboard...")
        dashboard_file = self.root_dir / "dashboard" / "app.py"
        
        if dashboard_file.exists():
            cmd = ['streamlit', 'run', str(dashboard_file)]
            self.log(f"Command: {' '.join(cmd)}")
            self.log("✓ Dashboard starting at http://localhost:8501")
            subprocess.Popen(cmd)
        else:
            self.log("❌ Dashboard file not found")
    
    def run_interactive_menu(self):
        """Run interactive menu"""
        while True:
            print("\n" + "=" * 80)
            print("Enterprise AI System - Main Menu")
            print("=" * 80)
            print("\n1. Check System Status")
            print("2. Setup Database")
            print("3. Train Models")
            print("4. Start API Server")
            print("5. Start Dashboard")
            print("6. Start Both (API + Dashboard)")
            print("7. View API Documentation")
            print("8. View System Status")
            print("9. Exit")
            
            choice = input("\nSelect option (1-9): ").strip()
            
            if choice == '1':
                self.check_system_status()
            elif choice == '2':
                self.setup_database()
            elif choice == '3':
                self.run_training()
            elif choice == '4':
                self.start_api()
                input("\nPress Enter to return to menu...")
            elif choice == '5':
                self.start_dashboard()
                input("\nPress Enter to return to menu...")
            elif choice == '6':
                self.start_api()
                print("\nWaiting for API to start...")
                import time
                time.sleep(3)
                self.start_dashboard()
                input("\nPress Enter to return to menu...")
            elif choice == '7':
                self.view_api_docs()
            elif choice == '8':
                self.view_system_status()
            elif choice == '9':
                self.log("Exiting. Thank you!")
                break
            else:
                self.log("❌ Invalid option")
    
    def check_system_status(self):
        """Check overall system status"""
        self.log("System Status Report")
        print("\n✓ Python version:", sys.version.split()[0])
        print("✓ Project location:", self.root_dir)
        print("✓ Configuration:", "✓ .env exists" if (self.root_dir / ".env").exists() else "⚠️  .env missing")
        
        # Try to import key modules
        try:
            from config.supabase_client import config
            print("✓ Supabase client: Ready")
        except:
            print("⚠️  Supabase client: Not configured")
        
        print("\nSystem ready for development!")
    
    def run_training(self):
        """Run model training"""
        self.log("Training Models")
        
        try:
            import numpy as np
            from ml_models.train_models import DelayRiskPredictor, BurnoutRiskPredictor
            
            self.log("Generating synthetic training data...")
            X = np.random.randn(100, 10)
            y_delay = np.random.randint(0, 2, 100)
            y_burnout = np.random.randint(0, 2, 100)
            
            self.log("Training Delay Risk Model...")
            delay_model = DelayRiskPredictor()
            metrics = delay_model.train(X, y_delay)
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
            print(f"  F1-Score: {metrics['f1']:.2%}")
            
            self.log("Training Burnout Risk Model...")
            burnout_model = BurnoutRiskPredictor()
            metrics = burnout_model.train(X, y_burnout)
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
            print(f"  F1-Score: {metrics['f1']:.2%}")
            
            self.log("✓ Training complete!")
        
        except Exception as e:
            self.log(f"❌ Training error: {e}")
    
    def view_api_docs(self):
        """Show API documentation"""
        print("\n" + "=" * 80)
        print("API Endpoints")
        print("=" * 80)
        
        endpoints = {
            "GET /": "Health check",
            "POST /predict-risk": "Predict delay and burnout risk",
            "GET /forecast": "Forecast future performance",
            "GET /employee-clusters": "Get employee clustering",
            "POST /generate-report": "Generate AI report",
            "POST /rl-recommendation": "Get RL-based recommendations",
            "GET /dashboard-data": "Get dashboard data",
            "GET /health": "System health check"
        }
        
        for endpoint, description in endpoints.items():
            print(f"\n{endpoint}")
            print(f"  {description}")
    
    def view_system_status(self):
        """View system status"""
        print("\n" + "=" * 80)
        print("System Status")
        print("=" * 80)
        
        status = {
            "API Server": "http://localhost:8000",
            "API Docs": "http://localhost:8000/docs",
            "Dashboard": "http://localhost:8501",
            "Database": "Configured" if (self.root_dir / ".env").exists() else "Not configured"
        }
        
        for service, url in status.items():
            print(f"\n{service}: {url}")
    
    def run(self):
        """Run quick start"""
        checks = [
            ("Python Version", self.check_python_version),
            ("Installation", self.check_dependencies),
            ("Environment", self.check_env_file)
        ]
        
        all_pass = True
        for check_name, check_func in checks:
            if not check_func():
                all_pass = False
        
        if all_pass:
            self.log("✓ All checks passed!")
            self.run_interactive_menu()
        else:
            self.log("⚠️  Please resolve issues above and try again")


def main():
    """Main entry point"""
    try:
        qs = QuickStart()
        qs.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
