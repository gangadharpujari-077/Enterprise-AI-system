# 🚀 Getting Started - Enterprise AI System

Quick setup guide to get the system running in 5 minutes.

## Prerequisites

- Python 3.10+
- Supabase account (or local PostgreSQL)
- OpenAI API key (optional, for GenAI features)
- Git

## ⚡ Quick Start (5 minutes)

### 1. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
copy .env.example .env
```

Edit `.env` and add your credentials:
```
SUPABASE_URL=your_url
SUPABASE_KEY=your_api_key
OPENAI_API_KEY=optional_openai_key
```

### 4. Setup Database (Optional)
```bash
python database_setup.py
```
Then execute the SQL in Supabase SQL Editor.

### 5. Run the System

**Terminal 1 - Start API:**
```bash
python api/main.py
```
✓ API running: http://localhost:8000
✓ API Docs: http://localhost:8000/docs

**Terminal 2 - Start Dashboard:**
```bash
streamlit run dashboard/app.py
```
✓ Dashboard running: http://localhost:8501

## 📊 What You Get

| Component | URL | Purpose |
|-----------|-----|---------|
| API Server | http://localhost:8000 | REST endpoints |
| API Docs | http://localhost:8000/docs | Interactive documentation |
| Dashboard | http://localhost:8501 | Visualization & analytics |

## 🔌 Example API Calls

### Predict Risk
```bash
curl -X POST http://localhost:8000/predict-risk \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, 0.3, 0.2, 0.4, 0.1, 0.6, 0.2, 0.3, 0.4, 0.5],
    "employee_id": "EMP123"
  }'
```

### Get Forecast
```bash
curl http://localhost:8000/forecast?employee_id=EMP123&metric=productivity&periods=7
```

### Get Recommendations
```bash
curl -X POST http://localhost:8000/rl-recommendation \
  -H "Content-Type: application/json" \
  -d '{
    "workload_score": 0.65,
    "delay_risk": 0.35,
    "burnout_risk": 0.25
  }'
```

## 🎯 Next Steps

1. **Explore the Dashboard**
   - View key metrics
   - Check risk distributions
   - Analyze employee clusters

2. **Use the API**
   - Make predictions
   - Generate forecasts
   - Get recommendations

3. **Train Models** (Optional)
   ```bash
   python quickstart.py
   # Select option 3: Train Models
   ```

4. **Integrate Data**
   - Connect to your Supabase
   - Load your work metrics
   - Run predictions on real data

## 📁 Key Files

| File | Purpose |
|------|---------|
| `api/main.py` | FastAPI backend |
| `dashboard/app.py` | Streamlit frontend |
| `config/supabase_client.py` | Database configuration |
| `ml_models/train_models.py` | ML model training |
| `data_pipeline/preprocessing.py` | Data processing |
| `requirements.txt` | Python dependencies |
| `README.md` | Full documentation |

## 🔧 Troubleshooting

### API Won't Start
```bash
# Check if port 8000 is available
netstat -ano | findstr :8000  # Windows
lsof -i :8000  # macOS/Linux

# Change port in api/main.py if needed
```

### Dashboard Connection Error
- Ensure API is running first
- Check API is accessible at http://localhost:8000

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Database Connection Error
- Verify SUPABASE_URL and SUPABASE_KEY are correct
- Check internet connection
- Ensure Supabase project is active

## 🚀 Advanced Options

### Use Interactive Menu
```bash
python quickstart.py
```

### Docker Deployment
```bash
docker-compose up
```

### Custom Configuration
Edit `config/config.py` for advanced settings

## 📚 Documentation

- **Full Guide**: See [README.md](README.md)
- **API Docs**: http://localhost:8000/docs
- **Architecture**: See project structure in README.md

## ✨ Features at a Glance

✅ Real-time risk prediction
✅ Performance forecasting
✅ Employee clustering
✅ Reinforcement learning recommendations
✅ Generative AI reports
✅ Interactive dashboard
✅ REST API

## 🤝 Need Help?

1. Check error messages
2. Review [README.md](README.md)
3. Look at API documentation at `/docs`
4. Check logs for details

## 🎉 You're All Set!

Your Enterprise AI System is now running. Start exploring and building intelligence into your workforce management!

---

**Pro Tips:**
- Use API docs at `/docs` to explore all endpoints
- Dashboard refreshes every 5 seconds with latest data
- Models improve with more data - train regularly
- Monitor logs for insights

**Next Module to Explore:**
→ Try the Employee Lookup feature in the dashboard
→ Test the RL recommendations API
→ Review generated reports for insights
