# Time Series Forecasting Backend (FastAPI + Prophet)

Backend service powering the Time Series Forecasting WebApp, built using FastAPI and enhanced with multiple time series and machine learning models — including Prophet, XGBoost, and CatBoost.

This backend provides APIs to upload data, train models, version them automatically, and generate future forecasts.
---

##  Features


**Multi-Model Support** – Train and forecast using Prophet, XGBoost, or CatBoost  
**CSV Upload API** – Upload your own dataset for on-demand training  
**Automated Model Versioning** – Auto-saves model versions (`model_v1.pkl`, `model_v2.pkl`, …)  
**Centralized Exception Handling** – Unified error responses through utilities  
**Logging System** – Tracks all events and errors in `logs/app.log`  
**Metrics Storage** – Tracks MAE, RMSE, and R² for every model version  
**Docker Support** – Build and run anywhere seamlessly  
**Modular Architecture** – Clean separation of routes, logic, and storage  
---



## Tech Stack

| Category | Technology |
|-----------|-------------|
| **Language** | Python 3.11 |
| **Framework** | FastAPI |
| **Forecasting Models** | Prophet, XGBoost, CatBoost |
| **Storage** | File-based (CSV, PKL, JSON) |
| **Deployment** | Docker, AWS ECS |
| **Logging** | Python logging with rotation |
| **Version Control** | GitHub |

---
```
time-series-backend/
├── api/ # FastAPI app & API routes
│ ├── main.py # Main entrypoint for FastAPI
│
│ 
├── model/ # Model artifacts & scripts
│ ├── train_and_evaluate.py # Train + evaluate Prophet model
│ ├── preprocess.py # Data preprocessing & feature engineering
│ ├── saved_model.pkl # Trained model file (Joblib)
  └── utils.py # Helper functions
│
│ └── model_metrics/ # Folder for metrics
│         └── metrics.json # MAE, RMSE values
│
├── data/ # Dataset files
│ ├── PJME_hourly.csv # Main dataset
│ └── raw/ # Optional raw datasets
│____logs  #for checking the logs 
├── requirements.txt # Python dependencies
├── Dockerfile # Docker build instructions
├── README.md # Backend documentation
└── venv/ # Virtual environment (not included in repo)
```

### **1. Run Locally**
```bash
# Clone repo
git clone https://github.com/manvith001/time-series-backend.git
cd time-series-backend

# Create virtual environment
python -m venv venv

# Activate venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn api.main:app --reload
Backend URL: http://127.0.0.1:8000
```

### **2. Run with Docker**
```
Docker hub link :https://hub.docker.com/r/manvithbhoomi/time_series_backend

# Pull Docker image
docker pull manvithbhoomi/time_series_backend:latest


Run the container:
docker run -d -p 80:80 manvithbhoomi/time_series_backend:latest
```
 Example Flow
1️ Upload a Dataset
POST /upload_csv
{
  "message": "File uploaded successfully",
  "file_name": "9c143a011ab74698a8126dbb50927fc1_PJME_hourly.csv"
}

2️ Train a Model
POST /train?model_name=prophet&file_name=9c143a011ab74698a8126dbb50927fc1_PJME_hourly.csv
{
  "message": "prophet model trained successfully",
  "version": 2,
  "metrics": { "mae": 12.34, "rmse": 18.91, "r2": 0.92 }
}

3️ Generate Forecast
POST /predict?model_name=prophet&version=2&periods=48
{
  "message": "Prediction successful for next 48 periods",
  "Data": [
    { "ds": "2024-01-01 00:00:00", "yhat": 1234.5 },
    ...
  ]
}


📊 Metrics Example

Each trained model saves metrics as JSON:

{
  "mae": 12.34,
  "rmse": 18.91,
  "r2": 0.92,
  "timestamp": "2025-10-18T12:32:00"
}

 Future Improvements

 Add LSTM / NeuralProphet deep learning model

 Add asynchronous training with background workers

 Integrate Redis caching for fast re-forecasts

 Enable Prometheus metrics & Grafana dashboards
