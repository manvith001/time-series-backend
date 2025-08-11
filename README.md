# Time Series Forecasting Backend (FastAPI + Prophet)

Backend service for the **Time Series Forecasting WebApp**.  
Built with **FastAPI**, powered by **Prophet**, and designed to serve forecasts via REST APIs.

---

##  Features
- Train forecasting model on **hourly energy consumption** dataset
- Predict for **next N hours/days**
- Saves & loads model using **Joblib**
- Returns **MAE** & **RMSE** metrics
- Docker-ready, deployed on **AWS ECS**

---

## 🛠 Tech Stack
- **Language:** Python 3.11
- **Libraries:** Prophet, Pandas, Joblib, FastAPI, Uvicorn
- **Deployment:** AWS ECS, Docker
- **Version Control:** GitHub

---
ime-series-backend/
├── api/ # FastAPI app & API routes
│ ├── main.py # Main entrypoint for FastAPI
│ ├── train.py # Training endpoint logic
│ ├── predict.py # Prediction endpoint logic
│ └── utils.py # Helper functions
│
├── model/ # Model artifacts & scripts
│ ├── train_and_evaluate.py # Train + evaluate Prophet model
│ ├── preprocess.py # Data preprocessing & feature engineering
│ ├── saved_model.pkl # Trained model file (Joblib)
│ └── model_metrics/ # Folder for metrics
│ └── metrics.json # MAE, RMSE values
│
├── data/ # Dataset files
│ ├── PJME_hourly.csv # Main dataset
│ └── raw/ # Optional raw datasets
│
├── requirements.txt # Python dependencies
├── Dockerfile # Docker build instructions
├── README.md # Backend documentation
└── venv/ # Virtual environment (not included in repo)
## How to Run 


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
Docker hub link :https://hub.docker.com/r/manvithbhoomi/time_series_model_backend

# Pull Docker image
docker pull manvithbhoomi/time_series_model_backend

Run the container:
docker run -d -p 8000:8000 manvithbhoomi/time_series_model_backend
```
