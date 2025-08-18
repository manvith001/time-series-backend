from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from fastapi import status
import math
import joblib
import os
from fastapi.encoders import jsonable_encoder
from model.train_and_evaluate import train_and_save
import pandas as pd
from model.preprocess import preprocess_data
import logging
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "PJME_hourly.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "prophet_model.pkl")
LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler()
    ]
)



model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None






@app.post("/train")
async def train_model():
    logging.info("API /train called")
    if not os.path.exists(DATA_PATH):
        return JSONResponse(
            status_code=404, content={"message": "Data file not found."}
        )
    try:
        global model
        model = train_and_save() 
        return JSONResponse(
            status_code=200, content={"message": "Model trained and saved successfully."}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": f"Error during training: {str(e)}"}
        )
    


@app.post("/predict")
def predict_endpoint(periods: int = Query(24, description="Number of hours to forecast")):
    logging.info("API /predict called with periods=%s", periods)
    
    if periods <= 0:
        return JSONResponse(status_code=400, content={"error": "Periods must be a positive integer"})
    global model
    if model is None:
            if not os.path.exists(MODEL_PATH):
                print("No pre-trained model found. Training a new one...")
                model = train_and_save()
            else:
                model = joblib.load(MODEL_PATH)
    try:
        
        history = preprocess_data(DATA_PATH)
        last_row = history.iloc[-1]
        before_last = history.iloc[-2]

        
        
        
        future = model.make_future_dataframe(periods=periods, freq="H")

        
        
        future["lag1"] = last_row["y"]
        future["lag2"] = before_last["y"]
        future["rolling_mean_24"] = history["rolling_mean_24"].iloc[-1]
        future["hour"] = future["ds"].dt.hour
        future["month"] = future["ds"].dt.month
        future["weekday"] = future["ds"].dt.weekday
        future["is_weekend"] = future["weekday"].apply(lambda x: 1 if x >= 5 else 0)

       
        forecast = model.predict(future)
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
       
        response = jsonable_encoder({
                "message": f"Successfully generated forecast for next {periods} hours.",
                "Data": result.to_dict("records")})
        
        return JSONResponse(status_code=status.HTTP_200_OK, content=response)
    
    
    except Exception as e:
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Prediction failed: {str(e)}"}
        )


