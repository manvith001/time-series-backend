import os
import logging
from uuid import uuid4

import pandas as pd
import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder

from model.train_and_evaluate import train_and_save, train_xgboost, train_catboost
from model.preprocess import preprocess_data
from model.utils import handle_exception




BASE_DIR = os.path.dirname(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "user_uploads")
MODEL_DIR = os.path.join(BASE_DIR, "model")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


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
MODEL_DIR = os.path.join(BASE_DIR, "model")  
LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "user_uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)
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


@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        
        if file is None:
            logging.warning("No file uploaded.")
            raise HTTPException(status_code=400, detail="No file uploaded.")

        
        if not file.filename.endswith(".csv"):
            logging.warning(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

      
        unique_filename = f"{uuid4().hex}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

       
        try:
            contents = await file.read()
            if not contents:
                logging.warning(f"Uploaded file is empty: {file.filename}")
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
            with open(file_path, "wb") as f:
                f.write(contents)

            logging.info(f"File uploaded successfully: {unique_filename}")
            return JSONResponse(
                status_code=200,
                content={"message": "File uploaded successfully", "file_name": unique_filename}
            )
        except Exception as file_error:
            logging.error(f"Error saving file {file.filename}: {file_error}")
            raise HTTPException(status_code=500, detail="Error saving the file.")

    except Exception as e:
        logging.error(f"Unexpected error during file upload: {e}")
        return handle_exception(e, context=f"File upload: {file.filename if file else 'No file'}")


@app.post("/train")
async def train_model(
    model_name: str = Query(..., description="prophet, xgboost, catboost"),
    file_name: str = Query(..., description="Uploaded CSV filename")
):
    try:
        
        file_path = os.path.join(UPLOAD_DIR, file_name)
        if not os.path.exists(file_path):
            logging.warning(f"CSV file not found: {file_name}")
            raise HTTPException(status_code=404, detail="CSV file not found.")

       
        model_name_lower = model_name.lower()
        if model_name_lower == "prophet":
            logging.info(f"Training Prophet model with file: {file_name}")
            model, metrics, version = train_and_save(file_path)
        elif model_name_lower == "xgboost":
            logging.info(f"Training XGBoost model with file: {file_name}")
            model, metrics, version = train_xgboost(file_path)
        elif model_name_lower == "catboost":
            logging.info(f"Training CatBoost model with file: {file_name}")
            model, metrics, version = train_catboost(file_path)
        else:
            logging.warning(f"Invalid model name: {model_name}")
            raise HTTPException(
                status_code=400,
                detail="Invalid model_name. Choose prophet, xgboost, or catboost."
            )

       
        response = {
            "message": f"{model_name} model trained successfully.",
            "model_name": model_name_lower,
            "version": version,
            "file_name": file_name,
            "metrics": metrics
        }

        logging.info(f"Model trained: {model_name}, version: {version}")
        return JSONResponse(status_code=200, content=response)

    except Exception as e:
        logging.error(f"Unexpected error during model training: {e}")
        return handle_exception(e, context=f"Training: {model_name} / {file_name}")
    
    
    
    
@app.post("/predict")
async def predict_endpoint(
    model_name: str = Query(..., description="prophet, xgboost, catboost"),
    version: str = Query(..., description="Model version timestamp returned by /train"),
    file_name: str = Query(..., description="CSV used to train the model"),
    periods: int = Query(24, description="Forecast horizon in hours")
):
    try:
        
        file_path = os.path.join(UPLOAD_DIR, file_name)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="CSV file not found.")

       
        model_path = os.path.join(MODEL_DIR, model_name.lower(), f"{model_name.lower()}_v{version}.pkl")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model version not found.")

        model = joblib.load(model_path)

        
        history = preprocess_data(file_path)

        # prophet
        if model_name.lower() == "prophet":
            last_row = history.iloc[-1]
            before_last = history.iloc[-2]
            future = model.make_future_dataframe(periods=periods, freq="H")
            
            future["lag1"] = last_row["y"]
            future["lag2"] = before_last["y"]
            daily_rolling_mean_7 = (
                    history.resample("D", on="ds")["y"].sum().rolling(7).mean().iloc[-1])
            future["rolling_mean_7"] = daily_rolling_mean_7
            future["hour"] = future["ds"].dt.hour
            future["month"] = future["ds"].dt.month
            future["weekday"] = future["ds"].dt.weekday
            future["is_weekend"] = future["weekday"].apply(lambda x: 1 if x >= 5 else 0)
            forecast = model.predict(future)
            result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)

        else:
            # For XGBoost / CatBoost
            
            features = ["lag1", "lag2", "rolling_mean_24", "hour", "month", "weekday", "is_weekend"]
            test_df = history.copy().tail(periods)
            result_values = model.predict(test_df[features])
            result = test_df[["ds"]].copy()
            result["yhat"] = result_values

        response={
                "message": f"Prediction successful for next {periods} periods",
                "model_name": model_name,
                "version": version,
                "file_name": file_name,
                "Data": result.to_dict("records")
            }
        return JSONResponse(status_code=200, content=jsonable_encoder(response))

        

   
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return handle_exception(e, context=f"Training: {model_name} / {file_name}")



