import os
import joblib
from prophet import Prophet
from model.preprocess import preprocess_data
from model.utils import split_data, evaluate
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "PJME_hourly.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "prophet_model.pkl")
METRIC_DIR = os.path.join(BASE_DIR, "model", "model_metrics")

os.makedirs(METRIC_DIR, exist_ok=True)
METRIC_PATH = os.path.join(METRIC_DIR, "metrics.json")

def train_and_save(force_retrain=False):
    # Skip retraining if model exists and retrain not forced
    if os.path.exists(MODEL_PATH) and not force_retrain:
        print("[INFO] Model already exists. Skipping retraining.")
        return joblib.load(MODEL_PATH)

    print("Loading and preprocessing data...")
    df = preprocess_data(DATA_PATH)

    # Split into train/test
    train_df, test_df = split_data(df)

    # Baseline model
    test_df["baseline_prev_hour"] = test_df['lag1']
    baseline_mae, baseline_rmse = evaluate(test_df["y"], test_df["baseline_prev_hour"], "Baseline Model")

    # Prophet model
    model = Prophet()
    
    model.add_regressor("lag1")
    model.add_regressor("lag2")
    model.add_regressor("rolling_mean_24")
    model.add_regressor("hour")
    model.add_regressor("month")
    model.add_regressor("weekday")
    model.add_regressor("is_weekend")

    model.fit(train_df[["ds", "y", "lag1", "lag2", "rolling_mean_24", "hour", "month", "weekday", "is_weekend"]])

    # Create future dataframe
    future = model.make_future_dataframe(periods=len(test_df), freq="H")
    last_row = train_df.iloc[-1]
    before_last = train_df.iloc[-2]
    future["lag1"] = last_row["y"]
    future["lag2"] = before_last["y"]
    future["rolling_mean_24"] = train_df["rolling_mean_24"].iloc[-1]
    future["hour"] = future["ds"].dt.hour
    future["month"] = future["ds"].dt.month
    future["weekday"] = future["ds"].dt.weekday
    future["is_weekend"] = future["weekday"].apply(lambda x: 1 if x >= 5 else 0)

    # Predict
    forecast = model.predict(future)
    prophet_preds = forecast.tail(len(test_df))["yhat"].values

    # Evaluate
    mae, rmse = evaluate(test_df["y"], prophet_preds, "Prophet")

    # Save model & metrics
    joblib.dump(model, MODEL_PATH)
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "timestamp": datetime.now().isoformat()
    }
    pd.DataFrame([metrics]).to_json(METRIC_PATH, orient="records", indent=2)

    print("[INFO] Model trained and saved.")
    return model
