import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from prophet import Prophet
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from model.preprocess import preprocess_data
from model.utils import split_data, evaluate


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "PJME_hourly.csv")


MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

def save_model_and_metrics(model, model_name, metrics):
    """Save model and metrics with versioning."""
    model_subdir = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_subdir, exist_ok=True)

    
    existing_versions = [
        f for f in os.listdir(model_subdir)
        if f.startswith(model_name) and f.endswith(".pkl")
    ]
    version = len(existing_versions) + 1  
    
    model_filename = f"{model_name}_v{version}.pkl"
    metric_filename = f"metrics_v{version}.json"

    model_path = os.path.join(model_subdir, model_filename)
    metric_path = os.path.join(model_subdir, metric_filename)

   
    try:
        joblib.dump(model, model_path)
    except Exception as e:
        print(f"[ERROR] Failed to save {model_name}: {e}")
        return None, None

    
    metrics["version"] = version
    metrics["timestamp"] = datetime.now().isoformat()
    pd.DataFrame([metrics]).to_json(metric_path, orient="records", indent=2)

    print(f"[INFO] {model_name} v{version} saved → {model_path}")
    print(f"[INFO] Metrics saved → {metric_path}")
    return model_path, metric_path


# ---------------- PROPHET ----------------
def train_and_save(file_path):
    print("Loading and preprocessing data for Prophet...")
    df = preprocess_data(file_path)

    
    df_daily = df.groupby(df['ds'].dt.date).agg({'y': 'sum'}).reset_index()
    df_daily.rename(columns={'ds': 'date', 'y': 'y'}, inplace=True)
    df_daily['ds'] = pd.to_datetime(df_daily['date'])
    df_daily.drop(columns=['date'], inplace=True)

    
    df_daily['lag1'] = df_daily['y'].shift(1)
    df_daily['lag2'] = df_daily['y'].shift(2)
    df_daily['rolling_mean_7'] = df_daily['y'].rolling(window=7).mean()
    df_daily['weekday'] = df_daily['ds'].dt.weekday
    df_daily['is_weekend'] = df_daily['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df_daily.dropna(inplace=True)

    train_df, test_df = split_data(df_daily, train_size=0.8)
    
    
    test_df['baseline_prev_day'] = test_df['lag1']
    baseline_mae, baseline_rmse, baseline_r2 = evaluate(
        test_df['y'], test_df['baseline_prev_day'], "Baseline Model")
    print(f"[INFO] Baseline Model - MAE: {baseline_mae:.4f}, RMSE: {baseline_rmse:.4f}, R2: {baseline_r2:.4f}") 
    model = Prophet(
        daily_seasonality=False,
        yearly_seasonality=True,
        changepoint_prior_scale=0.5,
        seasonality_mode='multiplicative'
    )
    for col in ['lag1', 'lag2', 'rolling_mean_7', 'weekday', 'is_weekend']:
        model.add_regressor(col)

    model.fit(train_df[['ds', 'y', 'lag1', 'lag2', 'rolling_mean_7', 'weekday', 'is_weekend']])
    future = test_df[['ds', 'lag1', 'lag2', 'rolling_mean_7', 'weekday', 'is_weekend']].copy()
    forecast = model.predict(future)

    mae, rmse, r2 = evaluate(test_df['y'], forecast['yhat'], "Prophet")
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "baseline_r2": baseline_r2,
        "timestamp": datetime.now().isoformat()
    }

    model_path, metric_path = save_model_and_metrics(model, "prophet", metrics)
    version = metrics["version"]

    print("[INFO] Prophet model trained and saved.")
    return model, metrics, version


# ---------------- XGBOOST ----------------
def train_xgboost(file_path):
    print("Loading and preprocessing data for XGBoost...")
    df = preprocess_data(file_path)
    train_df, test_df = split_data(df)

    features = ["lag1", "lag2", "rolling_mean_24", "hour", "month", "weekday", "is_weekend"]
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    print("[INFO] Training XGBoost model...")
    model.fit(train_df[features], train_df["y"])
    train_preds = model.predict(train_df[features])
    test_preds = model.predict(test_df[features])

    train_mae, train_rmse, train_r2 = evaluate(train_df["y"], train_preds, "Train")
    test_mae, test_rmse, test_r2 = evaluate(test_df["y"], test_preds, "Test")

    if train_r2 > 0.98 and test_r2 < 0.9:
        print("[WARNING] Model is likely OVERFITTING")
    elif train_r2 < 0.9 and test_r2 < 0.9:
        print("[WARNING] Model is likely UNDERFITTING")
    else:
        print("[INFO] Model is well-fitted")

    metrics = {
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "timestamp": datetime.now().isoformat()
    }

    model_path, metric_path = save_model_and_metrics(model, "xgboost", metrics)
    version = metrics["version"]

    print("[INFO] XGBoost model trained and saved.")
    return model, metrics, version


# ---------------- CATBOOST ----------------
def train_catboost(file_path):
    print("Loading and preprocessing data for CatBoost...")
    df = preprocess_data(file_path)
    train_df, test_df = split_data(df)

    features = ["lag1", "lag2", "rolling_mean_24", "hour", "month", "weekday", "is_weekend"]

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )

    print("[INFO] Training CatBoost model...")
    model.fit(train_df[features], train_df["y"])

    train_preds = model.predict(train_df[features])
    test_preds = model.predict(test_df[features])

    train_mae, train_rmse, train_r2 = evaluate(train_df["y"], train_preds, "Train")
    test_mae, test_rmse, test_r2 = evaluate(test_df["y"], test_preds, "Test")

    if train_r2 > 0.98 and test_r2 < 0.9:
        print("[WARNING] Model is likely OVERFITTING")
    elif train_r2 < 0.9 and test_r2 < 0.9:
        print("[WARNING] Model is likely UNDERFITTING")
    else:
        print("[INFO] Model is well-fitted")

    metrics = {
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "timestamp": datetime.now().isoformat()
    }

    model_path, metric_path = save_model_and_metrics(model, "catboost", metrics)
    version = metrics["version"]

    print("[INFO] CatBoost model trained and saved.")
    return model, metrics, version


