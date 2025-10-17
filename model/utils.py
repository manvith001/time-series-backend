import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from fastapi import HTTPException
from fastapi.responses import JSONResponse



def split_data(df, train_size=0.8):
    
    
    length = len(df)
    train_size_count = round(length * train_size)
    
    train_df = df[:train_size_count]
    test_df = df[train_size_count:]
    
    return train_df, test_df



def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{name}] MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
    
    return mae, rmse, r2


import logging
from fastapi import HTTPException
from fastapi.responses import JSONResponse

def handle_exception(exc: Exception, context: str = "") -> JSONResponse:
    
    if isinstance(exc, HTTPException):
        logging.warning(f"{context} HTTPException: {exc.detail}")
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
    
    elif isinstance(exc, FileNotFoundError):
        logging.error(f"{context} FileNotFoundError: {exc}")
        return JSONResponse(status_code=404, content={"error": "File not found."})
    
    elif isinstance(exc, ValueError):
        logging.error(f"{context} ValueError: {exc}")
        return JSONResponse(status_code=400, content={"error": str(exc)})
    
    else:
        logging.error(f"{context} Unexpected error: {exc}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

