import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def split_data(df, train_size=0.8):
    
    
    length = len(df)
    train_size_count = round(length * train_size)
    
    train_df = df[:train_size_count]
    test_df = df[train_size_count:]
    
    return train_df, test_df



def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return mae, rmse
