import pandas as pd


def preprocess_data(path):

    df = pd.read_csv(path)
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    df = df.dropna(subset=["ds"])

    df = df.drop_duplicates(subset=["ds"])

    df = df.set_index("ds").asfreq("h").reset_index()
    df["y"] = df["y"].interpolate(method="linear")

   
    df["lag1"] = df["y"].shift(1)
    df["lag2"] = df["y"].shift(24)

   

    df["rolling_mean_24"] = df["y"].rolling(window=24).mean()

    

    df["hour"] = df["ds"].dt.hour
    df["weekday"] = df["ds"].dt.weekday
    df["month"] = df["ds"].dt.month
    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)

   
    df.dropna(inplace=True)
    print("Preprocessing complete. Data shape:", df.shape)
    return df
