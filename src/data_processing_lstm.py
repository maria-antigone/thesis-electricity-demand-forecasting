import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import yaml

def load_config(horizon="short"):
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)[horizon]

def prepare_lstm_data(df, config):
    target_col = config.get("target_column", "actual_load")
    features = []

    # Step 1: Apply LSTM lag features
    for lag in config.get("lags", []):
        df[f"lag_{lag}"] = df[target_col].shift(lag)
        features.append(f"lag_{lag}")

    df = df.dropna()

    # Step 2: Train/val/test split (time-based)
    total_len = len(df)
    train_end = int(total_len * 0.8)
    val_end = int(total_len * 0.9)

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    # Step 3: Scaling without leakage
    scaler_type = config.get("scaler", "MinMax")
    scaler_cls = MinMaxScaler if scaler_type == "MinMax" else StandardScaler

    scaler = scaler_cls()
    df_train[features] = scaler.fit_transform(df_train[features])
    df_val[features] = scaler.transform(df_val[features])
    df_test[features] = scaler.transform(df_test[features])

    # Step 4: Sequence construction
    X_train, y_train = create_sequences(df_train, features, target_col, config)
    X_val, y_val = create_sequences(df_val, features, target_col, config)
    X_test, y_test = create_sequences(df_test, features, target_col, config)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

def create_sequences(df, features, target, config):
    input_window = config["input_window"]
    output_horizon = config["output_horizon"]

    X, y = [], []
    for i in range(len(df) - input_window - output_horizon):
        X.append(df[features].iloc[i:i+input_window].values)
        y.append(df[target].iloc[i+input_window:i+input_window+output_horizon].values)
    return np.array(X), np.array(y)
