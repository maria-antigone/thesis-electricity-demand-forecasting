import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import yaml
import os

def encode_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    if 'weekday' in df.columns:
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7.0)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7.0)
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    return df

def load_config(horizon="short"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)[horizon]

def prepare_lstm_data(df_original: pd.DataFrame, config: dict):
    df = df_original.copy()

    target_col = config.get("target_column", "actual_load")
    
    features_for_x = []

    # Step 1: Apply LSTM lag features
    for lag_val in config.get("lags", []):
        df[f"lag_{lag_val}"] = df[target_col].shift(lag_val)
        features_for_x.append(f"lag_{lag_val}")

    exogenous_feature_names = config.get("exogenous_features", [])

    if exogenous_feature_names:
        print(f"Attempting to add exogenous features: {exogenous_feature_names}", flush=True)
        for ex_feat in exogenous_feature_names:
            if ex_feat in df.columns:
                if ex_feat not in features_for_x:
                    features_for_x.append(ex_feat)
            else:
                print(f"Warning: Exogenous feature '{ex_feat}' not found in DataFrame columns. Skipping.", flush=True)
    else:
        print("No exogenous features provided in config. Only lags will be used for LSTM input.", flush=True)

    columns_to_check_for_nan = list(set(features_for_x + [target_col]))
    df = df.dropna(subset=columns_to_check_for_nan)

    # Step 2: Train/val/test split (time-based)
    total_len = len(df)
    train_split_ratio = config.get("train_split", 0.8) 
    val_split_ratio = config.get("val_split", 0.1)

    train_end_idx = int(total_len * train_split_ratio)
    val_end_idx = int(total_len * (train_split_ratio + val_split_ratio))


    df_train = df.iloc[:train_end_idx].copy()
    df_val = df.iloc[train_end_idx:val_end_idx].copy()
    df_test = df.iloc[val_end_idx:].copy()

    print(f"LSTM data splits (after lag drop): Train {len(df_train)}, Val {len(df_val)}, Test {len(df_test)}", flush=True)

    # Step 3: Scaling 
    # Scaler for input features (lags and exogenous features)
    feature_scaler_config_key = config.get("feature_scaler", "MinMax")
    if feature_scaler_config_key == "Standard":
        feature_scaler_cls = StandardScaler
    else: # defaulting to minmax
        feature_scaler_cls = MinMaxScaler
    feature_scaler = feature_scaler_cls()

    # Fit scaler ONLY on training data features and transform all sets
    if features_for_x: # Check if there are any features to scale
        df_train[features_for_x] = feature_scaler.fit_transform(df_train[features_for_x])
        df_val[features_for_x] = feature_scaler.transform(df_val[features_for_x])
        df_test[features_for_x] = feature_scaler.transform(df_test[features_for_x])
    else:
        print("Warning: No input features (lags) or exogenous features provided for X. LSTM will effectively only see sequence index if any.", flush=True)

    # Target Scaling
    target_scaler_config_key = config.get("target_scaler", "MinMax")
    if target_scaler_config_key == "Standard":
        target_scaler_cls = StandardScaler
    else:  # defaulting to minmax
        target_scaler_cls = MinMaxScaler
    target_scaler = target_scaler_cls()

    df_train[target_col] = target_scaler.fit_transform(df_train[[target_col]])
    df_val[target_col] = target_scaler.transform(df_val[[target_col]])
    df_test[target_col] = target_scaler.transform(df_test[[target_col]])
    
    print(f"Total number of input features for LSTM (X): {len(features_for_x)}", flush=True)

    # Step 4: Sequence construction (will use scaled features for X and scaled target for Y)
    X_train, y_train = create_sequences(df_train, features_for_x, target_col, config)
    X_val, y_val = create_sequences(df_val, features_for_x, target_col, config)
    X_test, y_test = create_sequences(df_test, features_for_x, target_col, config)

    # Return target_scaler also!
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler, target_scaler

def create_sequences(df_processed, list_of_x_input_features, target_column_name, config):
    input_window = config["input_window"]
    output_horizon = config["output_horizon"]

    X, y = [], []
    
    if list_of_x_input_features:
        x_data_values = df_processed[list_of_x_input_features].values
    else:
        print("Warning: 'list_of_x_input_features' is empty in create_sequences. This might lead to errors or unexpected behavior.", flush=True)
        pass

    y_data_values = df_processed[[target_column_name]].values

    # Corrected loop range to maximize data usage
    for i in range(len(df_processed) - input_window - output_horizon + 1):
        if list_of_x_input_features:
             X.append(x_data_values[i : i + input_window, :])

        y.append(y_data_values[i + input_window : i + input_window + output_horizon].flatten()) 
    
    X_arr = np.array(X)
    y_arr = np.array(y)

    return X_arr, y_arr