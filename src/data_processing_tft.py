import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader
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

def add_time_idx_and_series_id(df: pd.DataFrame):
    df = df.copy()
    df = df.sort_values("utc_timestamp")
    df["time_idx"] = ((df["utc_timestamp"] - df["utc_timestamp"].min()).dt.total_seconds() // (15 * 60)).astype(int)
    df["series_id"] = 0
    return df

def split_dataset(df: pd.DataFrame, config: dict):
    total_size = len(df)
    train_split_ratio = config.get("train_split", 0.8)
    val_split_ratio = config.get("val_split", 0.1)

    train_end = int(train_split_ratio * total_size)
    val_end = int((train_split_ratio + val_split_ratio) * total_size)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test

def create_tft_dataset(df: pd.DataFrame, config: dict, target_col: str):
    time_varying_known_reals_from_config = config.get("exogenous_features_tft", [])

    time_varying_known_reals_present_in_df = [
        col for col in time_varying_known_reals_from_config if col in df.columns
    ]

    missing_cols = set(time_varying_known_reals_from_config) - set(time_varying_known_reals_present_in_df)
    if missing_cols:
        print(f"Warning in create_tft_dataset: The following exogenous_features from config were not found in the DataFrame and will be omitted: {missing_cols}", flush=True)

    print(f"TFT will be trained with the following {len(time_varying_known_reals_present_in_df)} time-varying known features: {time_varying_known_reals_present_in_df}", flush=True)

    default_min_enc_calc = config['input_window'] // 7
    min_enc_len_default = max(1, default_min_enc_calc if default_min_enc_calc > 0 else 24)
    
    min_enc_len = config.get('min_input_window', min_enc_len_default)
    if min_enc_len > config['input_window']:
        min_enc_len = config['input_window']
    
    print(f"TimeSeriesDataSet using: max_encoder_length={config['input_window']}, min_encoder_length={min_enc_len}", flush=True)

    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["series_id"],
        max_encoder_length=config["input_window"],
        min_encoder_length=min_enc_len, 
        max_prediction_length=config["output_horizon"],
        time_varying_known_reals=time_varying_known_reals_present_in_df,
        time_varying_unknown_reals=[target_col], # The target is the only unknown future variable
        static_categoricals=[],
        static_reals=[],    
        add_relative_time_idx=True,
        add_target_scales=True, 
        add_encoder_length=True,
        allow_missing_timesteps=config.get("allow_missing_timesteps_tft", True) 
    )

def create_dataloaders(train_df, val_df, test_df, config, target_col):
    print(f"Inside create_dataloaders: train_df length: {len(train_df)}, val_df length: {len(val_df)}, test_df length: {len(test_df)}", flush=True)
    print(f"Config for DataLoaders: input_window (max_encoder_length): {config['input_window']}, output_horizon (max_prediction_length): {config['output_horizon']}", flush=True)
    
    min_rows_needed = config['input_window'] + config['output_horizon']
    if len(train_df) < min_rows_needed :
        print(f"CRITICAL WARNING: train_df (length {len(train_df)}) is shorter than min_rows_needed ({min_rows_needed}). No sequences can be created for training.", flush=True)
    if len(val_df) < min_rows_needed:
        print(f"CRITICAL WARNING: val_df (length {len(val_df)}) is shorter than min_rows_needed ({min_rows_needed}). No sequences can be created for validation.", flush=True)
    
    training_dataset = create_tft_dataset(train_df, config, target_col)
    validation_dataset = create_tft_dataset(val_df, config, target_col)
    testing_dataset = create_tft_dataset(test_df, config, target_col) 

    num_workers_val = config.get("num_workers", 0) 
    print(f"DataLoaders will use num_workers: {num_workers_val}", flush=True)
    
    persistent_flag = True if num_workers_val > 0 else False

    train_loader = training_dataset.to_dataloader(
        train=True, 
        batch_size=config["batch_size"], 
        num_workers=num_workers_val,       
        persistent_workers=persistent_flag 
    )
    val_loader = validation_dataset.to_dataloader(
        train=False, 
        batch_size=config["batch_size"], 
        num_workers=num_workers_val,     
        persistent_workers=persistent_flag  
    )
    test_loader = testing_dataset.to_dataloader(
        train=False, 
        batch_size=config["batch_size"], 
        num_workers=num_workers_val,         
        persistent_workers=persistent_flag 
    )

    return train_loader, val_loader, test_loader, training_dataset