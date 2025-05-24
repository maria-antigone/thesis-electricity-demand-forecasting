# data_processing_tft.py

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

def add_time_idx_and_series_id(df: pd.DataFrame):
    df = df.copy()
    df = df.sort_values("utc_timestamp")
    df["time_idx"] = ((df["utc_timestamp"] - df["utc_timestamp"].min()).dt.total_seconds() // (15 * 60)).astype(int)
    df["series_id"] = 0
    return df

def split_dataset(df: pd.DataFrame, config: dict):
    total_size = len(df)
    train_end = int(0.8 * total_size)
    val_end = int(0.9 * total_size)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test

def create_tft_dataset(df: pd.DataFrame, config: dict, target_col: str):
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["series_id"],
        max_encoder_length=config["input_window"],
        max_prediction_length=config["output_horizon"],
        time_varying_known_reals=[
            "hour", "is_daylight", "month", "year", "weekday", "weekend_flag", "holiday_flag", "temperature",
            "solar_capacity", "solar_generation", "wind_capacity", "wind_generation", "wind_offshore_capacity",
            "wind_offshore_generation", "wind_onshore_capacity", "wind_onshore_generation"
        ],
        time_varying_unknown_reals=[target_col],
        static_categoricals=[],
        static_reals=[],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

def create_dataloaders(train_df, val_df, test_df, config, target_col):
    training = create_tft_dataset(train_df, config, target_col)
    validation = create_tft_dataset(val_df, config, target_col)
    testing = create_tft_dataset(test_df, config, target_col)

    train_loader = training.to_dataloader(train=True, batch_size=config["batch_size"], num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=config["batch_size"], num_workers=0)
    test_loader = testing.to_dataloader(train=False, batch_size=config["batch_size"], num_workers=0)

    return train_loader, val_loader, test_loader, training
