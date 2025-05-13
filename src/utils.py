from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader

def create_tft_dataset(df, encoder_len=96, prediction_len=2880, batch_size=64, val_split=0.2):
    """
    Creates train and validation TimeSeriesDataSet and corresponding DataLoaders.
    """
    df = df.copy()

    df["time_idx"] = range(len(df))
    df["group_id"] = "DE"
    df["target"] = df["actual_load"]

    categorical_columns = ["month", "hour", "is_daylight", "weekend_flag", "holiday_flag"]
    continuous_columns = [
        "solar_capacity", "solar_generation",
        "wind_capacity", "wind_generation",
        "wind_offshore_capacity", "wind_offshore_generation",
        "wind_onshore_capacity", "wind_onshore_generation",
        "temperature"
    ]

    # Define training cutoff
    training_cutoff = int(df["time_idx"].max() * (1 - val_split))

    # Create training TimeSeriesDataSet
    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=encoder_len,
        max_prediction_length=prediction_len,
        time_varying_known_reals=["time_idx"] + categorical_columns,
        time_varying_unknown_reals=["target"] + continuous_columns,
        static_categoricals=["group_id"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Create validation TimeSeriesDataSet using the training dataset's config
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

    # Dataloaders
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    return training, train_dataloader, val_dataloader
