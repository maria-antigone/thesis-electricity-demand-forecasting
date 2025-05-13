# model building functions

import torch
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

from torch.utils.data import DataLoader
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

df_TFT = pd.read_csv('../data/processed/merged_dataset_cleaned.csv', sep = ';', index_col = 'utc_timestamp', parse_dates = True)
df_TFT = df_TFT.loc['2019-01-01':'2019-12-31']

print(df_TFT.head())
print(df_TFT.tail())
print(df_TFT.shape)
print(df_TFT.dtypes)
print(df_TFT.isnull().sum())

# Drop forecasting, radiation, cet timestamp, and all profile columns
cols_to_drop = [
    "cet_cest_timestamp",
    "DE_load_forecast_entsoe_transparency",
    "DE_radiation_direct_horizontal",
    "DE_radiation_diffuse_horizontal",
    "DE_solar_profile",
    "DE_wind_profile",
    "DE_wind_offshore_profile",
    "DE_wind_onshore_profile"
]

df_TFT = df_TFT.drop(columns=cols_to_drop)

df_TFT.shape

print(df_TFT.dtypes)
df_TFT.isnull().sum()

df_TFT=df_TFT.copy()
df_TFT["time_idx"] = range(len(df_TFT)) # sequential time index
df_TFT["group_id"] = "DE" # only one group
df_TFT["target"] = df_TFT["DE_load_actual_entsoe_transparency"] # target variable
print(df_TFT.head())

max_encoder_length = 96      # past 1 day (15-min intervals)
max_prediction_length = 2880 # future 30 days

categorical_columns = ["month", "hour", "is_daylight"]
continuous_columns = [
    "DE_solar_capacity", "DE_solar_generation_actual",
    "DE_wind_capacity", "DE_wind_generation_actual",
    "DE_wind_offshore_capacity", "DE_wind_offshore_generation_actual",
    "DE_wind_onshore_capacity", "DE_wind_onshore_generation_actual",
    "DE_temperature",
]

training_dataset = TimeSeriesDataSet(
    df_TFT,
    time_idx="time_idx",
    target="target",
    group_ids=["group_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=["time_idx"] + categorical_columns,
    time_varying_unknown_reals=["target"] + continuous_columns,
    static_categoricals=["group_id"],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Wrap in data loader
batch_size = 64  # can tune later
train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)

# intialize the model
tft = TemporalFusionTransformer.from_dataset(
    training_dataset,
    learning_rate=0.03,
    hidden_size=16, 
    attention_head_size=1,
    dropout=0.1,
    loss=MAE(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# set up trainer and train
early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min")

trainer = Trainer(
    max_epochs=30,
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
    accelerator="auto"
)

# Now call fit
trainer.fit(model=tft, train_dataloaders=train_dataloader)

