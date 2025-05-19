import os
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet

def load_config(horizon="short"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)[horizon]

def get_callbacks(config, model_prefix="tft"):
    patience = config.get("early_stopping_patience", 5)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        verbose=True,
        mode="min"
    )

    checkpoint = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=f"{model_prefix}-{{epoch:02d}}-{{val_loss:.2f}}",
        monitor="val_loss",
        save_top_k=1,
        mode="min"
    )

    return [early_stop, checkpoint]

def log_epoch_metrics(epoch, logs):
    print(f"[Epoch {epoch}] ", end="", flush=True)
    for key, value in logs.items():
        print(f"{key}: {value:.4f} ", end="", flush=True)
    print(flush=True)

def create_tft_dataset(df, config):
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="electricity_demand",
        group_ids=["series_id"],
        max_encoder_length=config["input_window"],
        max_prediction_length=config["output_horizon"],
        time_varying_known_reals=[
            "hour", "day", "day_of_week", "month", "radiation", "temperature"
        ],
        time_varying_unknown_reals=["electricity_demand"],
        static_categoricals=[],
        static_reals=[],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
