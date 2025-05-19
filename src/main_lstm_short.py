import os
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from utils_lstm import load_config, build_lstm_model, log_epoch_metrics
from data_processing_lstm import prepare_lstm_data

# 1. Load config
config = load_config("short")
forecast_horizon = config["output_horizon"]

# 2. Load and prepare data
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "processed", "merged_dataset_featurized.csv")
df = prepare_lstm_data(df = pd.read_csv(data_path, sep=";", parse_dates=["utc_timestamp"], index_col="utc_timestamp"), 
                       config=config)

X_train, y_train, X_val, y_val, X_test, y_test, scaler = df

# 3. Define model
input_shape = (config["input_window"], X_train.shape[2])
model = build_lstm_model(input_shape, forecast_horizon, config)

# 4. Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=config["early_stopping_patience"], restore_best_weights=True)

# 5. Training loop with GPU-friendly logging
for epoch in range(config["epochs"]):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1,
        batch_size=config["batch_size"],
        verbose=0,
        callbacks=[early_stop]
    )

    logs = {
        "loss": history.history["loss"][0],
        "val_loss": history.history["val_loss"][0],
        "mae": history.history["mae"][0],
        "val_mae": history.history["val_mae"][0],
    }
    log_epoch_metrics(epoch + 1, logs)
