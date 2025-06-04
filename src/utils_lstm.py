from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import os
import yaml

def load_config(horizon="short"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)[horizon]

def log_epoch_metrics(epoch, logs):
    print(f"[Epoch {epoch}] ", end="", flush=True)
    for key, value in logs.items():
        print(f"{key}: {value:.4f} ", end="", flush=True)
    print(flush=True)

def build_lstm_model(input_shape, forecast_horizon, config):
    model = Sequential()
    model.add(LSTM(config.get("lstm_units", 64), 
                   return_sequences=False, 
                   input_shape=input_shape))
    model.add(Dropout(config.get("dropout_rate", 0.2)))
    model.add(Dense(forecast_horizon))

    model.compile(
        optimizer=Adam(learning_rate=config.get("learning_rate", 0.001)),
        loss=config.get("loss_function", "mae"),
        metrics=["mae"]
    )
    return model

