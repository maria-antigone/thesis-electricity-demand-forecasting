import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from data_processing_lstm import scale_data, create_lstm_sequences
from utils_lstm import build_lstm_model

import os

#Loading data

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "processed", "merged_dataset_featurized.csv")
df = pd.read_csv(data_path, sep = ";")
scaled_data, scaler = scale_data(df)

#Defining constants
INPUT_WINDOW = 96
FORECAST_HORIZON = 2880
TARGET_COL = 0

#Creating sequences
X, y = create_lstm_sequences(scaled_data, input_window=INPUT_WINDOW, forecast_horizon=FORECAST_HORIZON, target_col=TARGET_COL)

#Split
train_size = int(0.8 * len(X))
val_size = int(0.1 * len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

#Model
input_shape = (INPUT_WINDOW, X.shape[2])
model = build_lstm_model(input_shape, FORECAST_HORIZON)
#model.summary()

#Callbacks
early_stop = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)

#Training
history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val), 
                    epochs=50, 
                    batch_size=32, 
                    callbacks=[early_stop], 
                    verbose=1)