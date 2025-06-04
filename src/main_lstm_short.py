# main_lstm_short.py

import os
import time
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger as KerasCSVLogger
from utils_lstm import load_config, build_lstm_model
from data_processing_lstm import prepare_lstm_data
from metrics import mae as calculate_mae_metric, rmse as calculate_rmse_metric, mape as calculate_mape_metric # Renamed to avoid conflict

import tensorflow as tf
print("GPU available:", tf.config.list_physical_devices('GPU'), flush=True)

# 1. Load config
config = load_config("short")
print(f"LSTM Config loaded: {config}", flush=True)
forecast_horizon = config["output_horizon"]

# Output directory setup
script_dir_main = os.path.dirname(os.path.abspath(__file__))
project_base_dir_main = os.path.dirname(script_dir_main)
run_name_lstm = f"lstm_{config.get('run_tag', 'short_v3.0_eval')}_h{forecast_horizon}_w{config['input_window']}"
output_dir_run_lstm = os.path.join(project_base_dir_main, "outputs", "lstm_runs", run_name_lstm)
os.makedirs(output_dir_run_lstm, exist_ok=True)
print(f"LSTM run outputs will be saved in: {output_dir_run_lstm}", flush=True)

# 2. Load and prepare data
data_file_name_main = "merged_dataset_featurized.csv"
data_path_main = os.path.join(project_base_dir_main, "data", "processed", data_file_name_main)
print(f"Loading LSTM data from: {data_path_main}", flush=True)
raw_df = pd.read_csv(data_path_main, sep=";", parse_dates=["utc_timestamp"], index_col="utc_timestamp")

X_train, y_train, X_val, y_val, X_test, y_test, input_feature_scaler, target_scaler = \
    prepare_lstm_data(raw_df, config=config)
# y_train, y_val, y_test are SCALED

print(f"LSTM target_scaler type: {type(target_scaler)}", flush=True)

# 3. Define model
if X_train.shape[2] == 0:
    print("Error: X_train has 0 features.", flush=True)
    exit()
input_shape_for_model = (X_train.shape[1], X_train.shape[2])
model = build_lstm_model(input_shape_for_model, forecast_horizon, config)
model.summary()

# 4. Callbacks
early_stop = EarlyStopping(monitor='val_loss', 
                           patience=config["early_stopping_patience"], 
                           restore_best_weights=True,
                           verbose=1)

csv_log_path = os.path.join(output_dir_run_lstm, "lstm_epoch_training_log.csv")
keras_csv_logger = KerasCSVLogger(csv_log_path, append=False) # Overwrite if file exists for a new run
print(f"Keras CSVLogger will save epoch logs to: {csv_log_path}", flush=True)

print(f"Input shape for LSTM model: {input_shape_for_model}", flush=True)
print(f"X_train shape: {X_train.shape}, y_train shape (scaled): {y_train.shape}", flush=True)

# 5. Training
print("Starting LSTM training (targets scaled, corrected loop)...", flush=True)
start_training_time = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=config["epochs"],
    batch_size=config["batch_size"],
    verbose=1,
    callbacks=[early_stop, keras_csv_logger]
)
end_training_time = time.time()
total_training_duration = end_training_time - start_training_time
print(f"LSTM training finished in {total_training_duration:.2f} seconds.", flush=True)

print("\nEvaluating LSTM model on test set...", flush=True)

# Step 1: Make predictions on X_test (these will be scaled)
y_pred_scaled_lstm = model.predict(X_test)
print(f"Shape of y_pred_scaled_lstm: {y_pred_scaled_lstm.shape}", flush=True)
print(f"Shape of y_test (scaled): {y_test.shape}", flush=True)

# Step 2: Inverse transform!
y_pred_mw = np.zeros_like(y_pred_scaled_lstm)
y_test_mw = np.zeros_like(y_test) # y_test from prepare_lstm_data is already scaled

for i in range(y_pred_scaled_lstm.shape[0]):
    pred_sample_reshaped = y_pred_scaled_lstm[i, :].reshape(-1, 1)
    true_sample_reshaped = y_test[i, :].reshape(-1, 1) # y_test is the scaled ground truth
    
    y_pred_mw[i, :] = target_scaler.inverse_transform(pred_sample_reshaped).flatten()
    y_test_mw[i, :] = target_scaler.inverse_transform(true_sample_reshaped).flatten()

print(f"Shape of y_pred_mw (original scale): {y_pred_mw.shape}", flush=True)
print(f"Shape of y_test_mw (original scale): {y_test_mw.shape}", flush=True)

# Step 3: Calculate metrics using values in original MW scale
final_mae_mw = calculate_mae_metric(y_test_mw, y_pred_mw)
final_rmse_mw = calculate_rmse_metric(y_test_mw, y_pred_mw)
final_mape_mw = calculate_mape_metric(y_test_mw, y_pred_mw)

print("\nLSTM Final Evaluation Metrics (Test Set, Original MW Scale):", flush=True)
print(f"  MAE:  {final_mae_mw:.4f} MW", flush=True)
print(f"  RMSE: {final_rmse_mw:.4f} MW", flush=True)
print(f"  MAPE: {final_mape_mw:.2f}%", flush=True)

# Step 4 (Optional): Save predictions in MW scale
predictions_mw_csv_path = os.path.join(output_dir_run_lstm, "lstm_predictions_mw.csv")
if forecast_horizon > 0:
    df_predictions_mw = pd.DataFrame({
        'actual_mw_step1': y_test_mw[:, 0].flatten(),
        'predicted_mw_step1': y_pred_mw[:, 0].flatten()
    })
        
    df_predictions_mw.to_csv(predictions_mw_csv_path, index=False)
    print(f"LSTM predictions (MW scale) saved to: {predictions_mw_csv_path}", flush=True)

print("\nLSTM script finished (Items 1, 2 & 3 addressed).", flush=True)