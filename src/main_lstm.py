import os
import time
import numpy as np
import pandas as pd
import argparse
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger as KerasCSVLogger
from utils_lstm import load_config as load_config_section, build_lstm_model
from data_processing_lstm import prepare_lstm_data, encode_cyclical_features
from metrics import mae as calculate_mae_metric, rmse as calculate_rmse_metric, mape as calculate_mape_metric

import tensorflow as tf

# GPU Memory configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
   try:
       for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
       logical_gpus = tf.config.experimental.list_logical_devices('GPU')
       print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs configured for memory growth.", flush=True)
   except RuntimeError as e:
       print(e)
else:
    print("No GPUs detected by TensorFlow.", flush=True)

# Argument parsing
parser = argparse.ArgumentParser(description="Run LSTM for a specific forecasting horizon.")
parser.add_argument("--horizon", type=str, required=True, choices=["short", "medium", "long"],
                    help="The forecasting horizon to run (short/medium/long). Corresponds to config.yaml.")
args = parser.parse_args()

# 1. Load config
config_horizon_name = args.horizon
full_horizon_config = load_config_section(config_horizon_name)
config = full_horizon_config['lstm'] # model specific params!
print(f"LSTM Config loaded: {config}", flush=True)
forecast_horizon = full_horizon_config["output_horizon"]

script_dir_main = os.path.dirname(os.path.abspath(__file__))
project_base_dir_main = os.path.dirname(script_dir_main)

run_tag = config.get('run_tag', f"lstm_{config_horizon_name}")
run_name_lstm = f"{run_tag}_h{forecast_horizon}_w{full_horizon_config['input_window']}"

output_dir_run_lstm = os.path.join(project_base_dir_main, "outputs", "lstm_runs", run_name_lstm)
os.makedirs(output_dir_run_lstm, exist_ok=True)
print(f"LSTM run outputs will be saved in: {output_dir_run_lstm}", flush=True)

# 2. Load and prepare data
data_file_name_main = "merged_dataset_featurized.csv"
data_path_main = os.path.join(project_base_dir_main, "data", "processed", data_file_name_main)
print(f"Loading LSTM data from: {data_path_main}", flush=True)
raw_df = pd.read_csv(data_path_main, sep=";", parse_dates=["utc_timestamp"], index_col="utc_timestamp")
raw_df = encode_cyclical_features(raw_df)

X_train, y_train, X_val, y_val, X_test, y_test, input_feature_scaler, target_scaler = \
    prepare_lstm_data(raw_df, config=full_horizon_config)

print(f"LSTM target_scaler type for horizon {config_horizon_name}: {type(target_scaler)}", flush=True)

# 3. Define model
if X_train.ndim < 3 or X_train.shape[2] == 0:
    print(f"Error: X_train shape is {X_train.shape}. Expected 3D array with features for LSTM input.", flush=True)
    exit()
input_shape_for_model = (X_train.shape[1], X_train.shape[2])
model = build_lstm_model(input_shape_for_model, forecast_horizon, config)
model.summary()

# 4. Callbacks
early_stop = EarlyStopping(monitor='val_loss', 
                           patience=full_horizon_config["early_stopping_patience"], 
                           restore_best_weights=True,
                           verbose=1)

csv_log_path = os.path.join(output_dir_run_lstm, "lstm_epoch_training_log.csv")
keras_csv_logger = KerasCSVLogger(csv_log_path, append=False)
print(f"Keras CSVLogger will save epoch logs to: {csv_log_path}", flush=True)

print(f"Input shape for LSTM model: {input_shape_for_model}", flush=True)
print(f"X_train shape: {X_train.shape}, y_train shape (scaled): {y_train.shape}", flush=True)

# 5. Training
print("Starting LSTM training (targets scaled, corrected loop)...", flush=True)
start_training_time = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=full_horizon_config["epochs"], 
    batch_size=config["batch_size"],      
    verbose=2,
    callbacks=[early_stop, keras_csv_logger]
)
end_training_time = time.time()
total_training_duration = end_training_time - start_training_time
print(f"LSTM training finished in {total_training_duration:.2f} seconds.", flush=True)

print("\nEvaluating LSTM model on test set...", flush=True)

# Dynamic Evaluation & Saving Logic!

# Step 1: Make predictions on X_test (these will be scaled)
y_pred_scaled_lstm = model.predict(X_test)

# Step 2: Inverse transform predictions and actuals to get them in MW scale
y_pred_mw = target_scaler.inverse_transform(y_pred_scaled_lstm)
y_test_mw = target_scaler.inverse_transform(y_test)

# Step 3: Dynamically select the final step for evaluation based on parent config
final_step = full_horizon_config["output_horizon"]
final_step_index = final_step - 1

# Step 4: Calculate metrics using only the FINAL step of the forecast horizon
final_mae_mw = calculate_mae_metric(y_test_mw[:, final_step_index], y_pred_mw[:, final_step_index])
final_rmse_mw = calculate_rmse_metric(y_test_mw[:, final_step_index], y_pred_mw[:, final_step_index])
final_mape_mw = calculate_mape_metric(y_test_mw[:, final_step_index], y_pred_mw[:, final_step_index])

print(f"\nLSTM Final Evaluation Metrics (Test Set, Horizon: Step {final_step}):", flush=True)
print(f"  MAE:  {final_mae_mw:.4f} MW", flush=True)
print(f"  RMSE: {final_rmse_mw:.4f} MW", flush=True)
print(f"  MAPE: {final_mape_mw:.2f}%", flush=True)

# Step 5: Dynamically save predictions (first and final step)
predictions_to_save = {
    'actual_mw_step_1': y_test_mw[:, 0].flatten(),
    'predicted_mw_step_1': y_pred_mw[:, 0].flatten()
}

if final_step > 1:
    predictions_to_save[f'actual_mw_step_{final_step}'] = y_test_mw[:, final_step_index].flatten()
    predictions_to_save[f'predicted_mw_step_{final_step}'] = y_pred_mw[:, final_step_index].flatten()

df_predictions_mw = pd.DataFrame(predictions_to_save)
predictions_mw_csv_path = os.path.join(output_dir_run_lstm, "lstm_predictions_mw.csv")
df_predictions_mw.to_csv(predictions_mw_csv_path, index=False)
print(f"LSTM predictions (MW scale) saved to: {predictions_mw_csv_path}", flush=True)

print(f"\nLSTM script for horizon '{config_horizon_name}' finished.", flush=True)