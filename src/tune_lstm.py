import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import os
import time
import numpy as np
import pandas as pd
import optuna
from tensorflow.keras.callbacks import EarlyStopping
from optuna.integration import KerasPruningCallback

from utils_lstm import load_config as load_config_section, build_lstm_model
from data_processing_lstm import prepare_lstm_data, encode_cyclical_features

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
        print(f"Error setting memory growth: {e}", flush=True)
else:
    print("No GPUs detected by TensorFlow.", flush=True)
print("Logical GPU devices available to TensorFlow:", tf.config.list_logical_devices('GPU'), flush=True)

def objective(trial: optuna.trial.Trial, base_config: dict, X_train_data, y_train_data, X_val_data, y_val_data) -> float:
    trial_config = base_config.copy()

    # Suggest hyperparameters for the trial
    trial_config['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    trial_config['lstm_units'] = trial.suggest_categorical('lstm_units', [16, 32, 48])
    trial_config['dropout_rate'] = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Using a safer, more practical range for batch size
    trial_config['batch_size'] = trial.suggest_categorical('batch_size', [16, 32])

    print(f"\nTrial {trial.number}: Starting with params: {trial.params}", flush=True)

    input_shape = (X_train_data.shape[1], X_train_data.shape[2])
    model = build_lstm_model(input_shape, trial_config["output_horizon"], trial_config)

    # Using strict callbacks for HPO efficiency
    hpo_epochs = trial_config.get('epochs_hpo', 10)
    hpo_patience = trial_config.get('early_stopping_patience_hpo', 3)

    early_stopping_cb = EarlyStopping(monitor='val_mae', patience=hpo_patience, restore_best_weights=True)
    pruning_cb = KerasPruningCallback(trial, 'val_mae')

    history = model.fit(
        X_train_data,
        y_train_data,
        validation_data=(X_val_data, y_val_data),
        epochs=hpo_epochs,
        batch_size=trial_config['batch_size'],
        # Change verbose to 2 for cleaner logs (one line per epoch)
        verbose=2, 
        callbacks=[early_stopping_cb, pruning_cb]
    )

    best_val_mae_for_trial = min(history.history['val_mae']) if 'val_mae' in history.history and history.history['val_mae'] else float('inf')

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned

    print(f"Trial {trial.number}: Completed with best val_mae: {best_val_mae_for_trial:.6f}", flush=True)

    return best_val_mae_for_trial

if __name__ == "__main__":
    HORIZON_TO_TUNE = "short"
    N_TRIALS = 20

    print(f"Starting hyperparameter tuning for LSTM with horizon: {HORIZON_TO_TUNE}", flush=True)

    base_config = load_config_section(HORIZON_TO_TUNE)
    print(f"Base configuration loaded: {base_config}", flush=True)

    # load and prepare data

    print("Preparing data for HPO...", flush=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_base_dir = os.path.dirname(script_dir)
    data_file_name = "merged_dataset_featurized.csv"
    data_path = os.path.join(project_base_dir, "data", "processed", data_file_name)
    raw_df_hpo = pd.read_csv(data_path, sep=";", parse_dates=["utc_timestamp"], index_col="utc_timestamp")
    raw_df_hpo = encode_cyclical_features(raw_df_hpo)

    # Using same prepare_lstm_data function
    X_train, y_train, X_val, y_val, _, _, _, _ = \
        prepare_lstm_data(raw_df_hpo, config=base_config)
    print(f"Data prepared: X_train shape {X_train.shape}, X_val shape {X_val.shape}", flush=True)

    # subset data for faster computation
    subset_percentage = 0.50
    subset_start_index = int(len(X_train) * (1 - subset_percentage))
    X_train_subset = X_train[subset_start_index:]
    y_train_subset = y_train[subset_start_index:]

    print(f"Using a {subset_percentage*100}% subset of training data for HPO. New shape: {X_train_subset.shape}")
    
    if X_train.size == 0 or X_val.size == 0:
        print("Error: Training or validation data is empty. Check data preparation and config.", flush=True)
        exit()

    # load optuna study
    study_name = f"lstm_{HORIZON_TO_TUNE}_hpo_study"

    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=base_config.get('hpo_pruner_warmup', 2))   
    )

    # start optimization
    print(f"Starting Optuna optimization with {N_TRIALS} trials...", flush=True)
    study.optimize(
        lambda trial: objective(trial, base_config, X_train_subset, y_train_subset, X_val, y_val),
        n_trials=N_TRIALS,
        timeout=base_config.get('hpo_timeout_seconds', None)
    )

    # Print results:
    print("\n--- Hyperparameter Optimization Finished ---", flush=True)
    print(f"Number of finished trials: {len(study.trials)}", flush=True)

    best_trial = study.best_trial
    print(f"Best trial for LSTM {HORIZON_TO_TUNE}:", flush=True)
    print(f"  Value (min val_mae_scaled): {best_trial.value:.6f}", flush=True)
    print("  Best Parameters:", flush=True)
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}", flush=True)