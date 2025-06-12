import os
import time
import numpy as np
import pandas as pd
import optuna
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback

from data_processing_tft import add_time_idx_and_series_id, split_dataset, create_dataloaders, encode_cyclical_features
from utils_tft import init_tft_model, get_callbacks as get_base_callbacks, load_config as load_config_section

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names*")
warnings.filterwarnings("ignore", message="Initializing zero-gradient BatchNorm module*")

# In tune_tft.py

# In tune_tft.py

import os
# ... other imports

# (Keep your warning filters and other setup)

def objective_tft(trial: optuna.trial.Trial, base_config: dict, train_df_hpo: pd.DataFrame, val_df_hpo: pd.DataFrame) -> float:
    trial_config = base_config.copy()

    # Suggest hyperparameters for the trial
    trial_config['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 5e-2, log=True)
    trial_config['hidden_size'] = trial.suggest_categorical('hidden_size', [16, 32, 64]) 
    trial_config['attention_head_size'] = trial.suggest_categorical('attention_head_size', [1, 4])
    trial_config['dropout'] = trial.suggest_float('dropout', 0.1, 0.4)
    trial_config['gradient_clip_val'] = trial.suggest_float('gradient_clip_val', 0.01, 1.0)

    print(f"\nTrial {trial.number}: Starting with params: {trial.params}", flush=True)

    min_len_for_test = trial_config['input_window'] + trial_config['output_horizon']
    dummy_test_df = val_df_hpo.head(min_len_for_test) if len(val_df_hpo) >= min_len_for_test else val_df_hpo
    
    train_loader, val_loader, _, training_dataset_for_trial = create_dataloaders(
        train_df_hpo, val_df_hpo, dummy_test_df,
        trial_config,
        trial_config["target_column"]
    )

    if not training_dataset_for_trial or len(train_loader) == 0:
        print(f"Trial {trial.number}: No training data after processing. Skipping.", flush=True)
        return float('inf')

    model = init_tft_model(training_dataset_for_trial, trial_config)

    # Callbacks (HPO specific) - Using aggressive settings for speed
    hpo_epochs = trial_config.get('epochs_hpo', 7) # Aggressive: 7 epochs for tuning
    pruning_cb_tft = PyTorchLightningPruningCallback(trial, monitor='val_loss')

    # Trainer
    trainer = Trainer(
        max_epochs=hpo_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[pruning_cb_tft],
        enable_progress_bar=True, # set to false for slurm job submissions
        enable_checkpointing=False,
        gradient_clip_val=trial_config['gradient_clip_val'],
    )

    # Train model
    try:
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
        print(f"Trial {trial.number} failed with an error: {e}", flush=True)
        # This can happen with OOM errors for large models
        return float('inf') 

    # Get the final validation loss
    final_val_loss = trainer.callback_metrics.get("val_loss")

    if final_val_loss is None:
        print(f"Trial {trial.number}: val_loss not found. Reporting as failed.", flush=True)
        return float('inf')

    final_val_loss_value = final_val_loss.item()
    print(f"Trial {trial.number}: Finished. Best val_loss: {final_val_loss_value:.6f}", flush=True)

    # Prune if the trial is unpromising
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return final_val_loss_value

if __name__ == "__main__":
    seed_everything(42, workers=True)
    HORIZON_TO_TUNE = "short"
    N_TRIALS = 20

    print(f"--- Starting Hyperparameter Optimization for TFT: {HORIZON_TO_TUNE} horizon ---", flush=True)

    base_config = load_config_section(HORIZON_TO_TUNE)
    print(f"Base TFT Config for {HORIZON_TO_TUNE} loaded: {base_config}", flush=True)

    print("Preparing data for TFT HPO...", flush=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_base_dir = os.path.dirname(script_dir)
    data_file_name = "merged_dataset_featurized.csv"
    data_path = os.path.join(project_base_dir, "data", "processed", data_file_name)
    raw_df_hpo_tft = pd.read_csv(data_path, sep=";", parse_dates=["utc_timestamp"])
    raw_df_hpo_tft = encode_cyclical_features(raw_df_hpo_tft)

    df_with_ids = add_time_idx_and_series_id(raw_df_hpo_tft)
    train_df, val_df, _ = split_dataset(df_with_ids, base_config)

    # aggressive subsampling for HPO!
    subset_percentage = 0.25
    subset_start_index = int(len(train_df) * (1 - subset_percentage))
    train_df_subset = train_df.iloc[subset_start_index:]

    print(f"--- Using AGGRESSIVE {subset_percentage*100}% subset for TFT HPO. New train shape: {train_df_subset.shape} ---")

    if train_df.empty or val_df.empty:
        print("Error: Training or validation DataFrame is empty. Check data preparation.", flush=True)
        exit()
        
    study_name_tft = f"tft_{HORIZON_TO_TUNE}_hpo_study"

    study = optuna.create_study(
        direction='minimize',
        study_name=study_name_tft,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps= 1)) # reduced from 2, more aggressive pruning
    

    print(f"Starting Optuna optimization for TFT with {N_TRIALS} trials...", flush=True)
    study.optimize(
        lambda trial_obj: objective_tft(trial_obj, base_config, train_df_subset, val_df),
        n_trials=N_TRIALS,
        timeout=base_config.get('hpo_timeout_seconds', None)
    )

    print("\n--- TFT Hyperparameter Optimization Finished ---", flush=True)
    print(f"Number of finished trials: {len(study.trials)}", flush=True)
    
    best_trial_tft = study.best_trial
    print(f"Best trial for TFT {HORIZON_TO_TUNE}:", flush=True)
    print(f"  Value (min val_loss): {best_trial_tft.value:.6f}", flush=True)
    print("  Best Parameters:", flush=True)
    for key, value in best_trial_tft.params.items():
        print(f"    {key}: {value}", flush=True)


