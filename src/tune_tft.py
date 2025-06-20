import os
import time
import numpy as np
import pandas as pd
import optuna
import torch
from pytorch_lightning import seed_everything, Trainer
from optuna.integration import PyTorchLightningPruningCallback
from torch.multiprocessing import set_start_method

from data_processing_tft import add_time_idx_and_series_id, split_dataset, create_dataloaders, encode_cyclical_features
from utils_tft import init_tft_model, load_config as load_config_section

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names*")
warnings.filterwarnings("ignore", message="Initializing zero-gradient BatchNorm module*")


def objective_tft(trial: optuna.trial.Trial, base_config: dict, train_df_hpo: pd.DataFrame, val_df_hpo: pd.DataFrame) -> float:
    trial_config = base_config.copy()

    # Suggest hyperparameters for the trial
    trial_config['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    trial_config['hidden_size'] = trial.suggest_categorical('hidden_size', [16, 32, 64])
    trial_config['attention_head_size'] = trial.suggest_categorical('attention_head_size', [1, 4])
    trial_config['dropout'] = trial.suggest_float('dropout', 0.1, 0.4)
    # not tuning gradient clip, low impact value to tune

    horizon = base_config.get('output_horizon')
    if horizon == 2880:
        trial_config['batch_size'] = trial.suggest_categorical('batch_size', [4, 8])
    else:
        trial_config['batch_size'] = trial.suggest_categorical('batch_size', [16, 32])

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
    pruning_cb_tft = PyTorchLightningPruningCallback(trial, monitor='val_loss')

    # hpo_epochs = trial_config.get('epochs_hpo', 7) # Aggressive: 7 epochs for tuning
    pruning_cb_tft = PyTorchLightningPruningCallback(trial, monitor='val_loss')

    # Trainer
    trainer = Trainer(
        max_epochs=trial_config.get('epochs_hpo', 15),
        accelerator="gpu",
        devices=1,
        callbacks=[pruning_cb_tft],
        enable_progress_bar=False, # Set to False for clean logs in batch jobs
        enable_checkpointing=False, # Disable saving checkpoints during HPO
        gradient_clip_val=trial_config.get('gradient_clip_val', 0.1),
    )

    # Train model
    try:
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
        print(f"Trial {trial.number} failed with an error: {e}", flush=True)
        return float('inf') 

    final_val_loss = trainer.callback_metrics.get("val_loss")

    if final_val_loss is None:
        print(f"Trial {trial.number}: val_loss not found. Reporting as failed.", flush=True)
        return float('inf')

    final_val_loss_value = final_val_loss.item()
    print(f"Trial {trial.number}: Finished. Best val_loss: {final_val_loss_value:.6f}", flush=True)

    return final_val_loss_value

if __name__ == "__main__":
    try:
        set_start_method("forkserver")
    except RuntimeError:
        pass

    seed_everything(42, workers=True)

    horizons_to_tune = ["short", "medium", "long"]
    N_TRIALS = 20

    print("Preparing full dataset for TFT HPO...", flush=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_base_dir = os.path.dirname(script_dir)
    data_file_name = "merged_dataset_featurized.csv"
    data_path = os.path.join(project_base_dir, "data", "processed", data_file_name)
    raw_df_hpo_tft = pd.read_csv(data_path, sep=";", parse_dates=["utc_timestamp"])

    for horizon in horizons_to_tune:
        print(f"\n{'='*20} Starting HPO for TFT HORIZON: {horizon.upper()} {'='*20}", flush=True)
        
        # 1. Load config for the current horizon
        base_config = load_config_section(horizon)
        print(f"Base TFT Config for {horizon} loaded: {base_config}", flush=True)

        # 2. Prepare data for the current horizon
        # (This assumes raw features are used for TFT as per your config)
        df_with_ids = add_time_idx_and_series_id(raw_df_hpo_tft)
        train_df, val_df, _ = split_dataset(df_with_ids, base_config)

        # 3. Use aggressive subsampling to speed up tuning
        subset_percentage = 0.25
        subset_start_index = int(len(train_df) * (1 - subset_percentage))
        train_df_subset = train_df.iloc[subset_start_index:]
        print(f"--- Using AGGRESSIVE {subset_percentage*100}% subset for TFT HPO. New train shape: {train_df_subset.shape} ---")

        if train_df_subset.empty or val_df.empty:
            print(f"Error: Data for horizon {horizon} is empty. Skipping.", flush=True)
            continue
        
        # 4. Create a unique study for this horizon
        study_name_tft = f"tft_{horizon}_hpo_study_final"
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name_tft,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=base_config.get('hpo_pruner_warmup', 2))
        )

        # 5. Run the optimization
        print(f"Starting Optuna optimization for TFT ({horizon}) with {N_TRIALS} trials...", flush=True)
        study.optimize(
            lambda trial_obj: objective_tft(trial_obj, base_config, train_df_subset, val_df),
            n_trials=N_TRIALS
        )

        # 6. Print results for this horizon
        print(f"\n--- TFT HPO Finished for HORIZON: {horizon.upper()} ---", flush=True)
        print(f"Number of finished trials: {len(study.trials)}", flush=True)
        
        if study.best_trial:
            best_trial_tft = study.best_trial
            print(f"Best trial for TFT {horizon}:", flush=True)
            print(f"  Value (min val_loss): {best_trial_tft.value:.6f}", flush=True)
            print("  Best Parameters:", flush=True)
            for key, value in best_trial_tft.params.items():
                print(f"    {key}: {value}", flush=True)
        else:
            print(f"No successful trials completed for horizon {horizon}.")