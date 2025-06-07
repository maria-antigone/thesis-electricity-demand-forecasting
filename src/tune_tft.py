import os
import time
import numpy as np
import pandas as pd
import optuna
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback

from utils_lstm import load_config as load_config_section
from data_processing_tft import add_time_idx_and_series_id, split_dataset, create_dataloaders
from utils_tft import init_tft_model, get_callbacks as get_base_callbacks

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names*")
warnings.filterwarnings("ignore", message="Initializing zero-gradient BatchNorm module*")

def objective_tft(trial: optuna.trial.Trial, base_config: dict, train_df_hpo: pd.DataFrame, val_df_hpo: pd.DataFrame) -> float:
    trial_config = base_config.copy()

    # suggest hyperparameters for trial
    trial_config['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 5e-2, log=True)
    trial_config['hidden_size'] = trial.suggest_categorical('hidden_size', [16, 32, 64])
    trial_config['attention_head_size'] = trial.suggest_categorical('attention_head_size', [1, 4])
    trial_config['dropout'] = trial.suggest_float('dropout', 0.1, 0.4)
    # trial_config['batch_size'] = trial.suggest_categorical('batch_size', [32, 64]) # If tuning batch_size
    trial_config['gradient_clip_val'] = trial.suggest_float('gradient_clip_val', 0.01, 1.0)

    print(f"\nTrial {trial.number}: Starting with params: {trial.params}", flush=True)
    
    # Prepare data loaders - if batch_size is fixed, dataloades could be prepared once outside. here: assume create_dataloaders uses trial_config['batch_size']
    train_loader, val_loader, _, training_dataset_for_trial = create_dataloaders(
        train_df_hpo, val_df_hpo, val_df_hpo.sample(min(len(val_df_hpo), 10)) if len(val_df_hpo) > 0 else val_df_hpo,
        trial_config, 
        trial_config["target_column"]
    )

    if not training_dataset_for_trial or len(train_loader) == 0:
        print(f"Trial {trial.number}: No training data after processing. Skipping.", flush=True)
        return float('inf')
    
    # Build Model
    model = init_tft_model(training_dataset_for_trial, trial_config)

    # Callbacks (HPO specific!)
    hpo_epochs = trial_config.get('epochs_hpo', 15)  # Few
    hpo_patience = trial_config.get('early_stopping_patience_hpo', 3)  # Strict

    early_stopping_cb_hpo = EarlyStopping(
        monitor = 'val_loss',
        patience = hpo_patience,
        mode = 'min',
        verbose = False,
    )

    pruning_cb_tft = PyTorchLightningPruningCallback(trial, monitor='val_loss')

    # Trainer
    trainer = Trainer(
        max_epochs=hpo_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[early_stopping_cb_hpo, pruning_cb_tft],
        logger=True,
        enable_progress_bar = False,
        enable_checkpointing=False,
        gradient_clip_val=trial_config['gradient_clip_val'],
    )

    # train model
    try:
        trainer.fit(model, train_loader, val_loader)
    except optuna.exceptions.TrialPruned:
        print(f"Trial {trial.number} was pruned.", flush=True)
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}", flush=True)
        return float('inf')
    
    # Return metric to be optimized
    val_loss_value = trainer.callback_metrics.get('val_loss')
    if val_loss_value is None:
        print(f"Trial {trial.number}: val_loss not found in callback_metrics. Pruner might have a value.", flush=True)
        if trial.state == optuna.trial.TrialState.PRUNED:
            pass
        best_val_loss_for_trial = val_loss_value.item() if val_loss_value is not None else float('inf')

    else:
        best_val_loss_for_trial
    if trial.should_prune():
        print(f"Trial {trial.number} explicitly checked as pruned after fit.", flush=True)
        raise optuna.exceptions.TrialPruned()
    
    print(f"Trial {trial.number}: Finished. Best val_loss: {best_val_loss_for_trial:.6f}", flush=True)
    return best_val_loss_for_trial

if __name__ == "__main__":
    seed_everything(42, workers=True)
    HORIZON_TO_TUNE = "short"
    N_TRIALS = 5

    print(f"--- Starting Hyperparameter Optimization for TFT: {HORIZON_TO_TUNE} horizon ---", flush=True)

    base_config = load_config_section(HORIZON_TO_TUNE)
    print(f"Base TFT Config for {HORIZON_TO_TUNE} loaded: {base_config}", flush=True)

    print("Preparing data for TFT HPO...", flush=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_base_dir = os.path.dirname(script_dir)
    data_file_name = "merged_dataset_featurized.csv"
    data_path = os.path.join(project_base_dir, "data", "processed", data_file_name)
    raw_df_hpo_tft = pd.read_csv(data_path, sep=";", parse_dates=["utc_timestamp"])

    df_with_ids = add_time_idx_and_series_id(raw_df_hpo_tft)
    train_df, val_df, _ = split_dataset(df_with_ids, base_config)
    print(f"Data prepared: train_df shape {train_df.shape}, val_df shape {val_df.shape}", flush=True)

    if train_df.empty or val_df.empty:
        print("Error: Training or validation DataFrame is empty. Check data preparation.", flush=True)
        exit()
        
    study_name_tft = f"tft_{HORIZON_TO_TUNE}_hpo_study"

    study = optuna.create_study(
        direction='minimize',
        study_name=study_name_tft,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=base_config.get('hpo_pruner_warmup', 2))
    )

    print(f"Starting Optuna optimization for TFT with {N_TRIALS} trials...", flush=True)
    study.optimize(
        lambda trial_obj: objective_tft(trial_obj, base_config, train_df, val_df),
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


