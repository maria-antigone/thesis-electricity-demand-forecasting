# main tft

import os
import time
import yaml
import torch
import pandas as pd
import argparse
from pytorch_lightning import seed_everything
from pytorch_forecasting import TemporalFusionTransformer

from data_processing_tft import add_time_idx_and_series_id, split_dataset, create_dataloaders
from utils_tft import init_tft_model, get_callbacks, get_trainer
from metrics import mae, rmse, mape

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names*")
warnings.filterwarnings("ignore", message="Initializing zero-gradient BatchNorm module*")

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    seed_everything(42, workers=True)

    parser = argparse.ArgumentParser(description="Run TFT model for a specific forecasting horizon.")
    parser.add_argument("--horizon", type=str, required=True, choices=["short", "medium", "long"],
                        help="The forecasting horizon to run (short/medium/long). Corresponds to config.yaml.")
    args = parser.parse_args()
    config_horizon_name = args.horizon

    print(f"--- Running TFT Training for {config_horizon_name.upper()} Horizon ---", flush=True)
    with open("src/config.yaml") as file:
        full_config_from_file = yaml.safe_load(file)
    
    full_horizon_config = full_config_from_file[config_horizon_name]
    
    config = full_horizon_config['tft']
    print(f"TFT-specific Config loaded: {config}", flush=True)
    forecast_horizon = full_horizon_config["output_horizon"]

    script_dir_main = os.path.dirname(os.path.abspath(__file__))
    project_base_dir_main = os.path.dirname(script_dir_main)
    
    run_tag = config.get('run_tag', f"tft_{config_horizon_name}")
    run_name_tft = f"{run_tag}_h{forecast_horizon}_w{full_horizon_config['input_window']}"
    
    output_dir_run_tft = os.path.join(project_base_dir_main, "outputs", "tft_runs", run_name_tft)
    os.makedirs(output_dir_run_tft, exist_ok=True)
    print(f"TT run outputs will be saved in: {output_dir_run_tft}", flush=True)

    data_file_name_main = "merged_dataset_featurized.csv"
    data_path_main = os.path.join(project_base_dir_main, "data", "processed", data_file_name_main)
    print(f"Loading TFT data from: {data_path_main}", flush=True)
    df = pd.read_csv(data_path_main, sep=";", parse_dates=["utc_timestamp"])
    df = add_time_idx_and_series_id(df)
    
    train_df, val_df, test_df = split_dataset(df, full_horizon_config)

    # if config_horizon_name == 'long':
        # Use only the most recent 60% of training data for the long run
        # subset_percentage = 0.60 
        # subset_start_index = int(len(train_df) * (1 - subset_percentage))
        # train_df = train_df.iloc[subset_start_index:]
        # print(f"--- LONG HORIZON: Using a {subset_percentage*100}% subset of training data to get a feasible runtime. New train shape: {train_df.shape} ---", flush=True)

    print("Creating Dataloaders...", flush=True)
    train_loader, val_loader, test_loader, training_dataset = create_dataloaders(
        train_df, val_df, test_df, full_horizon_config, full_horizon_config["target_column"]
    )

    print("Initializing TFT model...", flush=True)
    model = init_tft_model(training_dataset, config)

    callbacks = get_callbacks(output_dir_run_tft, full_horizon_config)
    trainer = get_trainer(full_horizon_config, output_dir_run_tft, callbacks)

    print("Starting TFT training...", flush=True)
    start_training_time = time.time()

    trainer.fit(model, train_loader, val_loader)
    end_training_time = time.time()
    total_training_duration = end_training_time - start_training_time
    print(f"TFT training finished in {total_training_duration:.2f} seconds.", flush=True)

    # Dynamic Evaluation and Saving logic

    print("\nEvaluating TFT model on test set...", flush=True)
    
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Loading best model for evaluation from: {best_model_path}")
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # Make predictions on the test set
    actuals = torch.cat([y[0] for x, y in iter(test_loader)])
    predictions = best_model.predict(test_loader, mode="prediction")

    final_step = full_horizon_config["output_horizon"]
    final_step_index = final_step - 1

    y_true_final_step = actuals[:, final_step_index].numpy()
    y_pred_final_step = predictions[:, final_step_index].numpy()

    # Calculate metrics using only the final step
    final_mae = mae(y_true_final_step, y_pred_final_step)
    final_rmse = rmse(y_true_final_step, y_pred_final_step)
    final_mape = mape(y_true_final_step, y_pred_final_step)

    print(f"\nTFT Final Evaluation Metrics (Test Set, Horizon: Step {final_step}):", flush=True)
    print(f"  MAE:  {final_mae:.4f}")
    print(f"  RMSE: {final_rmse:.4f}")
    print(f"  MAPE: {final_mape:.2f}%")

    # Dynamically save predictions (first and final step)
    predictions_to_save = {
        'actual_mw_step_1': actuals[:, 0].numpy().flatten(),
        'predicted_mw_step_1': predictions[:, 0].numpy().flatten()
    }
    if final_step > 1:
        predictions_to_save[f'actual_mw_step_{final_step}'] = y_true_final_step.flatten()
        predictions_to_save[f'predicted_mw_step_{final_step}'] = y_pred_final_step.flatten()
    
    df_predictions_mw = pd.DataFrame(predictions_to_save)
    predictions_mw_csv_path = os.path.join(output_dir_run_tft, "tft_predictions_mw.csv")
    df_predictions_mw.to_csv(predictions_mw_csv_path, index=False)
    print(f"TFT predictions (MW scale) saved to: {predictions_mw_csv_path}", flush=True)

    print(f"\nTFT script for horizon '{config_horizon_name}' finished.", flush=True)