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

if __name__ == "__main__":
    seed_everything(42, workers=True)

    parser = argparse.ArgumentParser(description="Run TFT model for a specific forecasting horizon.")
    parser.add_argument("--horizon", type=str, required=True, choices=["short", "medium", "long"],
                        help="The forecasting horizon to run (short/medium/long). Corresponds to config.yaml.")
    args = parser.parse_args()
    config_horizon_name = args.horizon

    print(f"--- Running TFT Training for {config_horizon_name.upper()} Horizon ---", flush=True)
    with open("src/config.yaml") as file:
        full_config = yaml.safe_load(file)
    config = full_config[config_horizon_name]
    print(f"TFT Config loaded: {config}", flush=True)
    forecast_horizon = config["output_horizon"]

    script_dir_main = os.path.dirname(os.path.abspath(__file__))
    project_base_dir_main = os.path.dirname(script_dir_main)
    
    run_tag = config.get('run_tag', config_horizon_name)
    run_name_tft = f"tft_{run_tag}_h{forecast_horizon}_w{config['input_window']}"
    
    output_dir_run_tft = os.path.join(project_base_dir_main, "outputs", "tft_runs", run_name_tft)
    os.makedirs(output_dir_run_tft, exist_ok=True)
    print(f"TFT run outputs will be saved in: {output_dir_run_tft}", flush=True)

    data_file_name_main = "merged_dataset_featurized.csv"
    data_path_main = os.path.join(project_base_dir_main, "data", "processed", data_file_name_main)
    print(f"Loading TFT data from: {data_path_main}", flush=True)
    df = pd.read_csv(data_path_main, sep=";", parse_dates=["utc_timestamp"])
    df = add_time_idx_and_series_id(df)
    train_df, val_df, test_df = split_dataset(df, config)

    print("Creating Dataloaders...", flush=True)
    train_loader, val_loader, test_loader, training_dataset = create_dataloaders(
        train_df, val_df, test_df, config, config["target_column"]
    )

    print("Initializing TFT model...", flush=True)
    model = init_tft_model(training_dataset, config)

    callbacks = get_callbacks(output_dir_run_tft, config)
    trainer = get_trainer(config, output_dir_run_tft, callbacks)

    print("Starting TFT training...", flush=True)
    start_training_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    end_training_time = time.time()
    total_training_duration = end_training_time - start_training_time
    print(f"TFT training finished in {total_training_duration:.2f} seconds.", flush=True)

    print("\nEvaluating TFT model on test set...", flush=True)
    
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Loading best model for evaluation from: {best_model_path}")
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    actuals = torch.cat([y[0] for x, y in iter(test_loader)])
    predictions = best_model.predict(test_loader, mode="prediction")

    predictions_csv_path = os.path.join(output_dir_run_tft, "tft_predictions.csv")
    pd.DataFrame({
        "actual": actuals.numpy().flatten(),
        "prediction": predictions.numpy().flatten()
    }).to_csv(predictions_csv_path, index=False)
    print(f"TFT predictions saved to: {predictions_csv_path}", flush=True)

    y_true = actuals.numpy()
    y_pred = predictions.numpy()

    final_mae = mae(y_true, y_pred)
    final_rmse = rmse(y_true, y_pred)
    final_mape = mape(y_true, y_pred)

    print(f"\nTFT Final Evaluation Metrics ({config_horizon_name.upper()} Horizon):", flush=True)
    print(f"  MAE:  {final_mae:.4f}")
    print(f"  RMSE: {final_rmse:.4f}")
    print(f"  MAPE: {final_mape:.2f}%")

    print(f"\nTFT script for horizon '{config_horizon_name}' finished.", flush=True)