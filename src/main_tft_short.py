# main_tft_short.py

import os
import yaml
import torch
import pandas as pd
from pytorch_lightning import seed_everything

from data_processing_tft import add_time_idx_and_series_id, split_dataset, create_dataloaders
from utils_tft import init_tft_model, get_callbacks, get_trainer
from metrics import mae, rmse, mape  # ← import directly

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names*")

if __name__ == "__main__":
    seed_everything(42)

    # === Load config ===
    with open("src/config.yaml") as file:
        full_config = yaml.safe_load(file)
    config = full_config["short"]

    # === Load & preprocess data ===
    df = pd.read_csv("data/processed/merged_dataset_featurized.csv", sep = ";", parse_dates=["utc_timestamp"])
    df = add_time_idx_and_series_id(df)
    train_df, val_df, test_df = split_dataset(df, config)

    # === Create dataloaders ===
    train_loader, val_loader, test_loader, training_dataset = create_dataloaders(
        train_df, val_df, test_df, config, config["target_column"]
    )

    # === Init model ===
    model = init_tft_model(training_dataset, config)

    # === Callbacks & Trainer ===
    output_dir = "checkpoints/tft_short"
    os.makedirs(output_dir, exist_ok=True)

    callbacks = get_callbacks(output_dir, config)
    trainer = get_trainer(config, output_dir, callbacks)

    # === Train ===
    trainer.fit(model, train_loader, val_loader)

    # === Evaluate ===
    actuals = torch.cat([y[0] for x, y in iter(test_loader)])
    predictions = model.predict(test_loader, mode="prediction")

    # === Save predictions ===
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame({
        "actual": actuals.numpy().flatten(),
        "prediction": predictions.numpy().flatten()
    }).to_csv("outputs/tft_short_predictions.csv", index=False)

    # === Print evaluation ===
    y_true = actuals.numpy()
    y_pred = predictions.numpy()

    print("TFT Evaluation Metrics (Short Horizon):")
    print(f"MAE:  {mae(y_true, y_pred):.4f}")
    print(f"RMSE: {rmse(y_true, y_pred):.4f}")
    print(f"MAPE: {mape(y_true, y_pred):.2f}%")
