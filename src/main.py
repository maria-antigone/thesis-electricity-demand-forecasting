# main.py

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE

from src.data_processing_tft import load_featurized_data
from src.utils_tft import create_tft_dataset

import warnings
from sklearn.exceptions import NotFittedError
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")


import os
import logging
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    filename = "outputs/training_log.txt",
    level = logging.INFO,
    format = "%(asctime)s - %(message)s",
    filemode = "a"
)

# logging validation loss after each epoch

from pytorch_lightning.callbacks import Callback
class FileLoggerCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            logging.info(f"Epoch {trainer.current_epoch}: val_loss: {val_loss:.4f}")

def main():
    logging.info("Training has begun!")

    # Load dataset
    df = load_featurized_data("data/processed/merged_dataset_featurized.csv")

    # Use only 2019 for now (as in your prototype)
    df = df.loc["2019-01-01":"2019-12-31"]

    # Create TFT dataset and dataloaders
    dataset, train_dataloader, val_dataloader = create_tft_dataset(df)

    # Define TensorBoard logger
    logger = TensorBoardLogger("lightning_logs", name="tft")

    # Define the model
    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        loss=MAE(),
        log_interval=10,
        reduce_on_plateau_patience=4
    )

    # Early stopping
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=True,
        mode="min"
    )

    # Trainer with GPU and logging
    trainer = Trainer(
        max_epochs=30,
        gradient_clip_val=0.1,
        callbacks=[early_stop],
        accelerator="gpu",
        devices=1,
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=True
    )

    # Fit the model
    trainer.fit(model=tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    logging.info("Training has finished!")


if __name__ == "__main__":
    main()
