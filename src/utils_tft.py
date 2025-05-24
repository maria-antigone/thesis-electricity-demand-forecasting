import os
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_forecasting.metrics import MAE

def init_tft_model(training, config):
    return TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=config["learning_rate"],
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        loss=MAE(),
        log_interval=10,
        log_val_interval=1,
        reduce_on_plateau_patience=4,
    )

def get_callbacks(output_dir, config):
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=config["early_stopping_patience"],
        mode="min"
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=output_dir,
        filename="tft-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min"
    )

    return [early_stop_callback, checkpoint_callback]

def get_trainer(config, output_dir, callbacks):
    return Trainer(
        max_epochs=config["epochs"],
        gradient_clip_val=0.1,
        callbacks=callbacks,
        default_root_dir=output_dir,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
