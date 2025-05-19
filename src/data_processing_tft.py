# feature engineering, standardization

import pandas as pd

import yaml

import os
import yaml

def load_config(horizon="short"):
    # Always resolve relative to the script’s own directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)[horizon]

# Use these to set up TimeSeriesDataSet
# max_encoder_length=..., max_prediction_length=...

def load_featurized_data(filepath: str) -> pd.DataFrame:
    """
    Load the pre-featurized dataset and apply any minimal preprocessing needed.
    """
    df = pd.read_csv(filepath, sep=';', index_col='utc_timestamp', parse_dates=True)

    # Optional: sort by datetime index just to be sure
    df = df.sort_index()

    return df
