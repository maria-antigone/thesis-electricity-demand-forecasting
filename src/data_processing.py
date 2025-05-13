# feature engineering, standardization

import pandas as pd

def load_featurized_data(filepath: str) -> pd.DataFrame:
    """
    Load the pre-featurized dataset and apply any minimal preprocessing needed.
    """
    df = pd.read_csv(filepath, sep=';', index_col='utc_timestamp', parse_dates=True)

    # Optional: sort by datetime index just to be sure
    df = df.sort_index()

    return df
