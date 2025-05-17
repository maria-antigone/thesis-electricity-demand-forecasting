import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_data(df):
    non_numeric = df.select_dtypes(exclude=['float64', 'int64'])  # e.g., timestamps
    numeric = df.select_dtypes(include=['float64', 'int64'])

    scaler = MinMaxScaler()
    scaled_numeric = scaler.fit_transform(numeric)

    return scaled_numeric, scaler

def create_lstm_sequences(data, input_window=96, forecast_horizon=96, target_col=0):
    X, y = [], []
    for i in range(len(data) - input_window - forecast_horizon):
        X_seq = data[i:i + input_window]
        y_seq = data[i + input_window:i + input_window + forecast_horizon, target_col]
        X.append(X_seq)
        y.append(y_seq)
    return np.array(X), np.array(y)