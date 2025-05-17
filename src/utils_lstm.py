from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape, forecast_horizon):
    model = Sequential()
    model.add(LSTM(64, return_sequences = False, input_shape = input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(forecast_horizon))

    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
    return model