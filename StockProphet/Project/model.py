import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.metrics import mean_absolute_error, mean_squared_error


def LSTM_model(input_shape):
    """
    Build and return an LSTM model.
    input_shape = (sequence_length, n_features)
    """
    model = Sequential()

    # First LSTM layer
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))

    # Second LSTM layer
    model.add(LSTM(64, return_sequences=False))

    # Dense layers
    model.add(Dense(32))
    model.add(Dense(1))

    return model


# ================================
# 2. COMPILE MODEL
# ================================
def compile_LSTM(model):
    """
    Compile LSTM model.
    """

    model.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ["accuracy"])
    return model


# ================================
# 3. TRAIN MODEL
# ================================
def train_LSTM(model, X_train, y_train, epochs=20, batch_size=32, verbose=1):
    """
    Fit LSTM model and return training history.
    """
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )
    return history


def test_lstm_model(model, X_test, y_test, scaler_y):
    """
    Loads the trained LSTM model and evaluates it on the test set.

    Args:
        model_path (str): Path to saved .keras model.
        X_test (np.ndarray): Test features, shape (samples, seq_len, features).
        y_test (np.ndarray): True values (scaled).
        scaler_y (MinMaxScaler): Scaler fitted on target.

    Returns:
        preds (np.ndarray): Inverse-transformed predictions.
        y_true (np.ndarray): Inverse-transformed ground-truth.
        metrics (dict): MAE, RMSE, MAPE
    """

    # Predict (still scaled)
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Some models return shape (n,1), some (n,)
    y_pred_scaled = y_pred_scaled.reshape(-1, 1)
    y_test_scaled = y_test.reshape(-1, 1)

    # Invert scaler
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test_scaled)

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape
    }

    return y_pred, y_true, metrics



def plot_predictions(y_true, y_pred, title="Stock Price Prediction vs Actual"):
    """
    Plots predicted vs actual stock prices.

    Args:
        y_true (np.ndarray): True stock prices.
        y_pred (np.ndarray): Predicted stock prices.
        title (str): Plot title.
    """

    plt.figure(figsize=(14,6))
    plt.plot(y_true, color='blue', label='Actual Price')
    plt.plot(y_pred, color='red', label='Predicted Price')
    plt.title(title, fontsize=16)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()
