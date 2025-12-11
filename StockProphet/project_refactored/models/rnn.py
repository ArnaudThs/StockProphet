"""
RNN (LSTM) model for stock price prediction.
Source: model.py + main.py (build_rnn_predictions)

Includes:
- Original simple LSTM for price prediction
- Probabilistic Multi-Horizon LSTM for uncertainty-aware forecasting
"""
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Input
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm

from project_refactored.config import (
    LSTM_WINDOW_SIZE, LSTM_EPOCHS, LSTM_BATCH_SIZE,
    LSTM_TRAIN_RATIO, LSTM_MODEL_PATH, PROB_LSTM_MODEL_PATH
)


def build_rnn_model(input_shape: tuple) -> Sequential:
    """
    Build RNN (LSTM) model architecture.

    Architecture:
        - LSTM(64, return_sequences=True)
        - LSTM(64, return_sequences=False)
        - Dense(32)
        - Dense(1)

    Args:
        input_shape: Tuple of (sequence_length, n_features)

    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(64, return_sequences=False),
        Dense(32),
        Dense(1)
    ])
    return model


def compile_model(model: Sequential) -> Sequential:
    """Compile model with MSE loss and Adam optimizer."""
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def train_model(model: Sequential, X_train: np.ndarray, y_train: np.ndarray,
                epochs: int = LSTM_EPOCHS, batch_size: int = LSTM_BATCH_SIZE,
                verbose: int = 1):
    """
    Train the LSTM model.

    Args:
        model: Compiled Keras model
        X_train: Training sequences, shape (samples, window_size, 1)
        y_train: Training targets, shape (samples, 1)
        epochs: Number of training epochs
        batch_size: Batch size
        verbose: Verbosity level

    Returns:
        Training history
    """
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )
    return history


def create_sequences(data: np.ndarray, window_size: int) -> tuple:
    """
    Create sequences for LSTM training.

    X[i] = data[i-window:i]
    y[i] = data[i]

    Args:
        data: Scaled price data, shape (n_samples, 1)
        window_size: Length of each sequence

    Returns:
        Tuple of (X, y) arrays
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y).reshape(-1, 1)


def generate_predictions(model: Sequential, close_prices: np.ndarray,
                         scaler: MinMaxScaler, window_size: int) -> np.ndarray:
    """
    Generate predictions for each timestep using sliding window.

    Args:
        model: Trained LSTM model
        close_prices: Array of close prices
        scaler: Fitted MinMaxScaler
        window_size: Sequence length

    Returns:
        Array of predictions aligned with original index (NaN for first window_size entries)
    """
    scaled = scaler.transform(close_prices.reshape(-1, 1))
    predictions = np.full(len(close_prices), np.nan)

    for i in range(window_size, len(scaled)):
        seq = scaled[i - window_size:i].reshape(1, window_size, 1)
        pred_scaled = model.predict(seq, verbose=0)
        predictions[i] = scaler.inverse_transform(pred_scaled)[0, 0]

    return predictions


def train_and_predict(df, target_col: str, window_size: int = LSTM_WINDOW_SIZE,
                      epochs: int = LSTM_EPOCHS, batch_size: int = LSTM_BATCH_SIZE,
                      train_ratio: float = LSTM_TRAIN_RATIO) -> tuple:
    """
    Full RNN pipeline: train LSTM on close prices, return predictions column.

    This is the main entry point called by pipeline.py to add rnn_pred_close
    to the DataFrame.

    Args:
        df: DataFrame with price data
        target_col: Column name for close prices (e.g., "AAPL_Close")
        window_size: Sequence length for LSTM
        epochs: Training epochs
        batch_size: Training batch size
        train_ratio: Fraction of data to use for training

    Returns:
        Tuple of (predictions, model, scaler)
    """
    close_prices = df[target_col].values

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices.reshape(-1, 1))

    # Create sequences
    X, y = create_sequences(scaled, window_size)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train/test split
    split = int(len(X) * train_ratio)
    X_train, y_train = X[:split], y[:split]

    print(f"Training RNN on {len(X_train)} samples...")

    # Build and train model
    model = build_rnn_model((window_size, 1))
    model = compile_model(model)
    train_model(model, X_train, y_train, epochs, batch_size)

    # Generate predictions for all rows
    print("Generating predictions...")
    predictions = generate_predictions(model, close_prices, scaler, window_size)

    return predictions, model, scaler


def save_model(model: Sequential, path=None):
    """Save trained model to disk."""
    if path is None:
        path = LSTM_MODEL_PATH
    model.save(path)
    print(f"Model saved to {path}")


def load_trained_model(path=None) -> Sequential:
    """Load trained model from disk."""
    if path is None:
        path = LSTM_MODEL_PATH
    return load_model(path)


def test_model(model: Sequential, X_test: np.ndarray, y_test: np.ndarray,
               scaler: MinMaxScaler) -> dict:
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        X_test: Test sequences
        y_test: Test targets (scaled)
        scaler: Scaler for inverse transform

    Returns:
        Dictionary with MAE, RMSE, MAPE metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred_scaled = y_pred_scaled.reshape(-1, 1)
    y_test_scaled = y_test.reshape(-1, 1)

    # Inverse transform
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test_scaled)

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape
    }

    return y_pred, y_true, metrics


# =============================================================================
# PROBABILISTIC MULTI-HORIZON LSTM
# =============================================================================

def gaussian_nll_loss(y_true, y_pred):
    """
    Gaussian Negative Log-Likelihood loss for probabilistic predictions.

    y_pred contains both mu and log_sigma stacked.
    For n horizons, y_pred shape is (batch, 2*n) where first n are mus, last n are log_sigmas.

    Args:
        y_true: Ground truth values, shape (batch, n_horizons)
        y_pred: Predicted [mu_1, ..., mu_n, log_sigma_1, ..., log_sigma_n]

    Returns:
        Negative log-likelihood loss
    """
    n_horizons = tf.shape(y_true)[1]
    mu = y_pred[:, :n_horizons]
    log_sigma = y_pred[:, n_horizons:]

    # Clamp log_sigma for numerical stability
    log_sigma = tf.clip_by_value(log_sigma, -10.0, 10.0)
    sigma = tf.exp(log_sigma) + 1e-6

    # Gaussian NLL: 0.5 * log(2*pi*sigma^2) + (y - mu)^2 / (2*sigma^2)
    nll = 0.5 * tf.math.log(2 * np.pi * sigma**2) + (y_true - mu)**2 / (2 * sigma**2)

    return tf.reduce_mean(nll)


def build_probabilistic_lstm(window_size: int, n_horizons: int, units: int = 64) -> Model:
    """
    Build Probabilistic Multi-Horizon LSTM using Keras Functional API.

    Outputs mu and log_sigma for each horizon, enabling uncertainty quantification.

    Architecture:
        Input -> LSTM(units) -> LSTM(units) -> Dense(32) -> [mu_outputs, log_sigma_outputs]

    Args:
        window_size: Sequence length
        n_horizons: Number of prediction horizons (e.g., 2 for t+1 and t+5)
        units: LSTM hidden units

    Returns:
        Keras Model with 2*n_horizons outputs (mu and log_sigma for each horizon)
    """
    inputs = Input(shape=(window_size, 1), name="price_sequence")

    # LSTM layers
    x = LSTM(units, return_sequences=True, name="lstm_1")(inputs)
    x = LSTM(units, return_sequences=False, name="lstm_2")(x)
    x = Dense(32, activation="relu", name="dense_shared")(x)

    # Output heads for each horizon
    mu_outputs = []
    log_sigma_outputs = []

    for i, h in enumerate([1, 5][:n_horizons]):  # Horizons t+1, t+5
        mu = Dense(1, activation="linear", name=f"mu_h{h}")(x)
        log_sigma = Dense(1, activation="linear", name=f"log_sigma_h{h}")(x)
        mu_outputs.append(mu)
        log_sigma_outputs.append(log_sigma)

    # Concatenate: [mu_1, mu_5, log_sigma_1, log_sigma_5]
    from keras.layers import Concatenate
    outputs = Concatenate(name="prob_outputs")(mu_outputs + log_sigma_outputs)

    model = Model(inputs=inputs, outputs=outputs, name="ProbabilisticMultiHorizonLSTM")
    return model


def create_multi_horizon_sequences(data: np.ndarray, window_size: int,
                                    horizons: list) -> tuple:
    """
    Create sequences for multi-horizon prediction.

    X[i] = data[i-window:i]
    y[i] = [data[i+h-1] for h in horizons]  (h-1 because horizon 1 means next step)

    Args:
        data: Scaled price data, shape (n_samples, 1)
        window_size: Sequence length
        horizons: List of prediction horizons [1, 5]

    Returns:
        Tuple of (X, y) where y has shape (samples, n_horizons)
    """
    max_horizon = max(horizons)
    X, y = [], []

    for i in range(window_size, len(data) - max_horizon + 1):
        X.append(data[i - window_size:i, 0])
        targets = [data[i + h - 1, 0] for h in horizons]
        y.append(targets)

    return np.array(X), np.array(y)


def train_probabilistic_lstm(df, target_col: str, window_size: int = LSTM_WINDOW_SIZE,
                              horizons: list = None, epochs: int = 30,
                              units: int = 64, train_ratio: float = LSTM_TRAIN_RATIO,
                              verbose: int = 1) -> tuple:
    """
    Train Probabilistic Multi-Horizon LSTM.

    Args:
        df: DataFrame with price data
        target_col: Column name for close prices
        window_size: Sequence length
        horizons: List of prediction horizons [1, 5]
        epochs: Training epochs
        units: LSTM hidden units
        train_ratio: Fraction for training
        verbose: Verbosity

    Returns:
        Tuple of (model, scaler, history)
    """
    from project_refactored.config import PROB_LSTM_HORIZONS, PROB_LSTM_EPOCHS, PROB_LSTM_UNITS

    if horizons is None:
        horizons = PROB_LSTM_HORIZONS

    close_prices = df[target_col].values

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices.reshape(-1, 1))

    # Create multi-horizon sequences
    X, y = create_multi_horizon_sequences(scaled, window_size, horizons)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train/test split
    split = int(len(X) * train_ratio)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    print(f"Training Probabilistic LSTM on {len(X_train)} samples...")
    print(f"Horizons: {horizons}, Units: {units}, Epochs: {epochs}")

    # Build and compile model
    model = build_probabilistic_lstm(window_size, len(horizons), units)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=gaussian_nll_loss
    )

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        verbose=verbose
    )

    return model, scaler, history


def generate_probabilistic_predictions(model: Model, close_prices: np.ndarray,
                                        scaler: MinMaxScaler, window_size: int,
                                        horizons: list) -> dict:
    """
    Generate probabilistic predictions for all timesteps.

    Args:
        model: Trained probabilistic LSTM
        close_prices: Array of close prices
        scaler: Fitted scaler
        window_size: Sequence length
        horizons: List of horizons

    Returns:
        Dictionary with prediction arrays for each feature:
        - rnn_mu_{h}d: Mean prediction for horizon h
        - rnn_sigma_{h}d: Std prediction for horizon h
        - rnn_prob_up_{h}d: P(price_t+h > price_t)
    """
    scaled = scaler.transform(close_prices.reshape(-1, 1))
    n_horizons = len(horizons)
    n_samples = len(close_prices)

    # Initialize output arrays
    results = {}
    for h in horizons:
        results[f'rnn_mu_{h}d'] = np.full(n_samples, np.nan)
        results[f'rnn_sigma_{h}d'] = np.full(n_samples, np.nan)
        results[f'rnn_prob_up_{h}d'] = np.full(n_samples, np.nan)

    # Generate predictions
    for i in range(window_size, n_samples):
        seq = scaled[i - window_size:i].reshape(1, window_size, 1)
        pred = model.predict(seq, verbose=0)[0]

        # Extract mu and log_sigma
        mus_scaled = pred[:n_horizons]
        log_sigmas = pred[n_horizons:]

        # Current price (scaled)
        current_scaled = scaled[i - 1, 0]

        for j, h in enumerate(horizons):
            mu_scaled = mus_scaled[j]
            sigma_scaled = np.exp(log_sigmas[j]) + 1e-6

            # Inverse transform mu to price space
            mu_price = scaler.inverse_transform([[mu_scaled]])[0, 0]

            # Transform sigma: since we're predicting scaled values,
            # sigma in price space ≈ sigma_scaled * price_scale
            price_scale = scaler.data_range_[0]
            sigma_price = sigma_scaled * price_scale

            # P(price_t+h > price_t) = P(return > 0)
            # Using scaled values: P(pred > current) = Phi((mu - current) / sigma)
            current_price = close_prices[i - 1]
            z_score = (mu_price - current_price) / (sigma_price + 1e-6)
            prob_up = norm.cdf(z_score)

            results[f'rnn_mu_{h}d'][i] = mu_price
            results[f'rnn_sigma_{h}d'][i] = sigma_price
            results[f'rnn_prob_up_{h}d'][i] = prob_up

    return results


def compute_derived_features(predictions: dict, horizons: list) -> dict:
    """
    Compute derived features from probabilistic predictions.

    Args:
        predictions: Dict from generate_probabilistic_predictions
        horizons: List of horizons

    Returns:
        Dictionary with additional derived features:
        - rnn_sharpe_{h}d: mu/sigma (expected return per unit risk)
        - rnn_trend_alignment: sign(mu_1d) * sign(mu_5d) agreement
        - rnn_conviction: Average probability strength across horizons
    """
    derived = {}

    # Sharpe-like signal for each horizon
    for h in horizons:
        mu = predictions[f'rnn_mu_{h}d']
        sigma = predictions[f'rnn_sigma_{h}d']
        # Use expected return (mu - current) / sigma, but we approximate with mu/sigma
        derived[f'rnn_sharpe_{h}d'] = mu / (sigma + 1e-6)

    # Trend alignment: do short-term and long-term agree?
    if len(horizons) >= 2:
        h1, h2 = horizons[0], horizons[1]
        mu_1 = predictions[f'rnn_mu_{h1}d']
        mu_5 = predictions[f'rnn_mu_{h2}d']

        # Get current prices (shifted by 1 since mu predicts future from previous point)
        # Alignment = 1 if both predict same direction, -1 if opposite, 0 if one is flat
        # We'll use the difference from a rolling baseline
        derived['rnn_trend_alignment'] = np.sign(mu_1) * np.sign(mu_5)

    # Conviction: how confident are we on average?
    # High conviction = probabilities far from 0.5
    prob_cols = [predictions[f'rnn_prob_up_{h}d'] for h in horizons]
    conviction = np.abs(np.nanmean(prob_cols, axis=0) - 0.5) * 2  # Scale to [0, 1]
    derived['rnn_conviction'] = conviction

    return derived


def train_and_predict_probabilistic(df, target_col: str, window_size: int = LSTM_WINDOW_SIZE,
                                     horizons: list = None, epochs: int = None,
                                     units: int = None, train_ratio: float = LSTM_TRAIN_RATIO,
                                     verbose: int = 1) -> tuple:
    """
    Full pipeline for Probabilistic Multi-Horizon LSTM.

    This is the main entry point called by pipeline.py to add probabilistic
    RNN features to the DataFrame.

    Args:
        df: DataFrame with price data
        target_col: Column name for close prices
        window_size: Sequence length
        horizons: Prediction horizons
        epochs: Training epochs
        units: LSTM units
        train_ratio: Training fraction
        verbose: Verbosity

    Returns:
        Tuple of (feature_dict, model, scaler) where feature_dict contains:
        - rnn_mu_{h}d for each horizon
        - rnn_sigma_{h}d for each horizon
        - rnn_prob_up_{h}d for each horizon
        - rnn_sharpe_{h}d for each horizon
        - rnn_trend_alignment
        - rnn_conviction
    """
    from project_refactored.config import PROB_LSTM_HORIZONS, PROB_LSTM_EPOCHS, PROB_LSTM_UNITS

    if horizons is None:
        horizons = PROB_LSTM_HORIZONS
    if epochs is None:
        epochs = PROB_LSTM_EPOCHS
    if units is None:
        units = PROB_LSTM_UNITS

    # Train model
    model, scaler, history = train_probabilistic_lstm(
        df, target_col, window_size, horizons, epochs, units, train_ratio, verbose
    )

    # Generate predictions
    close_prices = df[target_col].values
    predictions = generate_probabilistic_predictions(
        model, close_prices, scaler, window_size, horizons
    )

    # Compute derived features
    derived = compute_derived_features(predictions, horizons)

    # Merge all features
    all_features = {**predictions, **derived}

    print(f"Generated {len(all_features)} probabilistic RNN features")

    return all_features, model, scaler


def save_probabilistic_model(model: Model, path=None):
    """Save trained probabilistic model to disk."""
    if path is None:
        path = PROB_LSTM_MODEL_PATH
    model.save(path)
    print(f"Probabilistic model saved to {path}")


def load_probabilistic_model(path=None) -> Model:
    """Load trained probabilistic model from disk."""
    if path is None:
        path = PROB_LSTM_MODEL_PATH
    return load_model(path, custom_objects={'gaussian_nll_loss': gaussian_nll_loss})
