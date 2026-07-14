"""
model.py
========
LSTM-based Market Price Predictor

Architecture:
  Input  → LSTM(128) → Dropout → BatchNorm
         → LSTM(64)  → Dropout → BatchNorm
         → LSTM(32)  → Dropout
         → Dense(16, relu) → Dense(1)

Features used (multi-variate):
  Close, Volume, RSI, MACD, BB_Upper, BB_Lower,
  MA_20, MA_50, ATR, OBV, Daily_Return, Volatility_10d

Author : Ahmed Darwish
Email  : eahmeddarwish@gmail.com
GitHub : github.com/eahmeddarwish
"""

import os
import pickle
import hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF noise
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

LOOK_BACK    = 60          # days of history fed into LSTM
FORECAST_DAY = 1           # days ahead to predict
# NOTE: relative path — on Hugging Face Spaces (free tier) this directory
# lives on ephemeral storage and is wiped on every Space restart/rebuild,
# so cached models do not persist across restarts. Fine for a demo Space;
# for persistent caching, point CACHE_DIR at a mounted persistent disk or
# a Hugging Face Dataset repo instead.
CACHE_DIR    = "model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Feature columns the model is trained on
# These must exist in the dataframe AFTER add_all_indicators()
FEATURE_COLS = [
    "Close", "Volume",
    "RSI",
    "MACD", "MACD_Signal",
    "BB_Upper", "BB_Lower", "BB_Width",
    "MA_20", "MA_50",
    "ATR",
    "Daily_Return", "Volatility_10d",
    "Volume_Ratio",
    "Price_vs_MA20", "Price_vs_MA50",
]


# ─────────────────────────────────────────────
#  ARCHITECTURE
# ─────────────────────────────────────────────

def build_model(n_features: int, look_back: int = LOOK_BACK) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(look_back, n_features)),

        LSTM(128, return_sequences=True),
        Dropout(0.25),
        BatchNormalization(),

        LSTM(64, return_sequences=True),
        Dropout(0.20),
        BatchNormalization(),

        LSTM(32, return_sequences=False),
        Dropout(0.15),

        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="huber",
        metrics=["mae"],
    )
    return model


# ─────────────────────────────────────────────
#  DATA PREP
# ─────────────────────────────────────────────

def prepare_data(df: pd.DataFrame):
    """
    Scale features and build (X, y) sequences.

    IMPORTANT — no look-ahead leakage:
    The MinMaxScaler is fit ONLY on the portion of the raw series that will
    end up in the training windows. It is then applied (transform-only) to
    the full series, including the rows that fall inside the test windows.
    This mirrors how the model would behave in production, where future
    price ranges are unknown at "training time" — fitting the scaler on the
    full series (train + test) would leak information about future price
    levels into the model before that period ever hits the test set.

    Returns
    -------
    X_train, y_train, X_test, y_test,
    scaler, feature_cols, close_col_idx
    """
    # Keep only available feature cols
    available = [c for c in FEATURE_COLS if c in df.columns]
    data      = df[available].values
    close_idx = available.index("Close")

    n_samples     = len(data)
    total_windows = n_samples - LOOK_BACK - FORECAST_DAY + 1
    if total_windows < 20:
        raise ValueError(
            f"Not enough rows ({n_samples}) to build train/test windows "
            f"with a {LOOK_BACK}-day look-back. Try a longer history period."
        )
    train_windows = int(total_windows * 0.80)

    # Last raw-row index touched by any TRAINING window (input or target).
    # Everything from this point onward belongs to the test period and must
    # stay invisible to the scaler.
    raw_train_end = LOOK_BACK + train_windows + FORECAST_DAY - 1

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[:raw_train_end])   # fit on the training period only
    scaled = scaler.transform(data)    # apply that same fit to the full series

    # Build sliding-window sequences
    X, y = [], []
    for i in range(LOOK_BACK, len(scaled) - FORECAST_DAY + 1):
        X.append(scaled[i - LOOK_BACK : i])
        y.append(scaled[i + FORECAST_DAY - 1, close_idx])

    X, y = np.array(X), np.array(y)

    # 80 / 20 train-test split (no shuffle — time series!)
    X_train = X[:train_windows];  y_train = y[:train_windows]
    X_test  = X[train_windows:];  y_test  = y[train_windows:]

    return X_train, y_train, X_test, y_test, scaler, available, close_idx


def inverse_close(scaled_values: np.ndarray, scaler, n_features: int, close_idx: int) -> np.ndarray:
    """Inverse-transform only the Close price column."""
    dummy           = np.zeros((len(scaled_values), n_features))
    dummy[:, close_idx] = scaled_values.flatten()
    return scaler.inverse_transform(dummy)[:, close_idx]


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    epochs: int  = 25,
    batch_size: int = 32,
    verbose: int = 0,
    progress_cb=None,   # optional callable(fraction, description)
) -> dict:
    """
    Full training pipeline.

    Returns dict with:
        model, scaler, feature_cols, close_idx,
        test_actual, test_predicted, next_price,
        metrics, history
    """
    def _prog(frac, msg):
        if progress_cb:
            progress_cb(frac, msg)

    _prog(0.10, "⚙️  Preparing sequences…")
    X_train, y_train, X_test, y_test, scaler, feat_cols, close_idx = prepare_data(df)

    n_features = X_train.shape[2]

    _prog(0.20, "🧠  Building LSTM model…")
    model = build_model(n_features, LOOK_BACK)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    _prog(0.25, f"🏋️  Training on {len(X_train)} samples…")
    history = model.fit(
        X_train, y_train,
        epochs          = epochs,
        batch_size      = batch_size,
        validation_data = (X_test, y_test),
        callbacks       = callbacks,
        verbose         = verbose,
    )

    _prog(0.80, "🔮  Generating predictions…")

    # --- Test-set back-test ---
    test_preds_scaled = model.predict(X_test, verbose=0).flatten()
    test_actual_scaled = y_test

    test_actual    = inverse_close(test_actual_scaled,  scaler, len(feat_cols), close_idx)
    test_predicted = inverse_close(test_preds_scaled,   scaler, len(feat_cols), close_idx)

    # --- Next-day prediction ---
    last_seq          = scaler.transform(df[feat_cols].values)[-LOOK_BACK:]
    next_scaled       = model.predict(last_seq.reshape(1, LOOK_BACK, len(feat_cols)), verbose=0)
    next_price        = inverse_close(next_scaled.flatten(), scaler, len(feat_cols), close_idx)[0]

    # --- Metrics ---
    mae   = mean_absolute_error(test_actual, test_predicted)
    rmse  = np.sqrt(mean_squared_error(test_actual, test_predicted))
    r2    = r2_score(test_actual, test_predicted)
    mape  = np.mean(np.abs((test_actual - test_predicted) / (test_actual + 1e-9))) * 100

    dir_accuracy = (
        np.mean(np.sign(np.diff(test_actual)) == np.sign(np.diff(test_predicted))) * 100
        if len(test_actual) > 1
        else 50.0
    )

    _prog(1.00, "✅  Done!")

    return {
        "model"         : model,
        "scaler"        : scaler,
        "feature_cols"  : feat_cols,
        "close_idx"     : close_idx,
        "test_actual"   : test_actual,
        "test_predicted": test_predicted,
        "next_price"    : float(next_price),
        "metrics": {
            "MAE"               : round(float(mae),  4),
            "RMSE"              : round(float(rmse), 4),
            "R²"                : round(float(r2),   4),
            "MAPE (%)"          : round(float(mape), 2),
            "Direction Accuracy": round(float(dir_accuracy), 1),
            "Train Samples"     : int(len(X_train)),
            "Test  Samples"     : int(len(X_test)),
            "Epochs Run"        : int(len(history.history["loss"])),
            "Features Used"     : len(feat_cols),
        },
        "history": {
            "train_loss": [float(v) for v in history.history.get("loss", [])],
            "val_loss"  : [float(v) for v in history.history.get("val_loss", [])],
        },
    }


# ─────────────────────────────────────────────
#  DISK CACHE  (avoid re-training same ticker)
# ─────────────────────────────────────────────

def _cache_key(ticker: str, period: str, epochs: int) -> str:
    raw = f"{ticker}|{period}|{epochs}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def cache_path(ticker: str, period: str, epochs: int) -> str:
    key = _cache_key(ticker, period, epochs)
    return os.path.join(CACHE_DIR, f"{ticker.replace('/', '_')}_{key}.pkl")


def save_cache(path: str, result: dict):
    """Save training result (without Keras model — save separately)."""
    save_data = {k: v for k, v in result.items() if k != "model"}
    with open(path, "wb") as f:
        pickle.dump(save_data, f)
    result["model"].save(path.replace(".pkl", ".keras"))


def load_cache(path: str) -> dict | None:
    """Load cached result if it exists."""
    pkl_path   = path
    keras_path = path.replace(".pkl", ".keras")
    if not (os.path.exists(pkl_path) and os.path.exists(keras_path)):
        return None
    try:
        with open(pkl_path, "rb") as f:
            result = pickle.load(f)
        result["model"] = load_model(keras_path)
        return result
    except Exception:
        return None
