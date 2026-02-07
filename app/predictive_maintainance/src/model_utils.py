"""
Model building and utility functions for LSTM-based RUL prediction.
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from typing import Tuple


# Default hyperparameters
TIME_STEPS = 50
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.2


def create_sequences(X: np.ndarray, 
                     y: np.ndarray, 
                     time_steps: int = TIME_STEPS,
                     step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for LSTM input.
    
    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Target array of shape (n_samples, 1) or (n_samples,)
        time_steps: Number of time steps in each sequence
        step: Step size for sliding window
        
    Returns:
        Tuple of (X_sequences, y_sequences)
        - X_sequences: shape (n_sequences, time_steps, n_features)
        - y_sequences: shape (n_sequences, 1)
    """
    Xs, ys = [], []
    
    for i in range(0, len(X) - time_steps, step):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])  # Predict RUL at t+time_steps
    
    return np.array(Xs), np.array(ys)


def build_lstm_model(n_features: int,
                     time_steps: int = TIME_STEPS,
                     lstm_units_1: int = LSTM_UNITS_1,
                     lstm_units_2: int = LSTM_UNITS_2,
                     dense_units: int = DENSE_UNITS,
                     dropout_rate: float = DROPOUT_RATE,
                     learning_rate: float = 1e-3) -> Sequential:
    """
    Build LSTM model for RUL prediction.
    
    Args:
        n_features: Number of input features
        time_steps: Number of time steps in sequences
        lstm_units_1: Units in first LSTM layer
        lstm_units_2: Units in second LSTM layer
        dense_units: Units in dense layer
        dropout_rate: Dropout rate
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        LSTM(lstm_units_1, return_sequences=True, input_shape=(time_steps, n_features)),
        Dropout(dropout_rate),
        LSTM(lstm_units_2),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dense(1, activation='linear')  # Linear activation for regression
    ])
    
    model.compile(
        optimizer='adam',
        loss='mae',
        metrics=['mae']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    return model


def train_test_split_temporal(X_seq: np.ndarray,
                              y_seq: np.ndarray,
                              test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform time-ordered train-test split (no shuffling).
    
    Args:
        X_seq: Sequence features
        y_seq: Sequence targets
        test_size: Fraction of data to use for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    split_idx = int((1 - test_size) * len(X_seq))
    
    X_train = X_seq[:split_idx]
    X_test = X_seq[split_idx:]
    y_train = y_seq[:split_idx]
    y_test = y_seq[split_idx:]
    
    print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test
