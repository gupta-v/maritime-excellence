"""
Model training pipeline for RUL prediction.
Loads preprocessed data, builds LSTM model, trains with callbacks, and saves model.
"""

import os
import sys
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_utils import build_lstm_model


def train_model(X_train_path: str = "../models/X_train.npy",
               y_train_path: str = "../models/y_train.npy",
               X_test_path: str = "../models/X_test.npy",
               y_test_path: str = "../models/y_test.npy",
               model_save_path: str = "../models/rul_lstm_model.keras",
               epochs: int = 60,
               batch_size: int = 64,
               learning_rate: float = 1e-3):
    """
    Train LSTM model for RUL prediction.
    
    Args:
        X_train_path: Path to training features
        y_train_path: Path to training targets
        X_test_path: Path to test features
        y_test_path: Path to test targets
        model_save_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        
    Returns:
        Trained model and training history
    """
    print("=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)
    
    # Load preprocessed data
    print("\n[1/3] Loading preprocessed data...")
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")
    
    # Build model
    print("\n[2/3] Building LSTM model...")
    time_steps = X_train.shape[1]
    n_features = X_train.shape[2]
    
    model = build_lstm_model(
        n_features=n_features,
        time_steps=time_steps,
        learning_rate=learning_rate
    )
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\n[3/3] Training model (epochs={epochs}, batch_size={batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    print(f"\nSaving model to {model_save_path}...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final train loss: {history.history['loss'][-1]:.4f}")
    print(f"Final val loss: {history.history['val_loss'][-1]:.4f}")
    
    return model, history


if __name__ == "__main__":
    model, history = train_model()
