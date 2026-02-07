"""
Model evaluation pipeline for RUL prediction.
Loads trained model, makes predictions, calculates metrics, and generates plots.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate_model(model_path: str = "../models/rul_lstm_model.keras",
                  X_test_path: str = "../models/X_test.npy",
                  y_test_path: str = "../models/y_test.npy",
                  rul_scaler_path: str = "../models/rul_scaler.save",
                  plot_results: bool = True):
    """
    Evaluate trained model on test data.
    
    Args:
        model_path: Path to trained model
        X_test_path: Path to test features
        y_test_path: Path to test targets
        rul_scaler_path: Path to RUL scaler for inverse transform
        plot_results: Whether to generate plots
        
    Returns:
        Dictionary with metrics and predictions
    """
    print("=" * 60)
    print("EVALUATION PIPELINE")
    print("=" * 60)
    
    # Load model
    print("\n[1/4] Loading trained model...")
    model = load_model(model_path)
    
    # Load test data
    print("\n[2/4] Loading test data...")
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    print(f"Test data shape: {X_test.shape}")
    
    # Load RUL scaler
    print("\n[3/4] Loading RUL scaler...")
    with open(rul_scaler_path, "rb") as f:
        rul_scaler = pickle.load(f)
    
    # Make predictions
    print("\n[4/4] Making predictions and calculating metrics...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform to get actual RUL values
    y_pred = rul_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = rul_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Prediction range: [{y_pred.min():.0f}, {y_pred.max():.0f}]")
    print(f"True RUL range: [{y_true.min():.0f}, {y_true.max():.0f}]")
    
    # Plot results
    if plot_results:
        print("\nGenerating plots...")
        
        # True vs Predicted plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 
                'r--', lw=2, label='Perfect prediction')
        plt.xlabel('True RUL')
        plt.ylabel('Predicted RUL')
        plt.title('True vs Predicted RUL')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Time series plot
        plt.subplot(1, 2, 2)
        sample_size = min(500, len(y_true))
        plt.plot(y_true[:sample_size], label='True RUL', alpha=0.7)
        plt.plot(y_pred[:sample_size], label='Predicted RUL', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('RUL')
        plt.title(f'RUL Predictions (first {sample_size} samples)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../models/evaluation_results.png', dpi=150)
        print("Saved plot to ../models/evaluation_results.png")
        plt.show()
    
    results = {
        'mae': mae,
        'rmse': rmse,
        'y_true': y_true,
        'y_pred': y_pred
    }
    
    return results


if __name__ == "__main__":
    results = evaluate_model()
