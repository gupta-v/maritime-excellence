"""
Data preprocessing pipeline for RUL prediction.
Loads data, engineers features, scales, creates sequences, and saves preprocessed data.
"""

import os
import sys
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_and_prepare_data
from src.feature_engineering import calculate_health_index, map_health_to_rul, select_features
from src.model_utils import create_sequences, train_test_split_temporal


def preprocess_data(data_path: str = "../data/data.csv",
                   max_rul: int = 5000,
                   time_steps: int = 50,
                   test_size: float = 0.2,
                   save_scalers: bool = True,
                   scaler_dir: str = "../models"):
    """
    Complete preprocessing pipeline.
    
    Args:
        data_path: Path to input CSV data
        max_rul: Maximum RUL value for mapping
        time_steps: Number of time steps for sequences
        test_size: Fraction for test set
        save_scalers: Whether to save fitted scalers
        scaler_dir: Directory to save scalers
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, features)
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print("\n[1/7] Loading and preparing data...")
    df, decay_cols = load_and_prepare_data(data_path)
    
    # Step 2: Calculate health index
    print("\n[2/7] Calculating health index...")
    df = calculate_health_index(df, decay_cols)
    
    # Step 3: Map health to RUL
    print("\n[3/7] Mapping health to RUL...")
    df = map_health_to_rul(df, max_rul=max_rul)
    
    # Step 4: Select features
    print("\n[4/7] Selecting features...")
    features = select_features(df, exclude_columns=decay_cols)
    
    # Step 5: Scale features and target
    print("\n[5/7] Scaling features and target...")
    feature_scaler = MinMaxScaler()
    X_all = feature_scaler.fit_transform(df[features].values)
    
    rul_scaler = MinMaxScaler()
    y_all = rul_scaler.fit_transform(df[["RUL"]].values)
    
    print(f"Feature matrix shape: {X_all.shape}")
    print(f"RUL shape: {y_all.shape}")
    
    # Save scalers
    if save_scalers:
        os.makedirs(scaler_dir, exist_ok=True)
        with open(os.path.join(scaler_dir, "feature_scaler.save"), "wb") as f:
            pickle.dump(feature_scaler, f)
        with open(os.path.join(scaler_dir, "rul_scaler.save"), "wb") as f:
            pickle.dump(rul_scaler, f)
        print(f"Saved scalers to {scaler_dir}/")
    
    # Step 6: Create sequences
    print(f"\n[6/7] Creating sequences (time_steps={time_steps})...")
    X_seq, y_seq = create_sequences(X_all, y_all, time_steps=time_steps)
    print(f"Sequence shapes: X={X_seq.shape}, y={y_seq.shape}")
    
    # Step 7: Train-test split
    print(f"\n[7/7] Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split_temporal(
        X_seq, y_seq, test_size=test_size
    )
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(features)}")
    
    return X_train, X_test, y_train, y_test, features


if __name__ == "__main__":
    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test, features = preprocess_data()
    
    # Save preprocessed data
    print("\nSaving preprocessed data...")
    np.save("../models/X_train.npy", X_train)
    np.save("../models/X_test.npy", X_test)
    np.save("../models/y_train.npy", y_train)
    np.save("../models/y_test.npy", y_test)
    
    with open("../models/features.txt", "w") as f:
        f.write("\n".join(features))
    
    print("Preprocessed data saved to ../models/")
