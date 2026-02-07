"""
Data utilities for Operational Efficiency Model

This module provides functions for loading, preprocessing, and preparing
data for the fuel efficiency prediction model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple

from config import (
    FUEL_DATA_PATH,
    TARGET,
    ORIGINAL_TARGET,
    ALL_FEATURES,
    MIN_DISTANCE_THRESHOLD,
    TEST_SIZE,
    RANDOM_STATE
)


def load_fuel_data(data_path: Path = FUEL_DATA_PATH) -> pd.DataFrame:
    """
    Load fuel efficiency data from CSV file.
    
    Args:
        data_path: Path to the fuel data CSV file
        
    Returns:
        DataFrame containing the fuel efficiency data
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If the required columns are missing
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {data_path}\n"
            f"Please ensure the fuel data CSV is available."
        )
    
    df = pd.read_csv(data_path)
    print(f"✅ Successfully loaded data from {data_path}")
    print(f"   Shape: {df.shape}")
    
    # Validate required columns
    required_columns = ALL_FEATURES + [ORIGINAL_TARGET, "distance"]
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for the model.
    
    This function creates the target variable 'fuel_per_distance' by dividing
    fuel consumption by distance, and filters out invalid rows.
    
    Args:
        df: Input DataFrame with raw data
        
    Returns:
        DataFrame with engineered features
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Filter out rows where distance is too small
    initial_count = len(df)
    df = df[df['distance'] > MIN_DISTANCE_THRESHOLD].copy()
    filtered_count = initial_count - len(df)
    
    if filtered_count > 0:
        print(f"⚠️  Filtered out {filtered_count} rows with distance <= {MIN_DISTANCE_THRESHOLD}")
    
    # Create the target variable: fuel consumption per unit distance
    df[TARGET] = df[ORIGINAL_TARGET] / df['distance']
    
    print(f"✅ Created target variable '{TARGET}'")
    print(f"   Min: {df[TARGET].min():.4f}")
    print(f"   Max: {df[TARGET].max():.4f}")
    print(f"   Mean: {df[TARGET].mean():.4f}")
    print(f"   Median: {df[TARGET].median():.4f}")
    
    return df


def prepare_features_target(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features (X) and target (y) from the DataFrame.
    
    Args:
        df: DataFrame with engineered features
        
    Returns:
        Tuple of (X, y) where X is the feature DataFrame and y is the target Series
    """
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    
    print(f"\n📊 Feature Matrix Shape: {X.shape}")
    print(f"📊 Target Vector Shape: {y.shape}")
    
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\n📊 Training set size: {len(X_train)}")
    print(f"📊 Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def get_data_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of the dataset.
    
    Args:
        df: DataFrame to summarize
    """
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nUnique values per column:")
    print(df.nunique())
    print("\n" + "="*60)
