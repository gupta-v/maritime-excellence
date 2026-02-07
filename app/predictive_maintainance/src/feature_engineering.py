"""
Feature engineering utilities for predictive maintenance.
Includes health index calculation and RUL mapping.
"""

import pandas as pd
import numpy as np
from typing import List


# Default maximum RUL value
MAX_RUL = 5000


def calculate_health_index(df: pd.DataFrame, decay_columns: List[str]) -> pd.DataFrame:
    """
    Calculate health index from decay coefficients.
    
    Args:
        df: Input DataFrame
        decay_columns: List of column names containing decay coefficients
        
    Returns:
        DataFrame with added 'health_index_raw' and 'health_index' columns
    """
    # Calculate raw health index as mean of decay columns
    df["health_index_raw"] = df[decay_columns].mean(axis=1)
    
    # Normalize to [0, 1] range (0 = worst, 1 = best)
    h_min = df["health_index_raw"].min()
    h_max = df["health_index_raw"].max()
    
    print(f"Health index raw range: [{h_min:.4f}, {h_max:.4f}]")
    
    df["health_index"] = (df["health_index_raw"] - h_min) / (h_max - h_min + 1e-12)
    
    return df


def map_health_to_rul(df: pd.DataFrame, max_rul: int = MAX_RUL) -> pd.DataFrame:
    """
    Map health index to Remaining Useful Life (RUL).
    
    Args:
        df: Input DataFrame with 'health_index' column
        max_rul: Maximum RUL value (default: 5000)
        
    Returns:
        DataFrame with added 'RUL' column
    """
    # RUL decreases as health decreases
    # health 1.0 -> MAX_RUL, health 0.0 -> 0
    df["RUL"] = (df["health_index"] * max_rul).round().astype(float)
    
    print(f"RUL range: [{df['RUL'].min():.0f}, {df['RUL'].max():.0f}]")
    
    return df


def select_features(df: pd.DataFrame, 
                     feature_keywords: List[str] = None,
                     exclude_columns: List[str] = None) -> List[str]:
    """
    Select feature columns based on keywords.
    
    Args:
        df: Input DataFrame
        feature_keywords: List of keywords to match in column names
        exclude_columns: List of columns to exclude (e.g., decay columns, RUL)
        
    Returns:
        List of selected feature column names
    """
    if feature_keywords is None:
        feature_keywords = ["T", "P", "mf", "GTn", "GGn", "GTT", "Torque", "speed", "v", "Lever", "TIC"]
    
    if exclude_columns is None:
        exclude_columns = []
    
    # Find columns matching keywords
    features = []
    for col in df.columns:
        # Check if column matches any keyword
        if any(kw.lower() in col.lower() for kw in feature_keywords):
            # Exclude specified columns
            if col not in exclude_columns and \
               "rul" not in col.lower() and \
               "health" not in col.lower():
                features.append(col)
    
    # Remove duplicates while preserving order
    features = list(dict.fromkeys(features))
    
    # Fallback: use all numeric columns if no features found
    if len(features) == 0:
        print("Warning: No features found with keywords, using all numeric columns")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        features = [c for c in numeric_cols 
                   if c not in exclude_columns + ["RUL", "health_index", "health_index_raw"]]
    
    print(f"Selected {len(features)} features")
    print(f"Sample features: {features[:10]}")
    
    return features
