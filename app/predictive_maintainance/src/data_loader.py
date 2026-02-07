"""
Data loading and preprocessing utilities for predictive maintenance.
"""

import pandas as pd
from typing import List, Tuple


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    df = pd.read_csv(filepath)
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by removing extra spaces and non-breaking spaces.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned column names
    """
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("\xa0", " ", regex=True)
    return df


def detect_decay_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect columns containing 'decay' in their name (case-insensitive).
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of column names containing 'decay'
    """
    decay_cols = [col for col in df.columns if "decay" in col.lower()]
    return decay_cols


def load_and_prepare_data(filepath: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load data and perform initial preprocessing.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Tuple of (cleaned DataFrame, list of decay column names)
    """
    # Load data
    df = load_data(filepath)
    
    # Clean column names
    df = clean_column_names(df)
    
    # Detect decay columns
    decay_cols = detect_decay_columns(df)
    
    print(f"Loaded data with shape: {df.shape}")
    print(f"Detected {len(decay_cols)} decay columns: {decay_cols}")
    
    return df, decay_cols
