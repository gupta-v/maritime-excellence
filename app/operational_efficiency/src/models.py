"""
Model architecture for Operational Efficiency

This module defines the fuel efficiency prediction model using RandomForest
with preprocessing pipeline (StandardScaler and OneHotEncoder).
"""

import pickle
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Optional

from config import (
    MODEL_PARAMS,
    MODEL_SAVE_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    HANDLE_UNKNOWN
)


def create_preprocessor() -> ColumnTransformer:
    """
    Create the preprocessing pipeline for numeric and categorical features.
    
    Returns:
        ColumnTransformer with StandardScaler for numeric features
        and OneHotEncoder for categorical features
    """
    # Numeric preprocessing: StandardScaler
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing: OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown=HANDLE_UNKNOWN))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    
    return preprocessor


def create_model_pipeline() -> Pipeline:
    """
    Create the complete model pipeline with preprocessing and RandomForest.
    
    Returns:
        sklearn Pipeline with preprocessor and regressor
    """
    preprocessor = create_preprocessor()
    
    # Create the full pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(**MODEL_PARAMS))
    ])
    
    print("✅ Created model pipeline:")
    print(f"   Preprocessing: StandardScaler + OneHotEncoder")
    print(f"   Model: RandomForestRegressor")
    print(f"   Hyperparameters: {MODEL_PARAMS}")
    
    return model_pipeline


def save_model(model: Pipeline, save_path: Path = MODEL_SAVE_PATH) -> None:
    """
    Save the trained model pipeline to disk.
    
    Args:
        model: Trained sklearn Pipeline
        save_path: Path where to save the model
    """
    # Ensure the directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✅ Model successfully saved to {save_path}")


def load_model(model_path: Path = MODEL_SAVE_PATH) -> Pipeline:
    """
    Load a trained model pipeline from disk.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded sklearn Pipeline
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}\n"
            f"Please train the model first using pipeline/2_train_model.py"
        )
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✅ Successfully loaded model from {model_path}")
    
    return model


def predict_fuel_efficiency(
    model: Pipeline,
    input_data: pd.DataFrame
) -> pd.Series:
    """
    Predict fuel efficiency (fuel per distance) for given input data.
    
    Args:
        model: Trained model pipeline
        input_data: DataFrame with features matching the training data format
        
    Returns:
        Series with predicted fuel efficiency values
    """
    predictions = model.predict(input_data)
    return pd.Series(predictions, index=input_data.index)


def calculate_total_fuel(
    fuel_per_distance: float,
    distance: float
) -> float:
    """
    Calculate total fuel consumption for a given distance.
    
    Args:
        fuel_per_distance: Predicted fuel efficiency (fuel/km)
        distance: Distance to travel (km)
        
    Returns:
        Total fuel consumption
    """
    return fuel_per_distance * distance
