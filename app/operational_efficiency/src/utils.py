"""
Utility functions for Operational Efficiency Model

This module provides helper functions for model evaluation, feature importance
analysis, and result formatting.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, Tuple


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model pipeline
        X_test: Test feature matrix
        y_test: Test target vector
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mse': mean_squared_error(y_test, y_pred)
    }
    
    return metrics


def print_evaluation_metrics(metrics: Dict[str, float]) -> None:
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics: Dictionary with metric names and values
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"R-squared (R²):           {metrics['r2']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    print(f"Root Mean Squared Error:   {metrics['rmse']:.4f}")
    print(f"Mean Squared Error:        {metrics['mse']:.4f}")
    print("="*60)


def get_feature_importances(model: Pipeline) -> pd.Series:
    """
    Extract feature importances from the trained model.
    
    Args:
        model: Trained model pipeline
        
    Returns:
        Series with feature names and their importance scores
    """
    # Get the regressor from the pipeline
    regressor = model.named_steps['regressor']
    
    # Get raw importance scores
    importances = regressor.feature_importances_
    
    # Get feature names after preprocessing (including one-hot encoded features)
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    
    # Create a Series and sort by importance
    importance_series = pd.Series(importances, index=feature_names)
    importance_series = importance_series.sort_values(ascending=False)
    
    return importance_series


def print_feature_importances(
    importances: pd.Series,
    top_n: int = 10
) -> None:
    """
    Print the top N most important features.
    
    Args:
        importances: Series with feature importances
        top_n: Number of top features to display
    """
    print("\n" + "="*60)
    print(f"TOP {top_n} FEATURE IMPORTANCES")
    print("="*60)
    
    for feature, importance in importances.head(top_n).items():
        print(f"{feature:40s} {importance:.6f}")
    
    print("="*60)


def format_prediction_result(
    ship_type: str,
    weather: str,
    engine_eff: float,
    fuel_per_km: float,
    distance: float = 10.0
) -> str:
    """
    Format a prediction result for display.
    
    Args:
        ship_type: Type of ship
        weather: Weather condition
        engine_eff: Engine efficiency percentage
        fuel_per_km: Predicted fuel efficiency (fuel/km)
        distance: Distance for total fuel calculation
        
    Returns:
        Formatted string with prediction results
    """
    total_fuel = fuel_per_km * distance
    
    result = [
        f"  Input: {weather} | {ship_type} | {engine_eff}% eff",
        f"  → Predicted Efficiency (Fuel/km): {fuel_per_km:.4f}",
        f"  → Total Fuel for {distance}km step: {total_fuel:.4f}"
    ]
    
    return "\n".join(result)


def validate_input_data(
    df: pd.DataFrame,
    required_features: list
) -> Tuple[bool, str]:
    """
    Validate that input data has all required features.
    
    Args:
        df: Input DataFrame
        required_features: List of required feature names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_features = set(required_features) - set(df.columns)
    
    if missing_features:
        error_msg = f"Missing required features: {missing_features}"
        return False, error_msg
    
    return True, ""


def print_section_header(title: str, width: int = 60) -> None:
    """
    Print a formatted section header.
    
    Args:
        title: Title text
        width: Width of the header
    """
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width)
