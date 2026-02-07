"""
Configuration file for Operational Efficiency Model

This module contains all configuration parameters for the fuel efficiency
prediction model including data paths, feature definitions, and hyperparameters.
"""

import os
from pathlib import Path

# ============================================================================
# Directory Paths
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ============================================================================
# Data Paths
# ============================================================================
FUEL_DATA_PATH = DATA_DIR / "fuel_data.csv"
MODEL_SAVE_PATH = MODEL_DIR / "fuel_model_efficiency.pkl"

# ============================================================================
# Feature Definitions
# ============================================================================
# Target variable
TARGET = "fuel_per_distance"

# Original target (used for deriving fuel_per_distance)
ORIGINAL_TARGET = "fuel_consumption"

# Numeric features
NUMERIC_FEATURES = [
    "engine_efficiency"
]

# Categorical features
CATEGORICAL_FEATURES = [
    "ship_type",
    "month",
    "fuel_type",
    "weather_conditions"
]

# All features used for model input
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Features to exclude from the dataset
EXCLUDED_FEATURES = [
    "ship_id",       # Identifier, not predictive
    "route_id",      # Identifier, not predictive
    "distance",      # Not used as feature (incorporated in target)
    "fuel_consumption",  # Original target, causes data leakage
    "CO2_emissions"  # Highly correlated with fuel consumption
]

# ============================================================================
# Model Hyperparameters
# ============================================================================
MODEL_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1
}

# ============================================================================
# Training Parameters
# ============================================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# Preprocessing Parameters
# ============================================================================
# Minimum distance threshold to avoid division by zero
MIN_DISTANCE_THRESHOLD = 0.0

# Handle unknown categories during inference
HANDLE_UNKNOWN = "ignore"
