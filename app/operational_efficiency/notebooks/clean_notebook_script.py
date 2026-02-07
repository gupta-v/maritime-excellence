"""
Operational Efficiency - Clean Notebook Script

This is a cleaned-up Python script version of the notebook that uses
correct paths and can be run directly. It demonstrates the same functionality
as the notebook but with proper path references.

For production use, please use the pipeline scripts instead:
- pipeline/1_prepare_data.py
- pipeline/2_train_model.py  
- pipeline/3_run_inference.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "fuel_data.csv"
MODEL_PATH = BASE_DIR / "models" / "fuel_model_efficiency.pkl"

# ============================================================================
# 1. Load and Explore Data
# ============================================================================
print("="*70)
print("LOADING DATA")
print("="*70)

df = pd.read_csv(DATA_PATH)
print(f"✅ Successfully loaded data from {DATA_PATH}")
print(f"   Shape: {df.shape}\n")

print("Data Info:")
print(df.info())

print("\nUnique values per column:")
print(df.nunique())

# ============================================================================
# 2. Create Fuel Efficiency Target
# ============================================================================
print("\n" + "="*70)
print("FEATURE ENGINEERING")
print("="*70)

# Filter out zero-distance rows
df = df[df['distance'] > 0].copy()
print(f"Filtered data shape: {df.shape}")

# Create target: fuel per distance
df['fuel_per_distance'] = df['fuel_consumption'] / df['distance']
print(f"✅ Created 'fuel_per_distance' target")
print(f"   Min: {df['fuel_per_distance'].min():.4f}")
print(f"   Max: {df['fuel_per_distance'].max():.4f}")
print(f"   Mean: {df['fuel_per_distance'].mean():.4f}")

# ============================================================================
# 3. Prepare Features and Target
# ============================================================================
print("\n" + "="*70)
print("PREPARING FEATURES")
print("="*70)

target = 'fuel_per_distance'
features = [
    'ship_type', 'month', 'fuel_type', 
    'weather_conditions', 'engine_efficiency'
]

X = df[features]
y = df[target]

numeric_features = ['engine_efficiency']
categorical_features = ['ship_type', 'month', 'fuel_type', 'weather_conditions']

print(f"Features: {features}")
print(f"Target: {target}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ============================================================================
# 4. Create Preprocessing Pipeline
# ============================================================================
print("\n" + "="*70)
print("CREATING MODEL PIPELINE")
print("="*70)

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

print("✅ Model pipeline created")
print("   Preprocessor: StandardScaler + OneHotEncoder")
print("   Model: RandomForestRegressor(n_estimators=100)")

# ============================================================================
# 5. Train Model
# ============================================================================
print("\n" + "="*70)
print("TRAINING MODEL")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

print("\nFitting model...")
model_pipeline.fit(X_train, y_train)
print("✅ Model training complete")

# ============================================================================
# 6. Evaluate Model
# ============================================================================
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

y_pred = model_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# ============================================================================
# 7. Feature Importances
# ============================================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCES")
print("="*70)

model = model_pipeline.named_steps['regressor']
importances = model.feature_importances_
feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
importances_df = pd.Series(importances, index=feature_names)

print("\nTop 10 Most Important Features:")
for feature, importance in importances_df.sort_values(ascending=False).head(10).items():
    print(f"  {feature:40s} {importance:.6f}")

# ============================================================================
# 8. Save Model
# ============================================================================
print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model_pipeline, f)

print(f"✅ Model saved to {MODEL_PATH}")

# ============================================================================
# 9. Test Predictions
# ============================================================================
print("\n" + "="*70)
print("SAMPLE PREDICTIONS")
print("="*70)

test_cases = [
    {
        "ship_type": "Tanker Ship",
        "month": "January",
        "fuel_type": "HFO",
        "weather_conditions": "Stormy",
        "engine_efficiency": 90.0
    },
    {
        "ship_type": "Tanker Ship",
        "month": "January",
        "fuel_type": "HFO",
        "weather_conditions": "Calm",
        "engine_efficiency": 90.0
    },
]

for i, test_case in enumerate(test_cases, 1):
    input_df = pd.DataFrame([test_case])
    predicted_efficiency = model_pipeline.predict(input_df)[0]
    total_fuel_10km = predicted_efficiency * 10.0
    
    print(f"\nTest Case {i}:")
    print(f"  Weather: {test_case['weather_conditions']}")
    print(f"  Ship: {test_case['ship_type']}")
    print(f"  → Predicted Efficiency: {predicted_efficiency:.4f} fuel/km")
    print(f"  → Total Fuel (10km): {total_fuel_10km:.4f}")

print("\n" + "="*70)
print("✅ COMPLETE!")
print("="*70)
