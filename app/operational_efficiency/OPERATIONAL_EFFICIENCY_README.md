# Operational Efficiency Module Documentation

## To be inserted into README.md after line 809 (before "Setup and Usage" section)

---

## Operational Efficiency Module

### Overview

The Operational Efficiency module provides AI-powered fuel consumption prediction and optimization for maritime vessels. Using machine learning on historical fuel efficiency data, this module enables operators to:

- Predict fuel consumption based on vessel characteristics and environmental conditions
- Optimize operational parameters for fuel efficiency
- Analyze the impact of weather conditions on fuel consumption
- Make data-driven decisions about route and vessel operations

**Key Features:**

- RandomForest-based prediction model with scikit-learn preprocessing pipeline
- Feature importance analysis to understand fuel consumption drivers
- Production-ready modular architecture with `src/` and `pipeline/` structure
- Comprehensive data validation and error handling

### Model Architecture

#### Prediction Target

**Fuel Efficiency (`fuel_per_distance`)**: Fuel consumption per unit distance (fuel/km), calculated as:

```
fuel_per_distance = fuel_consumption / distance
```

This normalized metric allows fair comparison across different journey lengths and provides a direct measure of operational efficiency.

#### Input Features

**Categorical Features:**

- `ship_type`: Vessel classification (Tanker Ship, Surfer Boat, Fishing Trawler, Oil Service Boat)
- `month`: Temporal patterns (January-December)
- `fuel_type`: Fuel classification (Diesel, HFO)
- `weather_conditions`: Environmental state (Calm, Moderate, Stormy)

**Numeric Features:**

- `engine_efficiency`: Engine performance rating (percentage)

#### Model Performance

The trained RandomForestRegressor achieves strong predictive performance:

```
R² Score: 0.8329
Mean Absolute Error: 3.3205 fuel units/km
```

**Feature Importance Analysis:**

```
Top Contributing Features:
1. ship_type (Surfer Boat):     45.29%
2. ship_type (Tanker Ship):     35.61%
3. engine_efficiency:            8.67%
4. ship_type (Fishing Trawler):  1.82%
5. weather_conditions:           ~1.6% combined
```

**Key Insights:**

- **Ship type** is the dominant factor in fuel efficiency (>80% combined importance)
- **Engine efficiency** contributes meaningfully (~9%)
- **Weather conditions** have measurable but smaller impact
- **Month and fuel type** show minor seasonal/operational effects

### Technical Implementation

#### Preprocessing Pipeline

```python
ColumnTransformer:
  - StandardScaler for numeric features (engine_efficiency)
  - OneHotEncoder for categorical features (ship_type, month, fuel_type, weather)
```

#### Model Hyperparameters

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

### Directory Structure

```
operational_efficiency/
├── data/
│   └── fuel_data.csv              # Historical fuel consumption data
├── models/
│   └── fuel_model_efficiency.pkl  # Trained model pipeline
├── notebooks/
│   ├── data.ipynb                 # Original exploration notebook
│   ├── clean_notebook_script.py   # Clean Python version
│   └── NOTEBOOK_FIXES.md          # Path corrections reference
├── src/                           # Production source code
│   ├── config.py                  # Configuration and parameters
│   ├── models.py                  # Model architecture and utilities
│   ├── data_utils.py              # Data loading and preprocessing
│   └── utils.py                   # Evaluation and formatting utilities
└── pipeline/                      # Execution scripts
    ├── 1_prepare_data.py          # Data preparation pipeline
    ├── 2_train_model.py           # Model training pipeline
    └── 3_run_inference.py         # Inference and analysis
```

### Usage Instructions

#### 1. Data Preparation

```bash
cd app/operational_efficiency
py pipeline/1_prepare_data.py
```

**Output:**

- Loads and validates fuel efficiency data
- Creates `fuel_per_distance` target variable
- Displays comprehensive data summary

#### 2. Model Training

```bash
py pipeline/2_train_model.py
```

**Output:**

- Trains RandomForest model with preprocessing pipeline
- Evaluates on test set (R² and MAE metrics)
- Displays feature importance analysis
- Saves trained model to `models/fuel_model_efficiency.pkl`

**Training Time:** ~2-5 seconds on modern hardware

#### 3. Run Inference

```bash
py pipeline/3_run_inference.py
```

**Output:**

- Loads trained model
- Runs predictions on test cases
- Performs weather impact analysis
- Displays formatted prediction results

**Example Output:**

```
Test Case: Tanker in Storm
  Ship Type:          Tanker Ship
  Weather:            Stormy
  Engine Efficiency:  90.0%

  → Predicted Efficiency:  38.9880 fuel/km
  → Total Fuel (10km):     389.8800 units
```

### Weather Impact Analysis

The module includes automated weather impact comparison:

```python
Weather Conditions Impact (Tanker Ship, HFO fuel, 90% efficiency):
  Calm:     41.17 fuel/km (baseline)
  Moderate: 37.80 fuel/km (-8.17% vs Calm)
  Stormy:   38.99 fuel/km (-5.30% vs Calm)
```

**Insight:** Weather conditions show counter-intuitive patterns that may reflect optimized speed reductions or route adaptations in adverse conditions.

### Integration with Other Modules

The operational efficiency model can enhance route optimization by:

1. **Cost-Based Routing**: Incorporate fuel consumption predictions into route planning algorithms
2. **Vessel-Specific Optimization**: Use vessel type characteristics for personalized route recommendations
3. **Weather-Responsive Operations**: Adjust operational parameters based on forecast conditions
4. **Multi-Objective Optimization**: Balance safety (route_optimization) with efficiency (operational_efficiency)

### Research and Development

**Current Capabilities:**

- Static fuel efficiency prediction
- Single-vessel optimization
- Historical data analysis

**Future Enhancements:**

- Real-time operational optimization during voyage
- Integration with live fuel price data
- Multi-vessel fleet optimization
- Predictive maintenance integration based on efficiency degradation
- Carbon emission calculation and reporting

### Configuration

All module parameters are centralized in `src/config.py`:

```python
# Model hyperparameters
MODEL_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1
}

# Feature definitions
NUMERIC_FEATURES = ["engine_efficiency"]
CATEGORICAL_FEATURES = ["ship_type", "month", "fuel_type", "weather_conditions"]

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

### API Reference

#### Key Functions

**data_utils.py:**

```python
load_fuel_data(data_path) -> pd.DataFrame
engineer_features(df) -> pd.DataFrame
prepare_features_target(df) -> Tuple[pd.DataFrame, pd.Series]
split_train_test(X, y) -> Tuple[...]
```

**models.py:**

```python
create_model_pipeline() -> Pipeline
save_model(model, save_path) -> None
load_model(model_path) -> Pipeline
predict_fuel_efficiency(model, input_data) -> pd.Series
```

**utils.py:**

```python
evaluate_model(model, X_test, y_test) -> Dict[str, float]
get_feature_importances(model) -> pd.Series
print_evaluation_metrics(metrics) -> None
```

### Data Requirements

**Input Data Format (`fuel_data.csv`):**

| Column             | Type    | Description            |
| ------------------ | ------- | ---------------------- |
| ship_id            | object  | Vessel identifier      |
| ship_type          | object  | Vessel classification  |
| route_id           | object  | Route identifier       |
| month              | object  | Month of operation     |
| distance           | float64 | Journey distance (km)  |
| fuel_type          | object  | Fuel classification    |
| fuel_consumption   | float64 | Total fuel consumed    |
| CO2_emissions      | float64 | Carbon emissions       |
| weather_conditions | object  | Weather state          |
| engine_efficiency  | float64 | Engine performance (%) |

**Dataset Statistics:**

- Rows: 1,440 vessel journeys
- Ships: 120 unique vessels
- Ship Types: 4 categories
- Weather Conditions: 3 states

---
