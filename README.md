# Maritime Excellence Platform

> **Version 1.0**

A comprehensive AI-powered maritime management system integrating three advanced modules for route optimization, predictive maintenance, and operational efficiency.

## Overview

The Maritime Excellence Platform combines cutting-edge machine learning and graph neural network technologies to optimize maritime operations across multiple dimensions:

- **Route Optimization**: Weather-aware intelligent routing using Graph Neural Networks and Reinforcement Learning
- **Predictive Maintenance**: RUL (Remaining Useful Life) prediction for equipment using LSTM neural networks
- **Operational Efficiency**: Fuel consumption prediction and optimization using machine learning

## System Architecture

```
maritime-excellence/
├── app/
│   ├── route_optimization/         # GNN-based route planning
│   ├── predictive_maintainance/    # LSTM-based RUL prediction
│   └── operational_efficiency/     # Fuel efficiency optimization
├── documentation/
│   ├── route-and-operations.md     # Route optimization docs
│   └── predictive-maintenance.md   # Predictive maintenance docs
└── requirements.txt                # Project dependencies
```

---

## Module 1: Route Optimization

### Overview

An advanced maritime routing system that combines **Graph Neural Networks (GNN)** with **Reinforcement Learning (RL)** to create weather-aware, safe, and efficient vessel routes. The system learns from historical AIS (Automatic Identification System) data and optimizes for both safety and efficiency.

### Key Features

- **Two-Stage Learning Pipeline**: Imitation learning from historical trajectories followed by RL optimization
- **Weather Integration**: Real-time oceanographic data (wave height, period, ocean currents) as node features
- **Graph-Based Environment**: Maritime space modeled as navigable network for ML applications
- **AIS Data Processing**: Comprehensive pipeline for processing large-scale vessel tracking data

### Technical Approach

#### Stage 1: Graph Construction
- Discrete maritime graph with 0.05° resolution (~5.5km spacing)
- 8-directional connectivity for flexible routing
- Landmass exclusion using Natural Earth shapefiles
- Weather-enriched node features

#### Stage 2: Imitation Learning
- **GNNImitator**: Graph Convolutional Network learns navigation patterns from historical AIS trajectories
- Multi-class classification predicting next waypoint
- Training: 100 epochs, achieving loss reduction from 3.11 → 2.87

#### Stage 3: Reinforcement Learning
- **GNN_QNetwork**: Deep Q-Learning for policy optimization
- Custom OpenAI Gym environment (`VesselNavigationEnv`)
- Multi-objective reward function balancing progress, safety, and efficiency
- Training: 4,500 episodes with experience replay and target networks

### Model Performance

**Imitation Learning Results:**
- Training Dataset: 16,562 samples from 997 trajectories
- Final Loss: 2.8722
- Successfully learned spatial navigation patterns

**Reinforcement Learning Results:**
- Success Rate: **100%** on test routes
- Path Length: **2.4 steps average** (82-92% shorter than historical routes)
- Episode Training Time: ~3.5 hours for 4,500 episodes

### Directory Structure

```
route_optimization/
├── data/                      # AIS and shapefile data
├── models/                    # Trained GNN models
├── notebooks/                 # Analysis notebooks
├── output/                    # Route visualizations
├── pipeline/
│   ├── 1_create_graph.py     # Sea graph construction
│   ├── 2_prepare_data.py     # AIS data processing
│   ├── 3_train_model.py      # IL + RL training
│   └── 4_run_inference.py    # Route generation
└── src/
    ├── config.py             # Configuration parameters
    ├── graph_builder.py      # Graph construction utilities
    ├── data_utils.py         # AIS data processing
    ├── environment.py        # RL environment
    ├── models.py             # GNN architectures
    └── utils.py              # Evaluation utilities
```

### Usage

```bash
cd app/route_optimization

# 1. Create maritime graph
python pipeline/1_create_graph.py

# 2. Process AIS data
python pipeline/2_prepare_data.py

# 3. Train models (Imitation + RL)
python pipeline/3_train_model.py

# 4. Generate optimized routes
python pipeline/4_run_inference.py
```

### Key Technologies

- **Graph Processing**: NetworkX, GeoPandas
- **Machine Learning**: PyTorch, PyTorch Geometric
- **RL Framework**: Gymnasium
- **Weather Data**: Open-Meteo Marine API
- **Visualization**: Matplotlib, Plotly

---

## Module 2: Predictive Maintenance

### Overview

A machine learning system for predicting **Remaining Useful Life (RUL)** of maritime equipment using **LSTM (Long Short-Term Memory)** neural networks. The system analyzes sensor data to predict equipment failures, enabling proactive maintenance scheduling.

### Key Capabilities

1. **Health Index Calculation**: Derives equipment health scores (0-1) from decay coefficient measurements
2. **RUL Prediction**: Maps health indices to actual remaining operational hours using deep learning
3. **Time Series Analysis**: Processes sequential sensor readings to capture temporal degradation patterns
4. **Real-time Monitoring**: Continuous assessment of equipment condition

### How It Works

#### Data Processing
- Loads equipment sensor data (temperatures, pressures, flow rates, speeds, torque)
- Identifies decay coefficient columns indicating equipment degradation
- Normalizes and cleans column names

#### Feature Engineering
- **Health Index**: Normalized score (0-1) computed from mean of decay measurements
  - Health 1.0 = Healthy equipment
  - Health 0.0 = Critical condition
- **RUL Mapping**: Linear conversion from health index to operational hours
  - Health 1.0 → MAX_RUL (default: 5000 hours)
  - Health 0.0 → 0 hours
- **Sensor Features**: 16 features including temperature, pressure, flow rates, turbine metrics

#### Model Architecture

**LSTM Neural Network:**
```
Input: (50 timesteps, 16 sensor features)
├── LSTM Layer 1: 128 units, return_sequences=True
├── Dropout: 0.2
├── LSTM Layer 2: 64 units
├── Dropout: 0.2
├── Dense: 32 units, ReLU activation
└── Output: 1 unit (RUL prediction)
```

#### Training & Validation
- Time-ordered train/test split (preserves temporal ordering)
- Early stopping to prevent overfitting
- Dynamic learning rate adjustment
- Mean Absolute Error (MAE) optimization

### Model Performance

- **MAE**: ~165-200 on test set
- **RMSE**: ~270-300 on test set
- Successfully captures equipment degradation patterns

### Directory Structure

```
predictive_maintainance/
├── data/
│   └── data.csv                    # Sensor datasets
├── models/
│   ├── rul_lstm_model.keras       # Trained LSTM model
│   ├── rul_scaler.save            # RUL scaler
│   └── feature_scaler.save        # Feature scaler
├── notebooks/
│   ├── eda.ipynb                  # Exploratory analysis
│   ├── monitoring.ipynb           # Monitoring dashboard
│   └── rul.ipynb                  # Original RUL notebook
├── plots/                          # Visualization outputs
├── src/
│   ├── data_loader.py             # Data loading utilities
│   ├── feature_engineering.py     # Health index & RUL calculation
│   └── model_utils.py             # LSTM model building
└── pipeline/
    ├── preprocess.py              # Data preprocessing
    ├── train.py                   # Model training
    └── evaluate.py                # Model evaluation
```

### Usage

```bash
cd app/predictive_maintainance

# Option 1: Run Complete Pipeline
cd pipeline
python preprocess.py
python train.py
python evaluate.py

# Option 2: Use as Library
from src.data_loader import load_and_prepare_data
from src.feature_engineering import calculate_health_index, map_health_to_rul
from src.model_utils import build_lstm_model

# Load data
df, decay_cols = load_and_prepare_data("data/data.csv")

# Calculate health and RUL
df = calculate_health_index(df, decay_cols)
df = map_health_to_rul(df, max_rul=5000)

# Build and train model
model = build_lstm_model(n_features=16, time_steps=50)
```

### Configuration Parameters

```python
MAX_RUL = 5000              # Maximum RUL value (hours)
TIME_STEPS = 50             # Sequence length for LSTM
LSTM_UNITS_1 = 128         # First LSTM layer units
LSTM_UNITS_2 = 64          # Second LSTM layer units
DROPOUT_RATE = 0.2         # Dropout rate
```

### Real-World Applications

- **Manufacturing**: Predict machine tool wear and schedule maintenance
- **Aviation**: Monitor turbine engines for proactive servicing
- **Energy**: Track power generation equipment degradation
- **Transportation**: Assess vehicle component lifespans

---

## Module 3: Operational Efficiency

### Overview

AI-powered **fuel consumption prediction and optimization** system for maritime vessels. Using **RandomForest regression** on historical fuel efficiency data, this module enables data-driven decisions about vessel operations and route planning.

### Key Features

- **Fuel Efficiency Prediction**: Predict fuel consumption based on vessel characteristics and environmental conditions
- **Weather Impact Analysis**: Analyze how weather conditions affect fuel consumption
- **Feature Importance**: Understand key drivers of fuel consumption
- **Production-Ready**: Modular architecture with comprehensive validation

### Model Architecture

#### Prediction Target

**Fuel Efficiency (`fuel_per_distance`)**: 
```
fuel_per_distance = fuel_consumption / distance (fuel units/km)
```

This normalized metric enables fair comparison across different journey lengths.

#### Input Features

**Categorical Features:**
- `ship_type`: Vessel classification (Tanker Ship, Surfer Boat, Fishing Trawler, Oil Service Boat)
- `month`: Temporal patterns (January-December)
- `fuel_type`: Fuel classification (Diesel, HFO)
- `weather_conditions`: Environmental state (Calm, Moderate, Stormy)

**Numeric Features:**
- `engine_efficiency`: Engine performance rating (percentage)

#### Preprocessing Pipeline

```python
ColumnTransformer:
  - StandardScaler for numeric features (engine_efficiency)
  - OneHotEncoder for categorical features (ship_type, month, fuel_type, weather)

RandomForestRegressor:
  - n_estimators: 100
  - random_state: 42
  - n_jobs: -1 (parallel processing)
```

### Model Performance

```
R² Score: 0.8329
Mean Absolute Error: 3.3205 fuel units/km
```

**Feature Importance:**
1. Ship Type (Surfer Boat): 45.29%
2. Ship Type (Tanker Ship): 35.61%
3. Engine Efficiency: 8.67%
4. Ship Type (Fishing Trawler): 1.82%
5. Weather Conditions: ~1.6% combined

**Key Insight**: Ship type is the dominant factor (>80% combined importance), followed by engine efficiency.

### Directory Structure

```
operational_efficiency/
├── data/
│   └── fuel_data.csv                  # Historical fuel data
├── models/
│   └── fuel_model_efficiency.pkl      # Trained model pipeline
├── notebooks/
│   ├── data.ipynb                     # Exploration notebook
│   └── clean_notebook_script.py       # Clean Python version
├── src/
│   ├── config.py                      # Configuration
│   ├── models.py                      # Model utilities
│   ├── data_utils.py                  # Data preprocessing
│   └── utils.py                       # Evaluation utilities
└── pipeline/
    ├── 1_prepare_data.py              # Data preparation
    ├── 2_train_model.py               # Model training
    └── 3_run_inference.py             # Inference and analysis
```

### Usage

```bash
cd app/operational_efficiency

# 1. Prepare data
py pipeline/1_prepare_data.py

# 2. Train model
py pipeline/2_train_model.py

# 3. Run inference
py pipeline/3_run_inference.py
```

**Training Time**: ~2-5 seconds on modern hardware

### Dataset Information

- **Rows**: 1,440 vessel journeys
- **Unique Vessels**: 120 ships
- **Ship Types**: 4 categories
- **Weather Conditions**: 3 states

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for route optimization module)
- 16GB+ RAM (for processing large AIS datasets)

### Installation Steps

1. **Clone the repository**
```bash
cd d:\Downloads\be-proj\maritime-excellence
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Core Dependencies

```
# Scientific Computing
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Geospatial Processing
geopandas>=0.9.0
shapely>=1.7.0

# Graph Processing
networkx>=2.6.0

# Machine Learning
torch>=1.9.0
torch-geometric>=2.0.0
scikit-learn>=1.0.0

# RL Framework
gymnasium>=0.28.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.3.0

# Data APIs
openmeteo-requests>=1.0.0
requests-cache>=0.8.0

# Utilities
tqdm>=4.62.0
joblib>=1.1.0
```

---

## Integration & Cross-Module Synergies

### Multi-Objective Route Optimization

The three modules can be integrated for comprehensive maritime optimization:

1. **Route Optimization + Operational Efficiency**
   - Incorporate fuel consumption predictions into route cost functions
   - Generate vessel-specific optimal routes based on fuel efficiency characteristics
   - Balance distance optimization with fuel cost minimization

2. **Predictive Maintenance + Route Planning**
   - Avoid routes requiring high equipment stress when maintenance is due soon
   - Plan maintenance schedules based on optimized route profiles
   - Adjust maximum RUL thresholds based on upcoming voyage requirements

3. **Operational Efficiency + Predictive Maintenance**
   - Correlate equipment degradation with fuel efficiency patterns
   - Identify maintenance needs from abnormal fuel consumption
   - Optimize engine parameters for both efficiency and longevity

### Example Integration Workflow

```python
# 1. Predict equipment health
rul_prediction = predictive_maintenance.predict_rul(sensor_data)

# 2. Estimate fuel consumption for route options
fuel_estimates = operational_efficiency.predict_fuel(
    ship_type="Tanker Ship",
    weather=forecast_data,
    routes=candidate_routes
)

# 3. Generate optimal route considering all factors
optimal_route = route_optimization.generate_route(
    start=origin,
    end=destination,
    constraints={
        "max_equipment_stress": rul_prediction.threshold,
        "fuel_budget": fuel_estimates.budget,
        "weather_safety": weather_limits
    }
)
```

---

## Research Contributions

### Route Optimization Module
- Novel two-stage learning framework (IL → RL) for maritime routing
- Graph-based maritime environment with weather-aware features
- Demonstrated 82-92% route length reduction vs. historical trajectories

### Predictive Maintenance Module
- Unsupervised health index derivation from decay coefficients
- LSTM-based temporal degradation modeling
- Production-ready modular pipeline for equipment monitoring

### Operational Efficiency Module
- RandomForest-based fuel efficiency prediction with 83% R² score
- Feature importance analysis identifying ship type as primary driver
- Weather impact quantification for operational planning

---

## Technical Support & Documentation

### Module-Specific Documentation
- **Route Optimization**: See `documentation/route-optimization.md`
- **Predictive Maintenance**: See `documentation/predictive-maintenance.md`
- **Operational Efficiency**: See `documentation/operational-efficiency.md`

### Configuration Files
- Route Optimization: `app/route_optimization/src/config.py`
- Predictive Maintenance: Uses constants in `src/` modules
- Operational Efficiency: `app/operational_efficiency/src/config.py`

### Common Issues

**Import Errors**: Ensure you're running scripts from the correct directory or add parent path to `sys.path`

**Memory Issues**: 
- Reduce batch sizes in training configurations
- Process AIS data in smaller chunks
- Use smaller graph resolutions for testing

**Path Issues**: Update file paths in notebooks to reference `../models/` and `../data/` directories

---

## Data Sources

The Maritime Excellence Platform integrates data from multiple authoritative sources:

### 1. AIS - Marine Cadastre

**Description**: NOAA & BOEM project with free AIS ship tracking data for marine traffic analysis

**Link**: [marinecadastre.gov](https://marinecadastre.gov)

**Key Data Features**:
- **Vessel Data**: MMSI, IMO, Name, Type, Size
- **Signal Data**: Position, Speed, Course, Heading, Status

### 2. Weather - OpenMeteo API

**Description**: Free API for global forecast & historical weather, no API key needed

**Link**: [open-meteo.com](https://open-meteo.com)

**Key Data Features**:
- **Atmospheric Data**: Temperature, Humidity, Precipitation, Wind, Cloud Cover, Pressure, Solar Radiation

### 3. Natural Earth 10m Physical

**Description**: Public geospatial dataset for high-detail regional maps

**Link**: [naturalearthdata.com](https://naturalearthdata.com)

**Key Data Features**:
- **Vector**: Coastlines, Borders, Rivers, Lakes
- **Raster**: Relief, Bathymetry, Imagery

### 4. Naval Propulsion Maintenance (UCI)

**Description**: Dataset for predictive maintenance of simulated naval propulsion plants

**Link**: [UCI Dataset](https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants)

**Key Data Features**:
- **Operational Data**: Torque, RPM, Thrust
- **Environmental Data**: Temperature, Pressure, Humidity
- **Degradation Signals**: Equipment decay coefficients

