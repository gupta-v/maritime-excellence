# Predictive Maintenance - RUL Prediction

A machine learning project for predicting Remaining Useful Life (RUL) of equipment using LSTM neural networks.

## What This Project Does

This predictive maintenance system analyzes sensor data from industrial equipment to predict when machinery will fail, enabling proactive maintenance scheduling and reducing downtime costs.

### Key Capabilities

1. **Health Index Calculation**: Automatically derives equipment health scores from decay coefficient measurements
2. **RUL Prediction**: Maps health indices to Remaining Useful Life estimates using deep learning
3. **Time Series Analysis**: Processes sequential sensor readings to capture temporal patterns
4. **Real-time Monitoring**: Provides continuous assessment of equipment condition

### How It Works

The system follows a multi-stage pipeline:

#### Stage 1: Data Processing
- Loads equipment sensor data (temperatures, pressures, flow rates, speeds, torque, etc.)
- Cleans and normalizes column names
- Identifies decay coefficient columns that indicate equipment degradation

#### Stage 2: Feature Engineering
- **Health Index Calculation**: Computes a normalized health score (0-1) from decay coefficients
  - Higher values = healthier equipment
  - Calculated as the mean of all decay measurements
  - Normalized to account for varying baseline conditions
  
- **RUL Mapping**: Converts health index to actual RUL values
  - Linear mapping: Health 1.0 → MAX_RUL (default 5000), Health 0.0 → 0
  - Represents estimated operational hours remaining before failure

- **Sensor Feature Selection**: Automatically identifies relevant sensor measurements
  - Temperature readings (T)
  - Pressure measurements (P)
  - Mass flow rates (mf)
  - Gas turbine metrics (GTn, GGn, GTT)
  - Mechanical parameters (Torque, speed)

#### Stage 3: Deep Learning Model
- **Architecture**: Multi-layer LSTM (Long Short-Term Memory) network
  - Captures temporal dependencies in sensor patterns
  - Learns degradation trends over time
  - Predicts future RUL based on current and historical readings

- **Input**: 50-timestep sequences of 16 sensor features
- **Output**: Predicted RUL value for the equipment

#### Stage 4: Training & Validation
- Time-ordered train/test split (preserves temporal ordering)
- Early stopping to prevent overfitting
- Dynamic learning rate adjustment
- Mean Absolute Error (MAE) optimization

### Real-World Applications

- **Manufacturing**: Predict machine tool wear and schedule maintenance
- **Aviation**: Monitor turbine engines for proactive servicing
- **Energy**: Track power generation equipment degradation
- **Transportation**: Assess vehicle component lifespans

## Technical Approach

### Methodology

1. **Unsupervised Health Scoring**: The health index is derived directly from decay coefficients without requiring labeled failure data
2. **Supervised RUL Learning**: LSTM model learns the relationship between sensor patterns and remaining useful life
3. **Temporal Modeling**: 50-timestep sliding windows capture equipment behavior over time
4. **Regression Output**: Continuous RUL predictions (not just classification of "healthy" vs "failing")

## Directory Structure

```
pred_maintain/
├── data/                    # Data directory
│   └── data.csv            # Input dataset
├── models/                  # Saved models and scalers
│   ├── rul_lstm_model.keras
│   ├── rul_scaler.save
│   └── feature_scaler.save
├── notebooks/               # Jupyter notebooks
│   ├── eda.ipynb           # Exploratory Data Analysis
│   ├── monitoring.ipynb    # Monitoring dashboard
│   └── rul.ipynb          # Original RUL prediction notebook
├── plots/                   # Extracted visualization plots
│   ├── rul_plot_*.png      # 3 plots from RUL notebook
│   ├── eda_plot_*.png      # 1 plot from EDA notebook
│   └── README.md           # Plot descriptions
├── src/                     # Source utility modules
│   ├── data_loader.py      # Data loading and cleaning
│   ├── feature_engineering.py  # Health index and RUL calculation
│   └── model_utils.py      # LSTM model building utilities
├── pipeline/                # End-to-end pipeline scripts
│   ├── preprocess.py       # Data preprocessing pipeline
│   ├── train.py           # Model training pipeline
│   └── evaluate.py        # Model evaluation pipeline
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Data

Ensure `data/data.csv` exists and contains the required sensor data with decay coefficient columns.

## Usage

### Option 1: Run Complete Pipeline (Recommended)

From the project root directory:

```bash
# 1. Preprocess data
cd pipeline
python preprocess.py

# 2. Train model
python train.py

# 3. Evaluate model
python evaluate.py
```

### Option 2: Use as Library

Import and use modules in your own scripts or notebooks:

```python
from src.data_loader import load_and_prepare_data
from src.feature_engineering import calculate_health_index, map_health_to_rul
from src.model_utils import build_lstm_model

# Load data
df, decay_cols = load_and_prepare_data("data/data.csv")

# Calculate health and RUL
df = calculate_health_index(df, decay_cols)
df = map_health_to_rul(df, max_rul=5000)

# Build model
model = build_lstm_model(n_features=16, time_steps=50)
```

### Option 3: Use Jupyter Notebooks

The notebooks in `notebooks/` can now import and use the modular code:

```python
import sys
sys.path.append('..')

from src.data_loader import load_and_prepare_data
# ... rest of your notebook code
```

## Model Architecture

The LSTM model architecture:
- Input: (time_steps=50, n_features=16)
- LSTM Layer 1: 128 units, return_sequences=True
- Dropout: 0.2
- LSTM Layer 2: 64 units
- Dropout: 0.2
- Dense: 32 units, ReLU activation
- Output: 1 unit, linear activation (RUL prediction)

## Key Features

- **Health Index Calculation**: Derives equipment health from decay coefficients
- **RUL Mapping**: Maps health index (0-1) to actual RUL values
- **Time Series Processing**: Creates sliding window sequences for LSTM
- **Temporal Split**: Maintains time-order in train/test split
- **Callbacks**: Early stopping and learning rate reduction for optimal training

## Configuration

Key parameters can be adjusted in the pipeline scripts or by modifying constants in `src/` modules:

- `MAX_RUL`: Maximum RUL value (default: 5000)
- `TIME_STEPS`: Sequence length for LSTM (default: 50)
- `LSTM_UNITS_1`: First LSTM layer units (default: 128)
- `LSTM_UNITS_2`: Second LSTM layer units (default: 64)
- `DROPOUT_RATE`: Dropout rate (default: 0.2)

## Results

After training, the model typically achieves:
- MAE (Mean Absolute Error): ~165-200 on test set
- RMSE (Root Mean SquaredError): ~270-300 on test set

Results and plots are saved to `models/evaluation_results.png`.

## Next Steps

After running the pipelines:

1. **Update Notebook Paths**: The notebooks may need path updates to reference files in `../models/` instead of the root directory:
   - `../models/rul_lstm_model.keras`
   - `../models/rul_scaler.save`
   - `../models/feature_scaler.save`

2. **Experiment**: Modify hyperparameters in pipeline scripts or src modules

3. **Deploy**: Use the trained model for real-time predictions

## Troubleshooting

- **Import Errors**: Ensure you're running scripts from the correct directory
- **Path Issues**: Check that relative paths match your current working directory
- **Memory Issues**: Reduce `batch_size` or `time_steps` if needed

