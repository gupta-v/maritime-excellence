# src/config.py
import torch
from pathlib import Path

# Get the route_optimization directory (parent of src)
CONFIG_DIR = Path(__file__).resolve().parent  # src directory
ROUTE_OPT_DIR = CONFIG_DIR.parent  # route_optimization directory

# --- Core Paths (now absolute) ---
RAW_AIS_DIR = ROUTE_OPT_DIR / "data" / "raw_ais"
PROCESSED_DATA_DIR = ROUTE_OPT_DIR / "data" / "processed"
SHAPEFILE_DIR = ROUTE_OPT_DIR / "data" / "shapefile" / "ne_10m_land"
MODELS_DIR = ROUTE_OPT_DIR / "models"
OUTPUT_DIR = ROUTE_OPT_DIR / "plots"  

CLEANED_AIS_PATH = str(PROCESSED_DATA_DIR / "cleaned_guam_ais.csv")
GRAPH_PATH = str(PROCESSED_DATA_DIR / "sea_graph_guam.pkl") 
IMITATION_MODEL_PATH = str(MODELS_DIR / "gnn_imitator_v1.pth")
RL_MODEL_PATH = str(MODELS_DIR / "gnn_rl_agent_v1.pth")
OUTPUT_PLOT_PATH = str(OUTPUT_DIR / "route_comparison_plot.png")

# --- Graph & Grid Parameters ---
LAT_MIN, LAT_MAX = 13.0, 14.0
LON_MIN, LON_MAX = 144.0, 145.0
GRID_STEP = 0.05
LAND_SHAPEFILE = str(SHAPEFILE_DIR / "ne_10m_land.shp")

# --- Model & Training Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 256  # Updated to match Colab-trained models
MAX_NEIGHBORS = 8  # Corresponds to RL action space

# Imitation Learning (IL)
PRETRAIN_EPOCHS = 100  # Increased for better pretraining
IL_LEARNING_RATE = 1e-3  # Adjusted learning rate
SEQUENCE_LENGTH = 5  # Reduced sequence length
IL_BATCH_SIZE = 128

# Reinforcement Learning (RL)
RL_EPISODES = 4500  # Increased episodes
RL_LEARNING_RATE = 1e-4  # Adjusted learning rate
GAMMA = 0.95  # Updated discount factor
EPSILON_START = 1.0  # Start with full exploration
EPSILON_END = 0.01  # Lower minimum exploration
EPSILON_DECAY = 0.9995  # Slower decay
RL_BATCH_SIZE = 64  # Smaller batch size
REPLAY_BUFFER_CAPACITY = 50000  # Increased buffer capacity
TARGET_UPDATE_FREQ = 100  # Less frequent target updates
MAX_STEPS_PER_EPISODE = 200  # Reduced max steps
LOOP_MEMORY_SIZE = 15  # For loop detection