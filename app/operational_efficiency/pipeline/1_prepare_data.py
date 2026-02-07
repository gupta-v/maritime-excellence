"""
Pipeline Step 1: Data Preparation

This script loads the fuel efficiency data, performs feature engineering,
and displays data statistics for validation.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_utils import (
    load_fuel_data,
    engineer_features,
    get_data_summary
)
from config import FUEL_DATA_PATH


def main():
    """Main data preparation workflow."""
    print("\n" + "="*70)
    print("OPERATIONAL EFFICIENCY - DATA PREPARATION")
    print("="*70)
    
    # Step 1: Load data
    print("\n[1/3] Loading fuel efficiency data...")
    df = load_fuel_data(FUEL_DATA_PATH)
    
    # Step 2: Engineer features
    print("\n[2/3] Engineering features...")
    df = engineer_features(df)
    
    # Step 3: Display data summary
    print("\n[3/3] Data summary...")
    get_data_summary(df)
    
    print("\n✅ Data preparation complete!")
    print("   Ready for model training (pipeline/2_train_model.py)")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during data preparation: {e}")
        sys.exit(1)
