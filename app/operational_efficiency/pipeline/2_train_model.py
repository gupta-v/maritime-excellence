"""
Pipeline Step 2: Model Training

This script trains the fuel efficiency prediction model using RandomForest,
evaluates performance, and saves the trained model.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_utils import (
    load_fuel_data,
    engineer_features,
    prepare_features_target,
    split_train_test
)
from models import create_model_pipeline, save_model
from utils import (
    evaluate_model,
    print_evaluation_metrics,
    get_feature_importances,
    print_feature_importances,
    print_section_header
)
from config import FUEL_DATA_PATH, MODEL_SAVE_PATH


def main():
    """Main model training workflow."""
    print("\n" + "="*70)
    print("OPERATIONAL EFFICIENCY - MODEL TRAINING")
    print("="*70)
    
    # Step 1: Load and prepare data
    print("\n[1/6] Loading fuel efficiency data...")
    df = load_fuel_data(FUEL_DATA_PATH)
    
    print("\n[2/6] Engineering features...")
    df = engineer_features(df)
    
    print("\n[3/6] Preparing features and target...")
    X, y = prepare_features_target(df)
    
    print("\n[4/6] Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    # Step 2: Create and train model
    print("\n[5/6] Training model...")
    print_section_header("MODEL TRAINING")
    
    model_pipeline = create_model_pipeline()
    
    print("\nFitting model on training data...")
    model_pipeline.fit(X_train, y_train)
    print("✅ Model training complete!")
    
    # Step 3: Evaluate model
    print("\n[6/6] Evaluating model performance...")
    metrics = evaluate_model(model_pipeline, X_test, y_test)
    print_evaluation_metrics(metrics)
    
    # Display feature importances
    importances = get_feature_importances(model_pipeline)
    print_feature_importances(importances, top_n=15)
    
    # Step 4: Save model
    save_model(model_pipeline, MODEL_SAVE_PATH)
    
    print("\n✅ Training pipeline complete!")
    print(f"   Model saved to: {MODEL_SAVE_PATH}")
    print("   Ready for inference (pipeline/3_run_inference.py)")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during model training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
