"""
Pipeline Step 3: Run Inference

This script loads the trained model and runs predictions on test cases
to demonstrate fuel efficiency predictions.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models import load_model, predict_fuel_efficiency, calculate_total_fuel
from utils import print_section_header
from config import MODEL_SAVE_PATH, ALL_FEATURES


def create_test_cases():
    """
    Create test cases for inference demonstration.
    
    Returns:
        List of dictionaries with test case data
    """
    test_cases = [
        {
            "id": "Tanker in Storm",
            "ship_type": "Tanker Ship",
            "month": "January",
            "fuel_type": "HFO",
            "weather_conditions": "Stormy",
            "engine_efficiency": 90.0
        },
        {
            "id": "Tanker in Calm",
            "ship_type": "Tanker Ship",
            "month": "January",
            "fuel_type": "HFO",
            "weather_conditions": "Calm",
            "engine_efficiency": 90.0
        },
        {
            "id": "Surfer Boat in Storm",
            "ship_type": "Surfer Boat",
            "month": "January",
            "fuel_type": "Diesel",
            "weather_conditions": "Stormy",
            "engine_efficiency": 85.0
        },
        {
            "id": "Surfer Boat in Calm",
            "ship_type": "Surfer Boat",
            "month": "January",
            "fuel_type": "Diesel",
            "weather_conditions": "Calm",
            "engine_efficiency": 85.0
        },
        {
            "id": "Fishing Trawler in Moderate",
            "ship_type": "Fishing Trawler",
            "month": "June",
            "fuel_type": "Diesel",
            "weather_conditions": "Moderate",
            "engine_efficiency": 88.0
        },
        {
            "id": "Oil Service Boat in Calm",
            "ship_type": "Oil Service Boat",
            "month": "March",
            "fuel_type": "Diesel",
            "weather_conditions": "Calm",
            "engine_efficiency": 92.0
        }
    ]
    
    return test_cases


def run_inference(model, test_cases, test_distance_km=10.0):
    """
    Run inference on test cases and display results.
    
    Args:
        model: Trained model pipeline
        test_cases: List of test case dictionaries
        test_distance_km: Distance for total fuel calculation
    """
    print_section_header("INFERENCE RESULTS")
    print(f"\nTest Distance: {test_distance_km} km\n")
    
    for test_case in test_cases:
        print("-" * 70)
        print(f"Test Case: {test_case['id']}")
        print("-" * 70)
        
        # Create input DataFrame
        input_data = {k: v for k, v in test_case.items() if k != 'id'}
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        try:
            predicted_efficiency = model.predict(input_df)[0]
            total_fuel = calculate_total_fuel(predicted_efficiency, test_distance_km)
            
            print(f"  Ship Type:          {input_data['ship_type']}")
            print(f"  Weather:            {input_data['weather_conditions']}")
            print(f"  Engine Efficiency:  {input_data['engine_efficiency']}%")
            print(f"  Fuel Type:          {input_data['fuel_type']}")
            print(f"  Month:              {input_data['month']}")
            print(f"\n  → Predicted Efficiency:  {predicted_efficiency:.4f} fuel/km")
            print(f"  → Total Fuel ({test_distance_km}km): {total_fuel:.4f} units")
            print()
            
        except Exception as e:
            print(f"  ❌ Error predicting for this case: {e}\n")
    
    print("=" * 70)


def compare_weather_impact():
    """
    Compare the impact of weather conditions on fuel efficiency.
    """
    print_section_header("WEATHER IMPACT ANALYSIS")
    
    # Test with same ship, different weather
    weather_conditions = ["Calm", "Moderate", "Stormy"]
    results = {}
    
    for weather in weather_conditions:
        input_data = {
            "ship_type": "Tanker Ship",
            "month": "January",
            "fuel_type": "HFO",
            "weather_conditions": weather,
            "engine_efficiency": 90.0
        }
        
        input_df = pd.DataFrame([input_data])
        model = load_model(MODEL_SAVE_PATH)
        prediction = model.predict(input_df)[0]
        results[weather] = prediction
    
    print("\nFuel Efficiency (fuel/km) for Tanker Ship:")
    for weather, efficiency in results.items():
        print(f"  {weather:12s}: {efficiency:.4f}")
    
    # Calculate percentage differences
    if "Calm" in results:
        calm_efficiency = results["Calm"]
        print("\nImpact relative to Calm conditions:")
        for weather, efficiency in results.items():
            if weather != "Calm":
                pct_change = ((efficiency - calm_efficiency) / calm_efficiency) * 100
                print(f"  {weather:12s}: {pct_change:+.2f}%")
    
    print("=" * 70 + "\n")


def main():
    """Main inference workflow."""
    print("\n" + "="*70)
    print("OPERATIONAL EFFICIENCY - INFERENCE")
    print("="*70)
    
    # Step 1: Load model
    print("\n[1/3] Loading trained model...")
    model = load_model(MODEL_SAVE_PATH)
    
    # Step 2: Create test cases
    print("\n[2/3] Creating test cases...")
    test_cases = create_test_cases()
    print(f"   Created {len(test_cases)} test cases")
    
    # Step 3: Run inference
    print("\n[3/3] Running inference...\n")
    run_inference(model, test_cases, test_distance_km=10.0)
    
    # Bonus: Weather impact analysis
    compare_weather_impact()
    
    print("✅ Inference complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
