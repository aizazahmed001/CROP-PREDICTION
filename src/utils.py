"""
Utility functions for the Smart Agriculture Decision Support System
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime


def create_directory_structure():
    """Create required project directories if they don't exist"""
    directories = ['data', 'models', 'results', 'src']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("[INFO] Directory structure verified.")


def save_model(model, filename, directory='models'):
    """
    Save trained model to disk
    
    Args:
        model: Trained sklearn model
        filename: Name of the file (without extension)
        directory: Directory to save model
    """
    filepath = os.path.join(directory, f"{filename}.pkl")
    joblib.dump(model, filepath)
    print(f"[SUCCESS] Model saved: {filepath}")


def load_model(filename, directory='models'):
    """
    Load trained model from disk
    
    Args:
        filename: Name of the file (without extension)
        directory: Directory containing models
    
    Returns:
        Loaded model object
    """
    filepath = os.path.join(directory, f"{filename}.pkl")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found: {filepath}")
    model = joblib.load(filepath)
    print(f"[SUCCESS] Model loaded: {filepath}")
    return model


def log_metrics(metrics_dict, filename, directory='results'):
    """
    Save evaluation metrics to text file with timestamp
    
    Args:
        metrics_dict: Dictionary containing metric names and values
        filename: Name of the output file
        directory: Directory to save results
    """
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as f:
        f.write(f"Smart Agriculture AI System - Model Evaluation\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        for metric_name, metric_value in metrics_dict.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")
    print(f"[SUCCESS] Metrics saved: {filepath}")


def generate_synthetic_dataset(num_samples=1000, save=True):
    """
    Generate realistic synthetic agricultural dataset
    
    Args:
        num_samples: Number of data points to generate
        save: Whether to save dataset to CSV
    
    Returns:
        DataFrame containing synthetic agricultural data
    """
    np.random.seed(42)
    
    # Generate environmental features
    temperature = np.random.normal(25, 5, num_samples)  # Celsius
    humidity = np.random.normal(70, 15, num_samples)    # Percentage
    rainfall = np.random.gamma(2, 50, num_samples)      # mm
    soil_ph = np.random.normal(6.5, 0.8, num_samples)   # pH scale
    
    # Soil nutrients
    nitrogen = np.random.normal(140, 40, num_samples)   # kg/ha
    phosphorus = np.random.normal(40, 20, num_samples)   # kg/ha
    potassium = np.random.normal(200, 50, num_samples)   # kg/ha
    
    # Soil moisture
    moisture = np.random.normal(35, 10, num_samples)     # Percentage
    
    # Generate crop labels based on environmental conditions
    crop_labels = []
    crop_yields = []
    
    for i in range(num_samples):
        # Decision logic for crop recommendation
        if temperature[i] > 28 and rainfall[i] > 100:
            crop = 'Rice'
            base_yield = 4000
        elif temperature[i] > 22 and soil_ph[i] > 6.0 and nitrogen[i] > 120:
            crop = 'Wheat'
            base_yield = 3500
        elif temperature[i] < 20 and rainfall[i] < 80:
            crop = 'Barley'
            base_yield = 3000
        elif soil_ph[i] > 7.0 and temperature[i] > 25:
            crop = 'Cotton'
            base_yield = 2500
        elif phosphorus[i] > 50 and potassium[i] > 220:
            crop = 'Sugarcane'
            base_yield = 70000
        elif nitrogen[i] > 160 and moisture[i] > 40:
            crop = 'Maize'
            base_yield = 6000
        else:
            crop = 'Soybean'
            base_yield = 2800
        
        crop_labels.append(crop)
        
        # Generate yield with noise
        yield_value = base_yield + np.random.normal(0, base_yield * 0.1)
        crop_yields.append(max(0, yield_value))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Temperature': temperature,
        'Humidity': humidity,
        'Rainfall': rainfall,
        'Soil_pH': soil_ph,
        'Nitrogen': nitrogen,
        'Phosphorus': phosphorus,
        'Potassium': potassium,
        'Moisture': moisture,
        'Crop_Label': crop_labels,
        'Crop_Yield': crop_yields
    })
    
    # Clip values to realistic ranges
    df['Temperature'] = df['Temperature'].clip(0, 50)
    df['Humidity'] = df['Humidity'].clip(0, 100)
    df['Rainfall'] = df['Rainfall'].clip(0, 500)
    df['Soil_pH'] = df['Soil_pH'].clip(3, 10)
    df['Nitrogen'] = df['Nitrogen'].clip(0, 300)
    df['Phosphorus'] = df['Phosphorus'].clip(0, 100)
    df['Potassium'] = df['Potassium'].clip(0, 400)
    df['Moisture'] = df['Moisture'].clip(0, 100)
    
    if save:
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/agricultural_data.csv', index=False)
        print(f"[SUCCESS] Synthetic dataset generated with {num_samples} samples")
    
    return df


if __name__ == "__main__":
    # Test utility functions
    create_directory_structure()
    df = generate_synthetic_dataset(100)
    print(df.head())
    print(df.describe())