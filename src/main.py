import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from data.data_loader import DataLoader
from features.feature_engineering import FeatureEngineer
from models.orientation_model import OrientationModel

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "RSSI and Location Datasets"
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)
    
    # Initialize components
    data_loader = DataLoader(data_dir)
    feature_engineer = FeatureEngineer()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    rssi_df, acc_df = data_loader.load_and_preprocess_data()
    
    # Merge RSSI and acceleration data
    df = pd.merge_asof(rssi_df, acc_df, on='timestamp', direction='nearest')
    
    # Prepare features
    print("Preparing features...")
    X, y = feature_engineer.prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train model
    print("Training model...")
    model = OrientationModel(input_dim=X.shape[1], num_classes=len(feature_engineer.label_encoder.classes_))
    history = model.train(X_train, y_train, X_test, y_test)
    
    # Plot training history
    print("Plotting training history...")
    model.plot_training_history()
    
    # Evaluate model
    print("Evaluating model...")
    metrics = model.evaluate(X_test, y_test, feature_engineer.get_label_mapping())
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    model.plot_confusion_matrix(metrics)
    
    # Save model
    model_path = model_dir / "orientation_model.h5"
    model.save_model(str(model_path))
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main() 