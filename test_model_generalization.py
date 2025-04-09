import pandas as pd
import numpy as np
import joblib
import os
import glob
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import argparse

def load_latest_model(models_dir='trained_models'):
    """Load the most recently trained model from the models directory."""
    model_files = glob.glob(os.path.join(models_dir, '*.joblib'))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    
    # Sort by modification time (newest first)
    latest_model_path = max(model_files, key=os.path.getmtime)
    print(f"Loading model: {latest_model_path}")
    
    # Load the model
    model = joblib.load(latest_model_path)
    return model

def preprocess_data(df):
    """Preprocess data in the same way as during training."""
    print("Preprocessing data...")
    
    # Handle missing values
    df = df.fillna(0)
    
    # Convert date to features
    df['Date & Time'] = pd.to_datetime(df['Date & Time'])
    df['Year'] = df['Date & Time'].dt.year
    df['Month'] = df['Date & Time'].dt.month
    df['Day'] = df['Date & Time'].dt.day
    df['Hour'] = df['Date & Time'].dt.hour
    
    # Encode the district
    label_encoder = LabelEncoder()
    df['District_encoded'] = label_encoder.fit_transform(df['District'])
    
    # Extract numerical features
    numerical_cols = []
    for col in df.columns:
        if col not in ['Hazard Type', 'Date & Time', 'District', 'Source'] and 'Infrastructure' not in col and 'Response' not in col:
            try:
                pd.to_numeric(df[col], errors='raise')
                numerical_cols.append(col)
            except:
                continue
    
    print(f"Using numerical columns: {numerical_cols}")
    
    # Base features (same as training)
    features = ['Latitude', 'Longitude', 'Year', 'Month', 'Day', 'Hour', 'District_encoded']
    
    # Add other numerical columns if they have enough non-zero values
    for col in numerical_cols:
        if col not in features and df[col].count() > 0.5 * len(df):
            features.append(col)
    
    print(f"Final feature set: {features}")
    
    X = df[features]
    y = df['Hazard Type']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, features

def evaluate_model_with_cross_validation(model, X, y, cv=5):
    """Evaluate model using cross-validation."""
    print(f"\nEvaluating model with {cv}-fold cross-validation...")
    
    # Use stratified k-fold to maintain class distribution
    stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Get cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=stratified_kfold, scoring='accuracy')
    
    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
    print(f"Standard deviation: {cv_scores.std():.4f}")
    
    return cv_scores

def evaluate_model_with_holdout(model, X, y, test_size=0.3):
    """Evaluate model using a holdout validation set."""
    print(f"\nEvaluating model with holdout validation (test_size={test_size})...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Train model on training set
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Holdout accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    
    return accuracy, y_test, y_pred

def main():
    parser = argparse.ArgumentParser(description="Test model generalization using unseen data")
    parser.add_argument('--data', type=str, default='DataSet.csv',
                       help='Path to the data file (default: DataSet.csv)')
    parser.add_argument('--method', type=str, choices=['cv', 'holdout', 'both'], default='both',
                       help='Evaluation method: cross-validation (cv), holdout, or both (default: both)')
    args = parser.parse_args()
    
    # Load the trained model
    model = load_latest_model()
    
    # Load and preprocess data
    print(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)
    X, y, features = preprocess_data(df)
    
    # Print basic information about the data
    print("\nDataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Hazard Types: {df['Hazard Type'].unique()}")
    print(f"Number of records per hazard type:")
    print(df['Hazard Type'].value_counts())
    
    # Evaluate model generalization
    if args.method in ['cv', 'both']:
        cv_scores = evaluate_model_with_cross_validation(model, X, y)
    
    if args.method in ['holdout', 'both']:
        accuracy, _, _ = evaluate_model_with_holdout(model, X, y)
    
    print("\nGeneralization evaluation complete!")

if __name__ == '__main__':
    main() 