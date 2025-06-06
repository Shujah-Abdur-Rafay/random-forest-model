#!/usr/bin/env python3
"""
Random Forest Model Executor for GBDMS

This script:
1. Loads a trained Random Forest model
2. Processes input data
3. Makes predictions
4. Generates analysis outputs (metrics, visualizations)
5. Returns results in JSON format

Usage:
    python run_model.py --input dataset.csv [options]

Options:
    --input FILE         Input CSV dataset
    --trees INT          Number of trees in Random Forest (default: 100)
    --test-size FLOAT    Test set size (default: 0.3)
    --feature-importance BOOL   Generate feature importance (default: True)
    --confusion-matrix BOOL     Generate confusion matrix (default: True)
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import os
import json
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Random Forest model analysis')
    parser.add_argument('--input', required=True, help='Input CSV dataset')
    parser.add_argument('--trees', type=int, default=100, help='Number of trees in Random Forest')
    parser.add_argument('--test-size', type=float, default=0.3, help='Test set size')
    parser.add_argument('--feature-importance', type=bool, default=True, help='Generate feature importance')
    parser.add_argument('--confusion-matrix', type=bool, default=True, help='Generate confusion matrix')
    parser.add_argument('--output-dir', default='output', help='Output directory for results')
    
    return parser.parse_args()

def load_data(file_path):
    """Load and preprocess dataset."""
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df = df.fillna(0)
    
    # Return dataset
    return df

def prepare_features(df, target_col='Hazard Type'):
    """Prepare features and target variable."""
    # Check if target column exists
    if target_col not in df.columns:
        target_col = df.columns[-1]  # Use last column as default target
        
    # Extract target
    y = df[target_col]
    
    # Convert datetime columns
    datetime_cols = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                datetime_cols.append(col)
            except:
                pass
    
    # Extract features from datetime
    for col in datetime_cols:
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        if 'time' in col.lower():
            df[f'{col}_hour'] = df[col].dt.hour
    
    # Encode categorical columns
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' and col != target_col:
            categorical_cols.append(col)
    
    encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[f'{col}_encoded'] = encoder.fit_transform(df[col])
        encoders[col] = encoder
    
    # Drop unnecessary columns
    X = df.drop([target_col] + datetime_cols + categorical_cols, axis=1)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, list(X.columns), encoders

def train_and_evaluate(X, y, n_trees, test_size):
    """Train Random Forest model and evaluate performance."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Get feature importance
    feature_importances = model.feature_importances_
    
    return model, accuracy, conf_matrix, report, feature_importances, X_train, X_test, y_train, y_test

def generate_confusion_matrix_plot(conf_matrix, classes, output_path):
    """Generate confusion matrix visualization."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def generate_feature_importance_plot(importances, feature_names, output_path):
    """Generate feature importance visualization."""
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Limit to top 15 features for readability
    indices = indices[:15]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def save_model(model, output_dir):
    """Save the trained model."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"random_forest_model_{timestamp}.joblib")
    joblib.dump(model, model_path)
    return model_path

def format_classification_report(report):
    """Format classification report for JSON output."""
    classes = []
    
    for class_name, metrics in report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            classes.append({
                'class': class_name,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1-score'],
                'support': metrics['support']
            })
    
    return classes

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dict
    results = {
        'success': False,
        'timestamp': datetime.now().isoformat(),
        'input_file': args.input,
        'parameters': {
            'n_trees': args.trees,
            'test_size': args.test_size
        },
        'dataset': {},
        'model': {},
        'metrics': {},
        'visualizations': {}
    }
    
    try:
        # Load data
        print("Loading dataset...")
        df = load_data(args.input)
        
        # Basic dataset info
        results['dataset'] = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'preview': df.head(5).to_dict(orient='records')
        }
        
        # Prepare features
        print("Preparing features...")
        X, y, feature_names, encoders = prepare_features(df)
        
        # Train and evaluate
        print(f"Training Random Forest with {args.trees} trees...")
        model, accuracy, conf_matrix, report, feature_imp, X_train, X_test, y_train, y_test = train_and_evaluate(
            X, y, args.trees, args.test_size
        )
        
        # Save model
        model_path = save_model(model, output_dir)
        
        # Store metrics
        results['metrics'] = {
            'accuracy': accuracy,
            'classification_report': format_classification_report(report)
        }
        
        # Generate visualizations if requested
        if args.confusion_matrix:
            print("Generating confusion matrix...")
            cm_path = os.path.join(output_dir, 'confusion_matrix.png')
            cm_file = generate_confusion_matrix_plot(conf_matrix, list(set(y)), cm_path)
            results['visualizations']['confusion_matrix'] = cm_file
        
        if args.feature_importance:
            print("Generating feature importance plot...")
            fi_path = os.path.join(output_dir, 'feature_importance.png')
            fi_file = generate_feature_importance_plot(feature_imp, feature_names, fi_path)
            results['visualizations']['feature_importance'] = fi_file
        
        # Set success flag
        results['success'] = True
        results['model']['path'] = model_path
        
    except Exception as e:
        results['error'] = str(e)
        print(f"Error: {e}")
    
    # Output results as JSON
    print(json.dumps(results))
    
    return results

if __name__ == "__main__":
    main() 