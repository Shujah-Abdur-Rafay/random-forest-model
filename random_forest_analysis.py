import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('DataSet.csv')

# Display basic information about the dataset
print("\nDataset Info:")
print(f"Shape: {df.shape}")
print(f"Hazard Types: {df['Hazard Type'].unique()}")
print(f"Number of records per hazard type:")
print(df['Hazard Type'].value_counts())

# Preprocessing
print("\nPreprocessing data...")

# Handle missing values if any
df = df.fillna(0)

# Identify feature columns - using columns that have consistent data across hazard types
# Since different hazard types have different attribute structures, we'll use a subset of common features
common_features = ['District', 'Latitude', 'Longitude', 'Date & Time']

# Convert date to features
df['Date & Time'] = pd.to_datetime(df['Date & Time'])
df['Year'] = df['Date & Time'].dt.year
df['Month'] = df['Date & Time'].dt.month
df['Day'] = df['Date & Time'].dt.day
df['Hour'] = df['Date & Time'].dt.hour

# Extract numerical features based on position
# First, let's identify which columns contain primarily numerical data
numerical_cols = []
for col in df.columns:
    if col not in ['Hazard Type', 'Date & Time', 'District', 'Source'] and 'Infrastructure' not in col and 'Response' not in col:
        try:
            # Check if column can be converted to numeric
            pd.to_numeric(df[col], errors='raise')
            numerical_cols.append(col)
        except:
            continue

print(f"Using numerical columns: {numerical_cols}")

# Encode the district
label_encoder = LabelEncoder()
df['District_encoded'] = label_encoder.fit_transform(df['District'])

# Prepare features and target
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Train Random Forest model
print("\nTraining Random Forest classifier...")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
print("\nMaking predictions...")
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
print("\nEvaluating model performance:")
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create a timestamp for file naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create directories for trained models and results if they don't exist
models_dir = 'trained_models'
results_dir = f'evaluation_results_{timestamp}'

for directory in [models_dir, results_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"\nCreated directory: {directory}")

# Save the model in the models directory
print("\nSaving the trained model...")
model_filename = f"random_forest_hazard_classifier_{timestamp}.joblib"
model_path = os.path.join(models_dir, model_filename)

# Save only the model
joblib.dump(rf_classifier, model_path)
print(f"Model saved as: {model_path}")

# Save evaluation metrics
print("\nSaving evaluation metrics...")

# Extract metrics from classification report
metrics = {
    'accuracy': accuracy,
    'class_metrics': class_report
}

# Save metrics as JSON
metrics_path = os.path.join(results_dir, f"metrics_{timestamp}.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"Metrics saved to {metrics_path}")

# Create and save visualizations
print("\nGenerating visualizations...")

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
hazard_types = list(df['Hazard Type'].unique())
cm_display = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                         xticklabels=hazard_types, yticklabels=hazard_types)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"confusion_matrix_{timestamp}.png"))
plt.close()

# 2. Classification Metrics by Class
plt.figure(figsize=(12, 8))
classes = list(class_report.keys())
classes = [c for c in classes if c not in ['accuracy', 'macro avg', 'weighted avg']]

metrics_to_plot = ['precision', 'recall', 'f1-score']
metrics_df = pd.DataFrame({
    'Class': classes * len(metrics_to_plot),
    'Metric': [metric for metric in metrics_to_plot for _ in classes],
    'Value': [class_report[cls][metric] for metric in metrics_to_plot for cls in classes]
})

sns.barplot(x='Class', y='Value', hue='Metric', data=metrics_df)
plt.title('Performance Metrics by Class')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"class_metrics_{timestamp}.png"))
plt.close()

# 3. Feature Importance
plt.figure(figsize=(12, 8))
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(len(features)), importances[indices])
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"feature_importance_{timestamp}.png"))
plt.close()

# Create a summary text file
summary_path = os.path.join(results_dir, f"summary_{timestamp}.txt")
with open(summary_path, 'w') as f:
    f.write(f"Random Forest Classifier Evaluation Summary\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
    f.write(f"Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write(f"\nModel saved as: {model_path}\n")
    f.write(f"Evaluation metrics and visualizations saved in: {results_dir}\n")

print(f"\nSummary saved to {summary_path}")
print(f"\nAnalysis complete. Results have been saved in the '{results_dir}' folder.") 