import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
from datetime import datetime
import os

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
class_report = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Create a directory for trained models if it doesn't exist
models_dir = 'trained_models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"\nCreated directory: {models_dir}")

# Save the model in the models directory
print("\nSaving the trained model...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"random_forest_hazard_classifier_{timestamp}.joblib"
model_path = os.path.join(models_dir, model_filename)

# Save only the model
joblib.dump(rf_classifier, model_path)

print(f"Model saved as: {model_path}")

print("\nAnalysis complete. Trained model has been saved in the 'trained_models' folder.") 