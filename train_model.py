"""
train_model.py
---------------
Task 1: Data Understanding and Preprocessing
Task 2: Model Development

Run this file to train the model and save it as model.pkl
    python train_model.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# -------------------------------------------------
# TASK 1: Data Understanding and Preprocessing
# -------------------------------------------------

# 1. Load the dataset using Pandas
df = pd.read_csv("heart.csv")

# 2. Display the first five records
print("First 5 records:")
print(df.head())

# 3. Identify numerical features and target variable
target_column = "target"
numerical_features = [col for col in df.columns if col != target_column]
print("\nNumerical features:", numerical_features)
print("Target variable:", target_column)

# 4. Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# If any missing values exist, fill numeric columns with their median
if df.isnull().sum().sum() > 0:
    df = df.fillna(df.median(numeric_only=True))
    print("Missing values were filled using column median.")

# 5. Split the dataset into 80% training and 20% testing
X = df[numerical_features]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# -------------------------------------------------
# TASK 2: Model Development
# -------------------------------------------------

# Using Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate using Accuracy Score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# Save the trained model using Joblib
joblib.dump(model, "model.pkl")
# Also save the column order the model expects, so app.py builds
# the input row correctly regardless of JSON key order
joblib.dump(numerical_features, "model_columns.pkl")

print("\nModel saved as model.pkl")
print("Column order saved as model_columns.pkl")
