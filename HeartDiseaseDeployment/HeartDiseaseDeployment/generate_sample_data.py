"""
generate_sample_data.py
------------------------
This script creates a SAMPLE heart.csv with the exact same 14 columns
as the real Kaggle "Heart Disease Dataset" (johnsmith88).

WHY THIS FILE EXISTS:
I don't have internet access in this environment, so I can't download
the real dataset from Kaggle. This script generates realistic-looking
synthetic data with the correct column names and value ranges, so the
whole pipeline (train_model.py, app.py) runs end-to-end and you can
test everything immediately.

BEFORE FINAL SUBMISSION:
1. Go to: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
2. Download the real heart.csv
3. Replace this synthetic heart.csv with the real one
4. Re-run train_model.py to train on real data
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 300

data = {
    "age": np.random.randint(29, 78, n),
    "sex": np.random.randint(0, 2, n),
    "cp": np.random.randint(0, 4, n),
    "trestbps": np.random.randint(94, 201, n),
    "chol": np.random.randint(126, 565, n),
    "fbs": np.random.randint(0, 2, n),
    "restecg": np.random.randint(0, 3, n),
    "thalach": np.random.randint(71, 203, n),
    "exang": np.random.randint(0, 2, n),
    "oldpeak": np.round(np.random.uniform(0, 6.2, n), 1),
    "slope": np.random.randint(0, 3, n),
    "ca": np.random.randint(0, 5, n),
    "thal": np.random.randint(0, 4, n),
}

df = pd.DataFrame(data)

# Create a target that has some relationship to the features
# (so the model has something real to learn, not pure noise)
risk_score = (
    (df["age"] > 55).astype(int)
    + (df["chol"] > 240).astype(int)
    + (df["trestbps"] > 140).astype(int)
    + (df["thalach"] < 120).astype(int)
    + df["exang"]
    + (df["oldpeak"] > 2).astype(int)
)
df["target"] = (risk_score >= 3).astype(int)

df.to_csv("heart.csv", index=False)
print(f"Sample heart.csv created with {len(df)} rows.")
print(df.head())
