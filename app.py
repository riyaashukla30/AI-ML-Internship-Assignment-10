"""
app.py
------
Task 3: API Development

Runs a Flask app that:
  - Loads the trained model (model.pkl)
  - Serves a web page (/) with a form -> web interface
  - Serves a REST API (/predict) that accepts JSON and returns JSON
"""

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and the column order it expects
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")


def make_prediction(input_dict):
    """Takes a dict of patient details, returns prediction text + probability."""
    # Build a single-row DataFrame in the exact column order the model expects
    row = [float(input_dict.get(col, 0)) for col in model_columns]
    df = pd.DataFrame([row], columns=model_columns)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]  # probability of class "1"

    result_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
    return result_text, round(float(probability) * 100, 2)


@app.route("/")
def home():
    """Web interface: shows a form for entering patient details."""
    return render_template("index.html", columns=model_columns)


@app.route("/predict", methods=["POST"])
def predict():
    """
    REST API endpoint.
    Accepts JSON, e.g.:
    {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }
    Returns JSON:
    { "prediction": "Heart Disease Detected", "confidence": 87.4 }
    """
    try:
        data = request.get_json(force=True)
        result_text, confidence = make_prediction(data)
        return jsonify({"prediction": result_text, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # Render sets the PORT environment variable automatically
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
