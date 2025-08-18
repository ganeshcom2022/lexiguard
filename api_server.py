"""
LexiGuard Backend API Server
Serves predictions via REST API.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from feature_extractor import extract_lexical_features

# Load model and label encoder
model = joblib.load("xgb_model.pkl")
le = joblib.load("label_encoder.pkl")

# Flask app
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "LexiGuard API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "No URL provided."}), 400

    url = data["url"].strip()
    if not url:
        return jsonify({"error": "Empty URL."}), 400

    features = extract_lexical_features(url)
    X = pd.DataFrame([features])

    # Get numeric prediction and convert to int
    pred_numeric = int(model.predict(X)[0])

    # Convert numeric back to label
    pred_label = le.inverse_transform([pred_numeric])[0]

    return jsonify({"label": pred_label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
