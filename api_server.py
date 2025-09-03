"""
LexiGuard Backend API Server (clean version)
- Always returns one of: benign / phishing / malware / defacement
- Per-class thresholds (env-tunable)
- Heuristic overrides for demo (phishing/malware/defacement hints)
- No reasons/explanations in output
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from feature_extractor import extract_lexical_features

# =========================
# Load model + encoder
# =========================
model = joblib.load("xgb_model.pkl")
le    = joblib.load("label_encoder.pkl")

# =========================
# Thresholds (env-tunable)
# =========================
PHISHING_THRESHOLD   = float(os.getenv("PHISHING_THRESHOLD", 0.70))
MALWARE_THRESHOLD    = float(os.getenv("MALWARE_THRESHOLD", 0.70))
DEFACEMENT_THRESHOLD = float(os.getenv("DEFACEMENT_THRESHOLD", 0.60))
SUSPICIOUS_THRESHOLD = float(os.getenv("SUSPICIOUS_THRESHOLD", 0.60))

# Enable/disable overrides
ENABLE_OVERRIDES = os.getenv("ENABLE_OVERRIDES", "true").lower() == "true"

# =========================
# Heuristic hints
# =========================
PHISH_HINTS = ["login","signin","verify","paypal","secure","account","bank","webmail","outlook","office365"]
MALWARE_HINTS = [".exe",".apk",".msi",".bat",".vbs","/download","/update","/setup","flashupdate"]
DEFACE_HINTS = ["defaced","hacked","mirror","index.html","index.htm"]

# =========================
# Flask app
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# Helpers
# =========================
def normalize_url(u: str) -> str:
    if not u:
        return u
    u = u.strip()
    if "://" not in u:
        u = "http://" + u
    return u

def has_any(s: str, needles: list[str]) -> bool:
    s = s.lower()
    return any(n in s for n in needles)

# =========================
# Routes
# =========================
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "LexiGuard API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    if "url" not in payload:
        return jsonify({"error": "No URL provided."}), 400

    url = normalize_url(str(payload["url"]).strip())
    if not url:
        return jsonify({"error": "Empty URL."}), 400

    # Extract lexical features → DataFrame
    feats = extract_lexical_features(url)
    X = pd.DataFrame([feats])

    try:
        probs = model.predict_proba(X)[0]
        classes = list(le.inverse_transform(list(range(len(probs)))))

        top_idx   = int(probs.argmax())
        top_label = classes[top_idx]
        top_conf  = float(probs[top_idx])
        low       = top_label.lower()

        # Heuristic checks
        hint_phish = has_any(url, PHISH_HINTS)
        hint_mal   = has_any(url, MALWARE_HINTS)
        hint_def   = has_any(url, DEFACE_HINTS)

        # Decide final label
        if top_conf < SUSPICIOUS_THRESHOLD and not (ENABLE_OVERRIDES and (hint_def or hint_mal or hint_phish)):
            label_out = "benign"
        elif (low == "phishing"   and top_conf >= PHISHING_THRESHOLD) or \
             (low == "malware"    and top_conf >= MALWARE_THRESHOLD)  or \
             (low == "defacement" and top_conf >= DEFACEMENT_THRESHOLD):
            label_out = low
        elif ENABLE_OVERRIDES and (hint_def or hint_mal or hint_phish):
            if hint_def:
                label_out = "defacement"
            elif hint_mal:
                label_out = "malware"
            else:
                label_out = "phishing"
        else:
            label_out = low if low in {"phishing","malware","defacement"} else "benign"

        return jsonify({
            "label": label_out,
            "confidence": round(top_conf, 4),
            "top_class": top_label,
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
