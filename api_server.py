"""
LexiGuard Backend API Server (updated)
- Per-class thresholds (tunable via env vars)
- URL normalization (adds scheme if missing)
- Probabilities + concise reasons aligned with feature_extractor.py
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from feature_extractor import extract_lexical_features

# ---------------------------
# Load model and label encoder
# ---------------------------
model = joblib.load("xgb_model.pkl")
le    = joblib.load("label_encoder.pkl")

# ---------------------------
# Thresholds (adjust via env)
# ---------------------------
PHISHING_THRESHOLD   = float(os.getenv("PHISHING_THRESHOLD", 0.70))
MALWARE_THRESHOLD    = float(os.getenv("MALWARE_THRESHOLD", 0.70))
DEFACEMENT_THRESHOLD = float(os.getenv("DEFACEMENT_THRESHOLD", 0.60))
SUSPICIOUS_THRESHOLD = float(os.getenv("SUSPICIOUS_THRESHOLD", 0.60))

# Helpful keyword lists
SUSPICIOUS_KEYWORDS = [
    "login","signin","verify","update","secure","account","bank","wallet",
    "reset","confirm","invoice","pay","gift","bonus","win","free","click",
    "office365","outlook","webmail","microsoft","aws","apple","google"
]
DEFACEMENT_HINTS = ["defaced", "hacked", "index.html", "mirror"]

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------
# Helpers
# ---------------------------
def normalize_url(u: str) -> str:
    """Ensure scheme exists so parsing is consistent."""
    if not u:
        return u
    u = u.strip()
    if "://" not in u:
        u = "http://" + u
    return u

def _fv(feats: dict, key: str, default=0.0) -> float:
    """Fetch numeric feature safely."""
    try:
        return float(feats.get(key, default))
    except Exception:
        return default

def build_reasons(url: str, feats: dict, top_label: str, top_conf: float) -> list[str]:
    """
    Short, human-readable reasons based on *your* extractor's feature names.
    (Matches feature_extractor.py: num_hyphens, num_digits, special_char_ratio, etc.)
    """
    u = (url or "").lower()
    reasons = []

    # Length / complexity
    if _fv(feats, "url_length") > 75:
        reasons.append("Very long URL (> 75 characters)")
    if _fv(feats, "path_length") > 40:
        reasons.append("Long or complex path")

    # Characters / symbols
    if _fv(feats, "num_hyphens") >= 3:
        reasons.append("Many hyphens in URL")
    if _fv(feats, "num_digits") >= 6:
        reasons.append("Unusual amount of digits")
    if float(feats.get("special_char_ratio", 0)) > 0.30:
        reasons.append("High ratio of special characters")
    if _fv(feats, "num_at") >= 1:
        reasons.append("Contains '@' in URL")
    if _fv(feats, "num_dots") >= 4:
        reasons.append("Excessive subdomains/dots")

    # Host / addressing
    if bool(feats.get("has_ip_address", False)):
        reasons.append("Uses IP address instead of domain")
    if _fv(feats, "subdomain_count") > 3:
        reasons.append("Too many subdomains")
    if _fv(feats, "domain_length") > 30:
        reasons.append("Unusually long domain name")

    # Protocol
    if u.startswith("http://") and int(feats.get("has_https", 0)) == 0:
        reasons.append("Uses HTTP (no HTTPS)")

    # Phishing / defacement hints (non-decisive, for explanation only)
    for kw in SUSPICIOUS_KEYWORDS:
        if kw in u:
            reasons.append(f"Contains suspicious keyword: “{kw}”")
            break
    for kw in DEFACEMENT_HINTS:
        if kw in u:
            reasons.append(f"Contains defacement hint: “{kw}”")
            break

    # Confidence meta (only if very strong)
    if top_label.lower() in {"phishing","malware","defacement"} and top_conf >= 0.90:
        reasons.append("Model is highly confident")

    return reasons[:5]

# ---------------------------
# Routes
# ---------------------------
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

    # 1) Extract lexical features → DataFrame
    feats = extract_lexical_features(url)
    X = pd.DataFrame([feats])

    try:
        # 2) Predict probabilities
        probs = model.predict_proba(X)[0]  # [p0, p1, ...]
        # Map index → original string class via LabelEncoder
        classes = list(le.inverse_transform(list(range(len(probs)))))

        top_idx   = int(probs.argmax())
        top_label = classes[top_idx]
        top_conf  = float(probs[top_idx])
        low       = top_label.lower()

        # 3) Class-specific thresholds
        if   low == "phishing"   and top_conf >= PHISHING_THRESHOLD:   label_out = "phishing"
        elif low == "malware"    and top_conf >= MALWARE_THRESHOLD:    label_out = "malware"
        elif low == "defacement" and top_conf >= DEFACEMENT_THRESHOLD: label_out = "defacement"
        elif top_conf < SUSPICIOUS_THRESHOLD:                           label_out = "benign"
        else:                                                           label_out = "suspicious"

        # 4) Reasons
        reasons = build_reasons(url, feats, top_label, top_conf)

        return jsonify({
            "label": label_out,
            "confidence": round(top_conf, 4),
            "top_class": top_label,
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(classes, probs)},
            "reasons": reasons
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
