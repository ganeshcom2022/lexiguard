# api_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, joblib, pandas as pd

from feature_extractor import extract_lexical_features

# ---------------------------
# CONFIG / ARTIFACTS
# ---------------------------
MODEL_PATH          = os.getenv("MODEL_PATH", "xgb_model.pkl")
LABEL_ENCODER_PATH  = os.getenv("LABEL_ENCODER_PATH", "label_encoder.pkl")
FEATURE_COLS_PATH   = os.getenv("FEATURE_COLS_PATH", "feature_columns.pkl")  # optional

# Confidence thresholds (tune via Render env vars if you like)
MALWARE_THRESHOLD     = float(os.getenv("MALWARE_THRESHOLD", 0.80))  # >= → confident threat
SUSPICIOUS_THRESHOLD  = float(os.getenv("SUSPICIOUS_THRESHOLD", 0.60))  # < → treat as benign

# Optional API key protection
API_KEY = os.environ.get("LEXIGUARD_API_KEY")

# Load artifacts
model      = joblib.load(MODEL_PATH)
label_enc  = joblib.load(LABEL_ENCODER_PATH)

# Optional fixed column order (if you saved it during training)
feature_cols = None
if os.path.exists(FEATURE_COLS_PATH):
    try:
        feature_cols = joblib.load(FEATURE_COLS_PATH)
    except Exception:
        feature_cols = None

app = Flask(__name__)
CORS(app)


# ---------------------------
# UTILITIES
# ---------------------------
def normalize_url(u: str) -> str:
    """Add scheme if missing so parsing stays consistent."""
    if not u:
        return u
    u = u.strip()
    if "://" not in u:
        u = "http://" + u
    return u

def _fv(feats: dict, names, default=0.0) -> float:
    """
    Safe numeric fetch for one or more possible feature keys.
    Example: _fv(feats, ["count_hyphens","num_hyphen"])
    """
    if isinstance(names, str):
        names = [names]
    for n in names:
        if n in feats:
            try:
                return float(feats.get(n, default))
            except Exception:
                pass
    return default

SUSPICIOUS_KEYWORDS = [
    "login","signin","verify","update","secure","account","bank","wallet",
    "reset","confirm","invoice","pay","gift","bonus","win","free","click",
    "office365","outlook","webmail","microsoft","aws","apple","google"
]

def build_reasons(url: str, feats: dict, top_label: str, top_conf: float) -> list[str]:
    """
    Generate concise, human‑readable reasons based on lexical features + URL text.
    Returns up to 5 items to keep the popup readable.
    """
    u = (url or "").lower()
    reasons = []

    # Length / complexity
    if _fv(feats, "url_length") > 75:      reasons.append("Very long URL (> 75 characters)")
    if _fv(feats, "path_length") > 40:     reasons.append("Long or complex path")
    if _fv(feats, "query_length") > 40:    reasons.append("Long query string with many parameters")

    # Characters / symbols
    if _fv(feats, ["count_hyphens","num_hyphen"]) >= 3: reasons.append("Many hyphens in URL")
    if _fv(feats, ["count_digits","num_digits"]) >= 6:  reasons.append("Unusual amount of digits")
    if _fv(feats, ["count_special","num_special"]) >= 6:reasons.append("Many special symbols")
    if _fv(feats, ["count_at","num_at"]) >= 1:          reasons.append("Contains '@' in URL")
    if _fv(feats, ["count_dots","num_dots"]) >= 4:      reasons.append("Excessive subdomains/dots")

    # Host / addressing
    if bool(feats.get("has_ip", False)):   reasons.append("Uses IP address instead of domain")
    if _fv(feats, "subdomain_length") > 20:reasons.append("Very long subdomain (possible deception)")
    if _fv(feats, "hostname_length") > 30: reasons.append("Unusually long hostname")

    # Keywords exploited in phishing
    for kw in SUSPICIOUS_KEYWORDS:
        if kw in u:
            reasons.append(f"Contains suspicious keyword: “{kw}”")
            break

    # Protocol
    if u.startswith("http://"):            reasons.append("Uses HTTP (no HTTPS)")

    # Model meta (only add when truly strong)
    if top_label.lower() in {"phishing","malware","defacement"} and top_conf >= 0.90:
        reasons.append("Model is highly confident")

    return reasons[:5]


# ---------------------------
# ROUTES
# ---------------------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "LexiGuard API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    # optional API key
    if API_KEY:
        if request.headers.get("X-API-Key") != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    url = normalize_url(str(payload.get("url", "")).strip())
    if not url:
        return jsonify({"error": "No URL provided."}), 400

    # 1) Extract lexical features → DataFrame
    feats = extract_lexical_features(url)
    X = pd.DataFrame([feats])

    # 2) Align column order if available
    if feature_cols:
        X = X.reindex(columns=feature_cols, fill_value=0)

    # 3) Predict probabilities
    try:
        probs = model.predict_proba(X)[0]               # array of class probs
        classes = list(label_enc.inverse_transform(range(len(probs))))

        top_idx   = int(probs.argmax())
        top_label = classes[top_idx]
        top_conf  = float(probs[top_idx])

        # 4) Apply thresholds → final user-facing label
        if top_conf >= MALWARE_THRESHOLD and top_label.lower() in {"malware","phishing","defacement"}:
            label_out = top_label
        elif top_conf < SUSPICIOUS_THRESHOLD:
            label_out = "benign"
        else:
            label_out = "suspicious"

        # 5) Build reasons (explanations)
        explanations = build_reasons(url, feats, top_label, top_conf)

        return jsonify({
            "label": label_out,
            "confidence": round(top_conf, 4),
            "top_class": top_label,
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(classes, probs)},
            "reasons": explanations
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    # Local dev only; production uses Gunicorn (via Dockerfile)
    app.run(host="0.0.0.0", port=5000)
