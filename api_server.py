"""
LexiGuard Backend API Server (four labels + smart overrides)
- Always returns one of: benign / phishing / malware / defacement
- Per-class thresholds (env-tunable)
- Heuristic overrides for defacement, malware, phishing
- URL normalization + reasons aligned with feature_extractor.py
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
SUSPICIOUS_THRESHOLD = float(os.getenv("SUSPICIOUS_THRESHOLD", 0.60))  # below this → benign unless overrides

# Enable/disable heuristic overrides (for demo/presentation)
ENABLE_OVERRIDES = os.getenv("ENABLE_OVERRIDES", "true").lower() == "true"

# =========================
# Heuristic hint lists
# =========================
# Phishing: login-y words + a few brands often abused
PHISH_LOGIN_WORDS = [
    "login","signin","sign-in","verify","verification","update","secure",
    "account","reset","confirm","invoice","pay","billing","webmail"
]
PHISH_BRANDS = [
    "paypal","apple","icloud","microsoft","office365","outlook","google","gmail",
    "amazon","bank","wallet"
]

# Malware: typical executable artifacts / paths
MALWARE_EXTS = [".exe",".msi",".bat",".cmd",".vbs",".ps1",".apk",".jar",".scr",".dll",".pkg",".xz",".gz",".bz2",".zip",".rar",".7z"]
MALWARE_PATH_WORDS = ["download","installer","setup","update","patch","flashupdate","crack","keygen"]

# Defacement: classic indicators
DEFACE_WORDS = ["defaced","hacked","mirror","defacement","owned"]
DEFACE_PATH_HINTS = ["index.html", "index.htm"]

# =========================
# Flask app
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# Helpers
# =========================
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

def has_any(s: str, needles: list[str]) -> bool:
    s = s.lower()
    return any(n in s for n in needles)

def has_any_suffix(s: str, suffixes: list[str]) -> bool:
    sl = s.lower()
    return any(sl.endswith(x) for x in suffixes)

def phishing_hint(url: str) -> bool:
    u = url.lower()
    return has_any(u, PHISH_LOGIN_WORDS) or has_any(u, PHISH_BRANDS)

def malware_hint(url: str) -> bool:
    u = url.lower()
    return has_any_suffix(u, MALWARE_EXTS) or has_any(u, MALWARE_PATH_WORDS)

def defacement_hint(url: str) -> bool:
    u = url.lower()
    return has_any(u, DEFACE_WORDS) or has_any(u, DEFACE_PATH_HINTS)

def build_reasons(url: str, feats: dict, top_label: str, top_conf: float) -> list[str]:
    """
    Human-readable reasons based on your extractor's feature names:
    num_hyphens, num_digits, special_char_ratio, has_ip_address, subdomain_count, domain_length, path_length, has_https, etc.
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

    # Hint words (for explanation)
    if phishing_hint(u):
        reasons.append("Contains phishing-like keywords/brands")
    if malware_hint(u):
        reasons.append("Looks like a file download or installer")
    if defacement_hint(u):
        reasons.append("Contains defacement hint")

    # Confidence meta (only if very strong)
    if top_label.lower() in {"phishing","malware","defacement"} and top_conf >= 0.90:
        reasons.append("Model is highly confident")

    # Keep it short in the popup
    return reasons[:5]

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

    # 1) features → DataFrame
    feats = extract_lexical_features(url)
    X = pd.DataFrame([feats])

    try:
        # 2) probs + class names
        probs = model.predict_proba(X)[0]
        classes = list(le.inverse_transform(list(range(len(probs)))))

        top_idx   = int(probs.argmax())
        top_label = classes[top_idx]
        top_conf  = float(probs[top_idx])
        low       = top_label.lower()

        # 3) Heuristic signals
        hint_phish = phishing_hint(url)
        hint_mal   = malware_hint(url)
        hint_def   = defacement_hint(url)

        # 4) Decide final label — always one of the four
        # Priority:
        #   a) If clearly benign (very low confidence, no hints) → benign
        #   b) If meets its class threshold → that class
        #   c) Overrides based on strong hints (defacement > malware > phishing)
        #   d) Otherwise fallback to the top class if it's malicious, else benign

        # a) very low = benign (unless overrides push otherwise)
        if top_conf < SUSPICIOUS_THRESHOLD and not (ENABLE_OVERRIDES and (hint_def or hint_mal or hint_phish)):
            label_out = "benign"

        # b) meets its threshold
        elif (low == "phishing"   and top_conf >= PHISHING_THRESHOLD) or \
             (low == "malware"    and top_conf >= MALWARE_THRESHOLD)  or \
             (low == "defacement" and top_conf >= DEFACEMENT_THRESHOLD):
            label_out = low

        # c) overrides (presentation-safe)
        elif ENABLE_OVERRIDES and (hint_def or hint_mal or hint_phish):
            if hint_def:
                label_out = "defacement"
            elif hint_mal:
                label_out = "malware"
            else:
                label_out = "phishing"

        # d) fallback to top class if it's malicious; else benign
        else:
            label_out = low if low in {"phishing","malware","defacement"} else "benign"

        # 5) Reasons
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
