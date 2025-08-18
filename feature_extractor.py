import re
import math
from collections import Counter
from urllib.parse import urlparse, parse_qs

def calculate_entropy(s):
    prob = [n_x / len(s) for x, n_x in Counter(s).items()]
    entropy = -sum(p * math.log2(p) for p in prob)
    return entropy

def extract_lexical_features(url: str) -> dict:
    features = {}
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    tld = hostname.split(".")[-1] if hostname else ""

    # Basic counts
    features["url_length"] = len(url)
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_at"] = url.count("@")
    features["num_question_marks"] = url.count("?")
    features["num_equals"] = url.count("=")
    features["num_slashes"] = url.count("/")
    features["num_underscores"] = url.count("_")

    # Digits and letters
    num_digits = sum(c.isdigit() for c in url)
    num_letters = sum(c.isalpha() for c in url)
    features["num_digits"] = num_digits
    features["digit_ratio"] = num_digits / len(url) if len(url) > 0 else 0
    features["letter_ratio"] = num_letters / len(url) if len(url) > 0 else 0

    # Domain features
    features["domain_length"] = len(hostname)
    features["subdomain_count"] = hostname.count(".") - 1 if hostname else 0
    features["digits_in_domain"] = sum(c.isdigit() for c in hostname)

    # HTTPS
    features["has_https"] = int(url.startswith("https"))

    # IP Address
    ip_pattern = r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    features["has_ip_address"] = int(bool(re.match(ip_pattern, url)))

    # Suspicious keywords
    suspicious_keywords = ["login", "verify", "secure", "account", "update"]
    features["has_suspicious_keyword"] = int(
        any(keyword in url.lower() for keyword in suspicious_keywords)
    )
    features["suspicious_word_count"] = sum(url.lower().count(word) for word in suspicious_keywords)

    # Suspicious TLDs
    suspicious_tlds = ["tk", "ml", "ga", "cf", "gq"]
    features["suspicious_tld"] = int(tld in suspicious_tlds)

    # Shortened URLs
    shorteners = ["bit.ly", "tinyurl.com", "goo.gl"]
    features["is_shortened"] = int(any(short in hostname for short in shorteners))

    # Special character ratio
    special_chars = sum(not c.isalnum() for c in url)
    features["special_char_ratio"] = special_chars / len(url) if len(url) > 0 else 0

    # TLD length
    features["tld_length"] = len(tld)

    # Port
    features["has_port_in_url"] = int(":" in parsed.netloc)

    # Path length
    features["path_length"] = len(parsed.path)

    # Path depth
    features["path_depth"] = len([p for p in parsed.path.split("/") if p])

    # Entropy
    features["entropy"] = calculate_entropy(url)

    # Query params
    features["num_query_params"] = len(parse_qs(parsed.query))

    return features

if __name__ == "__main__":
    test_url = input("Enter URL: ").strip()
    output = extract_lexical_features(test_url)
    for k, v in output.items():
        print(f"{k}: {v}")
