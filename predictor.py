"""
predictor.py - integration layer for the application (Member 5).

Connects the three earlier pieces:
    Member 1: validation rules / data (train.csv, test.csv)
    Member 2: feature_extraction.py (URL validation + feature values for alerts)
    Member 3: predict.py + models/best_model.joblib (saved pipeline: raw URL -> probability)

Input goes through validation, then the saved model, then a verdict.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Member2_Phishing_Detection"))
sys.path.insert(0, str(ROOT / "Member3_Model_Development"))

import predict as member3                                   # noqa: E402  (loads best_model.joblib)
from feature_extraction import extract_url_features, is_valid_url  # noqa: E402


def explain(url):
    """Plain-English warning signs, taken from Member 2's feature values."""
    f = extract_url_features(url)
    r = []
    if f["is_ip_host"]:
        r.append("Uses a raw IP address instead of a domain name")
    if f["has_at_symbol"]:
        r.append("Contains '@', which can hide the real destination")
    if f["brand_in_subdomain"]:
        r.append("A well-known brand name appears in the subdomain of an unrelated site")
    if f["brand_in_sld_altered"]:
        r.append("A well-known brand name appears altered inside the domain (look-alike)")
    if f["host_keyword_count"] >= 1:
        r.append(f"Host name contains {int(f['host_keyword_count'])} suspicious word(s) such as login, account or verify")
    if f["suspicious_tld"]:
        r.append("Domain ending is often abused by phishers")
    if f["free_hosting_service"]:
        r.append("Hosted on a free website-builder or tunnelling service")
    if f["is_url_shortener"]:
        r.append("Shortened link hides the real destination")
    if f["is_idn_punycode"]:
        r.append("Uses punycode (possible look-alike characters)")
    if f["num_subdomains"] >= 3:
        r.append("Unusually long chain of subdomains")
    if f["host_digits"] >= 4:
        r.append("Many digits in the host name")
    if f["host_hyphens"] >= 3:
        r.append("Many hyphens in the host name")
    if f["hostname_length"] > 30:
        r.append("Unusually long host name")
    if not f["is_https"]:
        r.append("Does not use HTTPS")
    return r


class PhishingDetector:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def predict(self, url):
        """Return {'url','label','probability','reasons'}. Label is Phishing, Legitimate or Invalid."""
        if not isinstance(url, str) or not url.strip():
            return {"url": str(url), "label": "Invalid", "probability": 0.0,
                    "reasons": ["Empty or non-text input"]}
        url = url.strip()
        if not is_valid_url(url):
            return {"url": url, "label": "Invalid", "probability": 0.0,
                    "reasons": ["Not a valid http:// or https:// URL "
                                "(a scheme and a readable host name are required)"]}
        try:
            prob = member3.phishing_probability(url)
        except Exception as e:
            return {"url": url, "label": "Invalid", "probability": 0.0,
                    "reasons": [f"Could not analyse input: {e}"]}
        label = "Phishing" if prob >= self.threshold else "Legitimate"
        return {"url": url, "label": label, "probability": prob,
                "reasons": explain(url) if label == "Phishing" else []}
