import sys
import json
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
# Member 2's module must be importable, because the saved pipeline uses it
sys.path.insert(0, str(ROOT / "Member2_Phishing_Detection"))

_DIR = ROOT / "Member3_Model_Development" / "models"
_model = joblib.load(_DIR / "best_model.joblib")
_meta = json.load(open(_DIR / "model_metadata.json"))
PHISHING = _meta["phishing_value"]          # 0 = phishing in the original labels


def predict_url(url: str) -> str:
    """Return 'Phishing' or 'Legitimate'. Raises ValueError for an unusable URL."""
    try:
        pred = _model.predict([url])[0]
    except ValueError as exc:
        raise ValueError(f"Invalid URL: {exc}") from exc
    return "Phishing" if pred == PHISHING else "Legitimate"


def phishing_probability(url: str) -> float:
    """Probability (0 to 1) that the URL is phishing, for alert levels."""
    idx = list(_model.classes_).index(PHISHING)
    return float(_model.predict_proba([url])[0][idx])


if __name__ == "__main__":
    for u in ["http://secure-login-paypal.example.tk/verify-account",
              "https://www.wikipedia.org"]:
        print(u, "->", predict_url(u), f"(phishing probability {phishing_probability(u):.3f})")