"""Reusable URL feature extraction for Project 2 (AI-Powered Phishing Detection).

Member 2 deliverable. The same code is used for training (Member 3), evaluation
(Member 4) and live prediction in the application (Member 5), so a URL typed
into the app is converted exactly the way the training URLs were.

Design rules
------------
* Input is the URL string only. No network access, no page download, no DNS.
* Every handcrafted feature is a pure function of the URL (stateless), so
  nothing can leak from test data into training.
* Anything that is *learned* (scaling, TF-IDF vocabulary) lives in the sklearn
  pipeline built by ``make_feature_pipeline``. It is fitted on training data
  only; inside cross-validation it is re-fitted on each training fold.
* Malformed input is rejected with ``ValueError`` using the same rules as
  Member 1's ``prepare_data.py`` (HTTP/HTTPS scheme, parseable host, valid port).

Quick use
---------
    from feature_extraction import extract_features, make_feature_pipeline
    X = extract_features(["https://www.example.com", "http://paypa1-login.tk/a"])

    pipe = make_feature_pipeline(feature_set="all", scale=True)
    X_train = pipe.fit_transform(train_urls)   # fit on training data only
    X_test = pipe.transform(test_urls)         # never fit on test data

Command line demo:  python feature_extraction.py "http://secure-login.example.tk/verify"
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import ipaddress
import math
from pathlib import Path
import sys
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

HERE = Path(__file__).resolve().parent
PSL_CANDIDATES = [
    HERE / "resources" / "public_suffix_list.dat",
    HERE.parent / "Member1_Phishing_Detection" / "resources" / "public_suffix_list.dat",
]

# ---------------------------------------------------------------------------
# Hand-defined (a-priori) lists. They are fixed by us from general phishing
# awareness material, NOT learned from this dataset, so they cannot leak labels.
# ---------------------------------------------------------------------------
KEYWORDS = (
    "login", "log-in", "signin", "sign-in", "verify", "verification", "secure",
    "security", "account", "update", "confirm", "password", "banking", "bank",
    "wallet", "billing", "invoice", "payment", "support", "helpdesk", "recover",
    "unlock", "suspend", "alert", "authenticate", "validate", "webmail",
)
BRANDS = (
    "paypal", "apple", "icloud", "google", "gmail", "microsoft", "office365",
    "outlook", "amazon", "facebook", "instagram", "whatsapp", "netflix",
    "linkedin", "dropbox", "adobe", "docusign", "ebay", "alibaba", "steam",
    "telegram", "binance", "coinbase", "metamask", "dhl", "fedex", "usps",
    "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc", "santander",
    "sparkasse", "att", "verizon",
)
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "icu", "buzz", "click", "link",
    "work", "zip", "mov", "cyou", "rest", "monster", "cam", "live", "online",
    "site", "shop", "club", "info", "bid", "loan", "men", "win",
}
FREE_HOSTING = (  # free site builders / hosting / tunnelling services
    "firebaseapp.com", "web.app", "weebly.com", "wixsite.com", "blogspot.com",
    "github.io", "ipfs.io", "netlify.app", "herokuapp.com", "glitch.me",
    "pages.dev", "workers.dev", "000webhostapp.com", "godaddysites.com",
    "webflow.io", "ngrok.io", "repl.co", "vercel.app", "onrender.com",
    "sites.google.com", "myshopify.com", "square.site", "carrd.co",
)
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly", "cutt.ly",
    "rebrand.ly", "shorturl.at", "tiny.cc", "buff.ly", "rb.gy",
}
RISKY_EXTENSIONS = (
    ".exe", ".zip", ".scr", ".apk", ".js", ".jar", ".bat", ".php", ".asp",
    ".aspx", ".cgi",
)
VOWELS = set("aeiou")

# ---------------------------------------------------------------------------
# Feature registry: name -> group.
#   "host"      features describe the host name / authority only.
#   "structure" features describe scheme, www, path, query, fragment and overall
#               URL shape. In PhiUSIIL these are strongly tied to how the data was
#               collected (see analysis report), so they are kept separate and
#               can be switched off with feature_set="host".
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {
    # --- host / authority ---
    "hostname_length": "host",
    "num_dots_host": "host",
    "num_subdomains": "host",
    "subdomain_length": "host",
    "sld_length": "host",
    "tld_length": "host",
    "longest_label_length": "host",
    "is_ip_host": "host",
    "is_idn_punycode": "host",
    "has_at_symbol": "host",
    "has_port": "host",
    "host_hyphens": "host",
    "sld_hyphens": "host",
    "host_digits": "host",
    "host_digit_ratio": "host",
    "host_longest_digit_run": "host",
    "host_other_chars": "host",
    "host_entropy": "host",
    "sld_vowel_ratio": "host",
    "suspicious_tld": "host",
    "free_hosting_service": "host",
    "is_url_shortener": "host",
    "host_keyword_count": "host",
    "brand_in_sld_altered": "host",
    "brand_in_subdomain": "host",
    # --- structure ---
    "url_length": "structure",
    "is_https": "structure",
    "has_www": "structure",
    "path_length": "structure",
    "path_depth": "structure",
    "num_slashes_path": "structure",
    "has_double_slash_path": "structure",
    "query_length": "structure",
    "num_query_params": "structure",
    "has_fragment": "structure",
    "url_digits": "structure",
    "url_digit_ratio": "structure",
    "path_query_special_chars": "structure",
    "percent_encoded_count": "structure",
    "path_keyword_count": "structure",
    "brand_in_path": "structure",
    "risky_extension": "structure",
}
FEATURE_NAMES = list(FEATURE_GROUPS)


def feature_names(feature_set: str = "all") -> list[str]:
    """Names of the features in a set: 'all', 'host' (robust) or 'structure'."""
    if feature_set == "all":
        return list(FEATURE_NAMES)
    if feature_set in ("host", "structure"):
        return [n for n, g in FEATURE_GROUPS.items() if g == feature_set]
    raise ValueError("feature_set must be 'all', 'host' or 'structure'")


# ---------------------------------------------------------------------------
# Public Suffix List (same ICANN-only logic as Member 1, so domains agree)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _suffix_rules():
    for path in PSL_CANDIDATES:
        if path.exists():
            exact, wildcard, exceptions = set(), set(), set()
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line == "// ===BEGIN PRIVATE DOMAINS===":
                    break
                if not line or line.startswith("//"):
                    continue
                if line.startswith("!"):
                    exceptions.add(line[1:].encode("idna").decode("ascii"))
                elif line.startswith("*."):
                    wildcard.add(line[2:].encode("idna").decode("ascii"))
                else:
                    exact.add(line.encode("idna").decode("ascii"))
            return exact, wildcard, exceptions
    return None  # fall back to a crude rule; a warning is printed once


def registrable_domain(host: str) -> str:
    rules = _suffix_rules()
    parts = host.split(".")
    if rules is None:
        return ".".join(parts[-2:]) if len(parts) > 2 else host
    exact, wildcard, exceptions = rules
    suffix_length = 1
    for i in range(len(parts)):
        tail = ".".join(parts[i:])
        if tail in exceptions:
            suffix_length = len(parts) - i - 1
            break
        if tail in exact:
            suffix_length = max(suffix_length, len(parts) - i)
        if i > 0 and tail in wildcard:
            suffix_length = max(suffix_length, len(parts) - i + 1)
    return ".".join(parts[-(suffix_length + 1):])


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------
def parse_url(url):
    """Validate a URL like Member 1 did and return (clean_url, parts, host).

    Raises ValueError with a short reason code for unusable input.
    """
    if not isinstance(url, str):
        raise ValueError("not_a_string")
    url = url.strip()
    if not url:
        raise ValueError("empty_url")
    if any(ord(c) < 32 or ord(c) == 127 for c in url):
        raise ValueError("control_character")
    parts = urlsplit(url)  # may raise ValueError (e.g. malformed IPv6)
    if parts.scheme.lower() not in ("http", "https"):
        raise ValueError("unsupported_or_missing_scheme")
    host = parts.hostname
    if not host or any(c.isspace() for c in host):
        raise ValueError("missing_or_invalid_hostname")
    _ = parts.port  # raises ValueError for a malformed or out-of-range port
    host = host.rstrip(".").lower()
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        pass
    if not host:
        raise ValueError("missing_or_invalid_hostname")
    return url, parts, host


def is_valid_url(url) -> bool:
    try:
        parse_url(url)
        return True
    except ValueError:
        return False


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def _longest_digit_run(text: str) -> int:
    best = run = 0
    for ch in text:
        run = run + 1 if ch.isdigit() else 0
        best = max(best, run)
    return best


def _split_host(host_core: str):
    """Return (is_ip, subdomain, sld, suffix) for a host with 'www.' removed."""
    try:
        ipaddress.ip_address(host_core)
        return True, "", host_core, ""
    except ValueError:
        pass
    reg = registrable_domain(host_core)
    reg_parts = reg.split(".")
    sld = reg_parts[0]
    suffix = ".".join(reg_parts[1:])
    subdomain = host_core[: -len(reg)].rstrip(".") if host_core.endswith(reg) and host_core != reg else ""
    return False, subdomain, sld, suffix


def _keyword_hits(text: str) -> int:
    return sum(1 for k in KEYWORDS if k in text)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_url_features(url: str) -> dict:
    """Return {feature_name: number} for one URL. Raises ValueError if invalid."""
    url, parts, host = parse_url(url)
    has_www = host.startswith("www.") and len(host) > 4
    # 'www.' is removed before host features so that it does not distort subdomain
    # counts. It is exposed separately as the has_www (structure) feature.
    host_core = host[4:] if has_www else host
    is_ip, subdomain, sld, suffix = _split_host(host_core)
    labels = host_core.split(".")
    path = parts.path or ""
    query = parts.query or ""
    fragment = parts.fragment or ""
    path_l, query_l = path.lower(), query.lower()
    tail = path + query + fragment
    letters = [c for c in sld if c.isalpha()]
    brand_pool_host = subdomain
    f = {}

    # ---- host / authority ----
    f["hostname_length"] = len(host_core)
    f["num_dots_host"] = host_core.count(".")
    f["num_subdomains"] = len(subdomain.split(".")) if subdomain else 0
    f["subdomain_length"] = len(subdomain)
    f["sld_length"] = len(sld)
    f["tld_length"] = len(suffix)
    f["longest_label_length"] = max(len(x) for x in labels)
    f["is_ip_host"] = int(is_ip)
    f["is_idn_punycode"] = int("xn--" in host_core)
    f["has_at_symbol"] = int("@" in parts.netloc)
    f["has_port"] = int(parts.port is not None)
    f["host_hyphens"] = host_core.count("-")
    f["sld_hyphens"] = sld.count("-")
    digits = sum(c.isdigit() for c in host_core)
    f["host_digits"] = digits
    f["host_digit_ratio"] = digits / len(host_core)
    f["host_longest_digit_run"] = _longest_digit_run(host_core)
    f["host_other_chars"] = sum(not (c.isalnum() or c in ".-") for c in host_core)
    f["host_entropy"] = _entropy(host_core)
    f["sld_vowel_ratio"] = (sum(c in VOWELS for c in letters) / len(letters)) if letters else 0.0
    f["suspicious_tld"] = int(suffix.split(".")[-1] in SUSPICIOUS_TLDS) if suffix else 0
    f["free_hosting_service"] = int(any(host_core == d or host_core.endswith("." + d) for d in FREE_HOSTING))
    f["is_url_shortener"] = int(host_core in SHORTENERS)
    f["host_keyword_count"] = _keyword_hits(host_core)
    # Brand name inside the domain name but not equal to it (paypal-secure.com)
    f["brand_in_sld_altered"] = int(any(b in sld and sld != b for b in BRANDS)) if not is_ip else 0
    # Brand name used as a subdomain of a different domain (paypal.evil.com)
    f["brand_in_subdomain"] = int(any(b in brand_pool_host and sld != b for b in BRANDS))

    # ---- structure ----
    segments = [s for s in path.split("/") if s]
    f["url_length"] = len(url)
    f["is_https"] = int(parts.scheme.lower() == "https")
    f["has_www"] = int(has_www)
    f["path_length"] = len(path)
    f["path_depth"] = len(segments)
    f["num_slashes_path"] = path.count("/")
    f["has_double_slash_path"] = int("//" in path)
    f["query_length"] = len(query)
    f["num_query_params"] = len([p for p in query.split("&") if p]) if query else 0
    f["has_fragment"] = int(bool(fragment))
    url_digits = sum(c.isdigit() for c in url)
    f["url_digits"] = url_digits
    f["url_digit_ratio"] = url_digits / len(url)
    f["path_query_special_chars"] = sum(c in "-_~%=&?#+" for c in tail)
    f["percent_encoded_count"] = tail.count("%")
    f["path_keyword_count"] = _keyword_hits(path_l + "?" + query_l)
    f["brand_in_path"] = int(any(b in path_l and sld != b for b in BRANDS))
    f["risky_extension"] = int(path_l.endswith(RISKY_EXTENSIONS))
    return f


def extract_features(urls, on_invalid: str = "raise") -> pd.DataFrame:
    """Vectorised wrapper: iterable of URL strings -> DataFrame (columns FEATURE_NAMES).

    on_invalid='raise' stops at the first unusable URL; 'nan' returns a row of NaN
    for it (use ``is_valid_url`` to tell the application to show an error).
    """
    if on_invalid not in ("raise", "nan"):
        raise ValueError("on_invalid must be 'raise' or 'nan'")
    index = urls.index if isinstance(urls, (pd.Series, pd.DataFrame)) else None
    rows = []
    nan_row = {n: np.nan for n in FEATURE_NAMES}
    for u in urls:
        try:
            rows.append(extract_url_features(u))
        except ValueError:
            if on_invalid == "raise":
                raise
            rows.append(nan_row)
    return pd.DataFrame(rows, columns=FEATURE_NAMES, index=index)


def host_text(urls) -> list[str]:
    """Host name (without leading 'www.') used by the optional character TF-IDF."""
    out = []
    for u in urls:
        try:
            h = parse_url(u)[2]
            out.append(h[4:] if h.startswith("www.") and len(h) > 4 else h)
        except ValueError:
            out.append("")
    return out


# ---------------------------------------------------------------------------
# sklearn components. Anything fitted (scaling, TF-IDF) is learned in .fit()
# and therefore only ever sees the data passed to fit (the training fold).
# ---------------------------------------------------------------------------
class UrlFeatureTransformer(BaseEstimator, TransformerMixin):
    """URL strings -> numeric handcrafted feature matrix. Stateless (nothing learned)."""

    def __init__(self, feature_set: str = "all"):
        self.feature_set = feature_set

    def fit(self, X, y=None):
        self.feature_names_ = feature_names(self.feature_set)
        return self

    def transform(self, X):
        frame = extract_features(list(np.asarray(X, dtype=object).ravel()))
        return frame[feature_names(self.feature_set)].to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(feature_names(self.feature_set))


def make_feature_pipeline(feature_set: str = "all", scale: bool = True,
                          tfidf: bool = False, tfidf_max_features: int = 5000):
    """Build the (unfitted) preprocessing pipeline.

    feature_set : 'all' | 'host' | 'structure'
    scale       : log1p + StandardScaler (helps Logistic Regression; trees do not need it)
    tfidf       : add character 3-4-gram TF-IDF of the host name (fitted on training data)

    Call ``.fit(train_urls)`` / ``.fit_transform(train_urls)`` on training data only and
    ``.transform(...)`` on anything else. Inside cross-validation put this pipeline
    in front of the model so each fold fits it on its own training part.
    """
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer

    steps = [("extract", UrlFeatureTransformer(feature_set))]
    if scale:
        steps += [("log1p", FunctionTransformer(np.log1p, validate=False)),
                  ("scale", StandardScaler())]
    numeric = Pipeline(steps)
    if not tfidf:
        return numeric
    text = Pipeline([
        ("host", FunctionTransformer(host_text, validate=False)),
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 4), min_df=5,
                                  max_features=tfidf_max_features, sublinear_tf=True,
                                  dtype=np.float32)),
    ])
    return FeatureUnion([("numeric", numeric), ("host_ngrams", text)])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    try:
        feats = extract_url_features(sys.argv[1])
    except ValueError as exc:
        print(f"Invalid URL: {exc}")
        sys.exit(1)
    width = max(map(len, feats))
    for name, value in feats.items():
        print(f"{name:<{width}}  {FEATURE_GROUPS[name]:<9} {value:.4g}")
