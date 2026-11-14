# Member 2: feature extraction and analysis

Reusable URL feature extraction for Project 2 (AI-Powered Phishing Detection), plus the analysis of which features separate phishing from legitimate URLs.

Input is the URL string only. Nothing is downloaded or visited. Nothing is learned from `test.csv`; the analysis reads `train.csv` only.

## Files

| File | Purpose |
|---|---|
| `feature_extraction.py` | The extractor, the sklearn transformer and the preprocessing pipeline. **Import this** (Members 3, 4, 5). |
| `analyze_features.py` | Feature analysis on `train.csv` only. Writes the files in `reports/` and `figures/`. |
| `test_feature_extraction.py` | 16 unit tests (features, invalid input, no-leakage checks). |
| `reports/feature_summary.csv` | Per-feature class means, AUC, mutual information, direction. |
| `reports/feature_stability.csv` | Single-feature AUC in each of the 5 group-aware folds. |
| `reports/ablation.csv` | Diagnostic group-CV scores per feature set (training data only). |
| `reports/structure_audit.csv` | How the two classes differ in URL shape (dataset artefact audit). |
| `reports/real_world_legit_check.csv` | Probabilities for hand-written legitimate URLs with paths. |
| `reports/features_train.csv.gz` | Train feature matrix, for inspection. Rebuild with the code; do not use it for the test set. |
| `figures/*.png` | Figures for the report. |

## Setup and commands

Python 3.10 or later. From the repository root:

```
cd Member2_Phishing_Detection
pip install -r requirements.txt
python -m unittest -v                  # 16 tests
python analyze_features.py             # about 4 minutes on 2 CPU cores
python feature_extraction.py "http://secure-login.example.tk/verify"   # print one URL's features
```

`feature_extraction.py` finds the Public Suffix List in `../Member1_Phishing_Detection/resources/`, so keep both folders side by side. Without it the code falls back to a crude "last two labels" rule.

## How Members 3, 4 and 5 use it

```python
import pandas as pd
from feature_extraction import make_feature_pipeline, extract_url_features, is_valid_url

train = pd.read_csv("../Member1_Phishing_Detection/data/processed/train.csv")
y_phish = (train["label"] == 0).astype(int)      # phishing = positive class (original label 0)

pre = make_feature_pipeline(feature_set="host", scale=True)   # or "all"; tfidf=True adds char n-grams
X_train = pre.fit_transform(train["URL"])        # fitted on training data only
# later: pre.transform(new_urls)  (never fit on test data)
```

Inside cross-validation, put the pipeline in front of the model so each fold fits it on its own training part:

```python
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
model = make_pipeline(make_feature_pipeline("host", scale=True), LogisticRegression(max_iter=300))
cv = GroupKFold(n_splits=5)                       # pass groups=train["domain_group"]
```

Parameters of `make_feature_pipeline`:

- `feature_set`: `"all"` (42 features), `"host"` (25 host features), `"structure"` (17 structure features).
- `scale`: `log1p` then `StandardScaler`. Use it for Logistic Regression. Random Forest does not need it.
- `tfidf`: adds character 3 to 4-gram TF-IDF of the host name, learned from the training fold only.

For the application (Member 5): call `is_valid_url(text)` first and show an error if it is false; then pass `[text]` through the same saved pipeline. Use the same `feature_set` and `tfidf` settings as the saved model.

## Which feature set to use: read this

In PhiUSIIL **every legitimate URL in the training data is `https://www.<domain>` with no path, query or fragment**, while phishing URLs are varied. A model can therefore score very high from URL shape alone (see `reports/structure_audit.csv` and `reports/ablation.csv`). That is a property of how the dataset was collected, not proof that the model detects phishing on real websites. The extractor keeps these features in a separate `structure` group so the team can:

1. train and report **both** `all` and `host` feature sets, and
2. discuss the gap honestly in Results and Limitations (Member 4).

`reports/real_world_legit_check.csv` shows what happens with real legitimate URLs that have paths.

## Feature groups

**Host (25).** Computed after removing a leading `www.`: lengths, dots, subdomain count and length, second-level-domain length, TLD length, IP host, punycode, `@`, port, hyphens, digits, digit runs, odd characters, Shannon entropy, vowel ratio, suspicious TLD list, free hosting list, URL shortener list, keyword count, brand name inside or beside the domain.

**Structure (17).** Overall length, https, leading `www.`, path length and depth, slashes, double slash, query length and parameter count, fragment, digits in the URL, special characters, percent-encoding, keywords and brand names in the path, risky file extension.

The keyword, brand, TLD, hosting and shortener lists are written by hand in `feature_extraction.py` from general phishing-awareness knowledge. They are not learned from the dataset. The lists are a design choice and an incomplete one; see the report's limitations.

## AI use

The code, tests and report text in this folder were drafted with AI assistance (Claude, Anthropic) and run in a sandbox. The student must review and understand the work, record their own changes and actual effort, and keep the AI-use declaration (Appendix C of the team report) accurate.
