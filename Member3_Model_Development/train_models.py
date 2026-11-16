import sys
import json
import time
import joblib
import sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

# ---------------- Paths and Member 2's code ----------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Member2_Phishing_Detection"))
from feature_extraction import extract_features, feature_names, make_feature_pipeline

TRAIN_PATH = ROOT / "Member1_Phishing_Detection" / "data" / "processed" / "train.csv"
OUT_DIR = ROOT / "Member3_Model_Development"

# ---------------- Settings ----------------
RANDOM_STATE = 42
N_FOLDS = 5
PHISHING_VALUE = 0                  # Member 1: 0 = phishing, 1 = legitimate
FEATURE_SETS = ["host", "all"]      # Member 2: report both
DEPLOY_FEATURE_SET = "host"         # feature set of the model saved as best_model.joblib

# ---------------- Load training data ----------------
train = pd.read_csv(TRAIN_PATH)
assert set(train["label"].unique()) <= {0, 1}, "Unexpected label values"
urls = train["URL"]
y = train["label"]
groups = train["domain_group"]      # only used to keep related domains together in CV

print(f"Rows: {len(train)}, domain groups: {groups.nunique()}")
print("Class counts (0 = phishing, 1 = legitimate):")
print(y.value_counts().to_string())

# Member 2's handcrafted features are stateless, so computing them once is leak-free.
# Anything learned (scaling) is fitted inside each CV training fold below.
t0 = time.time()
print("Extracting features ...", flush=True)
X_all = extract_features(urls)
print(f"Done in {time.time() - t0:.0f} s, {X_all.shape[1]} features", flush=True)

# ---------------- CV and scoring (phishing = positive class) ----------------
cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "f1": make_scorer(f1_score, pos_label=PHISHING_VALUE),
    "precision": make_scorer(precision_score, pos_label=PHISHING_VALUE),
    "recall": make_scorer(recall_score, pos_label=PHISHING_VALUE),
    "accuracy": "accuracy",
}

def cv_pipeline(model_name):
    if model_name == "logistic_regression":
        pipe = Pipeline([
            ("log1p", FunctionTransformer(np.log1p, validate=False)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ])
        grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    else:
        pipe = Pipeline([("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1))])
        grid = {"clf__n_estimators": [100, 200],
                "clf__max_depth": [10, 20],
                "clf__min_samples_leaf": [2, 5]}
    return pipe, grid

# ---------------- Tune, then build a deployable pipeline per configuration ----------------
(OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "results").mkdir(parents=True, exist_ok=True)
rows, deployable = [], {}

for fs in FEATURE_SETS:
    X = X_all[feature_names(fs)]
    for model_name in ["logistic_regression", "random_forest"]:
        print(f"\n=== feature set: {fs} | model: {model_name} ===", flush=True)
        t0 = time.time()
        pipe, grid = cv_pipeline(model_name)
        gs = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, refit="f1", n_jobs=-1)
        gs.fit(X, y, groups=groups)
        i = gs.best_index_
        row = {"feature_set": fs, "model": model_name,
               "best_params": gs.best_params_,
               "cv_f1_phishing": gs.cv_results_["mean_test_f1"][i],
               "cv_f1_std": gs.cv_results_["std_test_f1"][i],
               "cv_precision_phishing": gs.cv_results_["mean_test_precision"][i],
               "cv_recall_phishing": gs.cv_results_["mean_test_recall"][i],
               "cv_accuracy": gs.cv_results_["mean_test_accuracy"][i]}
        rows.append(row)
        print(f"best params: {gs.best_params_}")
        print(f"CV F1 {row['cv_f1_phishing']:.4f} (+/- {row['cv_f1_std']:.4f}) | "
              f"precision {row['cv_precision_phishing']:.4f} | recall {row['cv_recall_phishing']:.4f} "
              f"| {time.time() - t0:.0f} s", flush=True)

        # Deployable model: raw URL string in -> label out, using Member 2's pipeline.
        final = Pipeline([
            ("prep", make_feature_pipeline(fs, scale=(model_name == "logistic_regression"))),
            ("clf", clone(gs.best_estimator_.named_steps["clf"])),
        ])
        final.fit(urls, y)
        sample = slice(0, 2000)     # sanity check: same predictions as the tuned model
        agree = (final.predict(urls.iloc[sample]) == gs.best_estimator_.predict(X.iloc[sample])).mean()
        print(f"Deployable pipeline agrees with tuned model on sample: {agree:.4f}")
        deployable[(fs, model_name)] = final
        joblib.dump(final, OUT_DIR / "models" / f"{fs}_{model_name}.joblib", compress=3)

results_df = pd.DataFrame(rows)
results_df.to_csv(OUT_DIR / "results" / "cv_results.csv", index=False)

# ---------------- Select the deployed model ----------------
cands = results_df[results_df["feature_set"] == DEPLOY_FEATURE_SET]
best = cands.sort_values("cv_f1_phishing", ascending=False).iloc[0]
joblib.dump(deployable[(best["feature_set"], best["model"])],
            OUT_DIR / "models" / "best_model.joblib", compress=3)

metadata = {
    "selected_model": best["model"],
    "selected_feature_set": best["feature_set"],
    "selected_best_params": {k: (int(v) if isinstance(v, (np.integer,)) else v)
                             for k, v in best["best_params"].items()},
    "feature_names": feature_names(best["feature_set"]),
    "input": "raw URL string (the saved pipeline extracts features itself)",
    "phishing_value": PHISHING_VALUE,
    "label_meaning": {"0": "phishing", "1": "legitimate"},
    "selection_rule": f"highest mean CV F1 (phishing = positive) among '{DEPLOY_FEATURE_SET}' feature-set models",
    "cv": f"{N_FOLDS}-fold StratifiedGroupKFold, grouped by domain_group, shuffle, seed {RANDOM_STATE}, training data only",
    "random_seed": RANDOM_STATE,
    "n_training_rows": int(len(train)),
    "n_domain_groups": int(groups.nunique()),
    "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
    "python_version": sys.version.split()[0],
    "sklearn_version": sklearn.__version__,
    "pandas_version": pd.__version__,
}
with open(OUT_DIR / "models" / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n==== SUMMARY ====")
print(results_df.drop(columns="best_params").round(4).to_string(index=False))
print(f"\nDeployed: {best['model']} on '{best['feature_set']}' features")