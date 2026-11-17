"""Evaluate Member 3's four saved models without training or tuning on test data."""

import json
import platform
import sys
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import matplotlib
matplotlib.use("Agg")  # Save figures in Colab and on computers without a display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "Member1_Phishing_Detection" / "data" / "processed"
MODEL_DIR = ROOT / "Member3_Model_Development" / "models"
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"
CV_PATH = ROOT / "Member3_Model_Development" / "results" / "cv_results.csv"
sys.path.insert(0, str(ROOT / "Member2_Phishing_Detection"))
from feature_extraction import extract_features

MODEL_NAMES = [
    "host_logistic_regression", "host_random_forest",
    "all_logistic_regression", "all_random_forest",
]
SHORT_NAMES = ["Host LR", "Host RF", "All LR", "All RF"]
PHISHING = 0  # The other class, 1, is legitimate.
SEED = 42


def check_data(train, test):
    """Reject missing labels, duplicate records and overlap between the splits."""
    required = ["source_row", "URL", "label", "domain_group"]
    for name, frame in [("Training", train), ("Test", test)]:
        assert set(required).issubset(frame.columns), f"{name}: required column missing"
        assert not frame[required].isna().any().any(), f"{name}: missing values"
        assert set(frame["label"]) == {0, 1}, f"{name}: unexpected labels"
        assert frame["URL"].map(lambda value: isinstance(value, str) and bool(value.strip())).all()
        for column in ["URL", "source_row"]:
            assert frame[column].is_unique, f"{name}: duplicate {column}"

    rows = []
    for column in ["URL", "domain_group", "source_row"]:
        shared = len(set(train[column]) & set(test[column]))
        assert shared == 0, f"Training/test overlap in {column}: {shared}"
        rows.append({"Field": column, "Shared_values": shared})
    return pd.DataFrame(rows)


def check_cv_folds(train):
    """Recreate Member 3's training folds; this does not retrain the models."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    validation_appearances = np.zeros(len(train), dtype=int)
    for fold, (fit_ids, val_ids) in enumerate(
        cv.split(train["URL"], train["label"], groups=train["domain_group"]), start=1
    ):
        fit_groups = set(train.iloc[fit_ids]["domain_group"])
        val_groups = set(train.iloc[val_ids]["domain_group"])
        shared = len(fit_groups & val_groups)
        assert shared == 0, f"Shared domains in CV fold {fold}"
        validation_appearances[val_ids] += 1
        rows.append({"Fold": fold, "Training_rows": len(fit_ids),
                     "Validation_rows": len(val_ids), "Shared_domains": shared})
    assert (validation_appearances == 1).all(), "Each training row must be validated once"
    return pd.DataFrame(rows)


def matches_dataset_pattern(url):
    """HTTPS + www hostname + empty path/query/fragment; a trailing slash is different."""
    parts = urlsplit(url)
    host = (parts.hostname or "").rstrip(".").lower()
    return (parts.scheme.lower() == "https" and host.startswith("www.")
            and len(host) > 4 and parts.path == ""
            and parts.query == "" and parts.fragment == "")


def save_figures(results, counts):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    scale_max = int(counts[["TN", "FP", "FN", "TP"]].to_numpy().max())
    for row, ax in zip(counts.itertuples(index=False), axes.flat):
        matrix = np.array([[row.TN, row.FP], [row.FN, row.TP]])
        display = ConfusionMatrixDisplay(matrix, display_labels=["Legitimate", "Phishing"])
        display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d",
                     im_kw={"vmin": 0, "vmax": scale_max})
        ax.set_title(row.Model.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(results))
    for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        ax.bar(x + (i - 1.5) * 0.2, results[metric], width=0.2, label=metric)
    ax.set_xticks(x, SHORT_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Test-set model comparison\nPrecision, recall and F1 refer to phishing")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text())
    # Saved sklearn models should be loaded with the versions recorded by Member 3.
    for name, current, expected in [
        ("scikit-learn", sklearn.__version__, metadata["sklearn_version"]),
        ("pandas", pd.__version__, metadata["pandas_version"]),
    ]:
        if current != expected:
            raise RuntimeError(f"{name}: installed {current}, model requires {expected}. "
                               "Install Member4_Evaluation/requirements.txt first.")
    assert metadata["phishing_value"] == PHISHING, "Label mapping changed"
    assert (ROOT / "Member1_Phishing_Detection/resources/public_suffix_list.dat").is_file()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    print("Checking training/test separation and the five training folds...", flush=True)
    check_data(train, test).to_csv(RESULTS_DIR / "split_checks.csv", index=False)
    class_counts = pd.DataFrame({"Training": train["label"].value_counts(),
                                "Test": test["label"].value_counts()}).sort_index()
    class_counts.to_csv(RESULTS_DIR / "class_counts.csv", index_label="label")
    folds = check_cv_folds(train)
    folds.to_csv(RESULTS_DIR / "cv_fold_checks.csv", index=False)
    print(f"Data checks passed: {len(train):,} training rows, {len(test):,} test rows.", flush=True)

    predictions = test.copy()
    metric_rows, count_rows, error_frames = [], [], []
    for name in MODEL_NAMES:
        print(f"Evaluating: {name}", flush=True)
        model = joblib.load(MODEL_DIR / f"{name}.joblib")
        predicted = model.predict(test["URL"])  # No fit or fit_transform on the test set.
        assert len(predicted) == len(test) and set(predicted) <= {0, 1}
        predictions[name] = predicted
        y_true = test["label"]
        metric_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_true, predicted),
            "Precision": precision_score(y_true, predicted, pos_label=PHISHING, zero_division=0),
            "Recall": recall_score(y_true, predicted, pos_label=PHISHING, zero_division=0),
            "F1": f1_score(y_true, predicted, pos_label=PHISHING, zero_division=0),
        })
        # Explicit [legitimate, phishing] order makes these TN, FP, FN, TP.
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[1, 0]).ravel()
        count_rows.append({"Model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
        errors = test.loc[y_true != predicted].copy()
        errors["prediction"] = predicted[y_true.to_numpy() != predicted]
        errors["Model"] = name
        errors["Error_type"] = np.where(errors["label"] == 1, "False positive", "False negative")
        error_frames.append(errors)

    results = pd.DataFrame(metric_rows)
    counts = pd.DataFrame(count_rows)
    results.to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
    counts.to_csv(RESULTS_DIR / "error_counts.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    save_figures(results, counts)

    # Reuse and attribute Member 3's CV scores; never include test rows in CV.
    cv_results = pd.read_csv(CV_PATH)
    cv_results["Model"] = cv_results["feature_set"] + "_" + cv_results["model"]
    assert cv_results["Model"].is_unique and set(cv_results["Model"]) == set(MODEL_NAMES)
    cv_results.to_csv(RESULTS_DIR / "member3_cv_results.csv", index=False)
    comparison = results[["Model", "F1"]].rename(columns={"F1": "Test_F1"}).merge(
        cv_results[["Model", "cv_f1_phishing", "cv_f1_std"]], on="Model", validate="one_to_one"
    ).rename(columns={"cv_f1_phishing": "CV_F1", "cv_f1_std": "CV_F1_SD"})
    comparison.to_csv(RESULTS_DIR / "cv_test_comparison.csv", index=False)

    all_errors = pd.concat(error_frames, ignore_index=True)
    all_errors.to_csv(RESULTS_DIR / "misclassified_urls.csv", index=False)
    samples = []
    for name in MODEL_NAMES:
        for error_type in ["False positive", "False negative"]:
            subset = all_errors[(all_errors["Model"] == name) & (all_errors["Error_type"] == error_type)]
            samples.append(subset.sample(n=min(2, len(subset)), random_state=SEED))
    examples = pd.concat(samples, ignore_index=True)
    examples = pd.concat([examples, extract_features(examples["URL"])], axis=1)
    examples.to_csv(RESULTS_DIR / "error_examples.csv", index=False)

    # Exploratory analysis of the collection pattern, not a new independent test set.
    pattern = predictions["URL"].apply(matches_dataset_pattern)
    structure = pd.crosstab(predictions["label"].map({0: "Phishing", 1: "Legitimate"}), pattern)
    structure = structure.reindex(columns=[False, True], fill_value=0).rename(
        columns={False: "Other structure", True: "Matching structure"})
    structure.columns.name = None
    structure.to_csv(RESULTS_DIR / "url_structure_summary.csv", index_label="Class")
    matching_phishing = predictions[(predictions["label"] == PHISHING) & pattern]
    structure_rows = []
    for name in MODEL_NAMES:
        total = len(matching_phishing)
        detected = int((matching_phishing[name] == PHISHING).sum())
        structure_rows.append({"Model": name, "Phishing_URLs": total, "Detected": detected,
                               "Missed": total - detected, "Recall": detected / total if total else np.nan})
    structure_results = pd.DataFrame(structure_rows)
    structure_results.to_csv(RESULTS_DIR / "matching_structure_recall.csv", index=False)

    run_info = {
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__,
        "matplotlib": matplotlib.__version__, "positive_class": PHISHING,
        "training_rows": len(train), "test_rows": len(test),
        "cv_source": str(CV_PATH.relative_to(ROOT)),
        "cv_method": metadata["cv"], "cv_training_rerun": False,
        "selected_model_unchanged": metadata["selected_feature_set"] + "_" + metadata["selected_model"],
    }
    (RESULTS_DIR / "evaluation_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    print("\nTest-set results:\n" + results.round(4).to_string(index=False))
    print("\nCV comparison (CV scores from Member 3):\n" + comparison.round(4).to_string(index=False))
    print("\nMatching phishing subgroup:\n" + structure_results.round(4).to_string(index=False))
    print(f"\nSaved results in {RESULTS_DIR}\nSaved figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
