"""Feature analysis for Member 2: which URL features separate phishing from legitimate?

Uses data/processed/train.csv ONLY. test.csv is never opened (it is reserved for
Member 4's final evaluation). Run:  python analyze_features.py

Outputs (reports/ and figures/):
  features_train.csv.gz  feature matrix + label (for Members 3-4, no test data)
  feature_summary.csv    per-feature statistics, one row per feature
  feature_stability.csv  single-feature AUC in each group-aware fold
  redundant_pairs.csv    feature pairs with |Spearman| >= 0.90
  ablation.csv           diagnostic group-CV scores per feature set (train only)
  rf_importance.csv      impurity importances of a Random Forest fitted on train
  structure_audit.csv    how the legitimate/phishing classes differ in URL shape
  figures/*.png
"""
from pathlib import Path
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from feature_extraction import (FEATURE_GROUPS, extract_features, feature_names,
                                make_feature_pipeline)

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent / "Member1_Phishing_Detection" / "data" / "processed" / "train.csv"
REPORTS, FIGS = HERE / "reports", HERE / "figures"
SEED, FOLDS = 42, 5
PHISH, LEGIT = "#C0392B", "#2C6E9B"  # phishing = red, legitimate = blue
REPORTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})


def auc_phishing(x, y_phish):
    """Rank-based AUC of x for the phishing class (phishing = positive)."""
    ranks = rankdata(x)
    n1 = y_phish.sum()
    n0 = len(y_phish) - n1
    return (ranks[y_phish == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def binned_mutual_information(x, y, bins=20):
    """Mutual information (bits) between a feature (quantile-binned) and the label."""
    x = pd.Series(x)
    if x.nunique() <= bins:
        codes = pd.factorize(x)[0]
    else:
        codes = pd.qcut(x, bins, duplicates="drop").cat.codes.to_numpy()
    joint = pd.crosstab(codes, y).to_numpy(float)
    p = joint / joint.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())


def main():
    t0 = time.time()
    train = pd.read_csv(TRAIN)
    assert set(train.columns) == {"source_row", "URL", "label", "domain_group"}
    y_phish = (train["label"] == 0).astype(int).to_numpy()  # phishing = positive class
    groups = train["domain_group"].to_numpy()
    print(f"Loaded {len(train):,} training rows ({y_phish.sum():,} phishing).")

    # ---------------------------------------------------------------- features
    X = extract_features(train["URL"])
    print(f"Extracted {X.shape[1]} features in {time.time() - t0:.0f}s")
    out = X.copy()
    out.insert(0, "source_row", train["source_row"])
    out["label"] = train["label"]
    out["domain_group"] = train["domain_group"]
    out.to_csv(REPORTS / "features_train.csv.gz", index=False)

    # -------------------------------------------------- 1. structure audit
    parsed = X.assign(label=train["label"])
    audit = []
    pretty = {"is_https": "uses https", "has_www": "starts with www.", "path_length": "has a path",
              "query_length": "has a query string", "has_fragment": "has a fragment", "has_port": "has a port",
              "is_ip_host": "IP-address host", "num_subdomains": "has subdomain (after www)",
              "free_hosting_service": "free hosting service"}
    for name in pretty:
        col = parsed[name]
        flag = col > 0
        audit.append({"feature": pretty[name],
                      "phishing_with_property_%": round(100 * flag[parsed.label == 0].mean(), 2),
                      "legitimate_with_property_%": round(100 * flag[parsed.label == 1].mean(), 2)})
    bare = (X["path_length"].isin([0, 1])) & (X["query_length"] == 0) & (X["has_fragment"] == 0)
    audit.append({"feature": "bare homepage (no path/query)",
                  "phishing_with_property_%": round(100 * bare[parsed.label == 0].mean(), 2),
                  "legitimate_with_property_%": round(100 * bare[parsed.label == 1].mean(), 2)})
    combo = (X["is_https"] == 1) & (X["has_www"] == 1) & bare
    audit.append({"feature": "https + www + bare homepage",
                  "phishing_with_property_%": round(100 * combo[parsed.label == 0].mean(), 2),
                  "legitimate_with_property_%": round(100 * combo[parsed.label == 1].mean(), 2)})
    audit = pd.DataFrame(audit)
    audit.to_csv(REPORTS / "structure_audit.csv", index=False)
    print("\nStructure audit (share of each class with the property):\n", audit.to_string(index=False))

    # ------------------------------------- 2. univariate feature statistics
    gkf = GroupKFold(n_splits=FOLDS)
    folds = list(gkf.split(X, y_phish, groups))
    rows, stab = [], []
    for name in X.columns:
        x = X[name].to_numpy(float)
        auc = auc_phishing(x, y_phish)
        fold_aucs = [auc_phishing(x[te], y_phish[te]) for _, te in folds]
        stab.append({"feature": name, **{f"auc_fold{i + 1}": a for i, a in enumerate(fold_aucs)}})
        rho = np.corrcoef(rankdata(x), y_phish)[0, 1] if np.ptp(x) > 0 else 0.0
        rows.append({
            "feature": name, "group": FEATURE_GROUPS[name],
            "mean_phishing": x[y_phish == 1].mean(), "mean_legitimate": x[y_phish == 0].mean(),
            "median_phishing": np.median(x[y_phish == 1]), "median_legitimate": np.median(x[y_phish == 0]),
            "auc_phishing": auc, "auc_fold_mean": np.mean(fold_aucs), "auc_fold_std": np.std(fold_aucs),
            "separation": abs(auc - 0.5) * 2,  # 0 = none, 1 = perfect
            "direction": "higher in phishing" if auc > 0.5 else "lower in phishing",
            "spearman_with_phishing": rho,
            "mutual_info_bits": binned_mutual_information(x, y_phish),
            "pct_nonzero": 100 * (x != 0).mean(),
        })
    summary = pd.DataFrame(rows).sort_values("separation", ascending=False)
    summary.to_csv(REPORTS / "feature_summary.csv", index=False, float_format="%.4f")
    pd.DataFrame(stab).to_csv(REPORTS / "feature_stability.csv", index=False, float_format="%.4f")
    print("\nTop 15 features by single-feature separation:\n",
          summary.head(15)[["feature", "group", "auc_phishing", "auc_fold_std", "mutual_info_bits",
                            "direction"]].round(3).to_string(index=False))

    # -------------------------------------------------- 3. redundancy
    corr = X.corr(method="spearman").fillna(0)
    pairs = []
    cols = list(X.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if abs(corr.loc[a, b]) >= 0.90:
                pairs.append({"feature_a": a, "feature_b": b, "spearman": round(corr.loc[a, b], 3)})
    pd.DataFrame(pairs).to_csv(REPORTS / "redundant_pairs.csv", index=False)
    print(f"\n{len(pairs)} feature pairs with |Spearman| >= 0.90")

    # ----------------------------- 4. group-aware diagnostic ablation (train only)
    # These are cross-validation diagnostics to choose / justify feature sets. They are
    # NOT final model results; Members 3 and 4 own model training and evaluation.
    urls = train["URL"].to_numpy(dtype=object)
    results = []

    def run(name, model_factory, make_pre):
        f1s, aucs = [], []
        for tr, te in folds:
            pipe = make_pipeline(make_pre(), model_factory())  # preprocessing fitted per training fold
            pipe.fit(urls[tr], y_phish[tr])
            proba = pipe.predict_proba(urls[te])[:, 1]
            f1s.append(f1_score(y_phish[te], proba >= 0.5))
            aucs.append(roc_auc_score(y_phish[te], proba))
        results.append({"setting": name, "phishing_F1_mean": np.mean(f1s), "phishing_F1_std": np.std(f1s),
                        "ROC_AUC_mean": np.mean(aucs), "ROC_AUC_std": np.std(aucs)})
        print(f"  {name:<46} F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  AUC={np.mean(aucs):.4f}")

    print("\nGroup-aware 5-fold diagnostic (training data only):")
    # Cached features make the RF settings fast; the pipeline-based runs below prove no leakage.
    Xarr = X.to_numpy(float)
    for fs in ("structure", "host", "all"):
        idx = [list(X.columns).index(n) for n in feature_names(fs)]
        f1s, aucs = [], []
        for tr, te in folds:
            rf = RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=3,
                                        n_jobs=-1, random_state=SEED)
            rf.fit(Xarr[tr][:, idx], y_phish[tr])
            p = rf.predict_proba(Xarr[te][:, idx])[:, 1]
            f1s.append(f1_score(y_phish[te], p >= 0.5))
            aucs.append(roc_auc_score(y_phish[te], p))
        results.append({"setting": f"Random Forest, {fs} features ({len(idx)})",
                        "phishing_F1_mean": np.mean(f1s), "phishing_F1_std": np.std(f1s),
                        "ROC_AUC_mean": np.mean(aucs), "ROC_AUC_std": np.std(aucs)})
        print(f"  {results[-1]['setting']:<46} F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  AUC={np.mean(aucs):.4f}")

    def lr():
        return LogisticRegression(max_iter=300, C=1.0, solver="lbfgs", random_state=SEED)

    run("Logistic Regression, host features (scaled)", lr, lambda: make_feature_pipeline("host", scale=True))
    run("Logistic Regression, host + char TF-IDF (scaled)", lr,
        lambda: make_feature_pipeline("host", scale=True, tfidf=True, tfidf_max_features=3000))
    ablation = pd.DataFrame(results)
    ablation.to_csv(REPORTS / "ablation.csv", index=False, float_format="%.4f")

    # Leave-one-group-out (host features only): which host features matter?
    host_cols = feature_names("host")
    host_idx = [list(X.columns).index(n) for n in host_cols]
    rf_all = RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=3,
                                    n_jobs=-1, random_state=SEED).fit(Xarr, y_phish)
    imp = pd.DataFrame({"feature": X.columns, "group": [FEATURE_GROUPS[c] for c in X.columns],
                        "rf_importance": rf_all.feature_importances_}).sort_values("rf_importance", ascending=False)
    imp.to_csv(REPORTS / "rf_importance.csv", index=False, float_format="%.4f")
    rf_host = RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=3,
                                     n_jobs=-1, random_state=SEED).fit(Xarr[:, host_idx], y_phish)
    imp_host = pd.DataFrame({"feature": host_cols, "rf_importance": rf_host.feature_importances_}
                            ).sort_values("rf_importance", ascending=False)
    imp_host.to_csv(REPORTS / "rf_importance_host_only.csv", index=False, float_format="%.4f")
    print("\nTop host-only RF importances:\n", imp_host.head(10).round(4).to_string(index=False))

    # Sanity check: well-known legitimate URLs that have paths or subdomains (hand-written,
    # not from the dataset). Shows what the dataset's structure shortcut does to real inputs.
    real_legit = ["https://www.wikipedia.org", "https://en.wikipedia.org/wiki/Phishing",
                  "https://github.com/python/cpython", "https://docs.python.org/3/library/urllib.parse.html",
                  "https://accounts.google.com/signin", "https://www.bbc.com/news",
                  "https://mail.google.com/mail/u/0/", "https://www.python.org/downloads/"]
    Xreal = extract_features(real_legit)
    chk = pd.DataFrame({"url": real_legit,
                        "P(phishing) all-features RF": rf_all.predict_proba(Xreal.to_numpy(float))[:, 1],
                        "P(phishing) host-only RF": rf_host.predict_proba(Xreal[host_cols].to_numpy(float))[:, 1]})
    chk.to_csv(REPORTS / "real_world_legit_check.csv", index=False, float_format="%.3f")
    print("\nReal-world legitimate URLs (all are legitimate):\n", chk.round(3).to_string(index=False))

    # ------------------------------------------------------------ 5. figures
    # (a) single-feature separation, coloured by group
    top = summary.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = ["#E08A1E" if g == "structure" else "#2C6E9B" for g in top["group"]]
    ax.barh(top["feature"], top["separation"], color=colors)
    ax.set_xlabel("Single-feature separation = 2 × |AUC − 0.5|  (0 none, 1 perfect)")
    ax.set_title("Top 20 features by univariate separation (training data)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#2C6E9B"),
                       plt.Rectangle((0, 0), 1, 1, color="#E08A1E")],
              labels=["host feature", "structure feature"], loc="lower right", frameon=False)
    fig.savefig(FIGS / "fig1_feature_separation.png")
    plt.close(fig)

    # (b) distributions of key features by class
    key = ["url_length", "hostname_length", "num_subdomains", "host_digit_ratio",
           "host_entropy", "sld_hyphens"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, name in zip(axes.ravel(), key):
        x = X[name].to_numpy(float)
        hi = np.percentile(x, 99.5)
        edges = np.linspace(0, hi if hi > 0 else 1, 31) if X[name].nunique() > 8 else \
            np.arange(0, min(int(hi), 8) + 2) - 0.5
        for lab, col, nm in ((1, PHISH, "phishing"), (0, LEGIT, "legitimate")):
            ax.hist(np.clip(x[y_phish == lab], edges[0], edges[-1]), bins=edges, density=True,
                    alpha=0.6, color=col, label=nm)
        ax.set_title(name)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Feature distributions by class (training data, clipped at 99.5th percentile)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_distributions.png")
    plt.close(fig)

    # (c) class mix by structure properties
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    a = audit.set_index("feature").iloc[[0, 1, 2, 9, 10]]
    pos = np.arange(len(a))
    ax.bar(pos - 0.2, a["phishing_with_property_%"], 0.4, color=PHISH, label="phishing")
    ax.bar(pos + 0.2, a["legitimate_with_property_%"], 0.4, color=LEGIT, label="legitimate")
    ax.set_xticks(pos)
    ax.set_xticklabels([s.replace(" (", "\n(").replace("https + www + ", "https + www +\n") for s in a.index], fontsize=8)
    ax.set_ylabel("% of class")
    ax.set_title("Structural shape of the two classes (training data)")
    ax.legend(frameon=False)
    fig.savefig(FIGS / "fig3_structure_audit.png")
    plt.close(fig)

    # (d) correlation heatmap of host features
    hc = X[host_cols].corr(method="spearman").fillna(0)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(hc, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(host_cols)))
    ax.set_xticklabels(host_cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(host_cols)))
    ax.set_yticklabels(host_cols, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman correlation")
    ax.set_title("Correlation between host features (training data)")
    fig.savefig(FIGS / "fig4_host_correlation.png")
    plt.close(fig)

    # (e) ablation
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ab = ablation.iloc[::-1]
    ax.barh(ab["setting"], ab["phishing_F1_mean"], xerr=ab["phishing_F1_std"], color="#5B7DB1")
    ax.set_xlim(max(0, ab["phishing_F1_mean"].min() - 0.1), 1.0)
    ax.set_xlabel("Phishing-class F1, group-aware 5-fold CV on training data (diagnostic)")
    fig.savefig(FIGS / "fig5_ablation.png")
    plt.close(fig)

    meta = {"train_rows": int(len(train)), "phishing_rows": int(y_phish.sum()),
            "n_features": int(X.shape[1]), "folds": FOLDS, "seed": SEED,
            "runtime_seconds": round(time.time() - t0)}
    (REPORTS / "analysis_meta.json").write_text(json.dumps(meta, indent=2))
    print("\nDone.", meta)


if __name__ == "__main__":
    main()
