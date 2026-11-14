# Member 2 report text: Section 2.3 Feature extraction and analysis

Paste into the team report as Section 2.3. Figures are in `figures/`. Every number below comes from `reports/*.csv`, produced by `analyze_features.py` on `train.csv` only. Edit the wording so it is yours, and check each number against the CSV files before submission.

---

## 2.3 Feature extraction and analysis

### 2.3.1 Approach

Each URL string is converted into numeric features without opening the link. The extractor (`feature_extraction.py`) is a set of pure functions of the URL, so training, evaluation and the application use exactly the same code. Input is validated with the rules Member 1 used: an HTTP or HTTPS scheme, a parseable host name and a valid port. An invalid input raises an error, which the application reports to the user instead of predicting.

We extract 42 features in two groups. 25 host features describe the domain name: lengths, number and length of subdomains, digits, hyphens, Shannon entropy, TLD, IP-address use, internationalised (punycode) names, free-hosting and URL-shortener domains, phishing keywords, and a brand name placed inside or beside an unrelated domain. 17 structure features describe the rest of the URL: total length, https, a leading `www.`, path length and depth, query length and parameters, fragment, digit counts, special and percent-encoded characters, keywords and brand names in the path, and risky file extensions. A leading `www.` is removed before host features are computed, so it does not distort subdomain counts; it is recorded separately as `has_www`. Registrable domains use the same saved Public Suffix List as Member 1.

The keyword, brand, TLD, free-hosting and shortener lists are written by hand from general phishing-awareness knowledge and are not derived from the dataset. They are a design choice and are incomplete.

### 2.3.2 Learned preprocessing and leakage control

The handcrafted features learn nothing from data. Everything that is learned lives in a scikit-learn pipeline (`make_feature_pipeline`): a `log1p` transform followed by standardisation (used for Logistic Regression), and an optional character 3 to 4-gram TF-IDF of the host name. The pipeline is fitted on training data only. Inside cross-validation it is placed in front of the model, so each fold fits it on its own training part. Folds are group-aware (`GroupKFold` on Member 1's `domain_group`), so URLs from one registrable domain never sit in both the training and validation parts of a fold. Unit tests check that the fitted mean equals the training mean, that `transform` changes nothing that was learned, and that the TF-IDF vocabulary contains no n-grams seen only in new data. The test file was not opened during this work.

TF-IDF is not applied to email or message text because this project uses URLs only. We tested character n-grams of the host name as an optional feature (Section 2.3.4).

### 2.3.3 Selected features and reasons

Table 2.3-1 lists the feature families and why each was chosen. The final feature set is decided together with Members 3 and 4 (Section 2.3.5).

| Family | Features | Reason for inclusion |
|---|---|---|
| Length and shape | `url_length`, `hostname_length`, `sld_length`, `longest_label_length`, `path_length`, `query_length` | Phishing URLs often hide the real domain in long or padded names. |
| Subdomains | `num_subdomains`, `subdomain_length`, `num_dots_host` | Attackers place a trusted-looking name before an unrelated domain. |
| Digits, hyphens, entropy | `host_digits`, `host_digit_ratio`, `host_longest_digit_run`, `host_hyphens`, `sld_hyphens`, `host_entropy`, `sld_vowel_ratio` | Machine-generated or look-alike names have more digits and randomness. |
| Host type | `is_ip_host`, `is_idn_punycode`, `has_port`, `has_at_symbol`, `suspicious_tld`, `free_hosting_service`, `is_url_shortener` | Known abuse patterns: raw IP hosts, look-alike characters, cheap TLDs, free site builders, link shorteners. |
| Lexical cues | `host_keyword_count`, `path_keyword_count`, `brand_in_sld_altered`, `brand_in_subdomain`, `brand_in_path`, `risky_extension` | Urgency, account and brand words, and brand impersonation. |
| URL structure | `is_https`, `has_www`, `path_depth`, `num_slashes_path`, `has_double_slash_path`, `num_query_params`, `has_fragment`, `url_digits`, `url_digit_ratio`, `path_query_special_chars`, `percent_encoded_count` | Generic shape of the address. These are strongly tied to dataset construction (Section 2.3.5). |

### 2.3.4 Feature analysis results (training data, 188,674 URLs)

**Single-feature separation.** We measured how well each feature alone ranks phishing above legitimate URLs (ROC AUC with phishing as the positive class, Figure 2.3-1), and checked the value in each of the five group-aware folds. The strongest features were `path_length` (AUC 0.804), `subdomain_length` (0.761), `num_subdomains` (0.757), `url_length` (0.718), `num_dots_host` (0.715), `host_entropy` (0.691), `hostname_length` (0.690) and `host_digits` (0.672). The AUC varied by 0.00 to 0.07 (standard deviation) across folds, so the ranking is stable. Phishing hosts had more subdomains (mean 0.68 against 0.03), more digits (mean 1.87 against 0.05 per host name), longer host names (mean 22.2 against 15.2 characters) and higher entropy (3.58 against 3.30 bits) (Figure 2.3-2).

**Precise indicators.** Several binary host features are rare but nearly pure. Of the URLs flagged, 100% were phishing for `free_hosting_service` (18,969 URLs) and `is_ip_host` (545), 99.9% for `brand_in_subdomain` (3,200), 94.1% for `suspicious_tld` (11,092), 88.3% for `host_keyword_count` > 0 (4,945) and 69.4% for `brand_in_sld_altered` (1,939). They miss most phishing URLs, so they support other features instead of working alone. `has_port`, `has_at_symbol`, `host_other_chars` and `is_idn_punycode` were present in under 0.1% of URLs and carry almost no information in this dataset.

**Redundancy.** Nine feature pairs had Spearman correlation of at least 0.90, for example `host_digits` with `host_digit_ratio` and `host_longest_digit_run` (0.998), `path_length` with `num_slashes_path` (0.996), and `num_subdomains` with `subdomain_length` (0.989). Tree models tolerate this; for Logistic Regression one feature per redundant group can be kept (Figure 2.3-4).

**Tuning-free feature-set comparison.** To justify feature choices we ran a group-aware 5-fold diagnostic on the training data (Random Forest, 100 trees, depth 14; Logistic Regression with the scaling pipeline fitted inside each fold). These are feature-selection diagnostics, not final model results, which Members 3 and 4 report.

| Setting (training-data group CV) | Phishing F1 (mean ± sd) | ROC AUC |
|---|---|---|
| Random Forest, structure features (17) | 0.9971 ± 0.0003 | 0.9983 |
| Random Forest, host features (25) | 0.7740 ± 0.0251 | 0.8690 |
| Random Forest, all features (42) | 0.9977 ± 0.0002 | 0.9987 |
| Logistic Regression, host features | 0.7800 ± 0.0058 | 0.8535 |
| Logistic Regression, host features + host character TF-IDF | 0.8208 ± 0.0018 | 0.9121 |

### 2.3.5 A dataset property that shapes feature selection

The scores above differ greatly between feature groups, and the reason is how the dataset was collected. In the training data **every legitimate URL (100%) uses https, starts with `www.` and has no path, query or fragment**, as the bare home page of a domain. Phishing URLs are varied: 48.5% use https, 41.4% start with `www.`, 60.7% have a path and 72.4% are bare home pages. Only 1.05% of phishing URLs have all three of https, `www.` and a bare home page (Figure 2.3-3). A model can therefore reach F1 of about 0.997 from URL shape alone, with the most important Random Forest features being `is_https`, `num_slashes_path`, `path_length` and `has_www`.

We checked what this does to real inputs by scoring eight well-known legitimate URLs that we wrote by hand (not from the dataset). A Random Forest trained on all 42 features gave a phishing probability of 0.99 or higher for seven of the eight, because they have a path (for example `https://github.com/python/cpython` and `https://accounts.google.com/signin`). The only low score, 0.02, was for the bare home page `https://www.wikipedia.org`. The host-only model was less extreme (0.11 to 0.89) but still unreliable, with several legitimate URLs above 0.5.

For this reason the feature set is not a simple "use everything" decision. We keep the structure features in a separate group and recommend that the team:

1. trains and reports **both** the `all` and `host` feature sets, and presents the `host` results as the more cautious estimate of what the features can detect;
2. does not describe the very high score on all features as evidence of detection on live websites; and
3. discusses this in the Discussion and Limitations sections.

The host-only features still carry some of the same bias, because 97.2% of legitimate URLs have no subdomain after `www.`, which is not true of many real sites (`mail.google.com`).

### 2.3.6 Limitations of the feature work

- Keyword, brand, TLD and hosting lists are fixed by hand and miss many brands, languages and services; they may become outdated.
- Features come from the URL text only. They cannot see page content, redirects, certificates or domain age.
- The data is a historical benchmark. Phishing and legitimate URLs were collected in different ways, so patterns learned here may not hold on current traffic.
- Domain-grouped splits remove one source of leakage but not similarity between phishing campaigns.
- Feature importance values show association in this dataset, not cause.

### Figure captions

- **Figure 2.3-1** (`figures/fig1_feature_separation.png`): Top 20 features by univariate separation in the training data. Orange bars are structure features and blue bars are host features.
- **Figure 2.3-2** (`figures/fig2_distributions.png`): Distribution of six features by class in the training data.
- **Figure 2.3-3** (`figures/fig3_structure_audit.png`): Share of each class with simple structural properties. All legitimate URLs are https, `www.` bare home pages.
- **Figure 2.3-4** (`figures/fig4_host_correlation.png`): Spearman correlation between host features.
- **Figure 2.3-5** (`figures/fig5_ablation.png`): Group-aware 5-fold diagnostic F1 by feature set (training data only).

---

## Append to the team report

**Appendix B, row for the Member 2 student.** Completed tasks and evidence: `feature_extraction.py`, `analyze_features.py`, `test_feature_extraction.py`, `reports/` and `figures/` for feature analysis; Section 2.3. Student review: [write what you checked and changed]. Effort: [your actual hours].

**Appendix C, add to the declaration.** "Claude (Anthropic) assisted Member 2 with designing and writing the feature extraction code and unit tests, running the feature analysis on the training data, and drafting Section 2.3 and the Member 2 README. The student reviewed [describe what you reviewed, ran and changed]."

## Notes for the other members

- **Member 3:** build models on `make_feature_pipeline`. Train and save both `feature_set="all"` and `feature_set="host"` versions, plus the tuned settings. Phishing is the positive class, original label 0.
- **Member 4:** report the confusion matrix and metrics for both feature sets, and use `reports/real_world_legit_check.csv` and false positives with paths as error-analysis examples.
- **Member 5:** call `is_valid_url` before prediction and tell the user when a URL is rejected. Warn that predictions on legitimate URLs with paths may be wrong with the `all` model.
