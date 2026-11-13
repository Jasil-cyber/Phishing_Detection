# Member 1 report contribution

## 1 1 Background and problem

Phishing uses impersonation and deceptive digital communication to persuade people to reveal sensitive information. NIST describes it as an attempt to obtain such information while appearing to be a legitimate organisation or trusted person (NIST, n.d.). A link may therefore look plausible while leading to a page controlled by an attacker.

This project investigates whether patterns in a URL string can help distinguish phishing URLs from legitimate ones. Our proposed approach converts each URL into measurable features and applies supervised binary classification. This offers a practical scope for comparing machine learning methods without retrieving live webpages during prediction. A prediction is an indicator for the demonstration, rather than proof that a website is safe.

## 1 2 Aim and objectives

The aim is to develop and evaluate a URL-based phishing detector that returns a Phishing or Legitimate label and presents a warning for a flagged input. The project objectives are to prepare a reproducible labelled dataset; extract URL features consistently for training and new inputs; compare Logistic Regression and Random Forest against a baseline; measure accuracy, phishing precision, recall and F1 score; and demonstrate prediction and warning behaviour through a simple interface.

At this stage, dataset acquisition, cleaning, auditing and the training/test split are complete. Feature extraction, training, model evaluation and the application remain planned work for Members 2 to 5. No model performance is claimed in these sections.

## 1 3 Scope

The selected scope is HTTP/HTTPS URL classification using features derived from URL strings. The dataset's labels are retained as 0 for Phishing and 1 for Legitimate. The proposed application will accept a URL, compute the same features used during training and display the predicted class. Its planned preventive action is a warning with a disabled simulated Open link action for flagged inputs; this has not yet been implemented.

Email bodies, attachments, webpage HTML, live redirection chains and automatic browser-wide blocking are outside this implementation scope. Dataset URLs are treated as text and are not opened during preparation. We will assess performance on a held-out benchmark partition, so the eventual results must not be presented as a guarantee of detection on current live websites.

## 2 1 Dataset source and suitability

We selected the PhiUSIIL Phishing URL dataset by Prasad and Chandra (2024), documented by the UCI Machine Learning Repository. It provides URL strings and binary labels suitable for supervised classification. The UCI documentation defines 1 as legitimate and 0 as phishing and provides the dataset under the Creative Commons Attribution 4.0 licence. These labels and attribution are retained.

The data was acquired on 13 November 2026 through the Ninjax26/phishing-ml CSV ZIP mirror, whose file commit is recorded in reports/acquisition.json. The downloaded CSV contains 55 columns and omits the publisher's unused FILENAME identifier. Common fields in the first 110,768 records matched the available UCI download prefix; this was not a comparison of the entire primary file. The downloaded CSV is preserved unchanged in data/raw, and its SHA256 fingerprint is saved in reports/audit.json.

## Dataset description

An independent scan of the complete downloaded CSV found 235,795 records: 100,945 phishing and 134,850 legitimate. These class totals match the publisher's documented totals. No blank cells were found in the 55 downloaded columns. The implemented detector will use only URL as input and label as target. The other supplied features are excluded from this handoff, allowing Member 2 to define features that can be reproduced from a newly entered URL.

The processed files also contain source_row, which identifies the original data-record number, and domain_group, which controls splitting. Both are audit or grouping fields and must not be model predictors. The target label must also be kept out of the feature matrix.

## 2 2 Data preparation and splitting

The preparation script checks URL and label fields, trims outer whitespace, and validates an HTTP/HTTPS scheme, a parseable hostname and a valid port when supplied. This establishes input usability, not website reachability or safety. The scan found no missing or invalid URL/label records, and no URLs required trimming.

For identical URL strings with conflicting labels, the script is designed to exclude all associated records and record the reason. No such conflicts were found. For repeated URLs with a consistent label, only the first record was kept. This removed 425 duplicate phishing records and retained 235,370 unique URLs: 100,520 phishing (42.71%) and 134,850 legitimate (57.29%). Every excluded record is listed in reports/excluded_rows.csv. No imputation, resampling or learned preprocessing was applied.

## Training and test partitions

To reduce leakage from related URLs, records were grouped by registrable domain using a saved Public Suffix List snapshot. ICANN wildcard and exception rules were applied, while private hosting suffixes were excluded so related hosted tenants stay together. IP-address hosts were grouped by normalized IP. This produced 175,509 domain groups.

Each domain group was assigned deterministically by hashing 42|domain_group with SHA256. Groups whose first 64 hash bits fall in the lowest 20% of the hash range were assigned to testing; all others were assigned to training. This targets an approximate 80/20 split at group level, rather than exact row-level class stratification. The split rule was fixed without using model performance.

The training file contains 188,674 records: 80,725 phishing and 107,949 legitimate. The test file contains 46,696 records: 19,795 phishing and 26,901 legitimate. Testing accounts for 19.84% of cleaned records. Phishing shares are 42.79% in training and 42.39% in testing, so both partitions preserve a similar class mix without resampling.

Checks confirmed zero shared URL strings and zero shared domain groups between partitions, both classes in each partition, and complete accounting of all source rows. A second execution produced identical hashes for all three processed CSV files. Five focused tests also passed for related hosts, suffix rules, IP/IDN handling, invalid inputs and repeatable assignment.

Members 2 and 3 should use train.csv for feature development and group-aware cross-validation. All fitted transformations must be learned inside the training folds. Member 4 should reserve test.csv for the final evaluation after model selection. Domain separation does not eliminate collection bias, campaign similarity or changes in phishing behaviour over time; the split is not chronological.

## References

Prasad, A. and Chandra, S. (2024). PhiUSIIL Phishing URL (Website). UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset (accessed 13 November 2026).

NIST (n.d.). Phishing. Computer Security Resource Center glossary. https://csrc.nist.gov/glossary/term/phishing (accessed 13 November 2026).

Public Suffix List (n.d.). View the Public Suffix List. https://publicsuffix.org/list/ (snapshot downloaded 13 November 2026; original licence notice retained).

Ninjax26 (n.d.). PhiUSIIL_Phishing_URL_Dataset.csv.zip, dataset mirror in phishing-ml, file commit a724836c8b165f73347ea087ae9946dad9e9e293. https://github.com/Ninjax26/phishing-ml/blob/a724836c8b165f73347ea087ae9946dad9e9e293/PhiUSIIL_Phishing_URL_Dataset.csv.zip (accessed 13 November 2026).

## AI use declaration

OpenAI Codex assisted with project planning, the report skeleton, dataset selection, the Member 1 preparation code and tests, execution of the data audit, and drafting the Member 1 report sections and documentation. Automated checks were executed by Codex. The student must record their own review and changes, and the team must add any further AI use in code, prose, images, graphics, tables or captions before submission.
