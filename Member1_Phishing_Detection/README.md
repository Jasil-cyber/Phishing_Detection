# Member 1 dataset preparation

This package prepares the URL-only data for Project 2, AI-Powered Phishing Detection. It contains Member 1's data work. Feature extraction, model training, model evaluation and the application are the other members' tasks.

## Run the preparation

Use Python 3.10 or later. No third-party libraries are required.

1. Extract the entire ZIP into a folder.
2. Open a terminal in that folder.
3. Run `python -m unittest -v` to check URL parsing and grouping.
4. Run `python prepare_data.py` to recreate the cleaned data, splits and audit.

On Windows, `py` can be used instead of `python` if that is how Python is installed. The script locates its data relative to its own location. It does not visit any URL from the dataset.

The downloaded mirror CSV is included unchanged. It has 55 columns and omits the publisher's unused FILENAME identifier. Its SHA256 fingerprint is recorded in `reports/audit.json`. The acquisition manifest records the mirror URL and file commit. Common fields in the first 110,768 records were compared with the UCI download and matched; the entire primary file was not compared. To obtain an original publisher copy, download the ZIP from the UCI dataset page and extract `PhiUSIIL_Phishing_URL_Dataset.csv` into `data/raw/`. A changed source file or Public Suffix List snapshot can change the results; keep the supplied copies for reproducibility. If replacing the raw file, update `reports/acquisition.json` to describe your source accurately.

## Files to give the team

- `data/processed/train.csv`: input for feature development, training and cross-validation.
- `data/processed/test.csv`: reserved for final evaluation after the model and settings are fixed.
- `data/processed/clean_urls.csv`: complete cleaned dataset for traceability. Do not use it to bypass the saved split.
- `reports/audit.json`: verified counts, exclusion reasons, split details, overlap checks and file hashes.
- `reports/acquisition.json`: exact source and mirror provenance.
- `reports/excluded_rows.csv`: excluded source records with explicit reasons.
- `reports/split_manifest.csv`: each retained source row and its fixed split.
- `prepare_data.py`: commented, repeatable preparation code.
- `test_prepare_data.py`: checks for grouping, malformed inputs and repeatability.
- `resources/public_suffix_list.dat`: saved domain suffix rules used offline.

## Meaning of the columns

- `source_row`: one-based data-record number in the original CSV, excluding its header. This is an audit identifier, not a predictor.
- `URL`: original URL string with outer whitespace removed. Paths, queries, fragments and case are retained.
- `label`: target class; **0 means phishing and 1 means legitimate**. Never include this column in the feature matrix. When reporting phishing precision, recall and F1 with the original labels, set `pos_label=0`.
- `domain_group`: registrable domain, normalized IP, or hostname fallback, used only to keep related URLs together. Do not use it as a model feature.

Member 2 should build the feature matrix from `URL` only. Keep the original row order or join by `source_row` so labels cannot become misaligned. Do not use the original precomputed URL/webpage features in this URL-only implementation. Do not pass all numeric columns to the model automatically.

## Cleaning rules

The script checks all original columns for blank cells, and validates the two fields the project actually uses. It strips outer URL whitespace, validates labels, checks for an HTTP/HTTPS scheme and a parseable hostname and port, and records invalid rows. It is a parsing check, not a check that a website is reachable or safe.

If an identical trimmed URL has both class labels, all its records are excluded rather than choosing a label. Otherwise, duplicate URL strings are reduced to their first occurrence. Other columns do not override the original target label. No imputation, oversampling, undersampling or feature scaling is performed.

## Splitting rules

Related domains are grouped with the ICANN section of the saved Public Suffix List. Wildcard and exception rules are handled. The private-domain section is intentionally not used: hosted tenants under a domain such as github.io stay in one group. IP hosts are grouped by normalized IP address.

For each domain group, SHA256 hashes the string `42|domain_group`. The first 64 bits determine its partition: the lowest 20% of the hash range is test and the rest is training. This is an approximately 80/20 domain-group split, not an exactly stratified row split. Exact class counts and proportions are recorded in the audit. No model results were used to select the split.

The script asserts zero shared URL strings and zero shared domain groups across the partitions, full row accounting, and the presence of both labels in both partitions. Keep the test partition fixed. Use group-aware cross-validation on training data, passing `domain_group` to the splitter. Any fitted preprocessing, feature selection and tuning must occur inside the training folds.

## Limitations

The supplied labels are accepted as benchmark labels, not independently verified current website verdicts. This dataset is a historical benchmark. Domain separation reduces one form of leakage but does not remove campaign similarities, collection bias or temporal drift. The split is not chronological. URL-only predictions cannot establish whether a page is safe and do not inspect HTML, redirects, attachments or email text.

## Sources and attribution

Prasad, A. and Chandra, S. (2024). *PhiUSIIL Phishing URL (Website)*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

Dataset licence: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/. Derived files select URL and label, add audit/group fields, remove excluded records and create partitions. They are modifications of the original dataset, not the creators' original published files. No endorsement is implied.

Download mirror: https://github.com/Ninjax26/phishing-ml/blob/a724836c8b165f73347ea087ae9946dad9e9e293/PhiUSIIL_Phishing_URL_Dataset.csv.zip. The mirror ZIP contains the CSV used here, without the publisher's FILENAME identifier.

Public Suffix List: https://publicsuffix.org/list/. The supplied snapshot retains its Mozilla Public License 2.0 notice. Its hash is recorded in the audit. Only ICANN rules are used by the script.

NIST. *Phishing*. Computer Security Resource Center glossary. https://csrc.nist.gov/glossary/term/phishing

Source access and dataset acquisition date: 13 November 2026, Dubai time.

## AI use and contribution record

OpenAI Codex assisted with dataset selection, preparation code, checks, analysis, documentation and the Member 1 report text. The execution checks listed in the package were run by Codex. The student should review and understand the work, record their own changes and actual effort, and retain an accurate AI-use declaration under the course rules. No personal review, effort hours or independent student implementation is claimed by this package.
