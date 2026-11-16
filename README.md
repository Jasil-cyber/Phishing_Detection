# Member 5: System integration, alerts and demonstration

Command-line application for Project 2 (AI-Powered Phishing Detection). It takes a URL, checks it is valid (Member 2), scores it with the saved model (Member 3), shows an immediate warning with reasons, and disables a simulated "Open link" action for flagged URLs. Every decision is logged.

Nothing is ever opened or downloaded: the URL is only analysed as text.

## Setup
Python 3.10 or later. From the repository root:

```
pip install -r Member2_Phishing_Detection/requirements.txt
cd Member5_Integration
python -m unittest -v          # 12 integration checks
python app.py --demo           # scripted demo
```

Keep the folders `Member1_...`, `Member2_...`, `Member3_...` and `Member5_Integration` side by side. The app imports Member 2's `feature_extraction.py` and Member 3's `predict.py` and `models/best_model.joblib` from the neighbouring folders. The model was saved with scikit-learn 1.9.1; other versions may fail to load it or warn.

## Usage
| Command | What it does |
|---|---|
| `python app.py --url "http://coinbase.com.ticket-94.id/"` | Check one URL |
| `python app.py --file sample_urls.txt` | Batch scan with a summary |
| `python app.py --interactive` | Type URLs one by one (`quit` to exit) |
| `python app.py --demo` | Simulated phishing email, invalid input, and known limitations |
| `--threshold 0.7` | Phishing probability cut-off (default 0.5) |

Input must start with `http://` or `https://`. Anything else (for example `www.google.com`) is reported as invalid, and the link action is blocked, following Member 2's validation rules.

## Workflow
```
URL input -> is_valid_url()  (Member 2)
          -> phishing_probability()  (Member 3, best_model.joblib: raw URL -> probability)
          -> verdict: Phishing (probability >= threshold) / Legitimate / Invalid
          -> alert with reasons (Member 2's feature values)
          -> SimulatedBrowser.open_link(): BLOCKED for Phishing and Invalid
          -> logs/alerts_log.csv
```

## Files
| File | Purpose |
|---|---|
| `app.py` | Interface, alerts, preventive action, logging |
| `predictor.py` | Integration layer connecting Members 2 and 3, and the alert reasons |
| `test_integration.py` | 12 integration checks (uses the real model and a fixed sample of `test.csv`) |
| `sample_urls.txt` | 10 phishing and 10 legitimate URLs, a random sample (seed 42) of Member 1's `test.csv` |
| `outputs/` | Saved demo, batch, invalid-input and interactive outputs |
| `logs/alerts_log.csv` | Decision log written by the app |

## Which model is used
`best_model.joblib` is Member 3's selected model: Logistic Regression on the 25 host features (selected by highest mean cross-validation phishing F1 among the host-feature models). Labels follow Member 1: 0 = phishing, 1 = legitimate. The app reads the probability of class 0 as the phishing probability.

## Limitations
- The warning reasons are simple rules read from the features. They explain what looks suspicious, but they are not the model's exact reasoning, and the model can flag a URL when no rule matches.
- The model flags some real sites, for example `https://accounts.google.com/signin` (92.9%) and `https://mail.google.com/mail/u/0/` (67.9%), and misses some phishing URLs, for example `https://sfr-commande-annulation.com/verification/loading.php` (36.4%). The demo shows these on purpose. Member 4 reports the model's measured performance.
- URL text only: no page content, redirects, certificates or email text.
- "Open link" is simulated. No real browser or email client is connected.
- The sample outputs are a demonstration, not an evaluation. The 20-URL sample is too small to measure performance.
