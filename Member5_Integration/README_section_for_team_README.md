
---

## Member 5 - System integration, alerts and demonstration

Folder: `Member5_Integration/`. Command-line app that validates a URL (Member 2), scores it with the saved model (Member 3), shows a warning with reasons and disables a simulated "Open link" for flagged URLs. Full instructions: `Member5_Integration/README.md`.

```
pip install -r Member2_Phishing_Detection/requirements.txt
cd Member5_Integration
python -m unittest -v          # 12 integration checks
python app.py --demo           # scripted demo
python app.py --url "http://coinbase.com.ticket-94.id/"
python app.py --file sample_urls.txt
```

Input must start with `http://` or `https://`. Demo and sample URLs come from `test.csv` (random sample, seed 42). Known false positives and misses are shown in the demo on purpose.
