"""Integration checks for the full workflow. Run from this folder: python -m unittest -v"""
import csv
import os
import tempfile
import unittest
from pathlib import Path

import app
import predictor
from predictor import PhishingDetector

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "Member1_Phishing_Detection" / "data" / "processed" / "test.csv"
MODEL = ROOT / "Member3_Model_Development" / "models" / "best_model.joblib"


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = PhishingDetector()
        cls.browser = app.SimulatedBrowser()

    def test_member3_model_file_exists_and_loads(self):
        self.assertTrue(MODEL.exists())
        self.assertIn(predictor.member3.predict_url("https://www.wikipedia.org"), {"Phishing", "Legitimate"})

    def test_label_orientation_matches_member1(self):
        # Member 1: 0 = phishing. The app must treat it that way.
        self.assertEqual(predictor.member3.PHISHING, 0)

    def test_obvious_phishing_is_flagged(self):
        for u in ["http://appleid.apple.com.sign-inx9p3fygrfpyvzdkn.top/",
                  "http://paypal.com@verify-account-882.xyz/login",
                  "http://185.23.77.9/secure/bank/confirm.php"]:
            self.assertEqual(self.d.predict(u)["label"], "Phishing", u)

    def test_normal_sites_pass(self):
        for u in ["https://www.wikipedia.org", "https://www.watson.ch"]:
            self.assertEqual(self.d.predict(u)["label"], "Legitimate", u)

    def test_app_matches_member3_predict_function(self):
        for u in ["https://www.watson.ch", "http://coinbase.com.ticket-94.id/",
                  "https://github.com/python/cpython"]:
            self.assertEqual(self.d.predict(u)["label"], predictor.member3.predict_url(u), u)

    def test_invalid_input_returns_invalid(self):
        for bad in ["", "   ", None, 12345, "not a url ::", "hello", "www.google.com", "ftp://example.com"]:
            self.assertEqual(self.d.predict(bad)["label"], "Invalid", repr(bad))

    def test_open_link_disabled_for_phishing_and_invalid(self):
        for u in ["http://185.23.77.9/secure/bank/confirm.php", "not a url ::"]:
            opened, msg = self.browser.open_link(self.d.predict(u))
            self.assertFalse(opened)
            self.assertIn("BLOCKED", msg)

    def test_open_link_allowed_for_legitimate(self):
        opened, _ = self.browser.open_link(self.d.predict("https://www.watson.ch"))
        self.assertTrue(opened)

    def test_alert_reasons_present_for_phishing(self):
        r = self.d.predict("http://paypal.com@verify-account-882.xyz/login")
        self.assertTrue(r["reasons"])

    def test_threshold_changes_decision(self):
        strict = PhishingDetector(threshold=1.01)
        self.assertEqual(strict.predict("http://185.23.77.9/secure/bank/confirm.php")["label"], "Legitimate")

    def test_fixed_test_csv_sample_workflow(self):
        """End to end on 20 URLs from Member 1's test.csv (seed 42): no crashes, valid labels,
        and the legitimate ones are accepted. Not a performance claim; Member 4 evaluates the model."""
        import pandas as pd
        t = pd.read_csv(TEST_CSV)
        sample = pd.concat([t[t.label == 0].sample(10, random_state=42),
                            t[t.label == 1].sample(10, random_state=42)])
        results = [self.d.predict(u) for u in sample.URL]
        self.assertTrue(all(r["label"] in {"Phishing", "Legitimate"} for r in results))
        legit = [r["label"] for r in results[10:]]
        self.assertEqual(legit.count("Legitimate"), 10)

    def test_logging_writes_row(self):
        original = app.LOG_PATH
        with tempfile.TemporaryDirectory() as tmp:
            app.LOG_PATH = os.path.join(tmp, "log.csv")
            try:
                app.log_result(self.d.predict("https://www.watson.ch"), True)
                with open(app.LOG_PATH, newline="") as f:
                    rows = list(csv.reader(f))
            finally:
                app.LOG_PATH = original
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
