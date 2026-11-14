"""Unit tests for feature_extraction.py. Run: python -m unittest -v"""
import unittest

import numpy as np

from feature_extraction import (FEATURE_GROUPS, FEATURE_NAMES, extract_features,
                                extract_url_features, feature_names, host_text,
                                is_valid_url, make_feature_pipeline,
                                registrable_domain)


class ExtractionTests(unittest.TestCase):
    def test_every_url_gets_every_feature(self):
        f = extract_url_features("https://www.example.com")
        self.assertEqual(list(f), FEATURE_NAMES)
        self.assertTrue(all(np.isfinite(v) for v in f.values()))

    def test_www_is_not_counted_as_subdomain(self):
        plain = extract_url_features("https://example.com")
        www = extract_url_features("https://www.example.com")
        self.assertEqual(www["has_www"], 1)
        self.assertEqual(plain["has_www"], 0)
        for name in feature_names("host"):  # host features identical with/without www
            self.assertEqual(plain[name], www[name], name)

    def test_phishing_style_url(self):
        f = extract_url_features("http://paypal.secure-login.evil-site.tk/verify/account.php?id=123")
        self.assertEqual(f["is_https"], 0)
        self.assertEqual(f["num_subdomains"], 2)
        self.assertEqual(f["suspicious_tld"], 1)
        self.assertEqual(f["brand_in_subdomain"], 1)
        self.assertEqual(f["risky_extension"], 1)
        self.assertEqual(f["num_query_params"], 1)
        self.assertGreaterEqual(f["host_keyword_count"], 1)

    def test_brand_exact_domain_is_not_flagged(self):
        self.assertEqual(extract_url_features("https://www.paypal.com")["brand_in_sld_altered"], 0)
        self.assertEqual(extract_url_features("https://paypal-secure.com")["brand_in_sld_altered"], 1)

    def test_ip_host_and_port(self):
        f = extract_url_features("http://192.168.0.1:8080/login")
        self.assertEqual(f["is_ip_host"], 1)
        self.assertEqual(f["has_port"], 1)
        self.assertEqual(f["tld_length"], 0)

    def test_suffix_rules_match_member1(self):
        self.assertEqual(registrable_domain("a.b.example.co.uk"), "example.co.uk")
        self.assertEqual(registrable_domain("shop.example.com"), "example.com")

    def test_free_hosting_and_shortener(self):
        self.assertEqual(extract_url_features("https://x1.firebaseapp.com/")["free_hosting_service"], 1)
        self.assertEqual(extract_url_features("https://bit.ly/abc")["is_url_shortener"], 1)

    def test_punycode_and_at_symbol(self):
        self.assertEqual(extract_url_features("http://xn--pple-43d.com")["is_idn_punycode"], 1)
        self.assertEqual(extract_url_features("http://user@evil.com/")["has_at_symbol"], 1)

    def test_groups_cover_all_features(self):
        self.assertEqual(set(FEATURE_GROUPS), set(FEATURE_NAMES))
        self.assertEqual(len(feature_names("host")) + len(feature_names("structure")), len(FEATURE_NAMES))


class InvalidInputTests(unittest.TestCase):
    BAD = ["", "   ", "example.com", "ftp://example.com", "http://", "http://exa mple.com",
           "http://example.com:99999", "javascript:alert(1)", "http://a.com/\x00", None, 42]

    def test_invalid_inputs_rejected(self):
        for bad in self.BAD:
            self.assertFalse(is_valid_url(bad), repr(bad))
            with self.assertRaises(ValueError):
                extract_url_features(bad)

    def test_nan_mode(self):
        df = extract_features(["https://ok.com", "nonsense"], on_invalid="nan")
        self.assertFalse(df.iloc[0].isna().any())
        self.assertTrue(df.iloc[1].isna().all())
        with self.assertRaises(ValueError):
            extract_features(["nonsense"])

    def test_host_text_handles_invalid(self):
        self.assertEqual(host_text(["https://www.a.com", "bad"]), ["a.com", ""])


class PipelineTests(unittest.TestCase):
    TRAIN = ["https://www.alpha.com", "http://beta-login.tk/verify", "https://www.gamma.org",
             "http://192.168.1.5/a.php", "https://www.delta.net", "http://x.web.app/signin"] * 5
    TEST = ["https://www.omega.com", "http://secure-pay.xyz/login"]

    def test_scaler_is_fitted_on_training_data_only(self):
        pipe = make_feature_pipeline("all", scale=True).fit(self.TRAIN)
        scaler = pipe.named_steps["scale"]
        expected = np.log1p(extract_features(self.TRAIN).to_numpy()).mean(axis=0)
        np.testing.assert_allclose(scaler.mean_, expected)
        before = scaler.mean_.copy()
        pipe.transform(self.TEST)  # transform must not change anything learned
        np.testing.assert_array_equal(scaler.mean_, before)

    def test_shapes_and_feature_sets(self):
        for fs in ("all", "host", "structure"):
            X = make_feature_pipeline(fs, scale=True).fit(self.TRAIN).transform(self.TEST)
            self.assertEqual(X.shape, (2, len(feature_names(fs))))

    def test_tfidf_vocabulary_comes_from_training_only(self):
        pipe = make_feature_pipeline("host", scale=True, tfidf=True).fit(self.TRAIN)
        vocab = pipe.transformer_list[1][1].named_steps["tfidf"].vocabulary_
        self.assertNotIn("meg", vocab)  # only appears in a test host (omega)
        X = pipe.transform(self.TEST)
        self.assertEqual(X.shape[0], 2)

    def test_training_and_prediction_use_same_code(self):
        batch = extract_features(self.TEST).iloc[1].to_dict()
        single = extract_url_features(self.TEST[1])
        self.assertEqual(batch, single)


if __name__ == "__main__":
    unittest.main()
