"""Run with python -m unittest -v. Tests do not contact any websites."""
import unittest
from prepare_data import domain_group, split_name, registrable_domain


class PreparationChecks(unittest.TestCase):
    def test_related_hosts_group_together(self):
        self.assertEqual(domain_group('https://login.example.co.uk/a'), 'example.co.uk')
        self.assertEqual(domain_group('http://WWW.EXAMPLE.CO.UK/b'), 'example.co.uk')
        self.assertEqual(domain_group('https://alice.github.io/'), 'github.io')

    def test_public_suffix_wildcard_and_exception(self):
        self.assertEqual(registrable_domain('a.b.ck'), 'a.b.ck')
        self.assertEqual(registrable_domain('a.www.ck'), 'www.ck')

    def test_ip_and_idn(self):
        self.assertEqual(domain_group('http://192.0.2.1/path'), '192.0.2.1')
        self.assertEqual(domain_group('http://[2001:db8::1]/'), '2001:db8::1')
        self.assertEqual(domain_group('https://bücher.de/'), 'xn--bcher-kva.de')

    def test_bad_input(self):
        for value in ['', 'javascript:alert(1)', 'https:///bad', 'https://x.com:abc/', 'https://bad host/']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                domain_group(value)

    def test_partition_is_repeatable(self):
        group = domain_group('https://a.example.com/')
        self.assertEqual(split_name(group), split_name(domain_group('http://b.example.com/path')))
        self.assertIn(split_name(group), ('train', 'test'))


if __name__ == '__main__':
    unittest.main()
