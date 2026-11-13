"""Prepare the PhiUSIIL URL-only dataset without visiting any dataset URLs.

Run: python prepare_data.py
Inputs remain untouched. Outputs are recreated deterministically.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import ipaddress
import json
from pathlib import Path
import platform
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data/raw/PhiUSIIL_Phishing_URL_Dataset.csv'
SEED = 42
PSL = ROOT / 'resources/public_suffix_list.dat'
FIELDS = ['source_row', 'URL', 'label', 'domain_group']


def load_suffix_rules(path):
    """Read the saved ICANN rules; group hosted tenants conservatively."""
    exact, wildcard, exceptions = set(), set(), set()
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line == '// ===BEGIN PRIVATE DOMAINS===':
            break
        if not line or line.startswith('//'):
            continue
        if line.startswith('!'):
            exceptions.add(line[1:].encode('idna').decode('ascii'))
        elif line.startswith('*.'):
            wildcard.add(line[2:].encode('idna').decode('ascii'))
        else:
            exact.add(line.encode('idna').decode('ascii'))
    return exact, wildcard, exceptions


RULES = load_suffix_rules(PSL)


def registrable_domain(host):
    exact, wildcard, exceptions = RULES
    parts = host.split('.')
    suffix_length = 1  # Default PSL rule is *.
    for i in range(len(parts)):
        tail = '.'.join(parts[i:])
        if tail in exceptions:
            suffix_length = len(parts) - i - 1
            break
        if tail in exact:
            suffix_length = max(suffix_length, len(parts) - i)
        if i > 0 and tail in wildcard:
            suffix_length = max(suffix_length, len(parts) - i + 1)
    return '.'.join(parts[-(suffix_length + 1):])


def split_name(group):
    """Fix each domain's partition by hash; target an approximate 80/20 split."""
    value = int.from_bytes(hashlib.sha256(f'{SEED}|{group}'.encode()).digest()[:8], 'big')
    return 'test' if value < (1 << 64) // 5 else 'train'


def domain_group(url):
    """Return registrable domain (or normalized IP), rejecting unusable URLs."""
    if any(ord(c) < 32 or ord(c) == 127 for c in url):
        raise ValueError('control_character')
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in ('http', 'https'):
        raise ValueError('unsupported_or_missing_scheme')
    host = parsed.hostname
    if not host or any(c.isspace() for c in host):
        raise ValueError('missing_or_invalid_hostname')
    _ = parsed.port  # Raises ValueError for malformed/out-of-range ports.
    host = host.rstrip('.').lower().encode('idna').decode('ascii')
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return registrable_domain(host)


def counts(rows):
    c = Counter(r['label'] for r in rows)
    return {'rows': len(rows), 'phishing_0': c[0], 'legitimate_1': c[1],
            'phishing_percent': round(c[0] * 100 / len(rows), 4) if rows else 0}


def write_csv(path, rows, fields=FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    if not RAW.exists():
        raise FileNotFoundError(f'Download the UCI dataset first: {RAW}')
    csv.field_size_limit(10_000_000)
    audit = {'source': 'https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset',
             'acquisition': json.loads((ROOT / 'reports/acquisition.json').read_text(encoding='utf-8')),
             'input_sha256': hashlib.sha256(RAW.read_bytes()).hexdigest(),
             'seed': SEED, 'label_mapping': {'0': 'Phishing', '1': 'Legitimate'},
             'python_version': platform.python_version(),
             'public_suffix_list_sha256': hashlib.sha256(PSL.read_bytes()).hexdigest()}
    candidates, excluded = [], []
    missing_by_column = Counter()
    labels_raw = Counter()
    trimmed = 0
    with RAW.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        audit['raw_columns'] = reader.fieldnames
        if not {'URL', 'label'}.issubset(reader.fieldnames or []):
            raise ValueError('Expected URL and label columns')
        for i, row in enumerate(reader, 1):
            for col, value in row.items():
                if value is None or not value.strip():
                    missing_by_column[col] += 1
            original = row.get('URL') or ''
            url = original.strip()
            raw_label = (row.get('label') or '').strip()
            labels_raw[raw_label] += 1
            trimmed += (url != original)
            try:
                if not url:
                    raise ValueError('missing_url')
                if raw_label not in ('0', '1'):
                    raise ValueError('missing_or_invalid_label')
                group = domain_group(url)
            except (ValueError, UnicodeError) as exc:
                excluded.append({'source_row': i, 'reason': str(exc),
                                 'URL': original, 'label': raw_label})
                continue
            candidates.append({'source_row': i, 'URL': url,
                               'label': int(raw_label), 'domain_group': group})
    audit['raw_rows'] = sum(labels_raw.values())
    audit['raw_label_counts'] = dict(sorted(labels_raw.items()))
    audit['blank_cells_by_column'] = dict(missing_by_column)
    audit['urls_trimmed'] = trimmed
    audit['invalid_rows'] = len(excluded)
    label_sets = defaultdict(set)
    for row in candidates:
        label_sets[row['URL']].add(row['label'])
    conflicts = {url for url, values in label_sets.items() if len(values) > 1}
    clean, seen = [], set()
    for row in candidates:
        reason = None
        if row['URL'] in conflicts:
            reason = 'conflicting_labels_same_url'
        elif row['URL'] in seen:
            reason = 'duplicate_url_keep_first'
        if reason:
            excluded.append({k: row[k] for k in ['source_row', 'URL', 'label']} | {'reason': reason})
        else:
            clean.append(row)
            seen.add(row['URL'])
    audit['excluded_reasons'] = dict(Counter(r['reason'] for r in excluded))
    audit['conflicting_url_count'] = len(conflicts)
    audit['clean'] = counts(clean)
    audit['unique_domain_groups'] = len({r['domain_group'] for r in clean})
    group_sizes = Counter(r['domain_group'] for r in clean)
    audit['largest_domain_group_rows'] = max(group_sizes.values())
    train = [r for r in clean if split_name(r['domain_group']) == 'train']
    test = [r for r in clean if split_name(r['domain_group']) == 'test']
    audit['split_method'] = 'SHA256 of 42|domain_group, first 64 bits; lowest 20 percent of hash space assigned to test. Approximate 80/20 grouped split, not exact class stratification.'
    audit['split_group_definition'] = 'Registrable domain using saved ICANN Public Suffix List rules; private suffixes excluded; normalized IP or hostname fallback'
    audit['train'], audit['test'] = counts(train), counts(test)
    audit['test_percent'] = round(len(test) / len(clean) * 100, 4)
    audit['url_overlap'] = len({r['URL'] for r in train} & {r['URL'] for r in test})
    audit['domain_overlap'] = len({r['domain_group'] for r in train} & {r['domain_group'] for r in test})
    assert audit['url_overlap'] == audit['domain_overlap'] == 0
    assert len(clean) + len(excluded) == audit['raw_rows']
    assert len(train) + len(test) == len(clean)
    assert {r['label'] for r in train} == {0, 1} == {r['label'] for r in test}
    processed = ROOT / 'data/processed'
    for name, rows in [('clean_urls', clean), ('train', train), ('test', test)]:
        write_csv(processed / f'{name}.csv', rows)
    write_csv(ROOT / 'reports/excluded_rows.csv', excluded,
              ['source_row', 'reason', 'URL', 'label'])
    manifest = []
    for split, rows in [('train', train), ('test', test)]:
        manifest.extend({'source_row': r['source_row'], 'split': split} for r in rows)
    write_csv(ROOT / 'reports/split_manifest.csv', manifest, ['source_row', 'split'])
    audit['output_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(processed.glob('*.csv'))}
    (ROOT / 'reports/audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k != 'raw_columns'}, indent=2))


if __name__ == '__main__':
    main()
