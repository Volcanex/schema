"""
Smart WDC URL extractor.
Parses already-downloaded WDC part files and extracts English-TLD URLs,
stopping early as soon as the per-type quota is reached.

Also downloads small types (FAQPage, Recipe, etc.) on demand — just enough
parts to hit their quota, not the whole dataset.

Usage:
    python3 scripts/wdc_extract.py
Output:
    data/raw/wdc_candidate_urls.jsonl
"""
import gzip
import json
import re
import time
import requests
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter

PROJECT   = Path(__file__).parent.parent
PARTS_DIR = PROJECT / 'data' / 'raw' / 'wdc' / '_parts'
OUT       = PROJECT / 'data' / 'raw' / 'wdc_candidate_urls.jsonl'
WDC_BASE  = 'https://data.dws.informatik.uni-mannheim.de/structureddata/2024-12/quads/classspecific/'

ENGLISH_TLDS = {
    'com', 'org', 'net', 'ie', 'us', 'biz', 'info',
    'co.uk', 'org.uk', 'me.uk',
    'ca',
    'com.au', 'net.au', 'org.au',
    'co.nz', 'net.nz', 'org.nz',
}

# How many URLs we want per type
QUOTAS = {
    'LocalBusiness': 30_000,
    'Product':       20_000,
    'Event':         12_000,
    'FAQPage':        8_000,
    'Recipe':        10_000,
    'Organization':  10_000,
    'Article':       15_000,
    'BlogPosting':    5_000,
    'NewsArticle':    5_000,
    'Person':         5_000,
    'WebSite':        5_000,
}

# Types with large already-downloaded data
AVAILABLE_LOCALLY = {
    'LocalBusiness': 176,
    'Product':      1562,
    'Event':         130,
}

# Small types to download on-demand (max parts to try)
DOWNLOAD_ON_DEMAND = {
    'FAQPage':    20,
    'Recipe':     30,
    'Organization': 5,   # stop early, it's huge
    'Article':     5,
    'BlogPosting': 5,
    'NewsArticle': 5,
    'Person':     10,
    'WebSite':    10,
}

NQ_SUBJECT_RE  = re.compile(r'^<([^>]+)>\s+<http://schema\.org/[^>]+>')
NQ_URL_RE      = re.compile(r'<([^>]+)>\s+<http://schema\.org/(?:url|sameAs)>\s+"([^"]+)"')
NQ_SUBJECT2_RE = re.compile(r'^<([^>]+)>')


def is_english_tld(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower().lstrip('www.')
        for tld in ENGLISH_TLDS:
            if host.endswith('.' + tld) or host == tld:
                return True
    except Exception:
        pass
    return False


def extract_urls_from_nq_gz(filepath: Path, schema_type: str, quota: int) -> list[dict]:
    """
    Parse a single .nq.gz file and extract (url, schema_type) pairs.
    The NQ format has one triple per line:
        <subject> <predicate> <object> <graph> .
    We extract the subject URL (which is the page URL) from triples.
    """
    urls = []
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8', errors='replace') as f:
            for line in f:
                # Extract subject URL from N-Quads
                m = NQ_SUBJECT2_RE.match(line)
                if not m:
                    continue
                url = m.group(1)
                if not url.startswith('http'):
                    continue
                if not is_english_tld(url):
                    continue
                # Deduplicate within file using a set (handled by caller)
                urls.append(url)
                if len(urls) >= quota * 2:  # extra for dedup
                    break
    except Exception as e:
        print(f'    Warning: error reading {filepath.name}: {e}')
    return urls


def process_type_locally(schema_type: str, quota: int) -> list[dict]:
    """Extract URLs from already-downloaded parts, stopping when quota reached."""
    parts_dir = PARTS_DIR / schema_type
    if not parts_dir.exists():
        print(f'  {schema_type}: parts dir not found, skipping')
        return []

    part_files = sorted(parts_dir.glob('*.gz'))
    print(f'  {schema_type}: {len(part_files)} parts available, quota={quota:,}')

    collected = []
    seen = set()

    for i, part in enumerate(part_files):
        batch = extract_urls_from_nq_gz(part, schema_type, quota - len(collected))
        new = 0
        for url in batch:
            if url not in seen:
                seen.add(url)
                collected.append({'url': url, 'schema_type': schema_type,
                                  'source': 'wdc', 'property_count': 0})
                new += 1

        if (i + 1) % 20 == 0 or len(collected) >= quota:
            print(f'    part {i+1}/{len(part_files)}: {len(collected):,} URLs so far')

        if len(collected) >= quota:
            print(f'    ✓ quota reached at part {i+1}')
            break

    print(f'  {schema_type}: {len(collected):,} URLs extracted')
    return collected


def download_and_extract(schema_type: str, max_parts: int, quota: int) -> list[dict]:
    """Download a few parts of a small type and extract URLs."""
    parts_dir = PARTS_DIR / schema_type
    parts_dir.mkdir(parents=True, exist_ok=True)

    collected = []
    seen = set()
    base_url = WDC_BASE + schema_type + '/'

    print(f'  {schema_type}: downloading up to {max_parts} parts...')

    for i in range(max_parts):
        fname = f'part_{i}.gz'
        fpath = parts_dir / fname

        if not fpath.exists():
            url = base_url + fname
            try:
                r = requests.get(url, stream=True, timeout=120)
                if r.status_code == 404:
                    print(f'    part_{i}: 404 — no more parts')
                    break
                r.raise_for_status()
                # Validate it's gzip
                first = b''
                chunks = []
                for chunk in r.iter_content(8192):
                    if not first:
                        first = chunk
                        if first[:2] != b'\x1f\x8b':
                            print(f'    part_{i}: not gzip, skipping')
                            break
                    chunks.append(chunk)
                if first[:2] != b'\x1f\x8b':
                    continue
                with open(fpath, 'wb') as f:
                    for chunk in chunks:
                        f.write(chunk)
                size_mb = fpath.stat().st_size / 1e6
                print(f'    part_{i}: downloaded {size_mb:.0f}MB')
            except Exception as e:
                print(f'    part_{i}: failed — {e}')
                continue
            time.sleep(0.5)

        # Extract
        batch = extract_urls_from_nq_gz(fpath, schema_type, quota - len(collected))
        for url in batch:
            if url not in seen:
                seen.add(url)
                collected.append({'url': url, 'schema_type': schema_type,
                                  'source': 'wdc', 'property_count': 0})

        if len(collected) >= quota:
            print(f'    ✓ quota reached at part_{i}')
            break

    print(f'  {schema_type}: {len(collected):,} URLs')
    return collected


def main():
    print('WDC URL Extractor')
    print(f'Output: {OUT}\n')

    all_records = []

    # Process already-downloaded types
    print('=== Processing locally available types ===')
    for schema_type, quota in QUOTAS.items():
        if schema_type in AVAILABLE_LOCALLY:
            records = process_type_locally(schema_type, quota)
            all_records.extend(records)
            print()

    # Download small types on demand
    print('=== Downloading small types on demand ===')
    for schema_type, max_parts in DOWNLOAD_ON_DEMAND.items():
        if schema_type not in AVAILABLE_LOCALLY:
            quota = QUOTAS.get(schema_type, 5_000)
            records = download_and_extract(schema_type, max_parts, quota)
            all_records.extend(records)
            print()

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        for r in all_records:
            f.write(json.dumps(r) + '\n')

    print(f'Total: {len(all_records):,} URLs → {OUT}')
    print('\nType breakdown:')
    for t, n in Counter(r['schema_type'] for r in all_records).most_common():
        print(f'  {t:25s} {n:6,}')


if __name__ == '__main__':
    main()
