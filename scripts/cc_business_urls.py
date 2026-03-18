"""
Source LocalBusiness and Product candidate URLs from Common Crawl CDX API.

Strategy: Query CC for .co.uk, .ie, .com.au, .ca homepages and pages
with business-like URL patterns. Not all will be businesses, but the
quality filter + Gemini teacher model will sort them.

For Product: query for /product/, /shop/, /store/ URLs on English TLDs.

Appends to wdc_candidate_urls.jsonl (deduplicates).

Usage:
    python3 -u scripts/cc_business_urls.py
"""
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

PROJECT    = Path(__file__).parent.parent
CANDIDATES = PROJECT / 'data' / 'raw' / 'wdc_candidate_urls.jsonl'
CANDIDATES.parent.mkdir(parents=True, exist_ok=True)

CC_INDEX = 'https://index.commoncrawl.org/CC-MAIN-2026-08-index'
TIMEOUT = 90

# ── Queries ───────────────────────────────────────────────────────────────────
# Each query: (url_pattern, schema_type, limit)
QUERIES = [
    # LocalBusiness: .co.uk homepages (overwhelmingly UK businesses)
    ('*.co.uk/', 'LocalBusiness', 3000),
    ('*.co.uk/about', 'LocalBusiness', 500),
    ('*.co.uk/contact', 'LocalBusiness', 500),
    # Irish businesses
    ('*.ie/', 'LocalBusiness', 2000),
    ('*.ie/about', 'LocalBusiness', 300),
    # Australian businesses
    ('*.com.au/', 'LocalBusiness', 1000),
    # Canadian businesses
    ('*.ca/', 'LocalBusiness', 500),
    # NZ businesses
    ('*.co.nz/', 'LocalBusiness', 300),

    # Product: English TLD shop/product pages
    ('*.co.uk/product/*', 'Product', 1000),
    ('*.co.uk/products/*', 'Product', 1000),
    ('*.co.uk/shop/*', 'Product', 1000),
    ('*.ie/product/*', 'Product', 300),
    ('*.ie/shop/*', 'Product', 300),
    ('*.com.au/product/*', 'Product', 300),
    ('*.com.au/shop/*', 'Product', 300),
]


def query_cdx(pattern, limit=1000):
    """Query CC CDX API and return URLs."""
    params = {
        'url': pattern,
        'output': 'json',
        'limit': limit,
        'fl': 'url',
        'filter': '=status:200',
    }
    try:
        r = httpx.get(CC_INDEX, params=params, timeout=TIMEOUT)
        if r.status_code == 504:
            print(f'    504 timeout — trying smaller limit')
            params['limit'] = min(limit, 200)
            r = httpx.get(CC_INDEX, params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            print(f'    HTTP {r.status_code}')
            return []

        urls = set()
        for line in r.text.strip().split('\n'):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if 'url' in rec:
                    urls.add(rec['url'])
            except json.JSONDecodeError:
                # Sometimes CDX returns plain text URLs
                if line.startswith('http'):
                    urls.add(line.strip())
        return list(urls)
    except Exception as e:
        print(f'    ERROR: {e}')
        return []


def main():
    # Load existing
    existing = set()
    if CANDIDATES.exists():
        with open(CANDIDATES) as f:
            for line in f:
                try:
                    existing.add(json.loads(line)['url'])
                except (json.JSONDecodeError, KeyError):
                    pass
    print(f'Existing pool: {len(existing):,} URLs')

    new_urls = []

    for pattern, schema_type, limit in QUERIES:
        print(f'\n  {schema_type:15s} {pattern:30s} (limit={limit})...', end=' ', flush=True)
        urls = query_cdx(pattern, limit)
        added = 0
        for u in urls:
            if u not in existing:
                new_urls.append({
                    'url': u,
                    'schema_type': schema_type,
                    'source': 'cc_cdx',
                })
                existing.add(u)
                added += 1
        print(f'+{added}')
        time.sleep(2)  # be polite to CDX

    # ── Save ──────────────────────────────────────────────────────────
    print(f'\n── Appending {len(new_urls):,} new URLs to {CANDIDATES}')
    with open(CANDIDATES, 'a') as f:
        for rec in new_urls:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    type_counts = {}
    for rec in new_urls:
        type_counts[rec['schema_type']] = type_counts.get(rec['schema_type'], 0) + 1
    print('\nNew URLs by type:')
    for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f'  {t:20s} {n:6,}')
    print(f'\nTotal pool now: {len(existing):,} URLs')
    print('✓ Done')


if __name__ == '__main__':
    main()
