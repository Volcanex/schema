"""
CC CDX API query script for 01_v2.
Queries Common Crawl index for typed business pages across English-language TLDs.
Uses URL path patterns per TLD (broad *.com queries time out).

WDC (02_v2) covers .com — this script covers .co.uk, .ie, .ca.
Output: data/raw/cc_candidate_urls.jsonl
"""
import json
import time
import requests
from pathlib import Path
from collections import Counter

CRAWL   = 'CC-MAIN-2026-08'
CDX_URL = f'https://index.commoncrawl.org/{CRAWL}-index'
OUT     = Path(__file__).parent.parent / 'data' / 'raw' / 'cc_candidate_urls.jsonl'

# (url_pattern, probable_type, per_page_limit)
# Only TLDs where path-filtered queries don't time out
QUERIES = [
    # UK
    ('*.co.uk/menu*',          'Restaurant',    5_000),
    ('*.co.uk/recipe*',        'Recipe',        5_000),
    ('*.co.uk/events/*',       'Event',         5_000),
    ('*.co.uk/jobs/*',         'JobPosting',    5_000),
    ('*.co.uk/faq*',           'FAQPage',       3_000),
    ('*.co.uk/accommodation*', 'Hotel',         2_000),
    ('*.co.uk/courses/*',      'Course',        2_000),
    ('*.co.uk/products/*',     'Product',       5_000),
    ('*.co.uk/blog/*',         'Article',       5_000),
    ('*.co.uk/about*',         'LocalBusiness', 8_000),
    ('*.co.uk/services*',      'LocalBusiness', 8_000),
    # Ireland
    ('*.ie/menu*',             'Restaurant',    2_000),
    ('*.ie/events/*',          'Event',         2_000),
    ('*.ie/jobs/*',            'JobPosting',    2_000),
    ('*.ie/faq*',              'FAQPage',       1_000),
    ('*.ie/products/*',        'Product',       2_000),
    ('*.ie/blog/*',            'Article',       2_000),
    ('*.ie/about*',            'LocalBusiness', 3_000),
    ('*.ie/services*',         'LocalBusiness', 3_000),
    # Canada
    ('*.ca/menu*',             'Restaurant',    3_000),
    ('*.ca/events/*',          'Event',         3_000),
    ('*.ca/jobs/*',            'JobPosting',    3_000),
    ('*.ca/recipe*',           'Recipe',        2_000),
    ('*.ca/products/*',        'Product',       3_000),
    ('*.ca/blog/*',            'Article',       3_000),
    ('*.ca/about*',            'LocalBusiness', 4_000),
]

def query_cdx(url_pattern: str, limit: int, retries: int = 3) -> list[dict]:
    params = {
        'url':    url_pattern,
        'output': 'json',
        'fl':     'url,languages,status',
        'limit':  limit,
        'filter': 'status:200',
    }
    for attempt in range(retries):
        try:
            r = requests.get(CDX_URL, params=params, timeout=60)
            if r.status_code == 504:
                print(f'    504 timeout on {url_pattern}, skipping')
                return []
            if r.status_code != 200:
                print(f'    HTTP {r.status_code} on {url_pattern}')
                return []
            results = []
            for line in r.text.strip().split('\n'):
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass
            return results
        except requests.Timeout:
            print(f'    Timeout attempt {attempt+1}/{retries} for {url_pattern}')
            time.sleep(5 * (attempt + 1))
    return []

def main():
    print(f'CC CDX query — crawl: {CRAWL}')
    print(f'Output: {OUT}\n')

    all_records = []
    seen_urls   = set()

    for url_pattern, probable_type, limit in QUERIES:
        print(f'  {probable_type:15s}  {url_pattern}  (limit={limit:,})')
        rows = query_cdx(url_pattern, limit)

        added = 0
        for row in rows:
            url  = row.get('url', '')
            lang = row.get('languages', '')
            # English-language filter (eng, or blank)
            if lang and lang not in ('eng', '') and not lang.startswith('eng'):
                continue
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_records.append({
                    'url':          url,
                    'probable_type': probable_type,
                    'source':       'cc',
                    'crawl':        CRAWL,
                })
                added += 1

        print(f'    → {added:,} URLs added  (total: {len(all_records):,})')
        time.sleep(1)  # polite

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        for r in all_records:
            f.write(json.dumps(r) + '\n')

    print(f'\nDone. {len(all_records):,} URLs → {OUT}')
    print('\nType breakdown:')
    for t, n in Counter(r['probable_type'] for r in all_records).most_common():
        print(f'  {t:20s} {n:6,}')

if __name__ == '__main__':
    main()
