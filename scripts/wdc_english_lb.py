"""
Extract English LocalBusiness and Product URLs from WDC data.

The main wdc_extract.py filters by English TLDs (.com, .co.uk etc.) but
most WDC LocalBusiness/Product URLs on .com are actually non-English.

This script downloads WDC parts and uses a broader filter:
- Keep .co.uk, .ie, .ca, .com.au, .co.nz URLs (always English)
- For .com/.org/.net: check if the N-Quads contain English-language
  markers (name/description in English) by checking for common English
  words in the schema property values

Appends to wdc_candidate_urls.jsonl.

Usage:
    python3 -u scripts/wdc_english_lb.py
"""
import gzip
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

PROJECT    = Path(__file__).parent.parent
CANDIDATES = PROJECT / 'data' / 'raw' / 'wdc_candidate_urls.jsonl'
CANDIDATES.parent.mkdir(parents=True, exist_ok=True)

WDC_BASE = 'https://data.dws.informatik.uni-mannheim.de/structureddata/2024-12/quads/classspecific'

ENGLISH_TLDS_STRICT = ('.co.uk', '.ie', '.ca', '.com.au', '.co.nz', '.org.uk')
# Common English words that appear in schema property values
ENGLISH_MARKERS = re.compile(
    r'\b(the|and|our|your|we|for|with|from|this|service|restaurant|'
    r'hotel|shop|store|salon|clinic|dental|repair|welcome|contact|'
    r'about|home|call|book|open|price|menu|hours|monday|tuesday|'
    r'wednesday|thursday|friday|saturday|sunday|location|address|'
    r'phone|email|quality|professional|experience|team|customer)\b',
    re.IGNORECASE,
)

TYPES = {
    'LocalBusiness': {'parts': 174, 'quota': 8_000},
    'Product':       {'parts': 1562, 'quota': 5_000},
}


def is_likely_english(nq_line):
    """Check if an N-Quads line contains English-language content."""
    # Extract string literals from the line (text between quotes)
    strings = re.findall(r'"([^"]{10,})"', nq_line)
    if not strings:
        return False
    combined = ' '.join(strings[:3])  # check first few values
    matches = len(ENGLISH_MARKERS.findall(combined))
    return matches >= 2


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

    for schema_type, cfg in TYPES.items():
        quota = cfg['quota']
        type_count = 0
        eng_strict = 0
        eng_detected = 0

        print(f'\n── {schema_type} (quota={quota:,}) ─────────────────────────────')

        for part_idx in range(min(cfg['parts'], 50)):  # cap at 50 parts
            if type_count >= quota:
                break

            url = f'{WDC_BASE}/{schema_type}/{schema_type}_part_{part_idx}.nq.gz'
            print(f'  part_{part_idx}...', end=' ', flush=True)
            try:
                r = requests.get(url, timeout=120)
                if r.status_code == 403:
                    print('403 — skipping')
                    break
                if r.status_code != 200:
                    print(f'{r.status_code}')
                    continue

                text = gzip.decompress(r.content).decode('utf-8', errors='replace')
                lines = text.strip().split('\n')

                added = 0
                for line in lines:
                    # Extract URL from N-Quads
                    url_match = re.search(r'<(https?://[^>]+)>', line)
                    if not url_match:
                        continue
                    page_url = url_match.group(1)
                    if page_url.startswith('http://schema.org') or page_url.startswith('https://schema.org'):
                        continue
                    if page_url in existing:
                        continue

                    host = urlparse(page_url).netloc.lower()

                    # Always keep strict English TLDs
                    if any(host.endswith(t) for t in ENGLISH_TLDS_STRICT):
                        new_urls.append({
                            'url': page_url,
                            'schema_type': schema_type,
                            'source': 'wdc_english',
                        })
                        existing.add(page_url)
                        type_count += 1
                        eng_strict += 1
                        added += 1
                    # For .com/.org/.net: check if content looks English
                    elif host.endswith(('.com', '.org', '.net')):
                        if is_likely_english(line):
                            new_urls.append({
                                'url': page_url,
                                'schema_type': schema_type,
                                'source': 'wdc_english',
                            })
                            existing.add(page_url)
                            type_count += 1
                            eng_detected += 1
                            added += 1

                    if type_count >= quota:
                        break

                print(f'+{added} (total={type_count})')
                time.sleep(1)

            except Exception as e:
                print(f'ERROR: {e}')
                time.sleep(3)

        print(f'  → {schema_type}: {type_count} URLs '
              f'(strict TLD={eng_strict}, detected={eng_detected})')

    # Save
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
