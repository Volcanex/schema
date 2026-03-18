"""
Expand candidate URLs by retrying WDC types that 403'd and CDX pagination.
Also derives Service/JobPosting/WebSite/BreadcrumbList from existing URLs.

Usage:
    python3 -u scripts/expand_candidates.py
"""
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

PROJECT    = Path(__file__).parent.parent
CANDIDATES = PROJECT / 'data' / 'raw' / 'wdc_candidate_urls.jsonl'

TIMEOUT = 60
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; SchemaBot/1.0; +https://github.com/Volcanex/schema)',
}

# ── WDC types that 403'd — retry with delays ─────────────────────────────────
WDC_BASE = 'https://data.commoncrawl.org/contrib/webdatacommons/structureddata/2024-12'
WDC_RETRY: dict[str, dict] = {
    'Article':       {'parts': 8,  'quota': 15_000},
    'BlogPosting':   {'parts': 6,  'quota': 5_000},
    'NewsArticle':   {'parts': 5,  'quota': 5_000},
    'WebSite':       {'parts': 5,  'quota': 5_000},
    'BreadcrumbList': {'parts': 4, 'quota': 3_000},
}

ENGLISH_TLDS = ('.com', '.co.uk', '.org', '.net', '.ie', '.ca',
                '.com.au', '.co.nz', '.org.uk', '.edu')


def is_english_tld(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host.endswith(t) for t in ENGLISH_TLDS)


def extract_urls_from_nq(text: str) -> list[str]:
    """Extract URLs from N-Quads text."""
    urls = []
    for m in re.finditer(r'<(https?://[^>]+)>', text):
        u = m.group(1)
        if not u.startswith('http://schema.org') and not u.startswith('https://schema.org'):
            urls.append(u)
    return urls


# ── Derive typed URLs from existing LocalBusiness pool ────────────────────────
def derive_urls(existing: set[str], all_records: list[dict]) -> list[dict]:
    """Derive Service/JobPosting/WebSite/BreadcrumbList from existing URLs."""
    new = []

    lb_domains = set()
    product_urls = []
    article_urls = []

    for rec in all_records:
        t = rec.get('schema_type', '')
        url = rec['url']
        domain = urlparse(url).netloc

        if t == 'LocalBusiness':
            lb_domains.add(domain)
        elif t == 'Product':
            product_urls.append(url)
        elif t == 'Article':
            article_urls.append(url)

    # Service: /services pages on LocalBusiness domains
    for domain in list(lb_domains)[:500]:
        for path in ('/services', '/our-services', '/what-we-do'):
            u = f'https://{domain}{path}'
            if u not in existing:
                new.append({'url': u, 'schema_type': 'Service', 'source': 'derived'})
                existing.add(u)

    # JobPosting: /careers on LocalBusiness domains
    for domain in list(lb_domains)[:300]:
        for path in ('/careers', '/jobs', '/vacancies'):
            u = f'https://{domain}{path}'
            if u not in existing:
                new.append({'url': u, 'schema_type': 'JobPosting', 'source': 'derived'})
                existing.add(u)

    # WebSite: homepage of LocalBusiness domains
    for domain in list(lb_domains)[:300]:
        u = f'https://{domain}/'
        if u not in existing:
            new.append({'url': u, 'schema_type': 'WebSite', 'source': 'derived'})
            existing.add(u)

    # BreadcrumbList: reuse Product + Article URLs (different schema_type label)
    for u in (product_urls + article_urls)[:2000]:
        bc_url = u  # same URL, just different type slot for sampling
        key = f'BC:{bc_url}'
        if key not in existing:
            new.append({'url': bc_url, 'schema_type': 'BreadcrumbList', 'source': 'derived'})
            existing.add(key)

    return new


def main():
    # Load existing
    existing = set()
    all_records = []
    if CANDIDATES.exists():
        with open(CANDIDATES) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    existing.add(rec['url'])
                    all_records.append(rec)
                except (json.JSONDecodeError, KeyError):
                    pass
    print(f'Existing pool: {len(existing):,} URLs')

    new_urls: list[dict] = []
    client = httpx.Client(headers=HEADERS, follow_redirects=True)

    # ── WDC retry ─────────────────────────────────────────────────────────
    print('\n── WDC retry (403\'d types) ─────────────────────────────────────')
    for schema_type, cfg in WDC_RETRY.items():
        type_count = 0
        quota = cfg['quota']
        for part_idx in range(cfg['parts']):
            if type_count >= quota:
                break

            url = f'{WDC_BASE}/class-specific/{schema_type}/{schema_type}_part_{part_idx}.nq.gz'
            print(f'  Trying {schema_type} part {part_idx}...', end=' ', flush=True)
            try:
                r = client.get(url, timeout=TIMEOUT)
                if r.status_code == 403:
                    print('403 — skipping type')
                    time.sleep(5)
                    break
                if r.status_code != 200:
                    print(f'{r.status_code}')
                    continue

                import gzip
                text = gzip.decompress(r.content).decode('utf-8', errors='replace')
                urls = extract_urls_from_nq(text)
                added = 0
                for u in urls:
                    if u not in existing and is_english_tld(u):
                        new_urls.append({'url': u, 'schema_type': schema_type, 'source': 'wdc_retry'})
                        existing.add(u)
                        type_count += 1
                        added += 1
                        if type_count >= quota:
                            break
                print(f'+{added} (total {type_count})')
                time.sleep(3)  # be polite

            except Exception as e:
                print(f'ERROR: {e}')
                time.sleep(5)

        print(f'  → {schema_type}: {type_count} URLs')

    # ── Derive URLs ───────────────────────────────────────────────────────
    print('\n── Deriving Service/JobPosting/WebSite/BreadcrumbList ──────────')
    derived = derive_urls(existing, all_records + new_urls)
    new_urls.extend(derived)
    print(f'  Derived {len(derived):,} URLs')

    # ── Save ──────────────────────────────────────────────────────────────
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

    client.close()


if __name__ == '__main__':
    main()
