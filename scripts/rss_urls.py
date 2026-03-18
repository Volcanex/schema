"""
Fetch Article, HowTo, JobPosting, and other URLs from RSS feeds + sitemaps.
Appends to wdc_candidate_urls.jsonl (deduplicates against existing).

Usage:
    python3 -u scripts/rss_urls.py
"""
import json
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

PROJECT    = Path(__file__).parent.parent
CANDIDATES = PROJECT / 'data' / 'raw' / 'wdc_candidate_urls.jsonl'
CANDIDATES.parent.mkdir(parents=True, exist_ok=True)

TIMEOUT = 15
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; SchemaBot/1.0; +https://github.com/Volcanex/schema)',
    'Accept': 'application/rss+xml, application/xml, text/xml, text/html',
}

# ── RSS feeds ─────────────────────────────────────────────────────────────────
RSS_FEEDS: dict[str, list[str]] = {
    'Article': [
        # BBC
        'http://feeds.bbci.co.uk/news/rss.xml',
        'http://feeds.bbci.co.uk/news/technology/rss.xml',
        'http://feeds.bbci.co.uk/news/business/rss.xml',
        'http://feeds.bbci.co.uk/news/science_and_environment/rss.xml',
        'http://feeds.bbci.co.uk/news/health/rss.xml',
        'http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml',
        'http://feeds.bbci.co.uk/sport/rss.xml',
        # Guardian
        'https://www.theguardian.com/uk/rss',
        'https://www.theguardian.com/technology/rss',
        'https://www.theguardian.com/business/rss',
        'https://www.theguardian.com/science/rss',
        'https://www.theguardian.com/environment/rss',
        'https://www.theguardian.com/lifeandstyle/rss',
        # Irish
        'https://www.rte.ie/news/rss/news-headlines.xml',
        'https://www.irishtimes.com/cmlink/news-1.1319192',
        # Canada
        'https://www.cbc.ca/webfeed/rss/rss-topstories',
        'https://www.cbc.ca/webfeed/rss/rss-technology',
        # Australia
        'https://www.abc.net.au/news/feed/51120/rss.xml',
        # NZ
        'https://www.stuff.co.nz/rss',
        # Tech
        'https://techcrunch.com/feed/',
        'https://www.wired.com/feed/rss',
        'https://arstechnica.com/feed/',
        'https://feeds.arstechnica.com/arstechnica/technology-lab',
        'https://www.theverge.com/rss/index.xml',
    ],
    'Recipe': [
        'https://www.bbcgoodfood.com/feed',
    ],
    'HowTo': [
        'https://www.wikihow.com/feed.rss',
    ],
    'JobPosting': [
        'https://www.reed.co.uk/rss',
    ],
}

# ── Sitemaps (can yield 10K+ URLs each) ───────────────────────────────────────
SITEMAPS: dict[str, list[str]] = {
    'Article': [
        'https://www.theguardian.com/sitemaps/news.xml',
        'https://www.independent.co.uk/sitemaps/googlenews',
        'https://www.stuff.co.nz/sitemap/news.xml',
        'https://techcrunch.com/news-sitemap.xml',
    ],
}

# ── Curated seed URLs ────────────────────────────────────────────────────────
SEED_URLS: dict[str, list[str]] = {
    'WebSite': [
        'https://www.bbc.co.uk', 'https://www.theguardian.com',
        'https://www.rte.ie', 'https://www.irishtimes.com',
        'https://www.cbc.ca', 'https://www.abc.net.au',
        'https://www.stuff.co.nz', 'https://www.nzherald.co.nz',
        'https://github.com', 'https://stackoverflow.com',
        'https://www.amazon.co.uk', 'https://www.ebay.co.uk',
        'https://www.etsy.com', 'https://www.shopify.com',
        'https://www.booking.com', 'https://www.tripadvisor.com',
        'https://www.linkedin.com', 'https://www.indeed.co.uk',
    ],
    'SoftwareApplication': [
        'https://slack.com', 'https://zoom.us', 'https://notion.so',
        'https://www.figma.com', 'https://code.visualstudio.com',
        'https://www.spotify.com', 'https://www.netflix.com',
        'https://www.dropbox.com', 'https://trello.com',
        'https://www.canva.com', 'https://www.grammarly.com',
    ],
    'Course': [
        'https://www.futurelearn.com/courses',
        'https://www.coursera.org/courses',
        'https://www.udemy.com/courses/',
        'https://www.edx.org/learn',
        'https://www.openlearn.com/courses',
        'https://www.khanacademy.org',
    ],
    'Service': [
        'https://www.checkatrade.com/trades/plumber',
        'https://www.checkatrade.com/trades/electrician',
        'https://www.checkatrade.com/trades/roofer',
        'https://www.checkatrade.com/trades/builder',
        'https://www.checkatrade.com/trades/painter-decorator',
        'https://www.checkatrade.com/trades/locksmith',
    ],
}

# ── extraction helpers ────────────────────────────────────────────────────────

def extract_urls_from_rss(xml: str) -> list[str]:
    """Extract URLs from <link> and <loc> tags in RSS/Atom/Sitemap XML."""
    urls = []
    # RSS <link>
    for m in re.finditer(r'<link[^>]*>([^<]+)</link>', xml):
        u = m.group(1).strip()
        if u.startswith('http'):
            urls.append(u)
    # Atom <link href="..."/>
    for m in re.finditer(r'<link[^>]+href="([^"]+)"', xml):
        u = m.group(1).strip()
        if u.startswith('http'):
            urls.append(u)
    # Sitemap <loc>
    for m in re.finditer(r'<loc>([^<]+)</loc>', xml):
        u = m.group(1).strip()
        if u.startswith('http'):
            urls.append(u)
    return urls


def is_english_tld(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host.endswith(t) for t in (
        '.com', '.co.uk', '.org', '.net', '.ie', '.ca',
        '.com.au', '.co.nz', '.org.uk', '.edu',
    ))


def fetch_url(client: httpx.Client, url: str) -> Optional[str]:
    try:
        r = client.get(url, timeout=TIMEOUT, follow_redirects=True)
        return r.text if r.status_code == 200 else None
    except Exception:
        return None


def main():
    # Load existing URLs for dedup
    existing = set()
    if CANDIDATES.exists():
        with open(CANDIDATES) as f:
            for line in f:
                try:
                    existing.add(json.loads(line)['url'])
                except (json.JSONDecodeError, KeyError):
                    pass
    print(f'Existing pool: {len(existing):,} URLs')

    new_urls: list[dict] = []
    client = httpx.Client(headers=HEADERS)

    # ── RSS feeds ─────────────────────────────────────────────────────────
    print('\n── RSS feeds ───────────────────────────────────────────────────')
    for schema_type, feeds in RSS_FEEDS.items():
        type_count = 0
        for feed_url in feeds:
            xml = fetch_url(client, feed_url)
            if not xml:
                print(f'  FAIL {feed_url}')
                continue
            urls = extract_urls_from_rss(xml)
            for u in urls:
                if u not in existing and is_english_tld(u):
                    new_urls.append({'url': u, 'schema_type': schema_type, 'source': 'rss'})
                    existing.add(u)
                    type_count += 1
            print(f'  {schema_type:15s} +{len(urls):4d} from {feed_url[:60]}')
            time.sleep(0.3)
        print(f'  → {schema_type} total new: {type_count}')

    # ── Sitemaps ──────────────────────────────────────────────────────────
    print('\n── Sitemaps ────────────────────────────────────────────────────')
    for schema_type, sitemap_urls in SITEMAPS.items():
        for sm_url in sitemap_urls:
            xml = fetch_url(client, sm_url)
            if not xml:
                print(f'  FAIL {sm_url}')
                continue
            urls = extract_urls_from_rss(xml)
            count = 0
            for u in urls:
                if u not in existing and is_english_tld(u):
                    new_urls.append({'url': u, 'schema_type': schema_type, 'source': 'sitemap'})
                    existing.add(u)
                    count += 1
            print(f'  {schema_type:15s} +{count:5d} from {sm_url[:60]}')
            time.sleep(1)

    # ── Seed URLs ─────────────────────────────────────────────────────────
    print('\n── Seed URLs ───────────────────────────────────────────────────')
    for schema_type, urls in SEED_URLS.items():
        count = 0
        for u in urls:
            if u not in existing:
                new_urls.append({'url': u, 'schema_type': schema_type, 'source': 'seed'})
                existing.add(u)
                count += 1
        print(f'  {schema_type:20s} +{count}')

    # ── Save ──────────────────────────────────────────────────────────────
    print(f'\n── Appending {len(new_urls):,} new URLs to {CANDIDATES}')
    with open(CANDIDATES, 'a') as f:
        for rec in new_urls:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # Summary
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
