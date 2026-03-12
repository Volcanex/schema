"""
Standalone script: replaces notebook 03.
Fetches live HTML for every URL in wdc_ie_records.jsonl + wdc_global_records.jsonl,
then writes warc_manifest.jsonl (url, html_path, wdc_jsonld, schema_match).

Why live fetch instead of CC WARC:
  - WDC 2024-12 is ~3 months old; .ie SME sites rarely change their schema.org
  - Avoids CC Index API (currently returning empty responses after heavy use)
  - Simpler, faster, resumable

Concurrency: CONCURRENCY parallel requests, domain-bucketed so we don't
hammer a single host.  Skips URLs already saved.
"""

import json
import logging
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv('.env', override=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path('data/raw')
HTML_DIR     = DATA_DIR / 'html'
IN_IE        = DATA_DIR / 'wdc_ie_records.jsonl'
IN_GLOB      = DATA_DIR / 'wdc_global_records.jsonl'
OUT_MANIFEST = DATA_DIR / 'warc_manifest.jsonl'

CONCURRENCY  = 30          # parallel HTTP fetches
TIMEOUT      = 15          # seconds per request
MAX_HTML_MB  = 5           # skip pages > 5 MB (JS-heavy SPAs)
MIN_DELAY    = 0.5         # seconds between requests to the SAME domain
MAX_HTML_BYTES = MAX_HTML_MB * 1024 * 1024

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (compatible; BaselineLabs-Research/1.0; '
        '+https://baseline.ie/research)'
    ),
    'Accept': 'text/html,application/xhtml+xml,*/*;q=0.8',
    'Accept-Language': 'en-IE,en;q=0.9',
}

HTML_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Load WDC records ───────────────────────────────────────────────────────
log.info('Loading WDC records...')
url_to_record: dict[str, dict] = {}
url_source:    dict[str, str]  = {}

for path, label in [(IN_IE, 'ie'), (IN_GLOB, 'global')]:
    if not path.exists():
        log.warning(f'{path} not found — skipping')
        continue
    count = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            url = rec.get('source_url', '')
            if url and url not in url_to_record:
                url_to_record[url] = rec
                url_source[url] = label
            count += 1
    log.info(f'  {label}: {count} records, {len(url_to_record)} unique URLs so far')

all_urls = list(url_to_record.keys())
log.info(f'Total unique URLs: {len(all_urls)}')

if not all_urls:
    log.error('No URLs found. Make sure run_nb02.py has completed.')
    sys.exit(1)


# ── 2. Resume: skip already-fetched ──────────────────────────────────────────
def url_to_filename(url: str) -> str:
    safe = url.replace('://', '_').replace('/', '_').replace('?', '_').replace('&', '_')
    return safe[:120] + '.html'

pending = [u for u in all_urls if not (HTML_DIR / url_to_filename(u)).exists()]
already_done = len(all_urls) - len(pending)
log.info(f'Already fetched: {already_done}  |  Remaining: {len(pending)}')


# ── 3. Per-domain rate limiting ───────────────────────────────────────────────
_domain_lock: dict[str, threading.Lock] = defaultdict(threading.Lock)
_domain_last: dict[str, float]          = defaultdict(float)

def _get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return url


def fetch_url(url: str) -> tuple[str, bytes | None]:
    """Fetch URL with per-domain rate limiting. Returns (url, html_bytes or None)."""
    domain = _get_domain(url)

    # Rate-limit per domain
    with _domain_lock[domain]:
        wait = MIN_DELAY - (time.monotonic() - _domain_last[domain])
        if wait > 0:
            time.sleep(wait)
        _domain_last[domain] = time.monotonic()

    try:
        resp = requests.get(
            url,
            headers=HEADERS,
            timeout=TIMEOUT,
            allow_redirects=True,
            stream=True,
        )
        if resp.status_code >= 400:
            log.debug(f'HTTP {resp.status_code} for {url}')
            return url, None

        content_type = resp.headers.get('content-type', '')
        if 'html' not in content_type.lower():
            log.debug(f'Skipping non-HTML content-type {content_type} for {url}')
            return url, None

        # Read with size cap
        chunks = []
        size = 0
        for chunk in resp.iter_content(chunk_size=65536):
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_HTML_BYTES:
                log.debug(f'Page too large (>{MAX_HTML_MB}MB), truncating: {url}')
                break

        return url, b''.join(chunks)

    except requests.exceptions.Timeout:
        log.debug(f'Timeout: {url}')
        return url, None
    except Exception as exc:
        log.debug(f'Fetch error {url}: {exc}')
        return url, None


# ── 4. Parallel fetch ─────────────────────────────────────────────────────────
log.info(f'Fetching {len(pending)} pages ({CONCURRENCY} threads)...')
fetched_ok = 0
fetch_fail  = 0

# Shuffle to spread load across domains
random.shuffle(pending)

with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
    futs = {pool.submit(fetch_url, url): url for url in pending}
    for fut in tqdm(as_completed(futs), total=len(futs), desc='Fetching HTML', unit='url'):
        url, raw_bytes = fut.result()
        if raw_bytes is None:
            fetch_fail += 1
            continue
        html = raw_bytes.decode('utf-8', errors='replace')
        out_path = HTML_DIR / url_to_filename(url)
        try:
            out_path.write_text(html, encoding='utf-8')
            fetched_ok += 1
        except Exception as e:
            log.warning(f'Save failed for {url}: {e}')
            fetch_fail += 1

log.info(f'Fetch done — saved: {fetched_ok + already_done}  |  failed/skipped: {fetch_fail}')


# ── 5. Build manifest ─────────────────────────────────────────────────────────
log.info('Building warc_manifest.jsonl...')
manifest_count = 0

with open(OUT_MANIFEST, 'w') as f:
    for url, wdc_rec in tqdm(url_to_record.items(), desc='Writing manifest', unit='url'):
        html_file = HTML_DIR / url_to_filename(url)
        if not html_file.exists():
            continue
        entry = {
            'url':            url,
            'html_path':      str(html_file),
            'wdc_jsonld':     wdc_rec.get('jsonld'),
            'schema_type':    wdc_rec.get('schema_type', ''),
            'property_count': wdc_rec.get('property_count', 0),
            'source':         url_source.get(url, 'global'),
        }
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        manifest_count += 1

log.info('── Done ──')
log.info(f'  HTML files:       {fetched_ok + already_done}')
log.info(f'  Manifest entries: {manifest_count}')
log.info(f'  Fetch failures:   {fetch_fail}')
log.info(f'  Output:           {OUT_MANIFEST}')
