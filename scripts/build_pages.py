"""
03_v2 pipeline as a standalone script.
Fetches HTML from WDC candidate URLs, takes Playwright screenshots,
quality-filters, and outputs pages_for_generation.jsonl.

Fully resumable — fetched HTML cached to disk, screenshots cached too.

Usage:
    python3 -u scripts/build_pages.py
"""
import asyncio
import hashlib
import json
import re
import random
import time
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import httpx
from playwright.async_api import async_playwright

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT       = Path(__file__).parent.parent
CANDIDATES    = PROJECT / 'data' / 'raw' / 'wdc_candidate_urls.jsonl'
HTML_CACHE    = PROJECT / 'data' / 'raw' / 'html_v2'
SHOT_DIR      = PROJECT / 'data' / 'screenshots_v2'
OUT           = PROJECT / 'data' / 'processed' / 'pages_for_generation.jsonl'
TYPE_CFG      = PROJECT / 'config' / 'type_distribution.json'

HTML_CACHE.mkdir(parents=True, exist_ok=True)
SHOT_DIR.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── config ───────────────────────────────────────────────────────────────────
FETCH_CONCURRENCY = 100
FETCH_TIMEOUT     = 8       # seconds
MIN_TEXT_CHARS    = 400
MAX_HTML_CHARS    = 200_000
TARGET_PAGES      = 20_000  # final output size
SCREENSHOT_WORKERS = 4      # parallel Playwright browsers (safe for 16GB RAM)

HEADERS = {
    'User-Agent':      'Mozilla/5.0 (compatible; SchemaBot/1.0; +https://github.com/Volcanex/schema)',
    'Accept':          'text/html,application/xhtml+xml',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
}

PARKED = re.compile(
    r'domain (is |for sale|parking)|coming soon|under construction|'
    r'buy this domain|this site is for sale|404 not found|403 forbidden|'
    r'account suspended|website coming|placeholder',
    re.IGNORECASE,
)

# ── helpers ──────────────────────────────────────────────────────────────────
def url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def html_cache_path(url: str) -> Path:
    return HTML_CACHE / f'{url_hash(url)}.html'

def shot_cache_path(url: str) -> Path:
    return SHOT_DIR / f'{url_hash(url)}.png'

def extract_text(html: str) -> str:
    html = re.sub(r'<(script|style)[^>]*>.*?</(script|style)>', '', html,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', html)
    return re.sub(r'\s+', ' ', text).strip()

def quality_ok(html: str) -> bool:
    text = extract_text(html)
    if len(text) < MIN_TEXT_CHARS:
        return False
    if PARKED.search(text[:2000]):
        return False
    # basic English heuristic (avoid importing langdetect for speed)
    # check for common English stopwords in first 500 chars
    sample = text[:500].lower()
    eng_words = sum(1 for w in ('the', 'and', 'for', 'with', 'this', 'that',
                                'our', 'your', 'you', 'are', 'we', 'has',
                                'from', 'more', 'about') if f' {w} ' in sample)
    return eng_words >= 2

# ── stage 1: async fetch ─────────────────────────────────────────────────────
async def fetch_one(client: httpx.AsyncClient, record: dict,
                    sem: asyncio.Semaphore):
    url   = record['url']
    cache = html_cache_path(url)

    if cache.exists():
        html = cache.read_text(errors='replace')
        return {**record, 'html': html}

    async with sem:
        try:
            r = await client.get(url, timeout=FETCH_TIMEOUT, follow_redirects=True)
            if r.status_code != 200:
                return None
            ct = r.headers.get('content-type', '')
            if 'html' not in ct and 'text' not in ct:
                return None
            html = r.text[:MAX_HTML_CHARS]
            if len(html) < MIN_TEXT_CHARS:
                return None
            cache.write_text(html, errors='replace')
            return {**record, 'html': html}
        except Exception:
            return None

async def fetch_all(records: list[dict]) -> list[dict]:
    sem     = asyncio.Semaphore(FETCH_CONCURRENCY)
    results = []
    done    = 0
    total   = len(records)

    async with httpx.AsyncClient(headers=HEADERS,
                                  limits=httpx.Limits(max_connections=FETCH_CONCURRENCY + 20)) as client:
        tasks = [fetch_one(client, r, sem) for r in records]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            done  += 1
            if result:
                results.append(result)
            if done % 500 == 0 or done == total:
                cached = sum(1 for r in results if html_cache_path(r['url']).stat().st_size > 0
                             and r.get('html'))
                print(f'  fetched {done:,}/{total:,}  ok={len(results):,}', flush=True)
    return results

# ── stage 2: parallel screenshots ─────────────────────────────────────────────
async def screenshot_worker(records: list[dict], shared: dict,
                            lock: asyncio.Lock) -> list[dict]:
    """Single Playwright worker processing its chunk of records."""
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            viewport={'width': 1280, 'height': 900},
            user_agent=HEADERS['User-Agent'],
        )
        for rec in records:
            url  = rec['url']
            path = shot_cache_path(url)
            ok = False

            if path.exists():
                results.append({**rec, 'screenshot_path': str(path)})
                ok = True
            else:
                page = await ctx.new_page()
                try:
                    await page.goto(url, timeout=12_000, wait_until='domcontentloaded')
                    await page.screenshot(path=str(path), full_page=False)
                    results.append({**rec, 'screenshot_path': str(path)})
                    ok = True
                except Exception:
                    pass
                finally:
                    await page.close()

            async with lock:
                shared['done'] += 1
                if ok:
                    shared['ok'] += 1
                if shared['done'] % 500 == 0 or shared['done'] == shared['total']:
                    print(f'  screenshots {shared["done"]:,}/{shared["total"]:,}  '
                          f'ok={shared["ok"]:,}', flush=True)

        await ctx.close()
        await browser.close()
    return results


async def screenshot_all(records: list[dict]) -> list[dict]:
    """Run SCREENSHOT_WORKERS parallel Playwright browsers."""
    n = SCREENSHOT_WORKERS
    chunk_size = max(1, (len(records) + n - 1) // n)
    chunks = [records[i:i+chunk_size] for i in range(0, len(records), chunk_size)]
    shared = {'done': 0, 'ok': 0, 'total': len(records)}
    lock = asyncio.Lock()
    print(f'  Launching {len(chunks)} parallel Playwright workers...', flush=True)
    results_nested = await asyncio.gather(
        *[screenshot_worker(chunk, shared, lock) for chunk in chunks]
    )
    return [page for worker_results in results_nested for page in worker_results]

# ── main ──────────────────────────────────────────────────────────────────────
async def main():
    t0 = time.time()

    # Load type distribution targets
    with open(TYPE_CFG) as f:
        cfg = json.load(f)
    weights     = {k: v['weight'] for k, v in cfg['types'].items()}
    total_w     = sum(weights.values())
    type_targets = {t: max(50, int(TARGET_PAGES * w / total_w))
                    for t, w in weights.items()}

    # ── load candidates ───────────────────────────────────────────────────────
    candidates = []
    with open(CANDIDATES) as f:
        for line in f:
            candidates.append(json.loads(line))
    print(f'Loaded {len(candidates):,} candidate URLs')

    # Sample per type with 4x oversampling (filter will cull ~60%)
    by_type = {}
    for r in candidates:
        by_type.setdefault(r['schema_type'], []).append(r)

    to_fetch = []
    for t, target in type_targets.items():
        pool = by_type.get(t, [])
        n    = min(len(pool), target * 4)
        to_fetch.extend(random.sample(pool, n) if n < len(pool) else pool)

    # Also include types not in config
    known = set(type_targets)
    for t, pages in by_type.items():
        if t not in known:
            to_fetch.extend(pages[:1000])

    random.shuffle(to_fetch)
    print(f'Fetching {len(to_fetch):,} URLs ({FETCH_CONCURRENCY} concurrent)…\n')

    # ── stage 1: fetch ────────────────────────────────────────────────────────
    print('── Stage 1: HTTP fetch ──────────────────────────────────────────')
    fetched = await fetch_all(to_fetch)
    print(f'\nFetch complete: {len(fetched):,} / {len(to_fetch):,} OK  '
          f'({len(fetched)/len(to_fetch)*100:.0f}%)')

    # ── stage 2: quality filter ───────────────────────────────────────────────
    print('\n── Stage 2: Quality filter ──────────────────────────────────────')
    filtered = [r for r in fetched if quality_ok(r.get('html', ''))]
    print(f'Quality filter: {len(filtered):,} / {len(fetched):,} passed '
          f'({len(filtered)/len(fetched)*100:.0f}%)')

    # ── stage 3: sample to targets ────────────────────────────────────────────
    print('\n── Stage 3: Sample to type targets ──────────────────────────────')
    by_type_filtered = {}
    for r in filtered:
        by_type_filtered.setdefault(r['schema_type'], []).append(r)

    pre_shot = []
    for t, target in type_targets.items():
        pool = by_type_filtered.get(t, [])
        n    = min(len(pool), target)
        pre_shot.extend(random.sample(pool, n) if n < len(pool) else pool)
        print(f'  {t:25s}  have={len(pool):5,}  taking={n:4,}')

    random.shuffle(pre_shot)
    print(f'\nPre-screenshot count: {len(pre_shot):,}')

    # ── stage 4: screenshots ──────────────────────────────────────────────────
    print('\n── Stage 4: Playwright screenshots ──────────────────────────────')
    with_shots = await screenshot_all(pre_shot)
    print(f'\nScreenshots: {len(with_shots):,} / {len(pre_shot):,} OK')

    # ── stage 5: save ─────────────────────────────────────────────────────────
    print('\n── Stage 5: Save ────────────────────────────────────────────────')
    with open(OUT, 'w') as f:
        for page in with_shots:
            f.write(json.dumps({
                'url':             page['url'],
                'schema_type':     page['schema_type'],
                'html':            page.get('html', ''),
                'screenshot_path': page.get('screenshot_path', ''),
                'source':          page.get('source', 'wdc'),
            }, ensure_ascii=False) + '\n')

    size_mb = OUT.stat().st_size / 1e6
    elapsed = (time.time() - t0) / 60
    print(f'\nSaved {len(with_shots):,} pages → {OUT} ({size_mb:.0f} MB)')
    print(f'Total time: {elapsed:.1f} min')

    print('\nType breakdown:')
    for t, n in Counter(p['schema_type'] for p in with_shots).most_common():
        print(f'  {t:25s} {n:5,}')

    print('\n✓ Done — ready for 03.5_v2_pages_review.ipynb')

if __name__ == '__main__':
    asyncio.run(main())
