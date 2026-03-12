"""
Standalone script: replaces notebook 04.
Reads warc_manifest.jsonl, renders each unique domain's HTML to a PNG
screenshot using Playwright (headless Chromium), then rewrites the manifest
with screenshot_path fields.

One screenshot per domain (first URL encountered).
Skips domains whose .png already exists (resume-safe).
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv('.env', override=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

from src.screenshot import batch_render

DATA_DIR        = Path('data/raw')
MANIFEST_PATH   = DATA_DIR / 'warc_manifest.jsonl'
SCREENSHOT_DIR  = Path('data/screenshots')
CONCURRENCY     = 8   # parallel Playwright pages

SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Load manifest ──────────────────────────────────────────────────────────
if not MANIFEST_PATH.exists():
    log.error(f'{MANIFEST_PATH} not found — run run_nb03.py first.')
    sys.exit(1)

log.info(f'Loading manifest from {MANIFEST_PATH}...')
manifest: list[dict] = []
with open(MANIFEST_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            manifest.append(json.loads(line))

log.info(f'Loaded {len(manifest)} manifest entries')

if not manifest:
    log.error('Manifest is empty.')
    sys.exit(1)


# ── 2. Build render items (one per unique domain) ─────────────────────────────
def url_to_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().lstrip('www.')
    except Exception:
        return url

log.info('Building render items (deduplicating by domain)...')
render_items: list[dict] = []
seen_domains: set[str] = set()
missing_html = 0

for entry in manifest:
    html_path = entry.get('html_path', '')
    url       = entry.get('url', '')
    if not html_path or not url:
        continue

    html_file = Path(html_path)
    if not html_file.exists():
        missing_html += 1
        continue

    domain = url_to_domain(url)
    if domain in seen_domains:
        continue
    seen_domains.add(domain)

    # Check if already rendered
    if (SCREENSHOT_DIR / f'{domain}.png').exists():
        continue

    try:
        html = html_file.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        log.warning(f'Could not read {html_file}: {e}')
        continue

    render_items.append({'id': domain, 'html': html})

log.info(f'Unique domains total:      {len(seen_domains)}')
log.info(f'Already rendered (skipped): {len(seen_domains) - len(render_items) - missing_html}')
log.info(f'Missing HTML files:         {missing_html}')
log.info(f'To render now:              {len(render_items)}')


# ── 3. Batch render ───────────────────────────────────────────────────────────
if render_items:
    log.info(f'Launching Playwright — {len(render_items)} screenshots ({CONCURRENCY} workers)...')
    results = asyncio.run(
        batch_render(
            items=render_items,
            output_dir=str(SCREENSHOT_DIR),
            concurrency=CONCURRENCY,
            skip_existing=True,
        )
    )
    success_count = sum(v for v in results.values())
    log.info(f'Rendered: {success_count}/{len(render_items)}  |  Failed: {len(render_items) - success_count}')
else:
    log.info('Nothing new to render.')


# ── 4. Update manifest with screenshot paths ──────────────────────────────────
log.info('Updating manifest with screenshot_path fields...')
updated: list[dict] = []
with_screenshots = 0

for entry in manifest:
    url    = entry.get('url', '')
    domain = url_to_domain(url)
    png    = SCREENSHOT_DIR / f'{domain}.png'
    entry['screenshot_path'] = str(png) if png.exists() else None
    if entry['screenshot_path']:
        with_screenshots += 1
    updated.append(entry)

with open(MANIFEST_PATH, 'w') as f:
    for entry in updated:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

log.info('── Done ──')
log.info(f'  Screenshots on disk:          {with_screenshots}')
log.info(f'  Manifest entries with shot:   {with_screenshots} / {len(updated)}')
log.info(f'  Manifest written:             {MANIFEST_PATH}')
log.info('  Next step: run_nb05.py (training data prep)')
