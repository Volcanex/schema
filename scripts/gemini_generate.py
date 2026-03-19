"""
Bulk schema.org JSON-LD generation using Gemini 2.5 Flash.

Reads pages_for_generation.jsonl, sends HTML + screenshot to Gemini,
validates the output, and writes results to data/generated/.

Usage:
    python3 -u scripts/gemini_generate.py [--workers 16] [--dry-run]

Requires: GEMINI_API_KEY in .env
"""
import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Setup paths
PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))
load_dotenv(PROJECT / '.env', override=True)

from src.schema_validator import validate_jsonld

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# --- Config ---
JSONL_PATH = PROJECT / 'data' / 'processed' / 'pages_for_generation.jsonl'
SHOT_DIR   = PROJECT / 'data' / 'screenshots_v2'
OUT_DIR    = PROJECT / 'data' / 'generated'
MAX_HTML_CHARS = 12_000   # ~3K tokens
MODEL = 'gemini-2.5-flash'

SYSTEM_PROMPT = """\
You are a schema.org expert. Given a webpage screenshot and its HTML source, \
generate the optimal JSON-LD markup for the page.

Rules:
- Output ONLY valid JSON-LD (no markdown, no explanation, no code fences)
- Use the most specific @type (e.g. Restaurant not LocalBusiness, NewsArticle not Article)
- Include ALL properties extractable from the visible content
- Use nested entities (PostalAddress, Offer, Rating, etc.)
- Always include "@context": "https://schema.org"
- For LocalBusiness: include name, address, telephone, openingHours if visible
- For Article: include headline, datePublished, author, publisher
- For Recipe: include name, recipeIngredient, recipeInstructions, cookTime
- For Event: include name, startDate, location, organizer
- For Product: include name, description, offers with price and currency
- Do NOT invent data not present on the page
- If the page clearly represents multiple schema types, output a @graph array"""


def load_pages():
    """Load unique pages from JSONL."""
    pages = []
    seen = set()
    with open(JSONL_PATH) as f:
        for line in f:
            rec = json.loads(line)
            url = rec['url']
            if url not in seen:
                pages.append(rec)
                seen.add(url)
    return pages


def filter_ready(pages):
    """Filter to pages that have both HTML and screenshot."""
    ready = []
    for p in pages:
        h = hashlib.md5(p['url'].encode()).hexdigest()
        shot_path = SHOT_DIR / f'{h}.png'
        if len(p.get('html', '')) > 200 and shot_path.exists():
            p['_shot_path'] = str(shot_path)
            p['_hash'] = h
            ready.append(p)
    return ready


async def generate_one(client, page, semaphore, stats):
    """Generate JSON-LD for a single page."""
    out_path = OUT_DIR / f"{page['_hash']}.json"

    # Skip if already generated
    if out_path.exists():
        stats['skipped'] += 1
        return

    html = page['html'][:MAX_HTML_CHARS]
    if len(page['html']) > MAX_HTML_CHARS:
        html += '\n<!-- [HTML truncated] -->'

    # Read screenshot
    with open(page['_shot_path'], 'rb') as f:
        shot_b64 = base64.b64encode(f.read()).decode()

    contents = [
        {'inline_data': {'mime_type': 'image/png', 'data': shot_b64}},
        {'text': f'Target schema type hint: {page["schema_type"]}\n\nHTML:\n{html}'},
    ]

    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=contents,
                    config={
                        'system_instruction': SYSTEM_PROMPT,
                        'max_output_tokens': 2000,
                        'temperature': 0.1,
                    },
                )
                raw = response.text.strip()
                break
            except Exception as e:
                if 'quota' in str(e).lower() or 'rate' in str(e).lower() or '429' in str(e):
                    wait = 2 ** attempt * 10
                    log.warning(f'Rate limited, waiting {wait}s...')
                    await asyncio.sleep(wait)
                elif attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    log.warning(f'Failed {page["url"][:60]}: {e}')
                    stats['failed'] += 1
                    return
        else:
            stats['failed'] += 1
            return

    # Strip markdown fences if present
    if raw.startswith('```'):
        lines = raw.split('\n')
        raw = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])

    # Validate
    validation = validate_jsonld(raw)

    result = {
        'url': page['url'],
        'schema_type_hint': page['schema_type'],
        'generated_jsonld': raw,
        'valid': validation['valid'],
        'quality_score': validation['quality_score'],
        'schema_types': validation['schema_types'],
        'property_count': validation['property_count'],
        'errors': validation['errors'],
    }

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if validation['valid']:
        stats['valid'] += 1
    else:
        stats['invalid'] += 1

    stats['done'] += 1
    if stats['done'] % 100 == 0:
        total = stats['done'] + stats['skipped']
        elapsed = time.time() - stats['start']
        rate = stats['done'] / elapsed * 60 if elapsed > 0 else 0
        log.info(
            f"Progress: {total:,}/{stats['total']:,} "
            f"({stats['valid']} valid, {stats['invalid']} invalid, "
            f"{stats['failed']} failed, {stats['skipped']} skipped) "
            f"[{rate:.0f}/min]"
        )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16,
                        help='Concurrent requests (default 16)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just show stats, don\'t generate')
    parser.add_argument('--limit', type=int, default=0,
                        help='Max pages to process (0=all)')
    args = parser.parse_args()

    # Load and filter
    log.info('Loading pages...')
    pages = load_pages()
    ready = filter_ready(pages)
    log.info(f'Total unique: {len(pages):,}, ready (HTML+screenshot): {len(ready):,}')

    # Check already done
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    already_done = len(list(OUT_DIR.glob('*.json')))
    log.info(f'Already generated: {already_done:,}')

    if args.dry_run:
        # Cost estimate
        n = len(ready) - already_done
        input_tokens = n * 5_300
        output_tokens = n * 1_000
        cost = (input_tokens / 1e6) * 0.15 + (output_tokens / 1e6) * 0.60
        log.info(f'Would generate {n:,} pages')
        log.info(f'Estimated cost: ${cost:.2f} (Gemini 2.5 Flash standard)')
        return

    if args.limit:
        ready = ready[:args.limit]

    # Init Gemini client
    from google import genai
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        log.error('GEMINI_API_KEY not set in .env')
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    stats = {
        'done': 0, 'valid': 0, 'invalid': 0, 'failed': 0,
        'skipped': 0, 'total': len(ready), 'start': time.time(),
    }

    semaphore = asyncio.Semaphore(args.workers)

    log.info(f'Starting generation with {args.workers} workers...')
    # Process in batches of 500 to avoid building too large a task list
    batch_size = 500
    for i in range(0, len(ready), batch_size):
        batch = ready[i:i + batch_size]
        tasks = [generate_one(client, page, semaphore, stats) for page in batch]
        await asyncio.gather(*tasks)
        log.info(f'Batch {i // batch_size + 1} done')

    elapsed = time.time() - stats['start']
    log.info(
        f'\n=== COMPLETE ===\n'
        f'Total processed: {stats["done"] + stats["skipped"]:,}\n'
        f'New generations: {stats["done"]:,}\n'
        f'  Valid:   {stats["valid"]:,}\n'
        f'  Invalid: {stats["invalid"]:,}\n'
        f'  Failed:  {stats["failed"]:,}\n'
        f'Skipped (existing): {stats["skipped"]:,}\n'
        f'Time: {elapsed / 60:.1f} min'
    )


if __name__ == '__main__':
    asyncio.run(main())
