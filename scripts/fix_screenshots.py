"""
Re-take missing screenshots for pages_for_generation.jsonl.
Checks which pages are missing screenshots and renders them with Playwright.

Usage:
    python3 -u scripts/fix_screenshots.py
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path

from playwright.async_api import async_playwright

PROJECT  = Path(__file__).parent.parent
JSONL    = PROJECT / 'data' / 'processed' / 'pages_for_generation.jsonl'
SHOT_DIR = PROJECT / 'data' / 'screenshots_v2'
WORKERS  = 8

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; SchemaBot/1.0; +https://github.com/Volcanex/schema)',
}


async def screenshot_worker(records, shared, lock):
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            viewport={'width': 1280, 'height': 900},
            user_agent=HEADERS['User-Agent'],
        )
        for rec in records:
            url = rec['url']
            path = SHOT_DIR / f"{hashlib.md5(url.encode()).hexdigest()}.png"

            if path.exists():
                async with lock:
                    shared['skip'] += 1
                continue

            page = await ctx.new_page()
            try:
                await page.goto(url, timeout=12_000, wait_until='domcontentloaded')
                await page.screenshot(path=str(path), full_page=False)
                results.append(str(path))
            except Exception:
                pass
            finally:
                await page.close()

            async with lock:
                shared['done'] += 1
                if shared['done'] % 200 == 0:
                    print(f"  rendered {shared['done']:,} / {shared['total']:,}  "
                          f"(skipped {shared['skip']:,} cached)", flush=True)

        await ctx.close()
        await browser.close()
    return results


async def main():
    # Load pages
    pages = []
    seen = set()
    with open(JSONL) as f:
        for line in f:
            rec = json.loads(line)
            url = rec['url']
            if url not in seen:
                pages.append(rec)
                seen.add(url)

    # Find missing
    missing = []
    for rec in pages:
        path = SHOT_DIR / f"{hashlib.md5(rec['url'].encode()).hexdigest()}.png"
        if not path.exists():
            missing.append(rec)

    print(f"Total unique pages: {len(pages):,}")
    print(f"Already have screenshot: {len(pages) - len(missing):,}")
    print(f"Need to render: {len(missing):,}")

    if not missing:
        print("Nothing to do!")
        return

    # Split into chunks
    n = WORKERS
    chunk_size = max(1, (len(missing) + n - 1) // n)
    chunks = [missing[i:i+chunk_size] for i in range(0, len(missing), chunk_size)]

    shared = {'done': 0, 'skip': 0, 'total': len(missing)}
    lock = asyncio.Lock()

    print(f"\nLaunching {len(chunks)} parallel Playwright workers...")
    await asyncio.gather(
        *[screenshot_worker(chunk, shared, lock) for chunk in chunks]
    )

    # Verify
    still_missing = 0
    for rec in pages:
        path = SHOT_DIR / f"{hashlib.md5(rec['url'].encode()).hexdigest()}.png"
        if not path.exists():
            still_missing += 1

    print(f"\nDone! Rendered {shared['done']:,} screenshots")
    print(f"Still missing: {still_missing:,} (pages that failed to render)")
    print(f"Total screenshots on disk: {len(list(SHOT_DIR.glob('*.png'))):,}")


if __name__ == '__main__':
    asyncio.run(main())
