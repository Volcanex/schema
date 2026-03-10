"""
Playwright-based screenshot rendering pipeline.
Renders saved HTML (from Common Crawl) to images — no live scraping needed.
"""

import asyncio
import base64
import logging
from pathlib import Path
from typing import Optional

from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)

VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 800


async def render_screenshot(
    html_content: str,
    output_path: str,
    width: int = VIEWPORT_WIDTH,
    height: int = VIEWPORT_HEIGHT,
    base_url: str = "about:blank",
) -> bool:
    """
    Render HTML to a PNG screenshot using Playwright (headless Chromium).

    Args:
        html_content: Full HTML string to render.
        output_path: Path to save the PNG.
        width: Viewport width in pixels.
        height: Viewport height in pixels.
        base_url: Base URL for resolving relative resources (usually not needed
                  for offline rendering, but helps some pages load correctly).

    Returns:
        True on success, False on failure.
    """
    from playwright.async_api import async_playwright

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page(viewport={"width": width, "height": height})
            await page.set_content(
                html_content,
                wait_until="domcontentloaded",
                timeout=15_000,
            )
            await page.screenshot(path=output_path, full_page=False)
            await browser.close()
        return True
    except Exception as exc:
        logger.warning(f"Screenshot failed for {output_path}: {exc}")
        return False


async def batch_render(
    items: list[dict],
    output_dir: str,
    concurrency: int = 8,
    width: int = VIEWPORT_WIDTH,
    height: int = VIEWPORT_HEIGHT,
    skip_existing: bool = True,
) -> dict[str, bool]:
    """
    Batch-render HTML to screenshots with controlled concurrency.

    Args:
        items: List of dicts with keys 'id' (str) and 'html' (str).
        output_dir: Directory to save PNGs.
        concurrency: Max parallel Playwright pages.
        skip_existing: Skip if {id}.png already exists.

    Returns:
        Dict mapping item id -> success bool.
    """
    from playwright.async_api import async_playwright

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, bool] = {}
    sem = asyncio.Semaphore(concurrency)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)

        async def process(item: dict) -> None:
            item_id = item["id"]
            out_path = out_dir / f"{item_id}.png"

            if skip_existing and out_path.exists():
                results[item_id] = True
                return

            async with sem:
                try:
                    page = await browser.new_page(
                        viewport={"width": width, "height": height}
                    )
                    await page.set_content(
                        item["html"],
                        wait_until="domcontentloaded",
                        timeout=15_000,
                    )
                    await page.screenshot(path=str(out_path), full_page=False)
                    await page.close()
                    results[item_id] = True
                except Exception as exc:
                    logger.warning(f"Failed to render {item_id}: {exc}")
                    results[item_id] = False

        await tqdm_asyncio.gather(
            *[process(item) for item in items],
            desc="Rendering screenshots",
        )
        await browser.close()

    success = sum(v for v in results.values())
    logger.info(f"Rendered {success}/{len(items)} screenshots successfully")
    return results


def html_to_screenshot_b64(
    html_content: str,
    width: int = VIEWPORT_WIDTH,
    height: int = VIEWPORT_HEIGHT,
) -> Optional[str]:
    """
    Synchronous wrapper: render HTML and return base64-encoded PNG string.
    Useful for synthetic data generation (feeding to Claude/GPT-4o).
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    success = asyncio.run(
        render_screenshot(html_content, tmp_path, width=width, height=height)
    )
    if not success:
        return None

    with open(tmp_path, "rb") as f:
        data = f.read()
    Path(tmp_path).unlink(missing_ok=True)
    return base64.b64encode(data).decode("utf-8")


def screenshot_path_to_b64(path: str) -> str:
    """Load an existing screenshot file and return base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
