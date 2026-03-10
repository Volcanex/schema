"""
Common Crawl utilities: index querying and WARC HTML extraction.
"""

import gzip
import json
import logging
import time
from pathlib import Path
from typing import Generator, Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

CC_INDEX_BASE = "https://index.commoncrawl.org"
CC_DATA_BASE = "https://data.commoncrawl.org"

DEFAULT_CRAWL = "CC-MAIN-2026-08"


# ---------------------------------------------------------------------------
# Index querying (CC Index Server API — no AWS account needed)
# ---------------------------------------------------------------------------

def query_cc_index(
    url_pattern: str,
    crawl: str = DEFAULT_CRAWL,
    match_type: str = "domain",
    limit: int = 10_000,
    output: str = "json",
    sleep_between_pages: float = 1.0,
) -> list[dict]:
    """
    Query the CC Index Server API for URLs matching a pattern.

    Args:
        url_pattern: Pattern to match, e.g. "*.ie" for all .ie domains.
        crawl: Crawl ID, e.g. "CC-MAIN-2026-08".
        match_type: "domain", "prefix", "exact", or "host".
        limit: Max records to return (paginated internally).
        output: "json" or "cdx".
        sleep_between_pages: Polite delay between paginated requests.

    Returns:
        List of index records as dicts.
    """
    endpoint = f"{CC_INDEX_BASE}/{crawl}-index"
    records = []
    resume_key = None

    while True:
        params = {
            "url": url_pattern,
            "matchType": match_type,
            "output": output,
            "limit": min(limit - len(records), 5_000),
        }
        if resume_key:
            params["resumeKey"] = resume_key

        resp = requests.get(endpoint, params=params, timeout=30)
        resp.raise_for_status()

        lines = [l for l in resp.text.strip().split("\n") if l]
        if not lines:
            break

        # Last line may be a resumeKey marker
        last = lines[-1]
        try:
            obj = json.loads(last)
            if "resumeKey" in obj:
                resume_key = obj["resumeKey"]
                lines = lines[:-1]
            else:
                resume_key = None
        except json.JSONDecodeError:
            resume_key = None

        for line in lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        logger.info(f"Fetched {len(records)} records so far...")

        if len(records) >= limit or resume_key is None:
            break

        time.sleep(sleep_between_pages)

    return records


def get_ie_domains(crawl: str = DEFAULT_CRAWL, limit: int = 50_000) -> list[dict]:
    """Convenience wrapper: fetch all .ie domain records from CC index."""
    logger.info(f"Querying CC index for .ie domains in {crawl}")
    # CDX API uses ".ie" (not "*.ie") with matchType=domain for TLD-level queries
    return query_cc_index(".ie", crawl=crawl, limit=limit)


def deduplicate_by_domain(records: list[dict]) -> dict[str, dict]:
    """
    Keep one record per registered domain (prefer homepage).
    Returns dict mapping domain -> record.
    """
    by_domain: dict[str, dict] = {}
    for rec in records:
        domain = rec.get("url", "").split("/")[2].lstrip("www.")
        if domain not in by_domain:
            by_domain[domain] = rec
    return by_domain


# ---------------------------------------------------------------------------
# WARC record fetching
# ---------------------------------------------------------------------------

def fetch_warc_record(
    filename: str,
    offset: int,
    length: int,
    retries: int = 3,
) -> Optional[str]:
    """
    Fetch a single HTML document from a Common Crawl WARC file using
    HTTP range requests. Runs from anywhere; free from EC2 us-east-1.

    Args:
        filename: WARC file path, e.g. "crawl-data/CC-MAIN-.../xxx.warc.gz".
        offset: Byte offset of the record in the file.
        length: Byte length of the compressed record.

    Returns:
        Decoded HTML string, or None if extraction fails.
    """
    url = f"{CC_DATA_BASE}/{filename}"
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            raw = gzip.decompress(resp.content)
            return _extract_html_from_warc(raw)
        except Exception as exc:
            logger.warning(f"Attempt {attempt + 1} failed for {filename}: {exc}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return None


def _extract_html_from_warc(raw_bytes: bytes) -> Optional[str]:
    """Extract the HTTP response body (HTML) from a raw WARC record."""
    # WARC record: WARC headers + blank line + HTTP headers + blank line + body
    first_sep = raw_bytes.find(b"\r\n\r\n")
    if first_sep == -1:
        return None
    second_sep = raw_bytes.find(b"\r\n\r\n", first_sep + 4)
    if second_sep == -1:
        return None
    html_bytes = raw_bytes[second_sep + 4:]
    return html_bytes.decode("utf-8", errors="replace")


def batch_fetch_warc(
    records: list[dict],
    output_dir: str,
    max_records: Optional[int] = None,
) -> Generator[dict, None, None]:
    """
    Batch-fetch HTML from WARC records. Yields dicts with url + html fields.
    Saves a .html file per record to output_dir.

    Args:
        records: List of CC index records (must have warc_filename, warc_record_offset,
                 warc_record_length, url).
        output_dir: Directory to save HTML files.
        max_records: Cap on number to process.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    to_process = records[:max_records] if max_records else records

    for rec in tqdm(to_process, desc="Fetching WARC records"):
        url = rec.get("url", "")
        filename = rec.get("filename") or rec.get("warc_filename")
        offset = int(rec.get("offset") or rec.get("warc_record_offset", 0))
        length = int(rec.get("length") or rec.get("warc_record_length", 0))

        if not filename:
            logger.warning(f"No WARC filename for {url}, skipping")
            continue

        safe_name = url.replace("://", "_").replace("/", "_")[:100]
        out_path = out / f"{safe_name}.html"

        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                html = f.read()
        else:
            html = fetch_warc_record(filename, offset, length)
            if html is None:
                logger.warning(f"Failed to fetch {url}")
                continue
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)

        yield {"url": url, "html": html, "warc_filename": filename}
