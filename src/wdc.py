"""
Web Data Commons (WDC) utilities: downloading and parsing schema.org subsets.

WDC has already extracted all schema.org markup from Common Crawl.
We use it to get (URL, JSON-LD) pairs for training.

2024-12 release URL structure:
  https://data.dws.informatik.uni-mannheim.de/structureddata/2024-12/quads/classspecific/{Type}/
  - part_0.gz ... part_N.gz  (N-Quads, gzip)
  - {Type}_lookup.csv        (PLD → chunk mapping for targeted downloads)
  - {Type}_sample.txt        (small sample for testing)
"""

import gzip
import io
import json
import logging
import re
from pathlib import Path
from typing import Generator, List, Optional, Union
from urllib.parse import urlparse

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

WDC_BASE = "https://data.dws.informatik.uni-mannheim.de/structureddata/2024-12/quads/classspecific/"

# Number of part files per class (upper bound; download stops at 404)
WDC_PART_COUNTS = {
    "LocalBusiness": 176,
    "Product": 120,
    "Organization": 80,
    "Article": 100,
    "NewsArticle": 40,
    "BlogPosting": 30,
    "Event": 40,
    "FAQPage": 20,
    "Recipe": 30,
    "Person": 50,
    "WebSite": 60,
    "BreadcrumbList": 40,
}

WDC_SUBSETS = {k: k for k in WDC_PART_COUNTS}  # kept for notebook compatibility


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_html_response(content: bytes) -> bool:
    """Return True if content looks like an HTML page rather than gzip data."""
    magic = content[:4]
    # Gzip magic bytes: 1f 8b
    if magic[:2] == b'\x1f\x8b':
        return False
    # HTML indicators
    if magic[:2] in (b'<!', b'<h', b'<H') or magic[:4] in (b'<htm', b'<HTM'):
        return True
    # Any non-gzip content is treated as bad
    return True


def _download_file(url: str, out_path: Path, desc: str = "") -> bool:
    """
    Download a single URL to out_path.
    Returns True on success, False if URL returns 404.
    Raises ValueError if the server returns HTML instead of binary data.
    """
    resp = requests.get(url, stream=True, timeout=120)
    if resp.status_code == 404:
        return False
    resp.raise_for_status()

    # Peek at first chunk to validate it's not HTML
    first_chunk = b""
    chunks = []
    for chunk in resp.iter_content(chunk_size=8192):
        if not first_chunk:
            first_chunk = chunk
            if _is_html_response(first_chunk):
                ct = resp.headers.get("content-type", "")
                raise ValueError(
                    f"Server returned HTML instead of gzip data for {url}\n"
                    f"Content-Type: {ct}\n"
                    f"First bytes: {first_chunk[:80]}"
                )
        chunks.append(chunk)

    with open(out_path, "wb") as f:
        for chunk in chunks:
            f.write(chunk)
    return True


# ---------------------------------------------------------------------------
# Download: lookup-based (only fetch parts containing target TLD)
# ---------------------------------------------------------------------------

def download_wdc_for_tld(
    schema_type: str,
    output_dir: str,
    tld: str = "ie",
    force: bool = False,
) -> List[Path]:
    """
    Download only the WDC parts that contain domains for a given TLD.

    Uses the {Type}_lookup.csv to identify relevant chunk numbers, then
    downloads only those part files.  Much faster than downloading everything.

    Args:
        schema_type: e.g. "LocalBusiness"
        output_dir:  Directory to save part files
        tld:         TLD to filter for, e.g. "ie"
        force:       Re-download even if files exist

    Returns:
        List of paths to downloaded part files
    """
    if schema_type not in WDC_SUBSETS:
        raise ValueError(f"Unknown schema type: {schema_type}. Available: {list(WDC_SUBSETS.keys())}")

    out_dir = Path(output_dir) / schema_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download lookup CSV
    lookup_url = f"{WDC_BASE}{schema_type}/{schema_type}_lookup.csv"
    lookup_path = out_dir / f"{schema_type}_lookup.csv"
    if not lookup_path.exists() or force:
        logger.info(f"Downloading lookup: {lookup_url}")
        resp = requests.get(lookup_url, timeout=120)
        resp.raise_for_status()
        lookup_path.write_bytes(resp.content)

    # Parse lookup CSV to find relevant chunks
    # Format: pld,chunk_id  (e.g. "example.ie,42")
    tld_suffix = f".{tld.lstrip('.')}"
    relevant_chunks = set()
    with open(lookup_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("pld"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                pld = parts[0].strip()
                chunk = parts[1].strip()
                if pld.endswith(tld_suffix) or pld == tld.lstrip("."):
                    try:
                        relevant_chunks.add(int(chunk))
                    except ValueError:
                        pass

    logger.info(f"Found {len(relevant_chunks)} chunks containing .{tld} domains for {schema_type}")

    if not relevant_chunks:
        logger.warning(f"No .{tld} chunks found for {schema_type} — downloading sample instead")
        return [download_wdc_sample(schema_type, output_dir, force=force)]

    # Download relevant parts
    downloaded = []
    for chunk_id in sorted(relevant_chunks):
        part_url = f"{WDC_BASE}{schema_type}/part_{chunk_id}.gz"
        part_path = out_dir / f"part_{chunk_id}.gz"
        if part_path.exists() and not force:
            downloaded.append(part_path)
            continue
        logger.info(f"Downloading {part_url}")
        try:
            ok = _download_file(part_url, part_path, desc=f"part_{chunk_id}")
            if ok:
                downloaded.append(part_path)
        except ValueError as e:
            logger.error(str(e))

    logger.info(f"Downloaded {len(downloaded)} parts for {schema_type}")
    return downloaded


def download_wdc_sample(
    schema_type: str,
    output_dir: str,
    force: bool = False,
) -> Optional[Path]:
    """Download the small sample file for a schema type (good for testing)."""
    out_dir = Path(output_dir) / schema_type
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_url = f"{WDC_BASE}{schema_type}/{schema_type}_sample.txt"
    sample_path = out_dir / f"{schema_type}_sample.txt"
    if sample_path.exists() and not force:
        return sample_path
    resp = requests.get(sample_url, timeout=60)
    resp.raise_for_status()
    sample_path.write_bytes(resp.content)
    return sample_path


def download_wdc_subset(
    schema_type: str,
    output_dir: str,
    force: bool = False,
    tld: Optional[str] = "ie",
) -> Optional[Path]:
    """
    Backwards-compatible wrapper: download WDC data for a schema type.

    If tld is set, uses the lookup CSV to download only relevant parts
    and merges them into a single .nq.gz file.  Returns that merged path.
    If no TLD chunks are found, falls back to the sample file.
    """
    if schema_type not in WDC_SUBSETS:
        raise ValueError(f"Unknown schema type: {schema_type}. Available: {list(WDC_SUBSETS.keys())}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_name = f"{schema_type}_{tld or 'all'}.nq.gz" if tld else f"{schema_type}_all.nq.gz"
    merged_path = out_dir / merged_name

    if merged_path.exists() and not force:
        logger.info(f"{merged_path} already exists, skipping download")
        return merged_path

    parts_dir = out_dir / "_parts"
    if tld:
        parts = download_wdc_for_tld(schema_type, str(parts_dir), tld=tld, force=force)
    else:
        # Download all parts
        parts = _download_all_parts(schema_type, str(parts_dir), force=force)

    if not parts:
        logger.warning(f"No parts downloaded for {schema_type}")
        return None

    # Merge parts into a single .nq.gz
    logger.info(f"Merging {len(parts)} parts → {merged_path}")
    with gzip.open(merged_path, "wt", encoding="utf-8") as out_f:
        for part_path in sorted(parts):
            open_fn = gzip.open if str(part_path).endswith(".gz") else open
            try:
                with open_fn(part_path, "rt", encoding="utf-8", errors="replace") as in_f:
                    for line in in_f:
                        out_f.write(line)
            except Exception as e:
                logger.warning(f"Error reading {part_path}: {e}")

    logger.info(f"Merged to {merged_path}")
    return merged_path


def _download_all_parts(schema_type: str, output_dir: str, force: bool = False) -> List[Path]:
    """Download all parts for a schema type (can be very large)."""
    out_dir = Path(output_dir) / schema_type
    out_dir.mkdir(parents=True, exist_ok=True)
    max_parts = WDC_PART_COUNTS.get(schema_type, 200)
    downloaded = []
    for i in range(max_parts):
        part_url = f"{WDC_BASE}{schema_type}/part_{i}.gz"
        part_path = out_dir / f"part_{i}.gz"
        if part_path.exists() and not force:
            downloaded.append(part_path)
            continue
        try:
            ok = _download_file(part_url, part_path)
            if not ok:
                break  # 404 = no more parts
            downloaded.append(part_path)
        except ValueError as e:
            logger.error(str(e))
            break
    return downloaded


# ---------------------------------------------------------------------------
# N-Quads parsing
# ---------------------------------------------------------------------------

def parse_nquads_file(filepath: str) -> Generator[dict, None, None]:
    """
    Parse a WDC N-Quads .nq.gz (or .txt) file into grouped entity dicts.

    N-Quads format: <subject> <predicate> <object> <graph> .
    The graph URL is the source web page.

    Yields dicts of the form:
        {"source_url": str, "properties": {predicate: [values]}, "subject": str}
    """
    path = Path(filepath)
    open_fn = gzip.open if path.suffix == ".gz" else open

    # Validate it's not HTML before parsing
    with open(filepath, "rb") as raw:
        magic = raw.read(4)
    if _is_html_response(magic):
        raise ValueError(
            f"File {filepath} appears to be HTML, not gzip/N-Quads. "
            "The WDC download may have failed — delete the file and retry."
        )

    current_subject = None
    current_source = None
    current_props: dict = {}

    with open_fn(filepath, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = _parse_nquad_line(line)
            if parts is None:
                continue

            subject, predicate, obj, graph = parts
            source_url = _extract_iri(graph)

            if subject != current_subject or source_url != current_source:
                if current_subject and current_props:
                    yield {
                        "subject": current_subject,
                        "source_url": current_source,
                        "properties": current_props,
                    }
                current_subject = subject
                current_source = source_url
                current_props = {}

            pred_local = _local_name(predicate)
            value = _extract_value(obj)
            if pred_local and value is not None:
                current_props.setdefault(pred_local, []).append(value)

    if current_subject and current_props:
        yield {
            "subject": current_subject,
            "source_url": current_source,
            "properties": current_props,
        }


def _parse_nquad_line(line: str) -> Optional[tuple]:
    """Parse a single N-Quads line into (subject, predicate, object, graph)."""
    pattern = r'(<[^>]+>|"(?:[^"\\]|\\.)*"(?:\^\^<[^>]+>|@\w+)?|_:\w+)'
    parts = re.findall(pattern, line)
    if len(parts) >= 4:
        return tuple(parts[:4])
    return None


def _extract_iri(token: str) -> str:
    return token.strip("<>")


def _local_name(iri: str) -> str:
    iri = iri.strip("<>")
    return iri.split("/")[-1].split("#")[-1]


def _extract_value(token: str) -> Optional[str]:
    token = token.strip()
    if token.startswith("<") and token.endswith(">"):
        return token[1:-1]
    if token.startswith('"'):
        match = re.match(r'"((?:[^"\\]|\\.)*)"', token)
        if match:
            return match.group(1).encode().decode("unicode_escape", errors="replace")
    return None


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_ie_domains(records: list) -> list:
    """Keep only records with .ie source URLs."""
    return [r for r in records if r.get("source_url", "").endswith(".ie")
            or ".ie/" in r.get("source_url", "")]


def filter_by_tld(records: list, tld: str) -> list:
    """Generic TLD filter."""
    tld = tld.lstrip(".")
    return [
        r for r in records
        if urlparse(r.get("source_url", "")).netloc.endswith(f".{tld}")
        or urlparse(r.get("source_url", "")).netloc == tld
    ]


def filter_rich_records(records: list, min_properties: int = 5) -> list:
    """Keep records with at least min_properties properties."""
    return [r for r in records if len(r.get("properties", {})) >= min_properties]


# ---------------------------------------------------------------------------
# Conversion to JSON-LD
# ---------------------------------------------------------------------------

def record_to_jsonld(record: dict, schema_type: str) -> dict:
    """
    Convert a WDC N-Quads record into a JSON-LD dict.
    Scalar properties use single values; multi-value properties use lists.
    """
    jsonld = {
        "@context": "https://schema.org",
        "@type": schema_type,
    }
    for prop, values in record.get("properties", {}).items():
        if prop in ("type", "context"):
            continue
        jsonld[prop] = values[0] if len(values) == 1 else values
    return jsonld


def load_and_filter_wdc(
    filepath: str,
    schema_type: str,
    tld_filter: Optional[str] = None,
    min_properties: int = 5,
    max_records: Optional[int] = None,
) -> list:
    """
    End-to-end: parse a WDC file, optionally filter by TLD and property count.

    Returns list of (source_url, jsonld) dicts ready for training data pipeline.
    """
    results = []
    for record in tqdm(parse_nquads_file(filepath), desc=f"Parsing {schema_type}"):
        if tld_filter and not (
            urlparse(record.get("source_url", "")).netloc.endswith(f".{tld_filter}")
        ):
            continue
        if len(record.get("properties", {})) < min_properties:
            continue
        jsonld = record_to_jsonld(record, schema_type)
        results.append({
            "source_url": record["source_url"],
            "jsonld": jsonld,
            "schema_type": schema_type,
            "property_count": len(jsonld) - 2,  # subtract @context and @type
        })
        if max_records and len(results) >= max_records:
            break

    logger.info(f"Loaded {len(results)} {schema_type} records from {filepath}")
    return results
