"""
Web Data Commons (WDC) utilities: downloading and parsing schema.org subsets.

WDC has already extracted all schema.org markup from Common Crawl.
We use it to get (URL, JSON-LD) pairs for training.
"""

import json
import logging
import re
from pathlib import Path
from typing import Generator, Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

WDC_BASE = "https://webdatacommons.org/structureddata/2024-12/files/"

# WDC hosts class-specific N-Quads files. These are the subsets we care about.
WDC_SUBSETS = {
    "LocalBusiness": "schema_org_LocalBusiness.nq.gz",
    "Product": "schema_org_Product.nq.gz",
    "Organization": "schema_org_Organization.nq.gz",
    "Article": "schema_org_Article.nq.gz",
    "NewsArticle": "schema_org_NewsArticle.nq.gz",
    "BlogPosting": "schema_org_BlogPosting.nq.gz",
    "Event": "schema_org_Event.nq.gz",
    "FAQPage": "schema_org_FAQPage.nq.gz",
    "Recipe": "schema_org_Recipe.nq.gz",
    "Person": "schema_org_Person.nq.gz",
    "WebSite": "schema_org_WebSite.nq.gz",
    "BreadcrumbList": "schema_org_BreadcrumbList.nq.gz",
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_wdc_subset(
    schema_type: str,
    output_dir: str,
    force: bool = False,
) -> Optional[Path]:
    """
    Download a WDC schema.org subset file.

    Args:
        schema_type: e.g. "LocalBusiness", "Product".
        output_dir: Directory to save the .nq.gz file.
        force: Re-download even if file exists.

    Returns:
        Path to the downloaded file, or None on failure.
    """
    if schema_type not in WDC_SUBSETS:
        raise ValueError(f"Unknown schema type: {schema_type}. "
                         f"Available: {list(WDC_SUBSETS.keys())}")

    filename = WDC_SUBSETS[schema_type]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    if out_path.exists() and not force:
        logger.info(f"{out_path} already exists, skipping download")
        return out_path

    url = WDC_BASE + filename
    logger.info(f"Downloading {url}")

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(out_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=filename) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info(f"Saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# N-Quads parsing
# ---------------------------------------------------------------------------

def parse_nquads_file(filepath: str) -> Generator[dict, None, None]:
    """
    Parse a WDC N-Quads .nq.gz file into grouped entity dicts.

    N-Quads format: <subject> <predicate> <object> <graph> .
    The graph URL is the source web page.

    Yields dicts of the form:
        {"source_url": str, "properties": {predicate: [values]}, "subject": str}
    """
    import gzip

    path = Path(filepath)
    open_fn = gzip.open if path.suffix == ".gz" else open

    current_subject = None
    current_source = None
    current_props: dict[str, list] = {}

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
    # Regex to match IRIs and literals
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
        # Literal — strip quotes and datatype/lang
        match = re.match(r'"((?:[^"\\]|\\.)*)"', token)
        if match:
            return match.group(1).encode().decode("unicode_escape", errors="replace")
    return None


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_ie_domains(records: list[dict]) -> list[dict]:
    """Keep only records with .ie source URLs."""
    return [r for r in records if r.get("source_url", "").endswith(".ie")
            or ".ie/" in r.get("source_url", "")]


def filter_by_tld(records: list[dict], tld: str) -> list[dict]:
    """Generic TLD filter."""
    tld = tld.lstrip(".")
    return [
        r for r in records
        if urlparse(r.get("source_url", "")).netloc.endswith(f".{tld}")
        or urlparse(r.get("source_url", "")).netloc == tld
    ]


def filter_rich_records(records: list[dict], min_properties: int = 5) -> list[dict]:
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
) -> list[dict]:
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
