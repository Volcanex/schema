"""
Training data assembly and formatting for fine-tuning Qwen2.5-VL.
Produces instruction-tuned conversation format with multimodal inputs.
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Optional, Union

from tqdm import tqdm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a schema.org JSON-LD expert. Given a screenshot and HTML of a web page, "
    "generate the optimal schema.org JSON-LD markup. Output ONLY valid JSON-LD with "
    "no markdown formatting, no explanation, and no code fences. "
    "Use the most specific @type available. Include all extractable properties. "
    'Always include "@context": "https://schema.org".'
)

# Per-type schema.org property hints injected at inference time.
# Keeps the model from hallucinating property names and reminds it of
# type-specific fields (especially useful for Irish SME patterns).
SCHEMA_TYPE_HINTS = {
    "LocalBusiness": (
        "Relevant LocalBusiness properties: name, url, telephone, email, address "
        "(streetAddress, addressLocality, addressRegion, postalCode, addressCountry), "
        "openingHoursSpecification, openingHours, priceRange, description, image, logo, "
        "geo (latitude, longitude), sameAs, currenciesAccepted, paymentAccepted. "
        "For Irish businesses: addressCountry='IE', postalCode is an Eircode (e.g. D01 AB23)."
    ),
    "Product": (
        "Relevant Product properties: name, description, image, brand, sku, gtin, mpn, "
        "offers (price, priceCurrency, availability, url), aggregateRating (ratingValue, reviewCount)."
    ),
    "Restaurant": (
        "Relevant Restaurant properties: name, url, telephone, address, servesCuisine, "
        "priceRange, openingHours, menu, hasMap, acceptsReservations, image."
    ),
    "Hotel": (
        "Relevant Hotel properties: name, url, telephone, address, starRating, "
        "amenityFeature, priceRange, image, checkinTime, checkoutTime."
    ),
}

def get_system_prompt(schema_type: str = None) -> str:
    """Return system prompt, optionally enriched with type-specific property hints."""
    if schema_type and schema_type in SCHEMA_TYPE_HINTS:
        return SYSTEM_PROMPT + "\n\n" + SCHEMA_TYPE_HINTS[schema_type]
    return SYSTEM_PROMPT

USER_TEMPLATE = "Generate schema.org JSON-LD for this web page.\n\nHTML:\n{html}"
MAX_HTML_CHARS = 48_000  # ~12K tokens after stripping; leaves ~2.5K headroom under 16384 max_seq_length


def _strip_html_noise(html: str) -> str:
    """
    Strip only content we are certain is noise for schema generation:
    - <style> blocks (pure CSS, no business info)
    - JavaScript <script> blocks
    - Existing <script type="application/ld+json"> stripped from INPUT so the model
      must generate schema from page content, not copy the existing markup.

    HTML comments are intentionally KEPT — devs sometimes embed addresses,
    phone numbers, or structured data hints in them.
    """
    # Remove existing JSON-LD from input — model must learn to generate it, not copy it
    html = re.sub(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>.*?</script>',
        '', html, flags=re.DOTALL | re.IGNORECASE
    )
    # Remove other <script> blocks (JavaScript — no business info)
    html = re.sub(r'<script(?:[^>]*)>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # Remove <style> blocks (CSS only)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # Collapse excessive blank lines
    html = re.sub(r'\n\s*\n+', '\n', html)
    return html.strip()


# ---------------------------------------------------------------------------
# Example formatting
# ---------------------------------------------------------------------------

def format_training_example(
    html: str,
    jsonld: Union[str, dict],
    screenshot_path: Optional[str] = None,
    example_id: Optional[str] = None,
    source: str = "wdc",
    schema_types: Optional[list[str]] = None,
    domain: Optional[str] = None,
    quality_score: Optional[float] = None,
) -> dict:
    """
    Format a single (html, jsonld) pair into the Qwen2.5-VL training format.

    The format is a list of messages (system, user, assistant) where
    the user message contains the screenshot (optional) + truncated HTML.
    """
    if isinstance(jsonld, dict):
        jsonld_str = json.dumps(jsonld, ensure_ascii=False)
    else:
        jsonld_str = jsonld

    clean_html = _strip_html_noise(html)
    truncated_html = clean_html[:MAX_HTML_CHARS]
    if len(clean_html) > MAX_HTML_CHARS:
        truncated_html += "\n<!-- [HTML truncated] -->"

    # Build user content: image (if available) + text
    user_content = []
    if screenshot_path and Path(screenshot_path).exists():
        user_content.append({"type": "image", "image": screenshot_path})
    user_content.append({
        "type": "text",
        "text": USER_TEMPLATE.format(html=truncated_html),
    })

    example = {
        "id": example_id or _generate_id(),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": jsonld_str},
        ],
        "metadata": {
            "source": source,
            "schema_types": schema_types or [],
            "domain": domain or "",
            "quality_score": quality_score or 0.0,
            "html_length": len(html),
            "has_screenshot": screenshot_path is not None and Path(screenshot_path).exists(),
        },
    }
    return example


def _generate_id() -> str:
    import uuid
    return str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def assemble_dataset(
    wdc_records: list[dict],
    screenshot_dir: Optional[str],
    output_path: str,
    max_examples: Optional[int] = None,
    min_quality: float = 0.3,
    shuffle: bool = True,
) -> int:
    """
    Assemble WDC records + screenshots into a JSONL training dataset.

    Args:
        wdc_records: Output from src.wdc.load_and_filter_wdc().
        screenshot_dir: Directory containing {domain}.png screenshots.
        output_path: Output .jsonl file path.
        max_examples: Cap on total examples.
        min_quality: Skip records below this quality score.
        shuffle: Shuffle before saving.

    Returns:
        Number of examples written.
    """
    from .schema_validator import validate_jsonld

    examples = []
    skipped = 0

    for record in tqdm(wdc_records, desc="Assembling dataset"):
        jsonld = record.get("jsonld", {})
        if not jsonld:
            skipped += 1
            continue

        jsonld_str = json.dumps(jsonld, ensure_ascii=False)
        validation = validate_jsonld(jsonld_str)

        if not validation["valid"] or validation["quality_score"] < min_quality:
            skipped += 1
            continue

        url = record.get("source_url", "")
        domain = _extract_domain(url)

        # Look for matching screenshot
        screenshot_path = None
        if screenshot_dir:
            candidate = Path(screenshot_dir) / f"{domain}.png"
            if candidate.exists():
                screenshot_path = str(candidate)

        example = format_training_example(
            html=record.get("html", ""),
            jsonld=jsonld,
            screenshot_path=screenshot_path,
            source="wdc",
            schema_types=validation["schema_types"],
            domain=domain,
            quality_score=validation["quality_score"],
        )
        examples.append(example)

        if max_examples and len(examples) >= max_examples:
            break

    if shuffle:
        random.shuffle(examples)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(examples)} examples to {output_path} ({skipped} skipped)")
    return len(examples)


def split_dataset(
    input_path: str,
    train_path: str,
    eval_path: str,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[int, int]:
    """Split a JSONL dataset into train and eval files."""
    random.seed(seed)

    with open(input_path) as f:
        lines = f.readlines()

    random.shuffle(lines)
    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    eval_lines = lines[split_idx:]

    for path, data in [(train_path, train_lines), (eval_path, eval_lines)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.writelines(data)

    logger.info(f"Split: {len(train_lines)} train, {len(eval_lines)} eval")
    return len(train_lines), len(eval_lines)


def dataset_stats(jsonl_path: str) -> dict:
    """Compute basic stats on an assembled dataset."""
    type_counts: dict[str, int] = {}
    quality_scores = []
    has_screenshot = 0
    total = 0

    with open(jsonl_path) as f:
        for line in f:
            ex = json.loads(line)
            meta = ex.get("metadata", {})
            for t in meta.get("schema_types", []):
                type_counts[t] = type_counts.get(t, 0) + 1
            if meta.get("quality_score"):
                quality_scores.append(meta["quality_score"])
            if meta.get("has_screenshot"):
                has_screenshot += 1
            total += 1

    return {
        "total": total,
        "with_screenshots": has_screenshot,
        "schema_type_distribution": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
        "avg_quality_score": round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0,
        "min_quality": round(min(quality_scores), 3) if quality_scores else 0,
        "max_quality": round(max(quality_scores), 3) if quality_scores else 0,
    }


def _extract_domain(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url).netloc.lstrip("www.")
