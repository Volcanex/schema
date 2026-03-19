"""
Schema.org JSON-LD validation utilities.
Checks structural validity, @type correctness, and property coverage.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEMA_TYPES_PATH = Path(__file__).parent.parent / "configs" / "schema_types.json"
_schema_config: Optional[dict] = None


def _load_config() -> dict:
    global _schema_config
    if _schema_config is None:
        with open(_SCHEMA_TYPES_PATH) as f:
            _schema_config = json.load(f)
    return _schema_config


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

def validate_jsonld(jsonld_str: str) -> dict:
    """
    Validate a JSON-LD string. Returns a result dict:
    {
        "valid": bool,
        "parsed": None,
        "errors": [str],
        "warnings": [str],
        "schema_types": [str],
        "property_count": int,
        "quality_score": float (0-1),
    }
    """
    result = {
        "valid": False,
        "parsed": None,
        "errors": [],
        "warnings": [],
        "schema_types": [],
        "property_count": 0,
        "quality_score": 0.0,
    }

    # 1. Parse JSON
    try:
        parsed = json.loads(jsonld_str.strip())
    except json.JSONDecodeError as e:
        result["errors"].append(f"Invalid JSON: {e}")
        return result

    result["parsed"] = parsed

    # 2. Handle top-level list (array of schemas) or @graph wrapper
    if isinstance(parsed, list):
        if len(parsed) == 0:
            result["errors"].append("Empty JSON array")
            return result
        entity = parsed[0]
        parsed = entity  # use first entity as the root for context check
    elif isinstance(parsed, dict) and "@graph" in parsed:
        entities = parsed["@graph"]
        if not isinstance(entities, list) or len(entities) == 0:
            result["errors"].append("@graph is empty or not a list")
            return result
        entity = entities[0]
    elif isinstance(parsed, dict):
        entity = parsed
    else:
        result["errors"].append(f"Unexpected JSON type: {type(parsed).__name__}")
        return result

    # 3. Check @context
    context = entity.get("@context") or parsed.get("@context")
    if not context:
        result["warnings"].append("Missing @context")
    elif "schema.org" not in str(context):
        result["warnings"].append(f"Unexpected @context: {context}")

    # 4. Check @type
    schema_type = entity.get("@type")
    if not schema_type:
        result["errors"].append("Missing @type")
        return result

    types = [schema_type] if isinstance(schema_type, str) else schema_type
    result["schema_types"] = types

    config = _load_config()
    known_types = set(config["priority_types"].keys())
    # Also include all subtypes
    for type_info in config["priority_types"].values():
        known_types.update(type_info.get("subtypes", []))

    for t in types:
        if t not in known_types:
            result["warnings"].append(f"Unknown schema type: {t} (may still be valid)")

    # 5. Check required properties
    primary_type = types[0]
    type_config = config["priority_types"].get(primary_type, {})
    required = type_config.get("required", ["@type", "@context"])

    root = result["parsed"]  # original top-level parsed object
    for req in required:
        prop = req.lstrip("@")
        # Check both the entity and the root (for @graph-wrapped schemas)
        in_entity = f"@{prop}" in entity or prop in entity
        in_root = isinstance(root, dict) and (f"@{prop}" in root or prop in root)
        if not in_entity and not in_root:
            result["errors"].append(f"Missing required property: {req}")

    # 6. Count meaningful properties (exclude @context, @type, @id)
    meta_keys = {"@context", "@type", "@id", "@graph"}
    props = {k: v for k, v in entity.items() if k not in meta_keys}
    result["property_count"] = len(props)

    # 7. Quality score (0-1)
    min_props = config.get("min_properties_for_quality", 5)
    high_props = config.get("high_quality_threshold", 10)
    recommended = type_config.get("recommended", [])

    prop_score = min(result["property_count"] / high_props, 1.0)
    coverage_score = sum(1 for r in recommended if r in entity) / max(len(recommended), 1)
    error_penalty = 0.3 * len(result["errors"])
    warning_penalty = 0.1 * len(result["warnings"])

    result["quality_score"] = max(
        0.0,
        round(0.6 * prop_score + 0.4 * coverage_score - error_penalty - warning_penalty, 3)
    )

    result["valid"] = len(result["errors"]) == 0
    return result


def validate_batch(jsonld_strings: list[str]) -> list[dict]:
    """Validate a batch of JSON-LD strings."""
    return [validate_jsonld(s) for s in jsonld_strings]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_jsonld_from_html(html: str) -> list[str]:
    """
    Extract JSON-LD blocks from HTML <script type="application/ld+json"> tags.
    Returns list of raw JSON strings (may be malformed).
    """
    import re
    pattern = r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
    matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)
    return [m.strip() for m in matches]


def parse_jsonld_from_html(html: str) -> list[dict]:
    """Extract and parse all valid JSON-LD blocks from HTML."""
    results = []
    for raw in extract_jsonld_from_html(html):
        try:
            parsed = json.loads(raw)
            results.append(parsed)
        except json.JSONDecodeError:
            continue
    return results


def schema_types_in_html(html: str) -> list[str]:
    """Quick check: what schema.org types does this page already declare?"""
    types = []
    for jsonld in parse_jsonld_from_html(html):
        t = jsonld.get("@type")
        if t:
            types.extend([t] if isinstance(t, str) else t)
    return list(set(types))


def has_quality_schema(html: str, min_score: float = 0.4) -> bool:
    """Return True if the page has at least one schema block meeting quality threshold."""
    for raw in extract_jsonld_from_html(html):
        result = validate_jsonld(raw)
        if result["valid"] and result["quality_score"] >= min_score:
            return True
    return False
