"""
Synthetic training data generation via Claude API.
Used for sites that lack schema markup — teaches the model to create schema from scratch.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import anthropic
from tqdm import tqdm

from .schema_validator import validate_jsonld

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_HTML_CHARS = 8_000


GENERATION_PROMPT = """\
Given this web page, generate the optimal schema.org JSON-LD markup.

Rules:
- Output ONLY valid JSON-LD (no markdown, no explanation, no code fences)
- Use the most specific @type available (e.g. Restaurant not LocalBusiness)
- Include all properties that can be extracted from the content
- Use nested entities where appropriate (e.g. address as PostalAddress, offers as Offer)
- Always include "@context": "https://schema.org"
- If multiple schema types are appropriate, use @type as an array
- For LocalBusiness subtypes, include openingHours if visible
- Do not invent data not present on the page

HTML content (may be truncated):
{html}"""


def generate_schema(
    html: str,
    screenshot_b64: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    client: Optional[anthropic.Anthropic] = None,
    max_tokens: int = 2_000,
    retries: int = 3,
) -> Optional[str]:
    """
    Generate schema.org JSON-LD for a page using the Claude API.

    Args:
        html: Page HTML (will be truncated to MAX_HTML_CHARS).
        screenshot_b64: Base64-encoded PNG screenshot (optional but improves quality).
        model: Claude model to use.
        client: Existing Anthropic client (creates one if None).
        max_tokens: Max output tokens.
        retries: Number of retry attempts on failure.

    Returns:
        JSON-LD string, or None on failure.
    """
    if client is None:
        client = anthropic.Anthropic()

    truncated_html = html[:MAX_HTML_CHARS]
    if len(html) > MAX_HTML_CHARS:
        truncated_html += "\n<!-- [HTML truncated] -->"

    content = []
    if screenshot_b64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": screenshot_b64,
            },
        })
    content.append({
        "type": "text",
        "text": GENERATION_PROMPT.format(html=truncated_html),
    })

    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": content}],
            )
            return resp.content[0].text.strip()
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            logger.warning(f"Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except Exception as exc:
            logger.warning(f"Generation attempt {attempt + 1} failed: {exc}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return None


def generate_batch(
    items: list[dict],
    output_dir: str,
    model: str = DEFAULT_MODEL,
    min_quality: float = 0.4,
    requests_per_minute: int = 40,
    skip_existing: bool = True,
) -> list[dict]:
    """
    Generate synthetic schema for a batch of pages.

    Args:
        items: List of dicts with 'id', 'html', and optionally 'screenshot_path'.
        output_dir: Directory to save individual .json result files.
        model: Claude model.
        min_quality: Discard generated results below this quality score.
        requests_per_minute: Rate limit (Claude default ~50 RPM on Sonnet).
        skip_existing: Skip if output file already exists.

    Returns:
        List of result dicts with generation outcomes.
    """
    client = anthropic.Anthropic()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    min_interval = 60.0 / requests_per_minute

    for item in tqdm(items, desc="Generating synthetic schema"):
        item_id = item["id"]
        out_path = out_dir / f"{item_id}.json"

        if skip_existing and out_path.exists():
            with open(out_path) as f:
                results.append(json.load(f))
            continue

        # Load screenshot if available
        screenshot_b64 = None
        if "screenshot_path" in item and Path(item["screenshot_path"]).exists():
            from .screenshot import screenshot_path_to_b64
            screenshot_b64 = screenshot_path_to_b64(item["screenshot_path"])

        start = time.time()
        raw_output = generate_schema(
            html=item["html"],
            screenshot_b64=screenshot_b64,
            model=model,
            client=client,
        )

        result = {
            "id": item_id,
            "source_url": item.get("source_url", ""),
            "generated_schema": None,
            "valid": False,
            "quality_score": 0.0,
            "error": None,
        }

        if raw_output:
            validation = validate_jsonld(raw_output)
            if validation["valid"] and validation["quality_score"] >= min_quality:
                result.update({
                    "generated_schema": raw_output,
                    "valid": True,
                    "quality_score": validation["quality_score"],
                    "schema_types": validation["schema_types"],
                    "property_count": validation["property_count"],
                })
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
            else:
                result["error"] = f"Quality too low ({validation['quality_score']:.2f}) or invalid"
        else:
            result["error"] = "Generation failed"

        results.append(result)

        # Polite rate limiting
        elapsed = time.time() - start
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    success = sum(1 for r in results if r["valid"])
    logger.info(f"Generated {success}/{len(items)} valid synthetic examples")
    return results


def cost_estimate(
    n_examples: int,
    avg_input_tokens: int = 3_000,
    avg_output_tokens: int = 500,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Estimate API cost for synthetic generation."""
    # Pricing as of early 2026 (update if needed)
    pricing = {
        "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},     # $/M tokens
        "claude-opus-4-6": {"input": 15.0, "output": 75.0},
        "claude-haiku-4-5-20251001": {"input": 0.25, "output": 1.25},
    }
    rates = pricing.get(model, pricing["claude-sonnet-4-6"])
    input_cost = (n_examples * avg_input_tokens / 1_000_000) * rates["input"]
    output_cost = (n_examples * avg_output_tokens / 1_000_000) * rates["output"]
    return {
        "model": model,
        "n_examples": n_examples,
        "estimated_input_tokens": n_examples * avg_input_tokens,
        "estimated_output_tokens": n_examples * avg_output_tokens,
        "input_cost_usd": round(input_cost, 2),
        "output_cost_usd": round(output_cost, 2),
        "total_cost_usd": round(input_cost + output_cost, 2),
    }
