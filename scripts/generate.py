"""
Gemini 2.5 Flash Batch Generation Pipeline for Schema.org JSON-LD
Generates synthetic training data by sending webpage screenshots + HTML
to Gemini 2.5 Flash and collecting JSON-LD outputs.
Supports:
- Batch API (50% cost reduction, higher latency)
- Standard API with rate limiting and retries
- Resume from interruptions
- Integrated validation with auto-retry on failure
Usage:
    python generate.py --input pages.jsonl --output results/ --mode batch
    python generate.py --input pages.jsonl --output results/ --mode standard --concurrency 5
"""
import json
import time
import base64
import hashlib
import argparse
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
# Add parent dir to path for validator import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from validators.schema_validator import validate, ValidationResult
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
# ============================================================================
# Configuration
# ============================================================================
@dataclass
class PipelineConfig:
    """Configuration for the generation pipeline."""
    # API settings
    model: str = "gemini-2.5-flash"
    api_key_env: str = "GEMINI_API_KEY"  # Environment variable name
    mode: str = "standard"  # "standard" or "batch"
    # Generation settings
    max_output_tokens: int = 4096
    temperature: float = 0.1  # Low temp for consistent structured output
    top_p: float = 0.95
    # Rate limiting (standard mode)
    concurrency: int = 5
    requests_per_minute: int = 60
    retry_max: int = 3
    retry_backoff: float = 2.0
    # Validation
    validate_output: bool = True
    auto_retry_invalid: bool = True
    max_validation_retries: int = 2
    # Paths
    system_prompt_path: str = "prompts/teacher_system_prompt.txt"
    input_path: str = "pages.jsonl"
    output_dir: str = "results"
    # Image settings
    max_image_dimension: int = 1280  # Resize screenshots to this max dimension
    image_quality: int = 85  # JPEG quality for compression
# ============================================================================
# Input/Output data models
# ============================================================================
@dataclass
class PageInput:
    """A single page to process."""
    page_id: str              # Unique identifier (e.g., domain + path hash)
    url: str                  # Original URL
    html: str                 # Trimmed HTML source
    screenshot_path: str      # Path to screenshot PNG/JPEG
    # Optional metadata
    domain: str = ""
    existing_schema: str = ""  # Any existing schema.org on the page
    category_hint: str = ""    # e.g., "restaurant", "product", "service"
@dataclass
class GenerationResult:
    """Result of generating JSON-LD for one page."""
    page_id: str
    url: str
    raw_output: str           # Raw model output
    jsonld: Optional[str]     # Cleaned JSON-LD (None if invalid)
    validation: dict          # Validation results
    valid: bool
    model: str
    generation_time_ms: int
    attempt: int = 1
    token_usage: dict = None
    def __post_init__(self):
        if self.token_usage is None:
            self.token_usage = {}
# ============================================================================
# HTML Trimming
# ============================================================================
def trim_html(html: str, max_tokens: int = 3000) -> str:
    """Trim HTML to fit within token budget while preserving useful content.
    Removes: scripts, styles, SVGs, comments, excessive whitespace.
    Preserves: semantic elements, text content, structural markup.
    """
    import re
    # Remove script tags and content
    html = re.sub(r"<script\b[^>]*>[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    # Remove style tags and content
    html = re.sub(r"<style\b[^>]*>[\s\S]*?</style>", "", html, flags=re.IGNORECASE)
    # Remove SVG tags
    html = re.sub(r"<svg\b[^>]*>[\s\S]*?</svg>", "", html, flags=re.IGNORECASE)
    # Remove HTML comments
    html = re.sub(r"<!--[\s\S]*?-->", "", html)
    # Remove existing JSON-LD (we're generating fresh)
    html = re.sub(r'<script\s+type="application/ld\+json"[^>]*>[\s\S]*?</script>', "", html, flags=re.IGNORECASE)
    # Remove data attributes
    html = re.sub(r'\s+data-[a-z-]+="[^"]*"', "", html)
    # Remove class attributes with very long values
    html = re.sub(r'\s+class="[^"]{100,}"', "", html)
    # Collapse whitespace
    html = re.sub(r"\s{2,}", " ", html)
    html = re.sub(r">\s+<", ">\n<", html)
    # Rough token estimate: ~4 chars per token
    max_chars = max_tokens * 4
    if len(html) > max_chars:
        # Keep head (for meta tags) and beginning of body
        head_match = re.search(r"<head\b[^>]*>([\s\S]*?)</head>", html, re.IGNORECASE)
        head_content = head_match.group(0) if head_match else ""
        body_match = re.search(r"<body\b[^>]*>([\s\S]*)", html, re.IGNORECASE)
        body_content = body_match.group(1) if body_match else html
        remaining = max_chars - len(head_content) - 100
        html = head_content + "\n<body>\n" + body_content[:remaining] + "\n<!-- truncated -->\n</body>"
    return html.strip()
# ============================================================================
# Gemini API Client
# ============================================================================
class GeminiClient:
    """Client for Gemini API with rate limiting and retries."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.api_key = self._get_api_key()
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self._request_times: list[float] = []
        self._system_prompt = self._load_system_prompt()
    def _get_api_key(self) -> str:
        import os
        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise ValueError(f"Set {self.config.api_key_env} environment variable")
        return key
    def _load_system_prompt(self) -> str:
        prompt_path = Path(__file__).parent.parent / self.config.system_prompt_path
        if prompt_path.exists():
            return prompt_path.read_text()
        # Fallback: try relative to CWD
        prompt_path = Path(self.config.system_prompt_path)
        if prompt_path.exists():
            return prompt_path.read_text()
        raise FileNotFoundError(f"System prompt not found: {self.config.system_prompt_path}")
    async def _rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()
        # Remove timestamps older than 60 seconds
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= self.config.requests_per_minute:
            wait = 60 - (now - self._request_times[0])
            if wait > 0:
                logger.debug(f"Rate limited, waiting {wait:.1f}s")
                await asyncio.sleep(wait)
        self._request_times.append(time.time())
    def _build_request(self, page: PageInput) -> dict:
        """Build the Gemini API request payload."""
        # Read and encode screenshot
        screenshot_path = Path(page.screenshot_path)
        if not screenshot_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {page.screenshot_path}")
        with open(screenshot_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        # Determine MIME type
        suffix = screenshot_path.suffix.lower()
        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(suffix, "image/png")
        # Trim HTML
        trimmed_html = trim_html(page.html)
        # Build user message
        user_content = f"URL: {page.url}\n\nHTML source:\n{trimmed_html}"
        if page.category_hint:
            user_content = f"Category hint: {page.category_hint}\n\n{user_content}"
        return {
            "system_instruction": {
                "parts": [{"text": self._system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_data,
                            }
                        },
                        {"text": user_content},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": self.config.temperature,
                "topP": self.config.top_p,
                "maxOutputTokens": self.config.max_output_tokens,
                "responseMimeType": "application/json",
            },
        }
    async def generate(self, page: PageInput) -> GenerationResult:
        """Generate JSON-LD for a single page with retries."""
        import aiohttp
        url = f"{self.base_url}/models/{self.config.model}:generateContent?key={self.api_key}"
        for attempt in range(1, self.config.retry_max + 1):
            try:
                await self._rate_limit()
                payload = self._build_request(page)
                start_time = time.time()
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                        elapsed_ms = int((time.time() - start_time) * 1000)
                        if resp.status == 429:
                            wait = self.config.retry_backoff ** attempt
                            logger.warning(f"Rate limited on {page.page_id}, waiting {wait}s")
                            await asyncio.sleep(wait)
                            continue
                        if resp.status != 200:
                            body = await resp.text()
                            logger.error(f"API error {resp.status} for {page.page_id}: {body[:200]}")
                            if attempt < self.config.retry_max:
                                await asyncio.sleep(self.config.retry_backoff ** attempt)
                                continue
                            return GenerationResult(
                                page_id=page.page_id, url=page.url,
                                raw_output=f"API_ERROR_{resp.status}: {body[:200]}",
                                jsonld=None, validation={}, valid=False,
                                model=self.config.model,
                                generation_time_ms=elapsed_ms, attempt=attempt,
                            )
                        result = await resp.json()
                # Extract text from response
                raw_output = ""
                token_usage = {}
                try:
                    candidates = result.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        raw_output = "".join(p.get("text", "") for p in parts)
                    usage = result.get("usageMetadata", {})
                    token_usage = {
                        "input_tokens": usage.get("promptTokenCount", 0),
                        "output_tokens": usage.get("candidatesTokenCount", 0),
                        "total_tokens": usage.get("totalTokenCount", 0),
                    }
                except (KeyError, IndexError) as e:
                    logger.error(f"Failed to parse response for {page.page_id}: {e}")
                # Validate
                validation_result = validate(raw_output, page.html) if self.config.validate_output else None
                is_valid = validation_result.valid if validation_result else True
                # Auto-retry if invalid
                if not is_valid and self.config.auto_retry_invalid and attempt <= self.config.max_validation_retries:
                    logger.info(f"Invalid output for {page.page_id} (attempt {attempt}), retrying...")
                    continue
                return GenerationResult(
                    page_id=page.page_id, url=page.url,
                    raw_output=raw_output,
                    jsonld=raw_output if is_valid else None,
                    validation=asdict(validation_result) if validation_result else {},
                    valid=is_valid,
                    model=self.config.model,
                    generation_time_ms=elapsed_ms,
                    attempt=attempt,
                    token_usage=token_usage,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {page.page_id} (attempt {attempt})")
                if attempt < self.config.retry_max:
                    await asyncio.sleep(self.config.retry_backoff ** attempt)
            except Exception as e:
                logger.error(f"Error for {page.page_id}: {e}")
                if attempt < self.config.retry_max:
                    await asyncio.sleep(self.config.retry_backoff ** attempt)
        return GenerationResult(
            page_id=page.page_id, url=page.url,
            raw_output="MAX_RETRIES_EXCEEDED", jsonld=None,
            validation={}, valid=False, model=self.config.model,
            generation_time_ms=0, attempt=self.config.retry_max,
        )
# ============================================================================
# Pipeline Orchestrator
# ============================================================================
class Pipeline:
    """Orchestrates the full generation pipeline."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.output_dir / "results.jsonl"
        self.stats_path = self.output_dir / "stats.json"
    def load_pages(self) -> list[PageInput]:
        """Load page inputs from JSONL file."""
        pages = []
        with open(self.config.input_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                pages.append(PageInput(**data))
        logger.info(f"Loaded {len(pages)} pages from {self.config.input_path}")
        return pages
    def get_completed_ids(self) -> set[str]:
        """Get IDs of already-processed pages (for resume)."""
        completed = set()
        if self.results_path.exists():
            with open(self.results_path) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get("valid"):
                            completed.add(data["page_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        logger.info(f"Found {len(completed)} completed pages (resuming)")
        return completed
    def save_result(self, result: GenerationResult):
        """Append a result to the output file."""
        with open(self.results_path, "a") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    async def run_standard(self, pages: list[PageInput]):
        """Run generation in standard (real-time) mode with concurrency."""
        client = GeminiClient(self.config)
        semaphore = asyncio.Semaphore(self.config.concurrency)
        stats = {"total": len(pages), "valid": 0, "invalid": 0, "errors": 0,
                 "total_input_tokens": 0, "total_output_tokens": 0}
        async def process_one(page: PageInput):
            async with semaphore:
                result = await client.generate(page)
                self.save_result(result)
                if result.valid:
                    stats["valid"] += 1
                elif result.jsonld is None and "API_ERROR" in result.raw_output:
                    stats["errors"] += 1
                else:
                    stats["invalid"] += 1
                if result.token_usage:
                    stats["total_input_tokens"] += result.token_usage.get("input_tokens", 0)
                    stats["total_output_tokens"] += result.token_usage.get("output_tokens", 0)
                processed = stats["valid"] + stats["invalid"] + stats["errors"]
                if processed % 50 == 0:
                    logger.info(
                        f"Progress: {processed}/{stats['total']} | "
                        f"Valid: {stats['valid']} | Invalid: {stats['invalid']} | "
                        f"Errors: {stats['errors']}"
                    )
        tasks = [process_one(page) for page in pages]
        await asyncio.gather(*tasks)
        # Save final stats
        with open(self.stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        return stats
    async def run(self):
        """Main entry point."""
        pages = self.load_pages()
        completed = self.get_completed_ids()
        remaining = [p for p in pages if p.page_id not in completed]
        if not remaining:
            logger.info("All pages already processed!")
            return
        logger.info(f"Processing {len(remaining)} remaining pages ({len(completed)} already done)")
        if self.config.mode == "batch":
            logger.info("Batch mode: preparing batch request file...")
            self._prepare_batch(remaining)
        else:
            stats = await self.run_standard(remaining)
            logger.info(f"\nFinal stats: {json.dumps(stats, indent=2)}")
    def _prepare_batch(self, pages: list[PageInput]):
        """Prepare a batch request file for Gemini Batch API."""
        client = GeminiClient(self.config)
        batch_file = self.output_dir / "batch_requests.jsonl"
        with open(batch_file, "w") as f:
            for page in pages:
                try:
                    payload = client._build_request(page)
                    batch_entry = {
                        "custom_id": page.page_id,
                        "request": {
                            "model": f"models/{self.config.model}",
                            "body": payload,
                        }
                    }
                    f.write(json.dumps(batch_entry, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.error(f"Failed to prepare {page.page_id}: {e}")
        logger.info(f"Batch file written: {batch_file} ({len(pages)} requests)")
        logger.info("Submit this file to the Gemini Batch API endpoint.")
        logger.info("See: https://ai.google.dev/gemini-api/docs/batch")
# ============================================================================
# Training Data Export
# ============================================================================
def export_training_data(results_path: str, output_path: str, min_factual_score: float = 0.5):
    """Export validated results as training data in chat format.
    Output format (JSONL, compatible with most fine-tuning APIs):
    {
        "messages": [
            {"role": "system", "content": "...system prompt..."},
            {"role": "user", "content": [image, html]},
            {"role": "assistant", "content": "...json-ld..."}
        ],
        "metadata": {"page_id": "...", "url": "...", "types": [...]}
    }
    """
    system_prompt_path = Path(__file__).parent.parent / "prompts" / "teacher_system_prompt.txt"
    system_prompt = system_prompt_path.read_text() if system_prompt_path.exists() else ""
    valid_count = 0
    total_count = 0
    with open(results_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            total_count += 1
            result = json.loads(line.strip())
            if not result.get("valid"):
                continue
            # Check factual score
            validation = result.get("validation", {})
            if validation.get("factual_score", 0) < min_factual_score:
                continue
            training_example = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"URL: {result['url']}\n\n[Screenshot attached]\n\n[HTML source attached]"},
                    {"role": "assistant", "content": result["jsonld"]},
                ],
                "metadata": {
                    "page_id": result["page_id"],
                    "url": result["url"],
                    "types": validation.get("stats", {}).get("types", []),
                    "factual_score": validation.get("factual_score", 0),
                },
            }
            f_out.write(json.dumps(training_example, ensure_ascii=False) + "\n")
            valid_count += 1
    logger.info(f"Exported {valid_count}/{total_count} examples to {output_path}")
    return valid_count
# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Schema.org JSON-LD synthetic data generation pipeline")
    subparsers = parser.add_subparsers(dest="command")
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate JSON-LD from pages")
    gen_parser.add_argument("--input", required=True, help="Input JSONL file with page data")
    gen_parser.add_argument("--output", default="results", help="Output directory")
    gen_parser.add_argument("--mode", choices=["standard", "batch"], default="standard")
    gen_parser.add_argument("--model", default="gemini-2.5-flash")
    gen_parser.add_argument("--concurrency", type=int, default=5)
    gen_parser.add_argument("--temperature", type=float, default=0.1)
    gen_parser.add_argument("--no-validate", action="store_true")
    gen_parser.add_argument("--no-auto-retry", action="store_true")
    # Export command
    export_parser = subparsers.add_parser("export", help="Export validated results as training data")
    export_parser.add_argument("--results", required=True, help="Path to results.jsonl")
    export_parser.add_argument("--output", required=True, help="Output training data path")
    export_parser.add_argument("--min-factual-score", type=float, default=0.5)
    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate a single JSON-LD file")
    val_parser.add_argument("jsonld_file", help="Path to JSON-LD file")
    val_parser.add_argument("--html", help="Path to source HTML")
    args = parser.parse_args()
    if args.command == "generate":
        config = PipelineConfig(
            input_path=args.input,
            output_dir=args.output,
            mode=args.mode,
            model=args.model,
            concurrency=args.concurrency,
            temperature=args.temperature,
            validate_output=not args.no_validate,
            auto_retry_invalid=not args.no_auto_retry,
        )
        pipeline = Pipeline(config)
        asyncio.run(pipeline.run())
    elif args.command == "export":
        export_training_data(args.results, args.output, args.min_factual_score)
    elif args.command == "validate":
        with open(args.jsonld_file) as f:
            raw = f.read()
        html = ""
        if args.html:
            with open(args.html) as f:
                html = f.read()
        result = validate(raw, html)
        print(json.dumps(asdict(result), indent=2))
    else:
        parser.print_help()
if __name__ == "__main__":
    main()
