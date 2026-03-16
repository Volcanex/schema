"""
Rescoring Pipeline — Gemini 2.5 Pro as Quality Judge
Takes validated generation results and re-scores a subset with a premium model.
Filters out examples that pass programmatic validation but have subtle quality issues.
Usage:
    python rescore.py --results results/results.jsonl --output results/rescored.jsonl \
                      --sample 2000 --min-score 3.5
"""
import json
import random
import asyncio
import logging
import time
import base64
import argparse
from pathlib import Path
from dataclasses import dataclass
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
@dataclass
class RescoreConfig:
    model: str = "gemini-2.5-pro"
    api_key_env: str = "GEMINI_API_KEY"
    concurrency: int = 3  # Lower concurrency for Pro model
    requests_per_minute: int = 30
    retry_max: int = 3
    retry_backoff: float = 3.0
    max_output_tokens: int = 1024
    temperature: float = 0.0
class Rescorer:
    def __init__(self, config: RescoreConfig):
        self.config = config
        self.api_key = self._get_api_key()
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self._request_times: list[float] = []
        self._rescore_prompt = self._load_rescore_prompt()
    def _get_api_key(self) -> str:
        import os
        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise ValueError(f"Set {self.config.api_key_env} environment variable")
        return key
    def _load_rescore_prompt(self) -> str:
        prompt_path = Path(__file__).parent.parent / "prompts" / "rescoring_prompt.txt"
        if prompt_path.exists():
            return prompt_path.read_text()
        raise FileNotFoundError("rescoring_prompt.txt not found")
    async def _rate_limit(self):
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= self.config.requests_per_minute:
            wait = 60 - (now - self._request_times[0])
            if wait > 0:
                await asyncio.sleep(wait)
        self._request_times.append(time.time())
    async def rescore_one(self, result: dict, screenshot_path: str, html: str) -> dict:
        """Rescore a single generation result with the Pro model."""
        import aiohttp
        url = f"{self.base_url}/models/{self.config.model}:generateContent?key={self.api_key}"
        # Build the rescoring request
        parts = []
        # Add screenshot if available
        if screenshot_path and Path(screenshot_path).exists():
            with open(screenshot_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            suffix = Path(screenshot_path).suffix.lower()
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
                suffix.lstrip("."), "image/png"
            )
            parts.append({"inline_data": {"mime_type": mime, "data": image_data}})
        # Add the text prompt
        user_text = (
            f"URL: {result['url']}\n\n"
            f"HTML source (trimmed):\n{html[:8000]}\n\n"
            f"Candidate JSON-LD:\n{result['jsonld']}"
        )
        parts.append({"text": user_text})
        payload = {
            "system_instruction": {"parts": [{"text": self._rescore_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
                "responseMimeType": "application/json",
            },
        }
        for attempt in range(1, self.config.retry_max + 1):
            try:
                await self._rate_limit()
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as resp:
                        if resp.status == 429:
                            wait = self.config.retry_backoff ** attempt
                            logger.warning(f"Rate limited, waiting {wait}s")
                            await asyncio.sleep(wait)
                            continue
                        if resp.status != 200:
                            body = await resp.text()
                            logger.error(f"API error {resp.status}: {body[:200]}")
                            if attempt < self.config.retry_max:
                                await asyncio.sleep(self.config.retry_backoff ** attempt)
                                continue
                            return {"error": f"API_ERROR_{resp.status}"}
                        api_result = await resp.json()
                # Parse response
                candidates = api_result.get("candidates", [])
                if candidates:
                    text = "".join(
                        p.get("text", "")
                        for p in candidates[0].get("content", {}).get("parts", [])
                    )
                    try:
                        scores = json.loads(text)
                        return scores
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse rescore response: {text[:200]}")
                        if attempt < self.config.retry_max:
                            continue
                        return {"error": "PARSE_ERROR", "raw": text[:500]}
            except asyncio.TimeoutError:
                logger.warning(f"Timeout (attempt {attempt})")
                if attempt < self.config.retry_max:
                    await asyncio.sleep(self.config.retry_backoff ** attempt)
            except Exception as e:
                logger.error(f"Error: {e}")
                if attempt < self.config.retry_max:
                    await asyncio.sleep(self.config.retry_backoff ** attempt)
        return {"error": "MAX_RETRIES_EXCEEDED"}
async def run_rescoring(
    results_path: str,
    pages_path: str,
    output_path: str,
    sample_size: int = 2000,
    min_score: float = 3.5,
    config: RescoreConfig = None,
):
    """Run rescoring on a sample of validated results."""
    if config is None:
        config = RescoreConfig()
    rescorer = Rescorer(config)
    # Load results (only valid ones)
    results = []
    with open(results_path) as f:
        for line in f:
            r = json.loads(line.strip())
            if r.get("valid") and r.get("jsonld"):
                results.append(r)
    logger.info(f"Loaded {len(results)} valid results")
    # Sample
    if len(results) > sample_size:
        results = random.sample(results, sample_size)
        logger.info(f"Sampled {sample_size} results for rescoring")
    # Load page data for HTML cross-referencing
    pages = {}
    if pages_path and Path(pages_path).exists():
        with open(pages_path) as f:
            for line in f:
                page = json.loads(line.strip())
                pages[page["page_id"]] = page
    # Process with concurrency
    semaphore = asyncio.Semaphore(config.concurrency)
    stats = {"total": len(results), "passed": 0, "failed": 0, "errors": 0}
    async def process_one(result: dict) -> dict:
        async with semaphore:
            page = pages.get(result["page_id"], {})
            scores = await rescorer.rescore_one(
                result,
                screenshot_path=page.get("screenshot_path", ""),
                html=page.get("html", ""),
            )
            result["rescore"] = scores
            if "error" in scores:
                stats["errors"] += 1
                result["rescore_pass"] = None
            else:
                overall = scores.get("overall_score", 0)
                passed = overall >= min_score and scores.get("pass", False)
                result["rescore_pass"] = passed
                if passed:
                    stats["passed"] += 1
                else:
                    stats["failed"] += 1
            return result
    tasks = [process_one(r) for r in results]
    rescored = await asyncio.gather(*tasks)
    # Save rescored results
    with open(output_path, "w") as f:
        for r in rescored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Save final training data (only rescored passes)
    training_path = Path(output_path).parent / "training_data_final.jsonl"
    final_count = 0
    with open(training_path, "w") as f:
        for r in rescored:
            if r.get("rescore_pass"):
                f.write(json.dumps({
                    "page_id": r["page_id"],
                    "url": r["url"],
                    "jsonld": r["jsonld"],
                    "rescore": r["rescore"],
                }, ensure_ascii=False) + "\n")
                final_count += 1
    logger.info(f"\nRescoring complete:")
    logger.info(f"  Passed: {stats['passed']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Errors: {stats['errors']}")
    logger.info(f"  Final training examples: {final_count}")
    logger.info(f"  Output: {training_path}")
    return stats
def main():
    parser = argparse.ArgumentParser(description="Rescore schema.org JSON-LD with premium model")
    parser.add_argument("--results", required=True, help="Path to validated results.jsonl")
    parser.add_argument("--pages", help="Path to original pages.jsonl (for HTML/screenshots)")
    parser.add_argument("--output", default="results/rescored.jsonl", help="Output path")
    parser.add_argument("--sample", type=int, default=2000, help="Number of results to rescore")
    parser.add_argument("--min-score", type=float, default=3.5, help="Minimum overall score to pass")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Rescoring model")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    config = RescoreConfig(model=args.model, concurrency=args.concurrency)
    asyncio.run(run_rescoring(
        results_path=args.results,
        pages_path=args.pages,
        output_path=args.output,
        sample_size=args.sample,
        min_score=args.min_score,
        config=config,
    ))
if __name__ == "__main__":
    main()
