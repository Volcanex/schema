"""
vLLM inference wrapper for the fine-tuned schema model.
Used for both local inference and batching against the RunPod serverless endpoint.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a schema.org JSON-LD expert. Given a screenshot and HTML of a web page, "
    "generate the optimal schema.org JSON-LD markup. Output ONLY valid JSON-LD with "
    "no markdown formatting, no explanation, and no code fences. "
    "Use the most specific @type available. Include all extractable properties. "
    'Always include "@context": "https://schema.org".'
)


# ---------------------------------------------------------------------------
# Local vLLM inference
# ---------------------------------------------------------------------------

class SchemaInferenceEngine:
    """
    vLLM-based inference engine. Loads the fine-tuned model and serves predictions.
    Run on a GPU node (L40S or A100). Not for local CPU use.
    """

    def __init__(
        self,
        model_path: str,
        quantization: str = "awq",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        enable_constrained_decoding: bool = True,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM not installed. Run: pip install vllm")

        logger.info(f"Loading model from {model_path}")
        self.llm = LLM(
            model=model_path,
            quantization=quantization,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.SamplingParams = SamplingParams
        self.enable_constrained_decoding = enable_constrained_decoding
        logger.info("Model loaded")

    def generate(
        self,
        html: str,
        screenshot_b64: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Generate schema JSON-LD for a single page."""
        prompt = self._build_prompt(html, screenshot_b64)
        params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.05,
        )
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()

    def generate_batch(
        self,
        items: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> list[str]:
        """
        Generate schema for a batch of pages (efficient batched inference).

        Args:
            items: List of dicts with 'html' and optionally 'screenshot_b64'.
        """
        prompts = [
            self._build_prompt(item["html"], item.get("screenshot_b64"))
            for item in items
        ]
        params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
        )
        outputs = self.llm.generate(prompts, params)
        return [o.outputs[0].text.strip() for o in outputs]

    def _build_prompt(self, html: str, screenshot_b64: Optional[str] = None) -> str:
        """Build the prompt string for vLLM (Qwen2.5-VL chat format)."""
        # Qwen2.5-VL uses its own special tokens for multimodal input.
        # When using vLLM with the Qwen processor, pass structured messages.
        # For simplicity here we return a plain string; in practice use
        # the tokenizer's apply_chat_template.
        truncated_html = html[:8_000]
        user_text = f"Generate schema.org JSON-LD for this web page.\n\nHTML:\n{truncated_html}"
        return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


# ---------------------------------------------------------------------------
# RunPod serverless client
# ---------------------------------------------------------------------------

class RunPodSchemaClient:
    """
    Async client for the RunPod serverless schema endpoint.
    Handles batching, retries, and progress tracking.
    """

    def __init__(
        self,
        endpoint_id: str,
        api_key: str,
        max_retries: int = 3,
    ):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.max_retries = max_retries
        self.base_url = f"https://api.runpod.io/v2/{endpoint_id}"

    async def submit_job(self, html: str, screenshot_b64: Optional[str] = None) -> str:
        """Submit a single inference job. Returns job ID."""
        import aiohttp

        payload = {
            "input": {
                "html": html[:8_000],
                "screenshot": screenshot_b64,
            }
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/run",
                json=payload,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["id"]

    async def poll_job(self, job_id: str, poll_interval: float = 2.0) -> Optional[str]:
        """Poll a job until complete. Returns output string or None."""
        import aiohttp
        import asyncio

        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/status/{job_id}"

        async with aiohttp.ClientSession() as session:
            for _ in range(120):  # max ~4 min wait
                async with session.get(url, headers=headers) as resp:
                    data = await resp.json()
                status = data.get("status")
                if status == "COMPLETED":
                    return data.get("output", {}).get("schema_jsonld")
                elif status in ("FAILED", "CANCELLED"):
                    logger.warning(f"Job {job_id} {status}")
                    return None
                await asyncio.sleep(poll_interval)

        logger.warning(f"Job {job_id} timed out")
        return None

    async def process_batch_async(
        self,
        items: list[dict],
        concurrency: int = 8,
    ) -> list[dict]:
        """
        Process a batch of pages with controlled concurrency.
        Each item should have 'id', 'html', and optionally 'screenshot_path'.
        """
        import asyncio

        sem = asyncio.Semaphore(concurrency)
        results = []

        async def process(item: dict) -> dict:
            async with sem:
                screenshot_b64 = None
                if "screenshot_path" in item and Path(item["screenshot_path"]).exists():
                    with open(item["screenshot_path"], "rb") as f:
                        screenshot_b64 = base64.b64encode(f.read()).decode()

                for attempt in range(self.max_retries):
                    try:
                        job_id = await self.submit_job(item["html"], screenshot_b64)
                        output = await self.poll_job(job_id)
                        if output:
                            return {
                                "id": item["id"],
                                "url": item.get("url", ""),
                                "schema_jsonld": output,
                                "success": True,
                            }
                    except Exception as exc:
                        logger.warning(f"Attempt {attempt + 1} failed for {item['id']}: {exc}")
                        await asyncio.sleep(2 ** attempt)

                return {"id": item["id"], "url": item.get("url", ""), "success": False}

        tasks = [process(item) for item in items]
        results = await asyncio.gather(*tasks)
        return list(results)

    def process_batch(self, items: list[dict], concurrency: int = 8) -> list[dict]:
        """Synchronous wrapper for process_batch_async."""
        import asyncio
        return asyncio.run(self.process_batch_async(items, concurrency=concurrency))
