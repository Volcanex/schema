"""
RunPod Serverless handler for schema.org JSON-LD inference.

This script is the entrypoint for the Docker container deployed to RunPod Serverless.

Build and deploy:
  docker build -t your-registry/schema-model:latest -f Dockerfile.serverless .
  docker push your-registry/schema-model:latest
  # Then create endpoint in RunPod dashboard pointing to your image.

The handler loads the model at cold start and processes requests.
Each request: { "input": { "html": "...", "screenshot": "<base64>" } }
Response:      { "schema_jsonld": "{ \"@context\": ... }" }
"""

import base64
import logging
import os
from pathlib import Path
from typing import Optional

import runpod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "/models/schema-qwen-7b-lora")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "4096"))
GPU_MEM_UTIL = float(os.getenv("GPU_MEM_UTIL", "0.85"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

SYSTEM_PROMPT = (
    "You are a schema.org JSON-LD expert. Given a screenshot and HTML of a web page, "
    "generate the optimal schema.org JSON-LD markup. Output ONLY valid JSON-LD with "
    "no markdown formatting, no explanation, and no code fences. "
    "Use the most specific @type available. Include all extractable properties. "
    'Always include "@context": "https://schema.org".'
)

# ---------------------------------------------------------------------------
# Cold start: load model once
# ---------------------------------------------------------------------------

logger.info(f"Loading model from {MODEL_PATH}")

try:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_PATH,
        quantization="awq",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.95,
        repetition_penalty=1.05,
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def build_prompt(html: str, has_image: bool = False) -> str:
    truncated = html[:8_000]
    user_text = f"Generate schema.org JSON-LD for this web page.\n\nHTML:\n{truncated}"
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def handler(event: dict) -> dict:
    """
    RunPod serverless handler.
    Called once per inference request.
    """
    try:
        input_data = event.get("input", {})
        html = input_data.get("html", "")
        screenshot_b64: Optional[str] = input_data.get("screenshot")

        if not html:
            return {"error": "No HTML provided"}

        prompt = build_prompt(html, has_image=screenshot_b64 is not None)

        # Note: for full multimodal support with screenshots, use the Qwen processor
        # to build proper multimodal prompts. This plain-text fallback works well
        # for HTML-rich pages; screenshots add marginal gain for most site types.
        # Full multimodal implementation: use transformers pipeline with processor.

        outputs = llm.generate([prompt], sampling_params)
        schema_jsonld = outputs[0].outputs[0].text.strip()

        return {"schema_jsonld": schema_jsonld}

    except Exception as exc:
        logger.error(f"Handler error: {exc}")
        return {"error": str(exc)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
