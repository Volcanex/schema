# Schema Model

A fine-tuned multimodal model that generates schema.org JSON-LD markup from web page screenshots + HTML. Built by Baseline Labs.

**Model**: Qwen2.5-VL-7B-Instruct (QLoRA fine-tuned, quantized Q4)
**Input**: Screenshot (1280×800 PNG) + HTML
**Output**: Valid schema.org JSON-LD

## What's Novel

No purpose-built multimodal model for schema.org generation exists. Closest alternatives are GPT wrappers or template engines. This model:
- Takes visual + structural cues (not text-only)
- Generates the correct schema *type* autonomously (not just extracts given a target type)
- Runs at ~$0.001/site vs $0.05–0.25/site with GPT-4o
- Produces a reusable model asset, not just API calls

## Repository Structure

```
schema-model/
├── notebooks/              # Numbered pipeline stages (run in order)
│   ├── 01_common_crawl_query.ipynb     # Find .ie domains in CC index
│   ├── 02_wdc_schema_download.ipynb    # Download WDC schema.org data
│   ├── 03_warc_extraction.ipynb        # Pull HTML from CC WARC files
│   ├── 04_screenshot_rendering.ipynb   # Render HTML → screenshots (Playwright)
│   ├── 05_training_data_prep.ipynb     # Assemble + validate training pairs
│   ├── 06_synthetic_data_gen.ipynb     # Claude API synthetic data generation
│   ├── 07_fine_tuning.ipynb            # QLoRA fine-tune on RunPod A100
│   ├── 08_evaluation.ipynb             # Benchmark vs GPT-4o, forgetting analysis
│   ├── 09_irish_web_inference.ipynb    # Batch process full Irish web
│   └── 10_analysis_dashboard.ipynb     # National schema audit results
├── src/                    # Importable Python modules
│   ├── common_crawl.py     # CC index querying + WARC fetching
│   ├── wdc.py              # WDC N-Quads parsing + filtering
│   ├── screenshot.py       # Playwright batch rendering
│   ├── schema_validator.py # JSON-LD validation + quality scoring
│   ├── training_data.py    # Dataset assembly + formatting
│   ├── synthetic_gen.py    # Claude API schema generation
│   ├── inference.py        # vLLM engine + RunPod serverless client
│   └── runpod_utils.py     # Pod/endpoint management
├── configs/
│   ├── schema_types.json       # Priority types + properties
│   ├── training_config.yaml    # QLoRA hyperparameters
│   └── inference_config.yaml   # vLLM serving config
└── scripts/
    ├── setup_runpod.sh         # RunPod pod environment setup
    └── deploy_serverless.py    # Serverless handler entrypoint
```

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt
playwright install chromium

# 2. Configure environment
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, RUNPOD_API_KEY, HF_TOKEN

# 3. Run notebooks in order (01 → 10)
jupyter lab
```

## Pipeline Phases

| Phase | Notebook | Compute | Cost | Time |
|-------|----------|---------|------|------|
| 1. Data collection | 01, 02 | CPU | $10–30 | 1–2 days |
| 2. Screenshot rendering | 04 | CPU (Playwright) | $0–30 | 1–2 days |
| 3. Synthetic data gen | 06 | API (Claude) | $50–200 | 1 day |
| 4. Fine-tuning | 07 | 1× A100 80GB | $50–100 | 4–8 hours |
| 5. Irish web inference | 09 | RunPod Serverless | $100–250 | 2–3 days |

**Total**: ~$210–610 for the full Irish web run.

Compare: GPT-4o API for inference alone = $6,400–32,000. No model asset at the end.

## Fine-Tuning (RunPod)

```bash
# 1. Spin up a RunPod A100 pod (~$1.39/hr)
# 2. SSH in or open Jupyter Lab from dashboard
bash /workspace/schema-model/scripts/setup_runpod.sh

# 3. Upload training data to /workspace/schema-model/data/processed/
# 4. Open and run notebook 07_fine_tuning.ipynb
```

## Inference

```python
from src.inference import RunPodSchemaClient

client = RunPodSchemaClient(endpoint_id='...', api_key='...')
results = client.process_batch(items=[
    {'id': 'example-ie', 'html': open('page.html').read()}
])
print(results[0]['schema_jsonld'])
```

## Dissertation Connection

The fine-tuning work connects to an MA AI dissertation on catastrophic forgetting in small LMs (1B–7B).

- Notebook 08 includes a forgetting evaluation: benchmark the fine-tuned model on general web QA vs the base model
- The QLoRA approach (LoRA rank 64, selective module targeting) is an instance of the parameter-efficient fine-tuning methods studied in EWC/LoRA forgetting research
- Comparing pre/post fine-tuning performance on held-out general tasks provides a concrete case study for the forgetting chapter

## Key Design Decisions

**Why Qwen2.5-VL-7B**: Native multimodal (no separate vision encoder to wrangle), Qwen2.5 was explicitly enhanced for structured data + JSON output, runs on single A100 for training and single L40S for inference.

**Why QLoRA**: 4-bit base + LoRA adapters lets us fine-tune a 7B model on one A100 80GB in 4–8 hours. Full fine-tuning would require 4–8× A100s.

**Why vLLM + XGrammar**: Continuous batching gives ~10× throughput vs sequential inference. XGrammar constrains output to valid JSON at ~40μs overhead per token — no post-processing needed.

**Why RunPod Serverless**: Scale-to-zero. Processing 640K pages takes ~2.5 days then we're done. No idle GPU cost. Auto-scales to meet the burst workload.
