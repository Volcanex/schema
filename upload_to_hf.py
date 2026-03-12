"""
Upload training dataset + screenshots to HuggingFace Hub as a private dataset.
Repo: gabriel-penman/schema-ie-training  (private)

Uploads:
  data/processed/train.jsonl
  data/processed/eval.jsonl
  data/screenshots/*.png  (1,488 files, ~330MB)

RunPod training script will download with:
  huggingface_hub.snapshot_download("gabriel-penman/schema-ie-training")
"""
import os, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('/home/gabriel/schema/.env', override=True)
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set in .env")
    sys.exit(1)

from huggingface_hub import HfApi, create_repo
from tqdm.auto import tqdm

REPO_ID   = "Volcanex/schema-ie-training"
REPO_TYPE = "dataset"
api       = HfApi(token=HF_TOKEN)

# ── Create private repo (idempotent) ──────────────────────────────────────
print(f"Creating/verifying repo: {REPO_ID}")
try:
    create_repo(REPO_ID, repo_type=REPO_TYPE, private=True, token=HF_TOKEN, exist_ok=True)
    print("  Repo ready.")
except Exception as e:
    print(f"  Repo create: {e}")

# ── Upload JSONL files ────────────────────────────────────────────────────
for fname in ['train.jsonl', 'eval.jsonl']:
    local = Path(f'data/processed/{fname}')
    print(f"Uploading {fname} ({local.stat().st_size/1e6:.1f} MB)...")
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=f'data/{fname}',
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=HF_TOKEN,
    )
    print(f"  Done.")

# ── Upload screenshots ────────────────────────────────────────────────────
screenshot_dir = Path('data/screenshots')
pngs = sorted(screenshot_dir.glob('*.png'))
print(f"\nUploading {len(pngs)} screenshots (~330MB)...")
print("(uploading as a folder — this takes a few minutes)")

api.upload_folder(
    folder_path=str(screenshot_dir),
    path_in_repo='data/screenshots',
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    token=HF_TOKEN,
)
print("  Screenshots done.")

print(f"""
Upload complete.
Repo: https://huggingface.co/datasets/{REPO_ID}

To use on RunPod, add to training script:
  from huggingface_hub import snapshot_download
  local = snapshot_download(repo_id="Volcanex/schema-ie-training", repo_type="dataset", token=HF_TOKEN)
  TRAIN_PATH = local + "/data/train.jsonl"
  SCREENSHOT_DIR = local + "/data/screenshots"
""")
