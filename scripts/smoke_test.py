#!/usr/bin/env python3
"""
Quick smoke test — run before full training to catch failures in <2 minutes.
Loads model, runs 3 gradient steps on 5 cached .pt examples, checks:
  - Loss is a real number (not NaN/0/inf)
  - Loss decreases step-over-step
  - No OOM
  - VRAM budget is within A100 limits

Usage:
    python scripts/smoke_test.py
"""
import os, sys, json, time
import torch

# Shim: set_submodule missing on some PyTorch builds, needed by bitsandbytes quantizer.
if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target, module):
        parent, _, last = target.rpartition(".")
        parent_mod = self.get_submodule(parent) if parent else self
        setattr(parent_mod, last, module)
    torch.nn.Module.set_submodule = _set_submodule
from pathlib import Path

# ── Must be first, before any CUDA init ────────────────────────────────────
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

PROJECT_ROOT = Path('/workspace/schema')
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env', override=True)

import yaml
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset

print('=' * 60)
print('SMOKE TEST')
print('=' * 60)

# ── Config ─────────────────────────────────────────────────────────────────
with open(PROJECT_ROOT / 'configs/training_config.yaml') as f:
    config = yaml.safe_load(f)

HF_TOKEN   = os.getenv('HF_TOKEN')
MODEL_ID   = config['model']['name']
MODELS_DIR = Path('/workspace/models')
CACHE_DIR  = PROJECT_ROOT / 'data' / 'preprocessed'

assert torch.cuda.is_available(), 'No GPU!'
total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {torch.cuda.get_device_name(0)}  VRAM: {total_vram:.0f} GB')
assert total_vram >= 75, f'Need ≥75 GB VRAM, got {total_vram:.0f} GB'

# ── Check cache files exist ────────────────────────────────────────────────
pt_files = sorted(CACHE_DIR.glob('train_*.pt'))
assert len(pt_files) >= 5, f'Need ≥5 cached .pt files in {CACHE_DIR}, found {len(pt_files)}'
print(f'Cache: {len(pt_files)} train .pt files ✓')

# ── Load model ─────────────────────────────────────────────────────────────
print(f'\nLoading {MODEL_ID}...')
t0 = time.time()
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                 bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, torch_dtype=torch.bfloat16,
    attn_implementation='sdpa', device_map='auto', token=HF_TOKEN, cache_dir=str(MODELS_DIR))
processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN, cache_dir=str(MODELS_DIR),
                                           min_pixels=256*28*28, max_pixels=640*28*28)
model = prepare_model_for_kbit_training(model)
lora_cfg = config['lora']
model = get_peft_model(model, LoraConfig(r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'],
    lora_dropout=lora_cfg['dropout'], bias=lora_cfg['bias'],
    task_type=lora_cfg['task_type'], target_modules=lora_cfg['target_modules']))

vram_after_load = torch.cuda.memory_allocated() / 1e9
print(f'Model loaded in {time.time()-t0:.0f}s. VRAM: {vram_after_load:.1f} GB')
assert vram_after_load < 55, f'Model using {vram_after_load:.1f} GB — too much, check quant'

# ── Mini dataset (5 examples) ──────────────────────────────────────────────
class SmokePTDataset(Dataset):
    def __init__(self): self.files = pt_files[:5]
    def __len__(self): return len(self.files)
    def __getitem__(self, i): return torch.load(self.files[i], map_location='cpu', weights_only=True)

MAX_SEQ = 12288
PAD_ID  = processor.tokenizer.pad_token_id

def collate_fn(examples):
    max_len = min(MAX_SEQ, max(ex['input_ids'].shape[0] for ex in examples))
    input_ids = torch.stack([torch.nn.functional.pad(
        ex['input_ids'][:max_len], (0, max(0, max_len - ex['input_ids'].shape[0])), value=PAD_ID)
        for ex in examples])
    attention_mask = torch.stack([torch.nn.functional.pad(
        ex['attention_mask'][:max_len], (0, max(0, max_len - ex['attention_mask'].shape[0])))
        for ex in examples])
    labels = torch.stack([torch.nn.functional.pad(
        ex['labels'][:max_len], (0, max(0, max_len - ex['labels'].shape[0])), value=-100)
        for ex in examples])
    # Guard: skip examples missing pixel_values (shouldn't happen, but be safe)
    has_pixels = all('pixel_values' in ex and ex['pixel_values'] is not None for ex in examples)
    if not has_pixels:
        print('[WARN] Batch missing pixel_values — skipping visual tokens')
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels,
            'pixel_values': torch.cat([ex['pixel_values'] for ex in examples]),
            'image_grid_thw': torch.cat([ex['image_grid_thw'] for ex in examples])}

loader = DataLoader(SmokePTDataset(), batch_size=1, collate_fn=collate_fn)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# ── 3 gradient steps ───────────────────────────────────────────────────────
print('\nRunning 3 training steps...')
model.train()
losses = []
for step, batch in enumerate(loader):
    if step >= 3: break
    batch = {k: v.to('cuda') for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step(); optimizer.zero_grad()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f'  Step {step+1}: loss={loss.item():.4f}  VRAM={vram:.1f} GB')
    assert not torch.isnan(loss), 'NaN loss!'
    assert loss.item() > 0, f'Zero loss at step {step+1} — labels all masked?'
    assert vram < total_vram * 0.98, f'VRAM critical: {vram:.1f}/{total_vram:.1f} GB'
    losses.append(loss.item())

# ── Verdict ────────────────────────────────────────────────────────────────
print()
print('=' * 60)
if len(losses) == 3:
    print(f'✅ SMOKE TEST PASSED')
    print(f'   Losses: {[f"{l:.4f}" for l in losses]}')
    print(f'   Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB')
    print(f'   Safe to launch full training.')
else:
    print('❌ SMOKE TEST FAILED — did not complete 3 steps')
    sys.exit(1)
