"""
NB05 — Training Data Prep
Filters manifest → assembles dataset.jsonl → splits train/eval
"""
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, '/home/gabriel/schema')
from dotenv import load_dotenv
load_dotenv('/home/gabriel/schema/.env', override=True)

from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

from src.schema_validator import validate_jsonld
from src.training_data import format_training_example, split_dataset

MANIFEST_PATH      = Path('data/raw/warc_manifest.jsonl')
UK_MANIFEST_PATH   = Path('data/raw/warc_manifest_uk.jsonl')   # optional, added later
PROCESSED_DIR      = Path('data/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MIN_QUALITY       = 0.3
MIN_HTML_LEN      = 200
MAX_PER_DOMAIN    = 5    # allow up to 5 pages per domain (different schema types)
MAX_PER_TYPE      = 5_000

# ── Load manifest ──────────────────────────────────────────────────────────
log.info('Loading manifest...')
manifest = [json.loads(l) for l in MANIFEST_PATH.open()]
if UK_MANIFEST_PATH.exists():
    uk = [json.loads(l) for l in UK_MANIFEST_PATH.open()]
    manifest += uk
    log.info(f'  {len(manifest)} entries (incl. {len(uk)} UK)')
else:
    log.info(f'  {len(manifest)} entries (.ie only — no UK manifest yet)')

# ── Quality filter ─────────────────────────────────────────────────────────
log.info('Quality filtering...')
stats = Counter()
valid_examples = []

for entry in tqdm(manifest, desc='Filtering'):
    html_path = entry.get('html_path')
    if not html_path or not Path(html_path).exists():
        stats['no_html'] += 1
        continue

    html = Path(html_path).read_text(errors='replace')
    if len(html) < MIN_HTML_LEN:
        stats['html_too_short'] += 1
        continue

    wdc_jsonld = entry.get('wdc_jsonld')
    if not wdc_jsonld:
        stats['no_jsonld'] += 1
        continue

    result = validate_jsonld(json.dumps(wdc_jsonld, ensure_ascii=False))
    if not result['valid']:
        stats['invalid_jsonld'] += 1
        continue
    if result['quality_score'] < MIN_QUALITY:
        stats['low_quality'] += 1
        continue

    # Drop encoding-corrupted labels — UTF-8 bytes misread as Latin-1
    # These would teach the model to output â¬ instead of € etc.
    jsonld_str = json.dumps(wdc_jsonld, ensure_ascii=False)
    if 'â¬' in jsonld_str or 'Ã' in jsonld_str or '\ufffd' in jsonld_str:
        stats['encoding_corrupted'] += 1
        continue

    # Drop examples with no screenshot — keep dataset fully multimodal
    if not entry.get('screenshot_path'):
        stats['no_screenshot'] += 1
        continue

    stats['accepted'] += 1
    valid_examples.append({
        'html':            html,
        'jsonld':          wdc_jsonld,
        'screenshot_path': entry.get('screenshot_path'),
        'source':          'wdc',
        'schema_types':    result['schema_types'],
        'domain':          entry.get('url', '').split('/')[2] if entry.get('url') else '',
        'quality_score':   result['quality_score'],
    })

log.info('Filter stats:')
for k, v in sorted(stats.items()):
    log.info(f'  {k}: {v}')

# ── Dedup by domain (cap at MAX_PER_DOMAIN per domain) ────────────────────
log.info(f'Deduplicating by domain (max {MAX_PER_DOMAIN} per domain)...')
domain_counts = Counter()
deduped = []
for ex in valid_examples:
    d = ex['domain']
    if domain_counts[d] < MAX_PER_DOMAIN:
        domain_counts[d] += 1
        deduped.append(ex)
log.info(f'  {len(deduped)} examples from {len(domain_counts)} unique domains')

# Type distribution
type_counts = Counter()
for ex in deduped:
    for t in ex.get('schema_types', ['Unknown']):
        type_counts[t] += 1
log.info('Schema type distribution:')
for t, c in type_counts.most_common(15):
    log.info(f'  {t}: {c}')

# ── Balance types ──────────────────────────────────────────────────────────
random.seed(42)
random.shuffle(deduped)
type_seen = Counter()
balanced = []
for ex in deduped:
    primary = (ex.get('schema_types') or ['Unknown'])[0]
    if type_seen[primary] < MAX_PER_TYPE:
        type_seen[primary] += 1
        balanced.append(ex)
log.info(f'After balancing: {len(balanced)} examples')

# ── Assemble JSONL ─────────────────────────────────────────────────────────
dataset_path = PROCESSED_DIR / 'dataset.jsonl'
log.info(f'Writing {len(balanced)} examples to {dataset_path}...')
with open(dataset_path, 'w') as f:
    for i, ex in enumerate(tqdm(balanced, desc='Assembling')):
        record = format_training_example(
            html=ex['html'],
            jsonld=ex['jsonld'],
            screenshot_path=ex.get('screenshot_path'),
            example_id=f'train_{i:06d}',
            source=ex['source'],
            schema_types=ex.get('schema_types'),
            domain=ex.get('domain'),
            quality_score=ex.get('quality_score'),
        )
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

log.info(f'Saved {len(balanced)} examples')

# ── Train/eval split ───────────────────────────────────────────────────────
train_n, eval_n = split_dataset(
    input_path=str(dataset_path),
    train_path=str(PROCESSED_DIR / 'train.jsonl'),
    eval_path=str(PROCESSED_DIR / 'eval.jsonl'),
    train_ratio=0.9,
)
log.info(f'Split: {train_n} train / {eval_n} eval')
log.info('── Done ──')
log.info(f'  Outputs: {PROCESSED_DIR}/train.jsonl, eval.jsonl')
