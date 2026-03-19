"""
Build train.jsonl + eval.jsonl from generated schema files + pages_for_generation.jsonl.

Usage:
    python3 -u scripts/build_training_data.py
"""
import json
import hashlib
import random
import re
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).parent.parent
GENERATED_DIR = PROJECT / "data" / "generated"
SHOT_DIR = PROJECT / "data" / "screenshots_v2"
PAGES_JSONL = PROJECT / "data" / "processed" / "pages_for_generation.jsonl"
OUT_DIR = PROJECT / "data" / "processed"

MIN_QUALITY = 0.3
MAX_HTML = 64_000
TRAIN_RATIO = 0.9
SEED = 42

SYSTEM_PROMPT = (
    "You are a schema.org JSON-LD expert. Given a screenshot and HTML of a web page, "
    "generate the optimal schema.org JSON-LD markup. Output ONLY valid JSON-LD with "
    "no markdown formatting, no explanation, and no code fences. "
    "Use the most specific @type available. Include all extractable properties. "
    'Always include "@context": "https://schema.org".'
)

USER_TEMPLATE = "Generate schema.org JSON-LD for this web page.\n\nHTML:\n{html}"

# Regex for stripping noise
_RE_JSONLD = re.compile(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>.*?</script>', re.DOTALL | re.IGNORECASE)
_RE_SCRIPT = re.compile(r'<script[^>]*>.*?</script>', re.DOTALL | re.IGNORECASE)
_RE_STYLE = re.compile(r'<style[^>]*>.*?</style>', re.DOTALL | re.IGNORECASE)
_RE_SVG = re.compile(r'<svg[\s>].*?</svg>', re.DOTALL | re.IGNORECASE)
_RE_LINK = re.compile(r'<link[^>]*/?>',  re.IGNORECASE)
_RE_DATA = re.compile(r'\s+data-[a-zA-Z][a-zA-Z0-9_:-]*(?:="[^"]*")?')
_RE_CLASS = re.compile(r'\s+class="[^"]*"')
_RE_STYLE_ATTR = re.compile(r'\s+style="[^"]*"')
_RE_BLANK = re.compile(r'\n\s*\n+')


def strip_html(html: str) -> str:
    for pat in [_RE_JSONLD, _RE_SCRIPT, _RE_STYLE, _RE_SVG, _RE_LINK, _RE_DATA, _RE_CLASS, _RE_STYLE_ATTR]:
        html = pat.sub('', html)
    html = _RE_BLANK.sub('\n', html)
    return html.strip()


def main():
    # Load pages for HTML content
    pages = {}
    with open(PAGES_JSONL) as f:
        for line in f:
            rec = json.loads(line)
            url = rec["url"]
            if url not in pages:
                pages[url] = rec
    print(f"Loaded {len(pages):,} unique pages")

    # Load generated schema files
    gen_files = sorted(GENERATED_DIR.glob("*.json"))
    print(f"Found {len(gen_files):,} generated schema files")

    examples = []
    skipped = {"invalid": 0, "low_quality": 0, "no_page": 0, "short_html": 0, "parse_err": 0}

    for gf in gen_files:
        try:
            gen = json.loads(gf.read_text())
        except Exception:
            skipped["parse_err"] += 1
            continue

        if not gen.get("valid", False):
            skipped["invalid"] += 1
            continue

        if gen.get("quality_score", 0) < MIN_QUALITY:
            skipped["low_quality"] += 1
            continue

        url = gen["url"]
        page = pages.get(url)
        if not page:
            skipped["no_page"] += 1
            continue

        html = page.get("html", "")
        if len(html) < 200:
            skipped["short_html"] += 1
            continue

        # Find screenshot
        h = hashlib.md5(url.encode()).hexdigest()
        shot_path = SHOT_DIR / f"{h}.png"

        # Strip and truncate HTML
        clean = strip_html(html)
        trunc = clean[:MAX_HTML]
        if len(clean) > MAX_HTML:
            trunc += "\n<!-- [HTML truncated] -->"

        # Build training example
        user_content = []
        if shot_path.exists():
            user_content.append({"type": "image", "image": str(shot_path)})
        user_content.append({"type": "text", "text": USER_TEMPLATE.format(html=trunc)})

        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": gen["generated_jsonld"]},
            ],
            "metadata": {
                "url": url,
                "schema_types": gen.get("schema_types", []),
                "quality_score": gen.get("quality_score", 0),
                "property_count": gen.get("property_count", 0),
                "has_screenshot": shot_path.exists(),
            },
        }
        examples.append(example)

    print(f"\nValid examples: {len(examples):,}")
    print(f"Skipped: {sum(skipped.values()):,}")
    for reason, count in skipped.items():
        print(f"  {reason}: {count:,}")

    # Split 90/10
    random.seed(SEED)
    random.shuffle(examples)
    split = int(len(examples) * TRAIN_RATIO)
    train = examples[:split]
    eval_ = examples[split:]

    # Write
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, data, name in [
        (OUT_DIR / "train.jsonl", train, "train"),
        (OUT_DIR / "eval.jsonl", eval_, "eval"),
    ]:
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"\n{name}: {len(data):,} examples -> {path}")

    # Stats
    types = Counter()
    with_shot = 0
    qualities = []
    for ex in examples:
        for t in ex["metadata"]["schema_types"]:
            types[t] += 1
        if ex["metadata"]["has_screenshot"]:
            with_shot += 1
        qualities.append(ex["metadata"]["quality_score"])

    print(f"\nWith screenshot: {with_shot:,}/{len(examples):,}")
    print(f"Quality: min={min(qualities):.2f} mean={sum(qualities)/len(qualities):.2f} max={max(qualities):.2f}")
    print(f"\nType distribution:")
    for t, n in types.most_common(20):
        print(f"  {t:30s} {n:5,}")


if __name__ == "__main__":
    main()
