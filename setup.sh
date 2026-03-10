#!/bin/bash
# Schema Model — local environment setup
# Run once after cloning: bash setup.sh

set -e

echo "=== Schema Model Setup ==="

# 1. Create virtualenv (prefer 3.12, fall back to 3.11 or default python3)
if command -v python3.12 &>/dev/null; then
    PYTHON=python3.12
elif command -v python3.11 &>/dev/null; then
    PYTHON=python3.11
else
    PYTHON=python3
fi

echo "Using: $($PYTHON --version)"

if [ ! -d ".venv" ]; then
    $PYTHON -m venv .venv
    echo "Created .venv"
else
    echo ".venv already exists, skipping"
fi

# 2. Activate
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip -q
pip install -r requirements.txt

# 4. Playwright browsers
playwright install chromium

# 5. Register as a Jupyter kernel so it shows up in JupyterLab
python -m ipykernel install --user --name=schema-model --display-name="Schema Model (.venv)"

# 6. Copy .env template if no .env exists yet
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "Created .env from template — fill in your API keys before running notebooks:"
    echo "  ANTHROPIC_API_KEY  — console.anthropic.com"
    echo "  RUNPOD_API_KEY     — runpod.io/console/user/settings"
    echo "  HF_TOKEN           — huggingface.co/settings/tokens"
fi

echo ""
echo "=== Setup complete ==="
echo "To start Jupyter:"
echo "  source .venv/bin/activate"
echo "  jupyter lab"
