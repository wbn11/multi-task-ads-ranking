#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements-a6000.txt}"

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r "${REQUIREMENTS_FILE}"

echo "Environment ready: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA build: {torch.version.cuda} | CUDA available: {torch.cuda.is_available()}')"
echo "Activate with: source .venv/bin/activate"
