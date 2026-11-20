#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install transformers huggingface_hub

printf 'Hugging Face environment ready at %s\n' "$VENV_DIR"
printf 'Activate it with: source %s/bin/activate\n' "$VENV_DIR"
