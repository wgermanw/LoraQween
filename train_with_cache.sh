#!/usr/bin/env bash
# Thin wrapper to set up Hugging Face cache paths before training.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Allow overriding with environment variables, but default to repo-local cache.
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

PERSON="${1:-Mikassa}"
TRIGGER="${2:-}"

echo "========================================"
echo "Hugging Face cache directory: $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "========================================"

CMD=(python scripts/train_lora.py --person "$PERSON")
if [[ -n "$TRIGGER" ]]; then
  CMD+=("--trigger_token" "$TRIGGER")
fi

echo "Starting training for person: $PERSON"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
