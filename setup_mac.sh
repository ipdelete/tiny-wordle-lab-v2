#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it with: brew install uv" >&2
  exit 1
fi

uv sync --extra dev

echo
echo "Environment synchronized."
echo "Run:"
echo "  export PYTORCH_ENABLE_MPS_FALLBACK=1"
echo "  uv run jupyter lab"
