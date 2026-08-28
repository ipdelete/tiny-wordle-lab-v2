#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

exec uv run --project "$root" --group notebooks jupyter lab \
  --no-browser \
  --ip=0.0.0.0 \
  --port=8888 \
  --ServerApp.allow_remote_access=True \
  --ServerApp.root_dir="$root"
