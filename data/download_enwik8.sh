#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_FILE="${ROOT_DIR}/enwik8.zip"

if [[ -f "${ROOT_DIR}/enwik8" ]]; then
  echo "enwik8 already exists at ${ROOT_DIR}/enwik8"
  exit 0
fi

echo "Downloading enwik8 to ${OUT_FILE}"
curl -L "http://mattmahoney.net/dc/enwik8.zip" -o "${OUT_FILE}"
python3 - <<'PY'
import zipfile
from pathlib import Path

root = Path("data")
archive = root / "enwik8.zip"
with zipfile.ZipFile(archive) as zf:
    zf.extractall(root)
print(f"Extracted to {root}")
PY
