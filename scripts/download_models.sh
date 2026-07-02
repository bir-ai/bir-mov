#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-server}"

usage() {
  cat <<'EOF'
Usage: scripts/download_models.sh

Downloads the bir.mov model artifacts from Hugging Face into ./model/.
The model directory is intentionally gitignored and should not be committed.

Environment overrides:
  PYTHON_BIN=/opt/homebrew/bin/python3.12 scripts/download_models.sh
  VENV_DIR=.venv-server scripts/download_models.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

is_supported_python() {
  "$1" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] in {(3, 10), (3, 11), (3, 12)} else 1)
PY
}

pick_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
      echo "PYTHON_BIN not found: $PYTHON_BIN" >&2
      return 1
    fi
    if ! is_supported_python "$PYTHON_BIN"; then
      echo "PYTHON_BIN must be Python 3.10, 3.11, or 3.12." >&2
      return 1
    fi
    command -v "$PYTHON_BIN"
    return 0
  fi

  local candidate
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && is_supported_python "$candidate"; then
      command -v "$candidate"
      return 0
    fi
  done

  echo "No supported Python found. Install Python 3.12 or set PYTHON_BIN." >&2
  return 1
}

PYTHON_BIN_RESOLVED="$(pick_python)"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating virtualenv: $VENV_DIR"
  "$PYTHON_BIN_RESOLVED" -m venv "$VENV_DIR"
fi

if ! "$VENV_DIR/bin/python" -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "Installing Hugging Face client..."
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install "huggingface_hub>=0.34"
fi

"$VENV_DIR/bin/python" - "$ROOT_DIR" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

root = Path(sys.argv[1])
repos = {
    "ml32m-als128-v1": "mskayacioglu/ml32m-als128-v1",
    "ml32m-mf128-v1": "mskayacioglu/ml32m-mf128-v1",
}

for name, repo_id in repos.items():
    target = root / "model" / name
    print(f"Downloading {repo_id} -> {target}")
    snapshot_download(repo_id=repo_id, local_dir=target)

print("Model artifacts are ready under ./model/ (gitignored).")
PY
