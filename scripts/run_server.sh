#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8360}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-server}"
REQUIREMENTS="$ROOT_DIR/requirements.txt"

usage() {
  cat <<'EOF'
Usage: scripts/run_server.sh [uvicorn args...]

Starts bir-mov in one command:
  - creates .venv-server with Python 3.12/3.11 when needed
  - installs requirements when requirements.txt changes
  - starts app.main:app with uvicorn

Environment overrides:
  HOST=0.0.0.0 PORT=8361 scripts/run_server.sh
  PYTHON_BIN=/opt/homebrew/bin/python3.12 scripts/run_server.sh
  VENV_DIR=.venv-server scripts/run_server.sh

Any extra args are passed to uvicorn, for example:
  scripts/run_server.sh --reload
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

python_minor() {
  "$1" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
}

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
      echo "PYTHON_BIN must be Python 3.10, 3.11, or 3.12 for the ML stack." >&2
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

hash_requirements() {
  shasum -a 256 "$REQUIREMENTS" | awk '{print $1}'
}

ensure_movielens_files() {
  local missing=()
  [[ -f "$ROOT_DIR/ml-32m/movies.csv" ]] || missing+=("ml-32m/movies.csv")
  [[ -f "$ROOT_DIR/ml-32m/links.csv" ]] || missing+=("ml-32m/links.csv")
  if (( ${#missing[@]} > 0 )); then
    printf 'Missing required MovieLens file: %s\n' "${missing[@]}" >&2
    echo "Download MovieLens ml-32m and unzip it into ./ml-32m before starting." >&2
    exit 1
  fi
}

ensure_venv() {
  local python_bin="$1"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating virtualenv: $VENV_DIR ($(python_minor "$python_bin"))"
    "$python_bin" -m venv "$VENV_DIR"
  fi

  if ! is_supported_python "$VENV_DIR/bin/python"; then
    echo "Existing venv uses Python $(python_minor "$VENV_DIR/bin/python"), which is not supported by the ML stack." >&2
    echo "Set VENV_DIR to a new path or remove $VENV_DIR and run this script again." >&2
    exit 1
  fi
}

install_requirements_if_needed() {
  local stamp="$VENV_DIR/.requirements.sha256"
  local current_hash
  current_hash="$(hash_requirements)"

  if [[ ! -f "$stamp" || "$(cat "$stamp")" != "$current_hash" ]]; then
    echo "Installing Python requirements..."
    "$VENV_DIR/bin/python" -m pip install --upgrade pip
    "$VENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS"
    echo "$current_hash" > "$stamp"
  fi
}

verify_runtime_imports() {
  "$VENV_DIR/bin/python" - <<'PY'
import importlib

for module in ("fastapi", "uvicorn", "pandas", "numpy", "torch", "implicit"):
    importlib.import_module(module)
PY
}

ensure_movielens_files
PYTHON_BIN_RESOLVED="$(pick_python)"
ensure_venv "$PYTHON_BIN_RESOLVED"
install_requirements_if_needed
verify_runtime_imports

echo "Starting bir-mov: http://$HOST:$PORT"
echo "IMDb ratings cache: data/title.ratings.tsv.gz (refreshed on every startup)"
exec "$VENV_DIR/bin/python" -m uvicorn app.main:app --host "$HOST" --port "$PORT" "$@"
