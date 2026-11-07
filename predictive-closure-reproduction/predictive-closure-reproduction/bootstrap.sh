#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${1:-$ROOT/upstream}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
PIN=2964fa8f737fa43871d47928edc9aac16fa9298f
NEW_CHECKOUT=0
if [[ ! -e "$UPSTREAM_DIR" ]]; then
    git clone https://github.com/Mosh0110/predictive-closure.git "$UPSTREAM_DIR"
    NEW_CHECKOUT=1
fi
if [[ "$NEW_CHECKOUT" == 1 ]]; then
    git -C "$UPSTREAM_DIR" checkout --detach "$PIN"
fi
[[ "$(git -C "$UPSTREAM_DIR" rev-parse HEAD)" == "$PIN" ]] || {
    echo "Existing upstream checkout has another revision; refusing to change it." >&2
    exit 1
}
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
if [[ ! -d "$ROOT/.venv" ]]; then
    "$PYTHON_BIN" -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -m pip install torch==2.11.0 --index-url "$TORCH_INDEX"
"$ROOT/.venv/bin/python" -m pip install -r "$UPSTREAM_DIR/requirements.txt"
"$ROOT/.venv/bin/python" -m pip freeze > "$ROOT/.venv/installed.txt"
echo "Isolated environment ready; no earlier study environment was changed."
