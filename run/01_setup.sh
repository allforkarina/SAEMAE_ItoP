#!/usr/bin/env bash
set -euo pipefail

# Activate my_py310 before invoking this script, or set PYTHON_BIN explicitly.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

"$python_bin" -m pip install -r "$project_root/requirements-target-statistics.txt"
"$python_bin" -m pip check
