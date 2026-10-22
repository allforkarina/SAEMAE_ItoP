#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
cd -- "$project_root"

for script in "$project_root"/run/*.sh; do
  bash -n "$script"
done
"$python_bin" -m unittest discover -s tests -v
