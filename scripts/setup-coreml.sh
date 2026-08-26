#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
venv_root="${repo_root}/.tools/coreml-venv-py313"
python_bin="${PYTHON_BIN:-python3.13}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "${python_bin} is required. Set PYTHON_BIN to a compatible Python 3.13 executable." >&2
  exit 1
fi

"${python_bin}" -m venv "${venv_root}"
"${venv_root}/bin/python" -m pip install --upgrade pip
"${venv_root}/bin/python" -m pip install 'coremltools==9.0'

echo "Installed coremltools 9.0 in ${venv_root}"
