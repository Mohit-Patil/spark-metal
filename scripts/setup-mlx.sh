#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
venv_root="${repo_root}/.tools/mlx-venv"

python3 -m venv "${venv_root}"
"${venv_root}/bin/python" -m pip install --upgrade pip
"${venv_root}/bin/python" -m pip install 'mlx==0.32.1'

echo "Installed MLX 0.32.1 in ${venv_root}"
