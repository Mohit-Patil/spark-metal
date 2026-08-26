#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_path="${repo_root}/.tools/mlx-venv/bin/python"

if [[ ! -x "${python_path}" ]]; then
  echo "MLX environment not found; run scripts/setup-mlx.sh first." >&2
  exit 1
fi

"${python_path}" "${repo_root}/prototypes/mlx/benchmark.py" "$@"
