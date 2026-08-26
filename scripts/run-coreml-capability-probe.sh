#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_path="${repo_root}/.tools/coreml-venv-py313/bin/python"

if [[ ! -x "${python_path}" ]]; then
  echo "Core ML environment not found; run scripts/setup-coreml.sh first." >&2
  exit 1
fi

"${python_path}" "${repo_root}/prototypes/coreml/capability_probe.py" "$@"
