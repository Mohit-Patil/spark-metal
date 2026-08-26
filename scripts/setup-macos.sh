#!/usr/bin/env bash

set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This setup currently supports ARM64 macOS hosts only." >&2
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required. Install it from https://brew.sh and retry." >&2
  exit 1
fi

brew install openjdk@21 apache-spark

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=project-env.sh
source "${script_dir}/project-env.sh"

java -version
spark-submit --version
