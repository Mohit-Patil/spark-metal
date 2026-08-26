#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=project-env.sh
source "${script_dir}/project-env.sh"

example="${SPARK_HOME}/examples/src/main/python/pi.py"
if [[ ! -f "${example}" ]]; then
  echo "Spark Python example not found at ${example}" >&2
  exit 1
fi

spark-submit \
  --master 'local[2]' \
  --conf spark.ui.enabled=false \
  "${example}" 4
