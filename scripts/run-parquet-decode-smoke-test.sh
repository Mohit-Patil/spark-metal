#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=project-env.sh
source "${script_dir}/project-env.sh"

"${script_dir}/build-spark-plugin.sh"
build_root="${repo_root}/build/spark-plugin"

spark-submit \
  --master 'local[8]' \
  --driver-memory 6g \
  --conf spark.ui.enabled=false \
  --conf spark.sql.adaptive.enabled=false \
  --class io.github.mohitpatil.sparkmetal.ParquetDecodeSmoke \
  "${build_root}/spark-metal-plugin.jar" \
  "${build_root}/libsparkmetal.dylib" \
  "${build_root}/kernels.metallib" \
  "${1:-100003}"

spark-submit \
  --master 'local[8]' \
  --driver-memory 6g \
  --conf spark.ui.enabled=false \
  --conf spark.sql.adaptive.enabled=false \
  --class io.github.mohitpatil.sparkmetal.ParquetDecodeSmoke \
  "${build_root}/spark-metal-plugin.jar" \
  exec \
  "${build_root}/libsparkmetal.dylib" \
  "${build_root}/kernels.metallib" \
  "${1:-100003}"

spark-submit \
  --master 'local[8]' \
  --driver-memory 6g \
  --conf spark.ui.enabled=false \
  --conf spark.sql.adaptive.enabled=false \
  --class io.github.mohitpatil.sparkmetal.ParquetDecodeSmoke \
  "${build_root}/spark-metal-plugin.jar" \
  measure \
  "${build_root}/libsparkmetal.dylib" \
  "${build_root}/kernels.metallib" \
  "${1:-100003}"
