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
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.ansi.enabled=false \
  --conf spark.sql.extensions=io.github.mohitpatil.sparkmetal.SparkMetalExtensions \
  --conf "spark.metal.nativeLibrary=${build_root}/libsparkmetal.dylib" \
  --conf "spark.metal.metalLibrary=${build_root}/kernels.metallib" \
  --class io.github.mohitpatil.sparkmetal.PluginSmoke \
  "${build_root}/spark-metal-plugin.jar" \
  "${1:-1000003}"

spark-submit \
  --master 'local[8]' \
  --driver-memory 6g \
  --conf spark.ui.enabled=false \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.ansi.enabled=true \
  --conf spark.sql.extensions=io.github.mohitpatil.sparkmetal.SparkMetalExtensions \
  --conf "spark.metal.nativeLibrary=${build_root}/libsparkmetal.dylib" \
  --conf "spark.metal.metalLibrary=${build_root}/kernels.metallib" \
  --class io.github.mohitpatil.sparkmetal.AnsiFallbackSmoke \
  "${build_root}/spark-metal-plugin.jar" \
  "${2:-100003}"
