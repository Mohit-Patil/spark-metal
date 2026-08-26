#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="${repo_root}/native/metal-prototype"
build_root="${repo_root}/build/metal-prototype"
result_root="${repo_root}/benchmark-results/metal-microbenchmark"

if ! xcrun -sdk macosx -find metal >/dev/null 2>&1; then
  echo "The Metal toolchain is missing. Install it with:" >&2
  echo "  xcodebuild -downloadComponent MetalToolchain" >&2
  exit 1
fi

mkdir -p "${build_root}" "${result_root}"
xcrun -sdk macosx metal \
  -c "${source_root}/kernels.metal" \
  -o "${build_root}/kernels.air"
xcrun -sdk macosx metallib \
  "${build_root}/kernels.air" \
  -o "${build_root}/kernels.metallib"
xcrun swiftc \
  -O \
  -framework Metal \
  "${source_root}/Benchmark.swift" \
  -o "${build_root}/metal-benchmark"

run_id="$(date -u '+%Y%m%dT%H%M%SZ')"
output="${METAL_BENCHMARK_OUTPUT:-${result_root}/${run_id}.json}"
"${build_root}/metal-benchmark" \
  "${build_root}/kernels.metallib" \
  --output "${output}" \
  "$@"

echo "Metal benchmark results written to ${output}."
