#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
mkdir -p "${repo_root}/build/native-tests"
xcrun clang++ -std=c++17 -O2 \
  "${repo_root}/native/jni/ParquetPageRuns.cpp" \
  "${repo_root}/native/tests/parquet_page_runs_test.cpp" \
  -o "${repo_root}/build/native-tests/parquet_page_runs_test"
"${repo_root}/build/native-tests/parquet_page_runs_test"
