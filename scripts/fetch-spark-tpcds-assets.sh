#!/usr/bin/env bash

set -euo pipefail

spark_revision="32f7299601108917fb01920a54e084595b7b3bf8"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_root="${repo_root}/.tools/spark-assets"

if [[ -d "${asset_root}/.git" ]]; then
  current_revision="$(git -C "${asset_root}" rev-parse HEAD)"
  if [[ "${current_revision}" == "${spark_revision}" ]]; then
    echo "Spark TPC-DS assets already match ${spark_revision}."
    exit 0
  fi
  echo "Existing asset checkout has an unexpected revision: ${current_revision}" >&2
  echo "Move ${asset_root} aside and rerun this script." >&2
  exit 1
fi

mkdir -p "$(dirname "${asset_root}")"
git clone \
  --filter=blob:none \
  --no-checkout \
  https://github.com/apache/spark.git \
  "${asset_root}"
git -C "${asset_root}" sparse-checkout init --cone
git -C "${asset_root}" sparse-checkout set sql/core/src/test/resources/tpcds
git -C "${asset_root}" fetch --depth 1 origin "${spark_revision}"
git -C "${asset_root}" checkout --detach FETCH_HEAD

echo "Queries are available under ${asset_root}/sql/core/src/test/resources/tpcds."
