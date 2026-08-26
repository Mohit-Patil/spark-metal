#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
generator="${repo_root}/.tools/tpcds-kit/tools/dsdgen"
output_dir="${TPCDS_RAW_DIR:-${repo_root}/benchmark-data/tpcds-sf10-raw}"
scale_factor="${TPCDS_SCALE_FACTOR:-10}"

if [[ ! -x "${generator}" ]]; then
  echo "TPC-DS generator is unavailable. Review its licence and run scripts/setup-tpcds-kit.sh." >&2
  exit 1
fi

if find "${output_dir}" -maxdepth 1 -name '*.dat' -print -quit 2>/dev/null | grep -q .; then
  echo "TPC-DS data already exists under ${output_dir}; refusing to mix generations." >&2
  exit 1
fi

mkdir -p "${output_dir}"
(
  cd "$(dirname "${generator}")"
  ./dsdgen \
    -SCALE "${scale_factor}" \
    -DIR "${output_dir}" \
    -FORCE Y
)

echo "Generated TPC-DS scale factor ${scale_factor} source data under ${output_dir}."
