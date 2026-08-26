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
generator_dir="$(dirname "${generator}")"
output_link="${generator_dir}/.spark-metal-output"
if [[ -e "${output_link}" && ! -L "${output_link}" ]]; then
  echo "Refusing to replace non-symlink generator path: ${output_link}" >&2
  exit 1
fi
ln -sfn "${output_dir}" "${output_link}"
trap 'unlink "${output_link}"' EXIT
(
  cd "${generator_dir}"
  ./dsdgen \
    -SCALE "${scale_factor}" \
    -DIR .spark-metal-output \
    -FORCE Y
)

echo "Generated TPC-DS scale factor ${scale_factor} source data under ${output_dir}."
