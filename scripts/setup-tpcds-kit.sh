#!/usr/bin/env bash

set -euo pipefail

tpcds_revision="1b7fb7529edae091684201fab142d956d6afd881"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kit_root="${repo_root}/.tools/tpcds-kit"

if [[ "${TPCDS_EULA_ACCEPTED:-}" != "yes" ]]; then
  cat >&2 <<'MESSAGE'
The TPC-DS kit has its own end-user licence agreement. This project cannot
accept it for you. Review:

  https://github.com/databricks/tpcds-kit/blob/master/EULA.txt

If you accept those terms, rerun with:

  TPCDS_EULA_ACCEPTED=yes scripts/setup-tpcds-kit.sh
MESSAGE
  exit 2
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This build recipe currently supports ARM64 macOS hosts only." >&2
  exit 1
fi

if [[ ! -d "${kit_root}/.git" ]]; then
  mkdir -p "$(dirname "${kit_root}")"
  git clone https://github.com/databricks/tpcds-kit.git "${kit_root}"
fi

git -C "${kit_root}" fetch --depth 1 origin "${tpcds_revision}"
git -C "${kit_root}" checkout --detach "${tpcds_revision}"

make -C "${kit_root}/tools" \
  OS=MACOS \
  MACOS_CFLAGS='-O3 -Wall -Wno-error=implicit-int -Wno-deprecated-non-prototype'

(
  cd "${kit_root}/tools"
  ./dsdgen -help >/dev/null
)
echo "TPC-DS generator built at ${kit_root}/tools/dsdgen."
