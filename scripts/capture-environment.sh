#!/usr/bin/env bash

set -euo pipefail

system_profiler SPHardwareDataType
sw_vers
uname -m

if command -v java >/dev/null 2>&1; then
  java -version
else
  echo "Java: not found"
fi

if command -v python3 >/dev/null 2>&1; then
  python3 --version
else
  echo "Python: not found"
fi

if command -v spark-submit >/dev/null 2>&1; then
  spark-submit --version
else
  echo "Apache Spark: not found"
fi

if command -v xcodebuild >/dev/null 2>&1; then
  xcodebuild -version
else
  echo "Xcode: not found"
fi

if command -v xcrun >/dev/null 2>&1; then
  xcrun -find metal || true
else
  echo "Metal compiler: xcrun not found"
fi

