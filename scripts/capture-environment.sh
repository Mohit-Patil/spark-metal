#!/usr/bin/env bash

set -euo pipefail

if command -v brew >/dev/null 2>&1; then
  if brew --prefix openjdk@21 >/dev/null 2>&1; then
    capture_jdk_prefix="$(brew --prefix openjdk@21)"
    export JAVA_HOME="${capture_jdk_prefix}/libexec/openjdk.jdk/Contents/Home"
    export PATH="${JAVA_HOME}/bin:${PATH}"
  fi
  if brew --prefix apache-spark >/dev/null 2>&1; then
    capture_spark_prefix="$(brew --prefix apache-spark)"
    export SPARK_HOME="${capture_spark_prefix}/libexec"
    export PATH="${SPARK_HOME}/bin:${PATH}"
  fi
fi

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
