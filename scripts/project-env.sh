#!/usr/bin/env bash

# Source this file from the repository root before running project scripts.

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required. Install it from https://brew.sh and retry." >&2
  return 1 2>/dev/null || exit 1
fi

spark_metal_jdk_prefix="$(brew --prefix openjdk@21 2>/dev/null)" || {
  echo "openjdk@21 is missing. Run scripts/setup-macos.sh first." >&2
  return 1 2>/dev/null || exit 1
}

spark_metal_spark_prefix="$(brew --prefix apache-spark 2>/dev/null)" || {
  echo "apache-spark is missing. Run scripts/setup-macos.sh first." >&2
  return 1 2>/dev/null || exit 1
}

export JAVA_HOME="${spark_metal_jdk_prefix}/libexec/openjdk.jdk/Contents/Home"
export SPARK_HOME="${spark_metal_spark_prefix}/libexec"
export PATH="${JAVA_HOME}/bin:${SPARK_HOME}/bin:${PATH}"
export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-127.0.0.1}"

unset spark_metal_jdk_prefix
unset spark_metal_spark_prefix
