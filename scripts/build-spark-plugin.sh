#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=project-env.sh
source "${script_dir}/project-env.sh"

native_root="${repo_root}/native"
plugin_root="${repo_root}/jvm/spark-plugin/src/main"
build_root="${repo_root}/build/spark-plugin"
classes="${build_root}/classes"
mkdir -p "${classes}"

xcrun -sdk macosx metal \
  -c "${native_root}/metal-prototype/kernels.metal" \
  -o "${build_root}/kernels.air"
xcrun -sdk macosx metallib \
  "${build_root}/kernels.air" \
  -o "${build_root}/kernels.metallib"
xcrun clang++ \
  -std=c++17 \
  -fobjc-arc \
  -dynamiclib \
  -framework Foundation \
  -framework Metal \
  -I "${JAVA_HOME}/include" \
  -I "${JAVA_HOME}/include/darwin" \
  "${native_root}/jni/SparkMetalBridge.mm" \
  "${native_root}/jni/ParquetPageRuns.cpp" \
  -o "${build_root}/libsparkmetal.dylib"

javac \
  -proc:none \
  -cp "${SPARK_HOME}/jars/*" \
  -d "${classes}" \
  "${plugin_root}/java/io/github/mohitpatil/sparkmetal/NativeBridge.java" \
  "${plugin_root}/java/io/github/mohitpatil/sparkmetal/OffHeapColumnVectorAccess.java"

java -cp "${SPARK_HOME}/jars/*" scala.tools.nsc.Main \
  -classpath "${SPARK_HOME}/jars/*:${classes}" \
  -d "${classes}" \
  "${plugin_root}/scala/io/github/mohitpatil/sparkmetal/SparkMetalNative.scala" \
  "${plugin_root}/scala/io/github/mohitpatil/sparkmetal/ParquetEligibility.scala" \
  "${plugin_root}/scala/io/github/mohitpatil/sparkmetal/SharedBufferPool.scala" \
  "${plugin_root}/scala/io/github/mohitpatil/sparkmetal/MetalFusedSumExec.scala" \
  "${plugin_root}/scala/io/github/mohitpatil/sparkmetal/MetalFusedMembershipCountExec.scala" \
  "${plugin_root}/scala/io/github/mohitpatil/sparkmetal/SparkMetalColumnarRule.scala" \
  "${plugin_root}/scala/io/github/mohitpatil/sparkmetal/SparkMetalExtensions.scala" \
  "${plugin_root}/scala/io/github/mohitpatil/sparkmetal/PluginSmoke.scala" \
  "${plugin_root}/scala/io/github/mohitpatil/sparkmetal/AnsiFallbackSmoke.scala" \
  "${plugin_root}/scala/io/github/mohitpatil/sparkmetal/SyntheticBenchmark.scala" \
  "${plugin_root}/scala/io/github/mohitpatil/sparkmetal/Q96SyntheticBenchmark.scala" \
  "${plugin_root}/scala/io/github/mohitpatil/sparkmetal/ParquetDecodeSmoke.scala"

jar --create \
  --file "${build_root}/spark-metal-plugin.jar" \
  -C "${classes}" .

echo "Built ${build_root}/spark-metal-plugin.jar"
