#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=project-env.sh
source "${script_dir}/project-env.sh"

native_root="${repo_root}/native"
java_main_root="${repo_root}/jvm/spark-plugin/src/main/java"
java_test_root="${repo_root}/jvm/spark-plugin/src/test/java"
build_root="${repo_root}/build/jni-smoke"
mkdir -p "${build_root}/classes"

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
  -o "${build_root}/libsparkmetal.dylib"

javac \
  -proc:none \
  -d "${build_root}/classes" \
  "${java_main_root}/io/github/mohitpatil/sparkmetal/NativeBridge.java" \
  "${java_test_root}/io/github/mohitpatil/sparkmetal/NativeBridgeSmoke.java"

java \
  -cp "${build_root}/classes" \
  io.github.mohitpatil.sparkmetal.NativeBridgeSmoke \
  "${build_root}/libsparkmetal.dylib" \
  "${build_root}/kernels.metallib" \
  "${1:-1000003}"
