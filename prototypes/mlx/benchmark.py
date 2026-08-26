#!/usr/bin/env python3

import argparse
from array import array
from datetime import datetime, timezone
import json
import platform
from statistics import median
import time

import mlx.core as mx


THRESHOLD = 100
MULTIPLIER = 3
ADDEND = 7


def make_input(count: int) -> tuple[array, int]:
    values = array("i")
    state = 0x5EED1234
    expected = 0
    for _ in range(count):
        state = (state * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
        value = state % 2_001 - 1_000
        values.append(value)
        if value > THRESHOLD:
            expected += value * MULTIPLIER + ADDEND
    return values, expected


def compiled_workload(stream):
    def workload(values):
        selected = mx.greater(values, THRESHOLD, stream=stream)
        multiplied = mx.multiply(values, MULTIPLIER, stream=stream)
        projected = mx.add(multiplied, ADDEND, stream=stream)
        filtered = mx.where(selected, projected, 0, stream=stream)
        widened = mx.astype(filtered, mx.int64, stream=stream)
        return mx.sum(widened, stream=stream)

    return mx.compile(workload)


def evaluate(workload, values, expected: int, include_materialization: bool):
    started = time.perf_counter_ns()
    mlx_values = mx.array(values, dtype=mx.int32) if include_materialization else values
    result = workload(mlx_values)
    mx.eval(result)
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
    actual = result.item()
    if actual != expected:
        raise RuntimeError(f"MLX result {actual} did not match reference {expected}")
    return elapsed


def measure(workload, values, expected: int, warmups: int, runs: int,
            include_materialization: bool = False):
    for _ in range(warmups):
        evaluate(workload, values, expected, include_materialization)
    return [
        evaluate(workload, values, expected, include_materialization)
        for _ in range(runs)
    ]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="65536,262144,1048576,4194304,8388608")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    arguments.sizes = [int(value) for value in arguments.sizes.split(",")]
    if any(value <= 0 for value in arguments.sizes):
        parser.error("all sizes must be positive")
    if arguments.warmups < 0 or arguments.runs <= 0:
        parser.error("warmups must be non-negative and runs must be positive")
    return arguments


def main():
    arguments = parse_arguments()
    cpu_workload = compiled_workload(mx.cpu)
    gpu_workload = compiled_workload(mx.gpu)
    configurations = []

    for row_count in arguments.sizes:
        print(f"Benchmarking {row_count} rows with MLX {mx.__version__}")
        host_values, expected = make_input(row_count)
        mlx_values = mx.array(host_values, dtype=mx.int32)
        mx.eval(mlx_values)

        cpu_times = measure(
            cpu_workload, mlx_values, expected, arguments.warmups, arguments.runs)
        gpu_times = measure(
            gpu_workload, mlx_values, expected, arguments.warmups, arguments.runs)
        gpu_materialization_times = measure(
            gpu_workload, host_values, expected, arguments.warmups, arguments.runs, True)
        cpu_median = median(cpu_times)
        gpu_median = median(gpu_times)
        materialization_median = median(gpu_materialization_times)
        configurations.append({
            "rows": row_count,
            "expectedSum": expected,
            "mlxCpuSeconds": cpu_times,
            "mlxGpuSeconds": gpu_times,
            "mlxGpuMaterializationInclusiveSeconds": gpu_materialization_times,
            "mlxCpuMedianSeconds": cpu_median,
            "mlxGpuMedianSeconds": gpu_median,
            "mlxGpuMaterializationInclusiveMedianSeconds": materialization_median,
            "steadyStateGpuSpeedup": cpu_median / gpu_median,
            "materializationInclusiveGpuSpeedup": cpu_median / materialization_median,
        })

    report = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "mlxVersion": mx.__version__,
        "operatingSystem": platform.platform(),
        "defaultDevice": str(mx.default_device()),
        "threshold": THRESHOLD,
        "multiplier": MULTIPLIER,
        "addend": ADDEND,
        "warmups": arguments.warmups,
        "runs": arguments.runs,
        "configurations": configurations,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")
        print(f"Wrote {arguments.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
