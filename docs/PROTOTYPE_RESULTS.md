# Prototype results

Captured on the first Apple M5 host on 2026-08-26. These results are feasibility
measurements, not TPC-DS results and not comparable to official TPC benchmark
results.

## Native fused operation

Workload:

```text
SUM(IF(value > 100, value * 3 + 7, 0))
```

Each configuration used two warm-ups and seven measured observations. The GPU
time below includes an entire input copy into Metal shared memory, command
encoding, synchronization, and final partial-sum merge.

| Rows | CPU median | Metal copy-inclusive median | Speedup |
|---:|---:|---:|---:|
| 65,536 | 0.159 ms | 0.295 ms | 0.54x |
| 262,144 | 0.653 ms | 0.337 ms | 1.94x |
| 1,048,576 | 2.935 ms | 0.684 ms | 4.29x |
| 4,194,304 | 10.731 ms | 1.292 ms | 8.31x |
| 8,388,608 | 20.759 ms | 2.277 ms | 9.12x |

The crossover for this standalone workload lies between 65,536 and 262,144 rows.

This speedup is relative to the simple scalar-style Swift CPU reference in this
prototype. The MLX results below show that a compiled/vectorized CPU
implementation is substantially stronger, so these native numbers establish
Metal feasibility but are not the project's final CPU comparison.

## MLX CPU/GPU comparison

MLX 0.32.1 executes the same input and exact integer expression using a compiled
graph. Each configuration used five warm-ups and 25 observations.

| Rows | MLX CPU | MLX GPU, input ready | MLX GPU, materialization included | Ready GPU speedup |
|---:|---:|---:|---:|---:|
| 65,536 | 0.028 ms | 0.566 ms | 0.262 ms | 0.05x |
| 262,144 | 0.070 ms | 0.301 ms | 0.302 ms | 0.23x |
| 1,048,576 | 0.238 ms | 0.445 ms | 0.422 ms | 0.54x |
| 4,194,304 | 0.999 ms | 1.099 ms | 1.367 ms | 0.91x |
| 8,388,608 | 1.923 ms | 1.978 ms | 2.204 ms | 0.97x |

There is no MLX GPU crossover in the tested range. Materialization timings can
occasionally be lower than input-ready timings because the observations are
independent and Apple GPU scheduling is noisy; the medians should not be
algebraically combined.

## Core ML capability result

A Core ML 9.0 ML Program can represent the `int32` form of the expression, and
its 4,194,304-row compute plan supports CPU and GPU and prefers GPU. None of the
operators supports the Neural Engine. Core ML cannot cast the intermediate to
`int64`, so the graph returns -1,166,372,896 instead of the required
3,128,594,400. It is therefore rejected as a Spark SQL backend for this vertical
slice regardless of its execution time.

## Spark SQL synthetic comparison

The Spark test reads 33,554,432 integers from the same Parquet dataset in two
separate local Spark processes. Both use eight local cores, off-heap column
vectors, one-million-row reader batches, five warm-ups, and eleven measured
runs. The only execution difference is the registered Metal columnar rule.

| Configuration | Median end-to-end time | Result |
|---|---:|---:|
| Vanilla Spark CPU | 76.298 ms | 1,688,850,044,797,455 |
| Spark Metal | 80.599 ms | 1,688,850,044,797,455 |

Observed end-to-end speedup: **0.95x**. The Metal physical operator and exact
result are verified, but this is not yet a performance win. This negative result
is retained because it shows that a fast kernel alone does not overcome Parquet
decode, column ownership, synchronization, scheduling, and final Spark work.

## Correctness evidence

- Native CPU and Metal results match for non-power-of-two input sizes.
- JNI CPU and Metal results match with every seventeenth value marked null.
- Spark physical plans contain `MetalFusedSum` after transition insertion.
- Spark SQL results match independent references for both nullable and non-null
  one-million-row inputs.
- Spark ANSI mode is verified to leave the expression on the CPU.
- MLX CPU and GPU results match the independent 64-bit reference.
- The Core ML probe deliberately demonstrates and records its 32-bit semantic
  mismatch instead of treating an overflowing result as valid.

## Remaining success gate

The project goal still requires an unmodified TPC-DS scale-factor-10 query with a
correct result and at least a 10% median end-to-end improvement over vanilla
Spark. No current result satisfies that gate.
