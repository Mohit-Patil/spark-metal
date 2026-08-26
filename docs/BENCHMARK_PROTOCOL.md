# TPC-DS benchmark protocol

## Dataset

- Benchmark: TPC-DS-derived workload for engineering comparison.
- Scale factor: 10, approximately 10 GB of generated source data before format-dependent compression.
- Primary storage: Parquet.
- Dataset generation command, schema, partitioning, and checksums must be recorded.
- Generated data is excluded from Git.

## Compared configurations

1. Vanilla local Spark CPU.
2. Native CPU reference where useful, such as Apache DataFusion Comet.
3. Spark with the experimental Apple accelerator enabled.

Standalone MLX, Core ML, MPSGraph, and Metal microbenchmarks are research controls and must not be presented as end-to-end Spark results.

## Controlled variables

- identical Spark, Scala, Java, and Python versions;
- identical query text and Parquet files;
- fixed local core count and Spark memory settings;
- fixed shuffle partitions;
- no unrelated foreground workload;
- power connected when possible;
- recorded macOS version and power mode;
- recorded thermal state before each measured group;
- identical correctness validation.

Because CPU and GPU share 16 GB on the first host, Spark heap, off-heap buffers, GPU buffers, filesystem cache, and the operating system compete for the same physical memory. Memory-pressure and swap metrics are mandatory.

## Run procedure

For each query and configuration:

1. Capture the executed physical plan.
2. Perform one unmeasured warm-up run.
3. Perform at least five measured runs.
4. Record wall-clock query completion time.
5. Record Spark stage and task metrics.
6. Record accelerator preparation, dispatch, synchronization, and fallback metrics.
7. Validate the output against vanilla Spark.
8. Report the median and all individual observations.

Cold filesystem-cache experiments must be reported separately from warm-cache experiments. They must not be mixed into one median.

## Primary metric

```text
end_to_end_speedup = median_vanilla_spark / median_accelerated_spark
```

The primary result includes all costs visible to the query caller.

## Secondary metrics

- accelerated region time;
- input batch preparation time;
- GPU command execution time;
- synchronization time;
- output materialization time;
- rows and bytes processed;
- accelerated versus fallback operator count;
- peak resident memory and swap;
- Spark shuffle read/write bytes;
- result checksum.

## Correctness

- Compare schemas exactly.
- For order-sensitive queries, compare ordered rows.
- For order-insensitive queries, canonicalize rows before hashing.
- Exact integer, decimal, date, timestamp, string, and null values must match.
- Approximate floating-point comparison is permitted only when Spark itself does not guarantee deterministic bitwise aggregation and the tolerance is declared before the run.

## Success threshold

At least one unmodified TPC-DS query must:

- execute a documented physical-plan region on the integrated Apple GPU;
- produce a correct result;
- improve median end-to-end runtime by at least 10%;
- sustain the result across at least five measured post-warm-up runs.

