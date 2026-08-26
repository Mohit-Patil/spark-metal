# Proposed architecture

## End-to-end flow

```text
Spark SQL / DataFrame
        |
        v
Spark Catalyst physical plan
        |
        v
Apple capability and cost rules
        |
        +---------------- unsupported ----------------> Spark CPU operator
        |
        v supported
Apple columnar physical operator
        |
        v
Spark ColumnarBatch bridge
        |
        v
Apple columnar runtime
        |
        +-------------> Metal backend --------> Apple GPU
        |
        +-------------> MLX prototype backend -> Apple GPU
        |
        +-------------> Core ML experiment ----> CPU/GPU/ANE
```

## Components

### Spark plugin

The Scala layer will:

- register a Spark columnar rule;
- inspect physical operators, expressions, types, and SQL settings;
- tag unsupported nodes with reasons;
- replace supported plan regions with Apple columnar operators;
- insert CPU/accelerator transitions;
- expose Spark SQL metrics;
- preserve CPU fallback.

### Columnar representation

The native representation will be Arrow-like:

```text
Fixed-width column
├── data buffer
├── validity bitmap
├── data type
└── row count

Variable-width column
├── offsets buffer
├── data buffer
├── validity bitmap
├── data type
└── row count
```

The first version supports fixed-width columns only.

### Native bridge

A narrow JNI API will pass native handles rather than per-row values. Ownership and lifetime must be explicit so that JVM cleanup cannot invalidate a Metal command in flight.

The implemented bridge has two paths. Spark off-heap integer vectors expose a
native address. The membership-count path aligns that address down to the
enclosing macOS page, wraps the page range with `newBufferWithBytesNoCopy`, and
passes the original displacement as the Metal buffer offset. If Metal rejects a
mapping, the bridge copies that column and increments an explicit Spark SQL
metric. The fused-sum path uses a direct page-aligned wrap when possible and a
reusable shared buffer otherwise. Both paths carry an explicit null mask.

### Apple columnar runtime

The runtime will provide device-independent operation contracts:

```text
supports(operation, schema, sqlContext)
execute(batch, operationFragment)
synchronize(result)
release(handle)
```

Backend adapters can implement the same representative workload for comparison, but the Spark-facing semantics remain centralized.

### Metal backend

The Metal backend will own:

- device and command queue;
- compiled pipeline cache;
- shared/private buffer policy;
- reusable buffer pool;
- command-buffer submission;
- completion and error propagation;
- kernel metrics.

Initial kernels:

1. validity bitmap operations;
2. integer comparisons;
3. Boolean predicate fusion;
4. selection/gather;
5. scaled-integer arithmetic;
6. global reductions.

The implemented q96 slice recognizes a deliberately narrow physical-plan
region: three build-right broadcast inner equi-joins over distinct integer fact
columns followed by a partial `count(1)`. It collects the three filtered
dimension-key columns, scans the Parquet fact columns as `ColumnarBatch` input,
and performs all three membership tests plus the partial count in one Metal
dispatch per batch. This removes three row-oriented hash joins and their
`ColumnarToRow` transition from the fact-side hot path.

Two membership kernels preserve both speed and SQL join multiplicity:

- unique build keys use dense one-byte presence maps;
- duplicate build keys use dense four-byte multiplicity maps and multiply the
  three match counts.

Null fact keys never match. Empty key sets return zero. On-heap vectors and key
domains wider than 16,777,216 entries execute through an exact CPU fallback.
Dense membership maps, the null placeholder, and the partial-count buffer are
prepared once and reused for every columnar batch in a Spark partition.

## Numeric policy

- Begin with 32-bit and 64-bit integers.
- Represent limited decimals as scaled signed integers.
- Detect overflow and fall back when exact Spark semantics cannot be preserved.
- Do not initially accelerate Spark `DoubleType`, because Metal shader execution does not provide the same straightforward native FP64 path expected by Spark.
- Add floating-point operations only with explicit compatibility tests and documented behavior.

## Memory policy

The q96 slice now maps Spark's decoded off-heap column pages directly into Metal
shared buffers for the duration of each synchronous command. Its
`inputCopyFallbacks` metric records any batch that cannot take that path. The
operator still includes Parquet decoding, command encoding, synchronization,
and result materialization in end-to-end measurements.

Measured stages:

```text
Spark decode
  + batch materialization
  + native/Metal buffer preparation
  + GPU execution
  + synchronization
  + result materialization
  + remaining Spark execution
```

This is borrowed, synchronous access rather than ownership transfer: Spark keeps
the column batch alive until the Metal command completes, then closes it.

## Fallback policy

An operator remains on the CPU when any required expression, data type, semantic mode, memory condition, or backend feature is unsupported. The plugin should report a reason for every rejected region.

The current prototype always falls back when `spark.sql.ansi.enabled=true`.
Its first Metal kernel implements Spark's non-ANSI 32-bit integer wrapping, not
ANSI overflow exceptions.

The three-way broadcast-membership/count replacement currently requires
`spark.sql.adaptive.enabled=false`; with AQE enabled it remains on Spark's CPU
plan. This is a temporary capability boundary until build-side key extraction is
integrated with adaptive broadcast query stages.
