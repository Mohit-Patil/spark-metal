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

## Numeric policy

- Begin with 32-bit and 64-bit integers.
- Represent limited decimals as scaled signed integers.
- Detect overflow and fall back when exact Spark semantics cannot be preserved.
- Do not initially accelerate Spark `DoubleType`, because Metal shader execution does not provide the same straightforward native FP64 path expected by Spark.
- Add floating-point operations only with explicit compatibility tests and documented behavior.

## Memory policy

The first implementation uses one explicit copy from Spark's decoded columnar input into reusable Metal shared buffers. This gives a correct baseline before attempting zero-copy ownership.

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

Zero-copy is an optimization milestone, not an assumption.

## Fallback policy

An operator remains on the CPU when any required expression, data type, semantic mode, memory condition, or backend feature is unsupported. The plugin should report a reason for every rejected region.

