# Direct Metal feasibility prototype

The first native experiment implements one fused, SQL-shaped operation over a
fixed-width integer column:

```text
SUM(IF(value > threshold, value * multiplier + addend, 0))
```

It is intentionally close to a Spark physical-plan fragment containing a filter,
projection, and global partial aggregation. The Metal kernel produces one signed
64-bit partial sum per threadgroup; the host performs the small final merge.

## Measurements

The benchmark reports three distinct times:

- CPU reference over a Swift integer array;
- GPU dispatch, synchronization, and partial-result merge with input already in a shared Metal buffer;
- the same GPU path including a complete input copy into that shared buffer.

Pipeline compilation, input generation, and warm-up are excluded from measured
runs. Exact CPU/GPU equality is checked on every observation. The copy-inclusive
number is the conservative feasibility measure until the Spark bridge establishes
a lower-copy ownership model.

## Run

```bash
scripts/run-metal-microbenchmark.sh
```

Override row counts or repetitions if needed:

```bash
scripts/run-metal-microbenchmark.sh --sizes 65536,1048576,8388608 --warmups 2 --runs 7
```

These are native feasibility measurements, not end-to-end Spark or TPC-DS results.

## Spark vertical slice

`SparkMetalExtensions` registers a Spark 4.2 `ColumnarRule`. Its current narrow
capability matcher recognizes an integer partial aggregate shaped as:

```sql
SUM(CASE WHEN value > constant THEN value * constant + constant ELSE 0 END)
```

The replacement `MetalFusedSumExec` consumes Spark `ColumnarBatch` input,
produces one 64-bit partial per input batch, and leaves final aggregation to
Spark. Unsupported expressions remain unchanged on the CPU. The current matcher
does not yet emit a fallback explanation.
