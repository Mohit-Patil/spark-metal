# GPU Parquet decoder spec (q96 fact-side scan)

## Goal

Replace the CPU Parquet decode of the three q96 fact-key columns with a Metal
GPU decode fused into the existing membership pipeline, targeting a 2x median
end-to-end q96 speedup over vanilla Spark at TPC-DS scale factor 10 (current
state: 1.52x with the CPU scan retained).

## Why this is the remaining lever

After streamed submission and dictionary-aware membership, `metalTime` is ~70ms
summed across eight tasks and fully overlapped; the ~135ms Metal-config median
is dominated by Spark's vectorized Parquet reader: RLE/bit-packed decode, null
expansion, and off-heap vector writes for ~86.4M values. IO and Snappy are
negligible (all three column chunks total ~40MB compressed on disk).

## Verified data-format facts (probe of SF10 `store_sales`, 2026-08-26)

- 30 files, one row group per file, ~978k rows per row group.
- Data pages are **V1**, ~20,000 values each, Snappy-compressed.
- Value encoding is **PLAIN_DICTIONARY** (1-byte bit width + RLE/bit-packed
  hybrid of dictionary ids); dictionary pages are PLAIN little-endian int32.
- Definition levels: RLE hybrid, `maxDef = 1`; `maxRep = 0` (no repetition
  bytes are present in V1 pages when maxRep is 0).
- Decompressed V1 page layout here: `[defLen: 4-byte LE][def RLE-hybrid:
  defLen bytes][1-byte value bit width][value RLE-hybrid to end of page]`.

## Architecture

Planner: `SparkMetalColumnarRule` gains a second, preferred replacement. When
the q96 region's fact child is a `FileSourceScanExec` over local Parquet whose
output is exactly the three int32 join keys, with no partition/bucket columns
and only `IsNotNull` data filters on those keys (subsumed by inner-join null
semantics), and a footer scan of every input file shows only supported
encodings, the whole region (scan + three joins + partial count) is replaced by
`MetalParquetMembershipCountExec`. Otherwise the existing
`MetalFusedMembershipCountExec` replacement (1.52x path) still applies.
Controlled by `spark.metal.parquetScan.enabled` (default true).

Execution, per (file, rowGroup) split:

1. **JVM** opens the file with parquet-mr (`ParquetFileReader`, already on
   Spark's classpath), which handles footer parsing, IO, and Snappy. The
   dictionary page of each column chunk is decoded to `int[]` and turned into
   the per-dictionary membership table (presence bytes for unique build keys,
   int multiplicities otherwise) using the shared table builders.
2. **Native CPU** parses each decompressed data page's two RLE/bit-packed
   hybrid streams into flat descriptor arrays — O(#runs), no value expansion:
   - value work items (≤256 values each): `{valueStart, count, kind,
     payload}` where kind is RLE (payload = id) or BITPACKED (payload = bit
     offset into the page's value bytes),
   - row segments from def levels: `{rowStart, valueStart, count, valid}` —
     contiguous all-valid or all-null row ranges with running value ordinals.
3. **GPU** (encoded asynchronously into the existing `MembershipStream`):
   - `expand_value_runs` writes each page's dictionary ids into a dense
     value-ordinal buffer (thread-per-value over work items),
   - `scatter_segments` writes validity bytes for every row and gathers ids
     from value space into row space (skipped when the page has no nulls: the
     value buffer region is targeted directly at the page's row offset),
   - once all pages of all three chunks are encoded, the **unchanged**
     membership kernels run over the row-group-aligned ids/validity buffers
     with the per-chunk dictionary tables (dense map, min 0, span dictSize).
4. One `waitUntilCompleted` per partition, exactly like the current stream.

## Fallbacks and safety

- Planning-time footer eligibility per file: value encodings within
  {RLE, BIT_PACKED, PLAIN_DICTIONARY, RLE_DICTIONARY}, codec in
  {SNAPPY, UNCOMPRESSED}, INT32, maxRep = 0, maxDef ≤ 1. Ineligible input
  keeps the current 1.52x operator.
- Runtime guard per row group: V2 pages, missing dictionary page, or value
  bit width > 24 fall back to a CPU decode of that row group via parquet-mr's
  `ColumnReader` feeding the existing `countOnCpu` semantics.
- All existing fallbacks (AQE, ANSI, non-dense domains) are unchanged.
- Correctness gates: the duplicate/null edge dataset must produce identical
  counts through the new scan; SF10 q96 must produce an exact result-hash,
  row-count, and schema match against vanilla Spark.

## Metrics

`numRowGroups`, `numPagesDecoded`, `cpuFallbackRowGroups`, `decodeParseTime`
(CPU run parsing), `metalTime` (submit + final wait), plus the existing
membership metrics.

## Non-goals

GPU Snappy decompression; encodings beyond the verified set; queries other
than the q96 region shape; distributed (non-local) execution.
