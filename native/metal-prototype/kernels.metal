#include <metal_stdlib>
using namespace metal;

struct FusedParameters {
    uint count;
    int threshold;
    int multiplier;
    int addend;
    uint has_nulls;
};

struct MembershipCountParameters {
    uint count;
    uint null_mask;
    int key_min_0;
    int key_min_1;
    int key_min_2;
    uint key_span_0;
    uint key_span_1;
    uint key_span_2;
};

inline bool dense_contains(device const uchar *keys, int minimum, uint span, int value)
{
    long offset = long(value) - long(minimum);
    return offset >= 0 && ulong(offset) < ulong(span) && keys[offset] != 0;
}

inline uint dense_multiplicity(device const uint *keys, int minimum, uint span, int value)
{
    long offset = long(value) - long(minimum);
    return offset >= 0 && ulong(offset) < ulong(span) ? keys[offset] : 0u;
}

kernel void fused_filter_project_sum(
    device const int *input [[buffer(0)]],
    device const uchar *validity [[buffer(1)]],
    device long *partial_sums [[buffer(2)]],
    constant FusedParameters &parameters [[buffer(3)]],
    uint global_index [[thread_position_in_grid]],
    uint local_index [[thread_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]])
{
    threadgroup long scratch[256];

    long contribution = 0;
    if (global_index < parameters.count &&
        (parameters.has_nulls == 0 || validity[global_index] == 0)) {
        int value = input[global_index];
        if (value > parameters.threshold) {
            int projected = value * parameters.multiplier + parameters.addend;
            contribution = long(projected);
        }
    }
    scratch[local_index] = contribution;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_index < stride) {
            scratch[local_index] += scratch[local_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_index == 0) {
        partial_sums[group_index] = scratch[0];
    }
}

kernel void fused_membership_count_3_unique(
    device const int *input_0 [[buffer(0)]],
    device const int *input_1 [[buffer(1)]],
    device const int *input_2 [[buffer(2)]],
    device const uchar *nulls_0 [[buffer(3)]],
    device const uchar *nulls_1 [[buffer(4)]],
    device const uchar *nulls_2 [[buffer(5)]],
    device const uchar *keys_0 [[buffer(6)]],
    device const uchar *keys_1 [[buffer(7)]],
    device const uchar *keys_2 [[buffer(8)]],
    device uint *partial_counts [[buffer(9)]],
    constant MembershipCountParameters &parameters [[buffer(10)]],
    uint global_index [[thread_position_in_grid]],
    uint simd_index [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]])
{
    // Apple GPUs execute 32-wide SIMD groups. Reduce inside each SIMD group,
    // then reduce the eight SIMD totals, avoiding the eight barriers used by
    // a conventional 256-thread tree reduction.
    threadgroup uint simd_totals[8];
    uint contribution = 0;
    if (global_index < parameters.count) {
        bool valid =
            ((parameters.null_mask & 1u) == 0u || nulls_0[global_index] == 0) &&
            ((parameters.null_mask & 2u) == 0u || nulls_1[global_index] == 0) &&
            ((parameters.null_mask & 4u) == 0u || nulls_2[global_index] == 0);
        if (valid) {
            contribution =
                dense_contains(keys_0, parameters.key_min_0, parameters.key_span_0, input_0[global_index]) &&
                dense_contains(keys_1, parameters.key_min_1, parameters.key_span_1, input_1[global_index]) &&
                dense_contains(keys_2, parameters.key_min_2, parameters.key_span_2, input_2[global_index])
                ? 1u : 0u;
        }
    }
    uint simd_total = simd_sum(contribution);
    if (simd_index == 0) {
        simd_totals[simdgroup_index] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0) {
        uint group_total = simd_sum(simd_index < 8 ? simd_totals[simd_index] : 0u);
        if (simd_index == 0) {
            partial_counts[group_index] = group_total;
        }
    }
}

struct ValueWorkItem { uint value_start; uint count; uint kind; uint payload; };
struct RowSegment { uint row_start; uint value_start; uint count; uint valid; };
// materialize and dictionary_count appended at the end of both params
// structs (rather than reordering existing fields) so the C++ mirrors in
// SparkMetalBridge.mm stay lockstep-compatible field-for-field with the
// pre-Task-2 layout.
struct ExpandParams { uint item_count; uint bit_width; uint value_bytes_offset; uint output_base; uint materialize; uint dictionary_count; };
struct ScatterParams { uint segment_count; uint row_base; uint materialize; uint dictionary_count; };

// dictionary is read only when params.materialize != 0 (key-column callers
// pass materialize = 0 and a small placeholder buffer that is never
// dereferenced). An out-of-range id -- bitWidth = ceil(log2(dictSize))
// permits ids up to 2^bitWidth-1, which can exceed dictionary_count-1 for a
// non-power-of-two dictionary, and a corrupt page could encode an
// out-of-range id directly -- materializes to 0 rather than reading past the
// dictionary buffer. A garbage id on a row the caller has already marked
// valid is corrupt input; 0 is a safe, deterministic placeholder consistent
// with the zero-filled-plane convention used everywhere else in this file.
kernel void expand_value_runs(
    device const uchar *page [[buffer(0)]],
    device const ValueWorkItem *items [[buffer(1)]],
    device int *output [[buffer(2)]],
    constant ExpandParams &params [[buffer(3)]],
    device const int *dictionary [[buffer(4)]],
    uint group_index [[threadgroup_position_in_grid]],
    uint local_index [[thread_index_in_threadgroup]])
{
    if (group_index >= params.item_count) return;
    ValueWorkItem item = items[group_index];
    if (local_index >= item.count) return;
    int value;
    if (item.kind == 0) {
        value = int(item.payload);
    } else {
        ulong bit = ulong(item.payload) + ulong(local_index) * params.bit_width;
        device const uchar *bytes = page + params.value_bytes_offset;
        uint window = uint(bytes[bit >> 3])
            | (uint(bytes[(bit >> 3) + 1]) << 8)
            | (uint(bytes[(bit >> 3) + 2]) << 16)
            | (uint(bytes[(bit >> 3) + 3]) << 24);
        value = int((window >> (bit & 7)) & ((1u << params.bit_width) - 1u));
    }
    if (params.materialize != 0) {
        uint id = uint(value);
        value = id < params.dictionary_count ? dictionary[id] : 0;
    }
    output[params.output_base + item.value_start + local_index] = value;
}

// dictionary is read only when params.materialize != 0. When materialize is
// set, `values` holds raw dictionary ids (as written by expand_value_runs
// with materialize = 0) and this kernel gathers dict[id] instead of id into
// the output plane -- the measure-column with-nulls path. Out-of-range ids
// are handled the same way as expand_value_runs above: materialize to 0
// instead of reading past the dictionary buffer.
kernel void scatter_segments(
    device const int *values [[buffer(0)]],
    device const RowSegment *segments [[buffer(1)]],
    device int *ids [[buffer(2)]],
    device uchar *validity [[buffer(3)]],
    constant ScatterParams &params [[buffer(4)]],
    device const int *dictionary [[buffer(5)]],
    uint group_index [[threadgroup_position_in_grid]],
    uint local_index [[thread_index_in_threadgroup]])
{
    if (group_index >= params.segment_count) return;
    RowSegment segment = segments[group_index];
    if (local_index >= segment.count) return;
    uint row = params.row_base + segment.row_start + local_index;
    if (segment.valid != 0) {
        int value = values[segment.value_start + local_index];
        if (params.materialize != 0) {
            uint id = uint(value);
            value = id < params.dictionary_count ? dictionary[id] : 0;
        }
        ids[row] = value;
    } else {
        validity[row] = 1;
    }
}

// Task 3 (grouped aggregate). One thread per row of a decoded row group.
//
// Gating is exactly the membership kernels' rule, generalized to key_count
// columns: a row survives only if every key column is non-null AND every key
// column's code lookup yields a non-negative code.
//
// Codes: code_tables[k] is indexed by column k's id-plane int32, which is
// either a dictionary id (a chunk decoded through parquetDecodePage's
// dictionary path) or, since Task 6b, a raw non-negative key VALUE (a chunk
// decoded through its PLAIN path) -- the kernel does not distinguish the two,
// it just indexes. Each entry is either -1 (this key is not in the build
// side, so the row is dropped) or the column's PREMULTIPLIED group component,
// so that group_id is simply the sum of the per-column codes. This is safe
// for value space with NO min offset: the JVM caller sizes code_tables[k] to
// dimMaxKey + 1 and leaves every low, unpopulated entry -1, so a negative
// fact-side value is still caught by the negative-identifier check below and
// an out-of-domain fact-side value is still caught by the code_length[k]
// bounds check next -- the same two guards that, for dictionary-id space,
// defend only against a corrupt plane. An index outside code_length[k] is
// likewise treated as a non-member rather than read (a corrupt page or a
// mis-sized table must never read past the buffer).
//
// Factors: factor_tables[k], when factor_length[k] != 0, holds the
// duplicate-build-key multiplicity for that key (1 for unique keys); the row's
// factor is the product across columns. A column with factor_length[k] == 0
// contributes 1 and its factor buffer is never dereferenced (the JNI side
// binds a shared one-element placeholder there).
//
// Accumulation: each aggregate's int64 contribution is added into
// partials[(group_id * agg_count + a) * 2] as a (lo, hi) uint pair, using two
// relaxed 32-bit atomics plus an explicit carry. Metal has no device-wide
// 64-bit atomic add; splitting one into two 32-bit adds is exact for signed
// 64-bit values -- including negative ones -- because two's-complement
// addition is addition modulo 2^64, and lo/hi accumulate the low and high
// halves of that same modular sum (the carry out of the low half is recovered
// from the value the low add returned). Reading the pair back as
// (long)((ulong)hi << 32 | lo) therefore reproduces the exact signed total,
// independent of the order threads or command buffers accumulated in.
struct GroupedAggParams {
    uint row_count;
    uint key_count;
    uint agg_count;
    uint group_count;
    uint key_null_mask;
    uint measure_null_mask;
    uint code_length[4];
    uint factor_length[4];
    uint agg_measure[8];
    uint agg_kind[8];
};

inline void aggregate_add_int64(device atomic_uint *partials, uint index, long value)
{
    ulong bits = ulong(value);
    uint low = uint(bits & 0xFFFFFFFFul);
    uint high = uint(bits >> 32);
    uint previous = atomic_fetch_add_explicit(&partials[2u * index], low, memory_order_relaxed);
    uint carry = previous > (0xFFFFFFFFu - low) ? 1u : 0u;
    atomic_fetch_add_explicit(&partials[2u * index + 1u], high + carry, memory_order_relaxed);
}

kernel void fused_grouped_aggregate(
    device const int *key_ids_0 [[buffer(0)]],
    device const int *key_ids_1 [[buffer(1)]],
    device const int *key_ids_2 [[buffer(2)]],
    device const int *key_ids_3 [[buffer(3)]],
    device const uchar *key_nulls_0 [[buffer(4)]],
    device const uchar *key_nulls_1 [[buffer(5)]],
    device const uchar *key_nulls_2 [[buffer(6)]],
    device const uchar *key_nulls_3 [[buffer(7)]],
    device const int *code_table_0 [[buffer(8)]],
    device const int *code_table_1 [[buffer(9)]],
    device const int *code_table_2 [[buffer(10)]],
    device const int *code_table_3 [[buffer(11)]],
    device const uint *factor_table_0 [[buffer(12)]],
    device const uint *factor_table_1 [[buffer(13)]],
    device const uint *factor_table_2 [[buffer(14)]],
    device const uint *factor_table_3 [[buffer(15)]],
    device const int *measure_values_0 [[buffer(16)]],
    device const int *measure_values_1 [[buffer(17)]],
    device const int *measure_values_2 [[buffer(18)]],
    device const int *measure_values_3 [[buffer(19)]],
    device const uchar *measure_nulls_0 [[buffer(20)]],
    device const uchar *measure_nulls_1 [[buffer(21)]],
    device const uchar *measure_nulls_2 [[buffer(22)]],
    device const uchar *measure_nulls_3 [[buffer(23)]],
    device atomic_uint *partials [[buffer(24)]],
    constant GroupedAggParams &params [[buffer(25)]],
    uint global_index [[thread_position_in_grid]])
{
    if (global_index >= params.row_count) return;

    device const int *key_ids[4] = {key_ids_0, key_ids_1, key_ids_2, key_ids_3};
    device const uchar *key_nulls[4] = {key_nulls_0, key_nulls_1, key_nulls_2, key_nulls_3};
    device const int *code_tables[4] = {code_table_0, code_table_1, code_table_2, code_table_3};
    device const uint *factor_tables[4] =
        {factor_table_0, factor_table_1, factor_table_2, factor_table_3};
    device const int *measure_values[4] =
        {measure_values_0, measure_values_1, measure_values_2, measure_values_3};
    device const uchar *measure_nulls[4] =
        {measure_nulls_0, measure_nulls_1, measure_nulls_2, measure_nulls_3};

    uint group = 0u;
    // ulong, not uint: factor is a product across up to 4 columns of
    // per-column duplicate-key multiplicities, and a uint accumulator can
    // wrap at 2^32 well within the range a handful of moderately-duplicated
    // dimensions can reach. The CPU fallback (aggregateRowGroupOnCpu in
    // MetalParquetGroupedAggregateExec) already accumulates its factor as a
    // Scala Long for exactly this reason -- ulong here matches that width.
    // Actual multiplicities are always small non-negative counts, so the
    // product is expected to stay well inside the signed 64-bit range that
    // long(factor) below reinterprets it into (matching the CPU fallback's
    // `measureValues(slot) * factor` Long multiply).
    ulong factor = 1ul;
    for (uint column = 0u; column < params.key_count && column < 4u; ++column) {
        if ((params.key_null_mask & (1u << column)) != 0u &&
            key_nulls[column][global_index] != 0) {
            return;
        }
        int identifier = key_ids[column][global_index];
        // Defensive only: a dictionary id is never negative (see the header).
        if (identifier < 0) return;
        uint entry = uint(identifier);
        if (entry >= params.code_length[column]) return;
        int code = code_tables[column][entry];
        if (code < 0) return;
        group += uint(code);
        uint factor_length = params.factor_length[column];
        if (factor_length != 0u) {
            if (entry >= factor_length) return;
            factor *= ulong(factor_tables[column][entry]);
        }
    }
    // Defensive: a well-formed set of code tables can only sum to a group id
    // inside the space the caller sized the partial table for, but a bad table
    // must not turn into an out-of-bounds device write.
    if (group >= params.group_count) return;

    for (uint aggregate = 0u; aggregate < params.agg_count && aggregate < 8u; ++aggregate) {
        uint kind = params.agg_kind[aggregate];
        long contribution;
        if (kind == 0u) {
            // count(*): every surviving row contributes its factor.
            contribution = long(factor);
        } else {
            uint slot = params.agg_measure[aggregate];
            if (slot >= 4u) continue;
            if ((params.measure_null_mask & (1u << slot)) != 0u &&
                measure_nulls[slot][global_index] != 0) {
                continue;
            }
            if (kind == 1u) {
                // sum(column)
                contribution = long(measure_values[slot][global_index]) * long(factor);
            } else {
                // count(column): non-null measures only, weighted by factor.
                contribution = long(factor);
            }
        }
        aggregate_add_int64(partials, group * params.agg_count + aggregate, contribution);
    }
}

kernel void fused_membership_count_3_multiplicity(
    device const int *input_0 [[buffer(0)]],
    device const int *input_1 [[buffer(1)]],
    device const int *input_2 [[buffer(2)]],
    device const uchar *nulls_0 [[buffer(3)]],
    device const uchar *nulls_1 [[buffer(4)]],
    device const uchar *nulls_2 [[buffer(5)]],
    device const uint *keys_0 [[buffer(6)]],
    device const uint *keys_1 [[buffer(7)]],
    device const uint *keys_2 [[buffer(8)]],
    device long *partial_counts [[buffer(9)]],
    constant MembershipCountParameters &parameters [[buffer(10)]],
    uint global_index [[thread_position_in_grid]],
    uint local_index [[thread_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]])
{
    threadgroup long scratch[256];
    long contribution = 0;
    if (global_index < parameters.count) {
        bool valid =
            ((parameters.null_mask & 1u) == 0u || nulls_0[global_index] == 0) &&
            ((parameters.null_mask & 2u) == 0u || nulls_1[global_index] == 0) &&
            ((parameters.null_mask & 4u) == 0u || nulls_2[global_index] == 0);
        if (valid) {
            uint multiplicity_0 = dense_multiplicity(
                keys_0, parameters.key_min_0, parameters.key_span_0, input_0[global_index]);
            uint multiplicity_1 = dense_multiplicity(
                keys_1, parameters.key_min_1, parameters.key_span_1, input_1[global_index]);
            uint multiplicity_2 = dense_multiplicity(
                keys_2, parameters.key_min_2, parameters.key_span_2, input_2[global_index]);
            contribution = long(multiplicity_0) * long(multiplicity_1) * long(multiplicity_2);
        }
    }
    scratch[local_index] = contribution;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_index < stride) {
            scratch[local_index] += scratch[local_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_index == 0) {
        partial_counts[group_index] = scratch[0];
    }
}
