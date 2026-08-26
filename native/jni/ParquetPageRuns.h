#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sparkmetal {

constexpr uint32_t kDecodeChunk = 256;  // max values per work item / segment

struct ValueWorkItem {   // one threadgroup of expand_value_runs
    uint32_t valueStart;  // ordinal in the page's non-null value space
    uint32_t count;       // <= kDecodeChunk
    uint32_t kind;        // 0 = RLE, 1 = BITPACKED
    uint32_t payload;     // RLE: dictionary id; BITPACKED: bit offset into value bytes
};

struct RowSegment {      // one threadgroup of scatter_segments
    uint32_t rowStart;    // row ordinal within the page
    uint32_t valueStart;  // first non-null value ordinal covered (unused when !valid)
    uint32_t count;       // <= kDecodeChunk
    uint32_t valid;       // 1 = rows carry values, 0 = rows are null
};

struct PageRuns {
    std::vector<ValueWorkItem> items;
    std::vector<RowSegment> segments;  // empty when allValid
    uint32_t bitWidth = 0;
    uint32_t valueBytesOffset = 0;     // offset of the value hybrid payload in the page
    uint32_t nonNullCount = 0;
    // Longest count of any item / segment. Both kernels map one threadgroup to
    // one item (or segment) and one thread to one of its values, so the
    // dispatch only needs a threadgroup this wide -- not the full kDecodeChunk.
    // Real data is full of short runs (TPC-DS store_sales averages ~11 values
    // per run), where a fixed 256-wide threadgroup leaves 95% of its threads
    // idle.
    uint32_t maxItemCount = 0;
    uint32_t maxSegmentCount = 0;
    bool allValid = true;
};

// Threadgroup width for a dispatch whose widest work item holds maxCount
// values: the smallest SIMD-width multiple that still covers it.
inline uint32_t decodeThreadgroupWidth(uint32_t maxCount) {
    if (maxCount == 0) return 32;
    uint32_t width = ((maxCount + 31) / 32) * 32;
    return width > kDecodeChunk ? kDecodeChunk : width;
}

// Parses one decompressed V1 data page. valueCount is the page's total row
// count (nulls included). hasDefLevels is maxDefinitionLevel == 1.
// Returns false when the page uses anything outside the supported subset.
bool parseDataPageV1(const uint8_t *page, size_t length, uint32_t valueCount,
                     bool hasDefLevels, PageRuns &out);

}  // namespace sparkmetal
