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
    bool allValid = true;
};

// Parses one decompressed V1 data page. valueCount is the page's total row
// count (nulls included). hasDefLevels is maxDefinitionLevel == 1.
// Returns false when the page uses anything outside the supported subset.
bool parseDataPageV1(const uint8_t *page, size_t length, uint32_t valueCount,
                     bool hasDefLevels, PageRuns &out);

}  // namespace sparkmetal
