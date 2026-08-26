#include <cassert>
#include <cstdio>
#include <cstring>
#include "../jni/ParquetPageRuns.h"

using namespace sparkmetal;

static void appendUleb(std::vector<uint8_t> &b, uint32_t v) {
    while (v >= 0x80) { b.push_back(uint8_t(v) | 0x80); v >>= 7; }
    b.push_back(uint8_t(v));
}

// Reference expansion of the parsed descriptors, mirroring what the GPU does.
static void expand(const PageRuns &runs, const std::vector<uint8_t> &page,
                   std::vector<int32_t> &values) {
    values.assign(runs.nonNullCount, -1);
    const uint8_t *bytes = page.data() + runs.valueBytesOffset;
    for (const ValueWorkItem &item : runs.items) {
        for (uint32_t i = 0; i < item.count; ++i) {
            if (item.kind == 0) { values[item.valueStart + i] = int32_t(item.payload); continue; }
            uint64_t bit = uint64_t(item.payload) + uint64_t(i) * runs.bitWidth;
            uint32_t window = 0;
            for (int b = 0; b < 4; ++b) window |= uint32_t(bytes[(bit >> 3) + b]) << (8 * b);
            values[item.valueStart + i] =
                int32_t((window >> (bit & 7)) & ((1u << runs.bitWidth) - 1));
        }
    }
}

// The 4-byte bit-extraction window can touch up to 3 bytes past the last
// payload byte; every constructed page is padded with 4 zero bytes before use
// (the parser receives the unpadded length).
static void pad(std::vector<uint8_t> &page) { page.resize(page.size() + 4, 0); }

int main() {
    // Page: def levels [1]x5, [0]x3, [1]x2 (10 rows, 7 values), then
    // bitWidth=3 values: RLE run 5x id=4, bit-packed group of 8 ids 0..7 (2 used).
    std::vector<uint8_t> def;
    appendUleb(def, 5 << 1); def.push_back(1);
    appendUleb(def, 3 << 1); def.push_back(0);
    appendUleb(def, 2 << 1); def.push_back(1);
    std::vector<uint8_t> page;
    page.push_back(uint8_t(def.size())); page.push_back(0); page.push_back(0); page.push_back(0);
    page.insert(page.end(), def.begin(), def.end());
    page.push_back(3);                       // value bit width
    appendUleb(page, 5 << 1); page.push_back(4);           // RLE 5 x 4
    appendUleb(page, (1 << 1) | 1);                        // 1 bit-packed group
    page.push_back(0x88); page.push_back(0xC6); page.push_back(0xFA);  // ids 0..7, 3-bit LSB-first
    size_t pageLength = page.size(); pad(page);

    PageRuns runs;
    assert(parseDataPageV1(page.data(), pageLength, 10, true, runs));
    assert(runs.bitWidth == 3 && !runs.allValid && runs.nonNullCount == 7);
    // Segments: rows 0-4 valid (values 0-4), rows 5-7 null, rows 8-9 valid (values 5-6).
    assert(runs.segments.size() == 3);
    assert(runs.segments[0].rowStart == 0 && runs.segments[0].count == 5 && runs.segments[0].valid == 1);
    assert(runs.segments[1].rowStart == 5 && runs.segments[1].count == 3 && runs.segments[1].valid == 0);
    assert(runs.segments[2].rowStart == 8 && runs.segments[2].valueStart == 5 &&
           runs.segments[2].count == 2 && runs.segments[2].valid == 1);
    std::vector<int32_t> values;
    expand(runs, page, values);
    const int32_t expected[7] = {4, 4, 4, 4, 4, 0, 1};
    assert(std::memcmp(values.data(), expected, sizeof expected) == 0);

    // All-valid page without nulls plus a bit-packed tail clamp: 12 values,
    // bit-packed 2 groups (16 slots) — parser must clamp to 12.
    std::vector<uint8_t> page2;
    std::vector<uint8_t> def2; appendUleb(def2, 12 << 1); def2.push_back(1);
    page2.push_back(uint8_t(def2.size())); page2.push_back(0); page2.push_back(0); page2.push_back(0);
    page2.insert(page2.end(), def2.begin(), def2.end());
    page2.push_back(1);                      // bit width 1
    appendUleb(page2, (2 << 1) | 1); page2.push_back(0xAA); page2.push_back(0xAA);
    size_t page2Length = page2.size(); pad(page2);
    PageRuns runs2;
    assert(parseDataPageV1(page2.data(), page2Length, 12, true, runs2));
    assert(runs2.allValid && runs2.segments.empty() && runs2.nonNullCount == 12);
    std::vector<int32_t> values2; expand(runs2, page2, values2);
    for (uint32_t i = 0; i < 12; ++i) assert(values2[i] == int32_t(i & 1));

    // Bit-packed *definition* levels with scattered nulls -- the shape real
    // data takes (TPC-DS store_sales) and the one the transition-scanning
    // expansion in parseDataPageV1 exists for. 9 groups = 72 rows:
    //   byte 0xFF -> rows 0-7 defined
    //   byte 0xEB -> 1,1,0,1,0,1,1,1 (LSB first) for rows 8-15
    //   byte 0x00 -> rows 16-23 null
    //   6x 0xFF   -> rows 24-71 defined
    // The first window therefore has to stop at a transition 10 bits in, later
    // ones start mid-byte, and the last spans 48 bits of a short (6-byte)
    // window -- covering every branch of the window walk.
    std::vector<uint8_t> def4;
    appendUleb(def4, (9 << 1) | 1);
    def4.push_back(0xFF); def4.push_back(0xEB); def4.push_back(0x00);
    for (int i = 0; i < 6; ++i) def4.push_back(0xFF);
    std::vector<uint8_t> page4;
    page4.push_back(uint8_t(def4.size())); page4.push_back(0); page4.push_back(0); page4.push_back(0);
    page4.insert(page4.end(), def4.begin(), def4.end());
    page4.push_back(2);                                    // value bit width
    appendUleb(page4, 62 << 1); page4.push_back(3);        // RLE 62 x id 3
    size_t page4Length = page4.size(); pad(page4);
    PageRuns runs4;
    assert(parseDataPageV1(page4.data(), page4Length, 72, true, runs4));
    assert(!runs4.allValid && runs4.nonNullCount == 62);
    const uint32_t expectedRows[7]   = {0, 10, 11, 12, 13, 16, 24};
    const uint32_t expectedCounts[7] = {10, 1, 1, 1, 3, 8, 48};
    const uint32_t expectedValid[7]  = {1, 0, 1, 0, 1, 0, 1};
    const uint32_t expectedValueStarts[7] = {0, 0, 10, 0, 11, 0, 14};
    assert(runs4.segments.size() == 7);
    for (uint32_t s = 0; s < 7; ++s) {
        assert(runs4.segments[s].rowStart == expectedRows[s]);
        assert(runs4.segments[s].count == expectedCounts[s]);
        assert(runs4.segments[s].valid == expectedValid[s]);
        if (runs4.segments[s].valid) {
            assert(runs4.segments[s].valueStart == expectedValueStarts[s]);
        }
    }
    std::vector<int32_t> values4; expand(runs4, page4, values4);
    for (uint32_t i = 0; i < 62; ++i) assert(values4[i] == 3);

    // Unsupported: bit width > 24 must be rejected.
    std::vector<uint8_t> page3(page2); page3[4 + def2.size()] = 30;
    PageRuns runs3;
    assert(!parseDataPageV1(page3.data(), page2Length, 12, true, runs3));

    // Corrupted defLength that overflows the page: must be rejected.
    // Create a page with defLength > remaining buffer size.
    std::vector<uint8_t> pageCorrupted;
    pageCorrupted.push_back(0xFF); pageCorrupted.push_back(0xFF);  // defLength = 65535 (too large)
    pageCorrupted.push_back(0x00); pageCorrupted.push_back(0x00);
    pageCorrupted.push_back(1);                                     // bit width 1
    size_t corruptedLength = pageCorrupted.size();
    pageCorrupted.resize(pageCorrupted.size() + 4, 0);  // pad for bit extraction
    PageRuns runsCorrupted;
    assert(!parseDataPageV1(pageCorrupted.data(), corruptedLength, 10, true, runsCorrupted));

    std::puts("parquet_page_runs_test OK");
    return 0;
}
