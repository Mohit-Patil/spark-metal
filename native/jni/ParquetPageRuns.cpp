#include "ParquetPageRuns.h"
#include <cstring>

namespace sparkmetal {
namespace {

struct ByteCursor {
    const uint8_t *data; size_t length; size_t offset = 0;
    bool readUleb(uint32_t &value) {
        value = 0;
        for (int shift = 0; shift < 35; shift += 7) {
            if (offset >= length) return false;
            uint8_t byte = data[offset++];
            value |= uint32_t(byte & 0x7F) << shift;
            if ((byte & 0x80) == 0) return true;
        }
        return false;
    }
    bool readLittleEndian(uint32_t byteCount, uint32_t &value) {
        if (offset + byteCount > length) return false;
        value = 0;
        for (uint32_t b = 0; b < byteCount; ++b) value |= uint32_t(data[offset + b]) << (8 * b);
        offset += byteCount;
        return true;
    }
};

// Walks one RLE/bit-packed hybrid stream, invoking run(count, isRle, value,
// byteOffset) which returns false to signal an error, with counts clamped so
// the total equals expectedValues.
template <typename Run>
bool walkHybrid(ByteCursor &cursor, uint32_t bitWidth, uint32_t expectedValues, Run run) {
    uint32_t produced = 0;
    while (produced < expectedValues) {
        uint32_t header;
        if (!cursor.readUleb(header)) return false;
        if (header & 1) {
            uint32_t groups = header >> 1;
            uint32_t count = groups * 8;
            size_t payloadBytes = size_t(groups) * bitWidth;  // groups * 8 * width / 8
            if (cursor.offset + payloadBytes > cursor.length) return false;
            // Bit offset is relative to the start of this run's payload; the
            // caller receives the byte offset so items are self-contained.
            if (count > expectedValues - produced) count = expectedValues - produced;
            if (!run(count, false, 0u, cursor.offset)) return false;
            cursor.offset += payloadBytes;
            produced += count;
        } else {
            uint32_t count = header >> 1;
            if (count == 0) return false;
            uint32_t value;
            if (!cursor.readLittleEndian((bitWidth + 7) / 8, value)) return false;
            if (count > expectedValues - produced) count = expectedValues - produced;
            if (!run(count, true, value, size_t(0))) return false;
            produced += count;
        }
    }
    return true;
}

}  // namespace

bool parseDataPageV1(const uint8_t *page, size_t length, uint32_t valueCount,
                     bool hasDefLevels, PageValueEncoding encoding, PageRuns &out) {
    // Reset in place rather than `out = PageRuns{}`: assigning a fresh PageRuns
    // frees both vectors' storage, so every page re-grew them from nothing.
    // Callers reuse one PageRuns across a column chunk's pages, which keeps the
    // capacity (~80 items for a 20k-value page) and makes this allocation-free
    // after the first page.
    out.items.clear();
    out.segments.clear();
    out.bitWidth = 0;
    out.valueBytesOffset = 0;
    out.plainBytesOffset = 0;
    out.nonNullCount = 0;
    out.maxItemCount = 0;
    out.maxSegmentCount = 0;
    out.allValid = true;
    out.plain = encoding == PageValueEncoding::Plain;
    ByteCursor cursor{page, length};

    // Definition levels: 4-byte LE length, then a bitWidth-1 hybrid stream.
    // Collected first as (count, defined) pairs, then folded into segments.
    // thread_local for the same reason: one scratch vector per decoding thread,
    // reused across pages. Cleared here, never read before it is written.
    static thread_local std::vector<std::pair<uint32_t, bool>> defRuns;
    defRuns.clear();
    if (hasDefLevels) {
        uint32_t defLength;
        if (!cursor.readLittleEndian(4, defLength)) return false;
        if (cursor.offset + defLength > length) return false;
        ByteCursor def{page + cursor.offset, defLength};
        bool ok = walkHybrid(def, 1, valueCount,
            [&](uint32_t count, bool isRle, uint32_t value, size_t byteOffset) {
                if (isRle) {
                    defRuns.emplace_back(count, value != 0);
                } else {
                    // Bit-packed definition levels. This is the common case for
                    // real data with scattered nulls -- TPC-DS SF10
                    // store_sales, say -- so it walks *transitions*, not bits:
                    // load a 64-bit window at the current bit position, flip it
                    // if the current run is of ones, and the first set bit is
                    // the end of the run. A bit-at-a-time loop with a coalescing
                    // branch per value cost ~300us on a 20k-value page (an
                    // unpredictable branch and a push_back capacity check per
                    // value), which was 95% of the whole GPU decode path's CPU
                    // budget on SF10 q96.
                    const uint8_t *bits = def.data + byteOffset;
                    size_t availableBytes = def.length - byteOffset;
                    uint32_t i = 0;
                    while (i < count) {
                        size_t byteIndex = i >> 3;
                        if (byteIndex >= availableBytes) return false;
                        size_t windowBytes = availableBytes - byteIndex;
                        if (windowBytes > 8) windowBytes = 8;
                        // Assembled explicitly rather than memcpy'd so the
                        // stream's LSB-first bit order does not depend on the
                        // host's endianness.
                        uint64_t window = 0;
                        for (size_t b = 0; b < windowBytes; ++b) {
                            window |= uint64_t(bits[byteIndex + b]) << (8 * b);
                        }
                        uint32_t bitInByte = i & 7;
                        window >>= bitInByte;
                        // Bits at or above validBits are zero in window, so the
                        // inverted search below always finds a set bit by then.
                        uint32_t validBits =
                            static_cast<uint32_t>(windowBytes * 8) - bitInByte;
                        bool defined = (window & 1) != 0;
                        uint64_t search = defined ? ~window : window;
                        uint32_t run = search == 0
                            ? validBits
                            : static_cast<uint32_t>(__builtin_ctzll(search));
                        if (run > validBits) run = validBits;
                        if (run > count - i) run = count - i;
                        if (run == 0) return false;  // defensive: never advance by 0
                        if (!defRuns.empty() && defRuns.back().second == defined) {
                            defRuns.back().first += run;
                        } else {
                            defRuns.emplace_back(run, defined);
                        }
                        i += run;
                    }
                }
                return true;
            });
        if (!ok) return false;
        cursor.offset += defLength;
    } else {
        defRuns.emplace_back(valueCount, true);
    }

    uint32_t nonNull = 0;
    for (auto &run : defRuns) if (run.second) nonNull += run.first;
    out.nonNullCount = nonNull;
    out.allValid = nonNull == valueCount;

    if (!out.allValid) {
        uint32_t row = 0, value = 0;
        for (auto &run : defRuns) {
            uint32_t remaining = run.first;
            while (remaining > 0) {
                uint32_t chunk = remaining < kDecodeChunk ? remaining : kDecodeChunk;
                out.segments.push_back({row, value, chunk, run.second ? 1u : 0u});
                if (chunk > out.maxSegmentCount) out.maxSegmentCount = chunk;
                row += chunk;
                if (run.second) value += chunk;
                remaining -= chunk;
            }
        }
    }

    if (encoding == PageValueEncoding::Plain) {
        // Plain values: no bit-width byte -- straight into a packed
        // little-endian int32 array of the nonNull values. No work items are
        // produced; the caller (CPU memcpy for an all-valid page, or a
        // value-space memcpy ahead of scatter_segments for a page with
        // nulls) reads the values directly from plainBytesOffset.
        out.plainBytesOffset = uint32_t(cursor.offset);
        if (nonNull == 0) return true;
        size_t neededBytes = size_t(nonNull) * sizeof(int32_t);
        if (cursor.offset + neededBytes > length) return false;
        return true;
    }

    // Values: 1-byte bit width, then a hybrid stream of nonNull dictionary ids.
    if (cursor.offset >= length) return false;
    out.bitWidth = page[cursor.offset++];
    if (out.bitWidth > 24) return false;
    out.valueBytesOffset = uint32_t(cursor.offset);
    if (nonNull == 0) return true;
    if (out.bitWidth == 0) {
        // Every value is dictionary id 0.
        for (uint32_t start = 0; start < nonNull; start += kDecodeChunk) {
            uint32_t chunk = nonNull - start < kDecodeChunk ? nonNull - start : kDecodeChunk;
            out.items.push_back({start, chunk, 0, 0});
            if (chunk > out.maxItemCount) out.maxItemCount = chunk;
        }
        return true;
    }
    uint32_t valueStart = 0;
    return walkHybrid(cursor, out.bitWidth, nonNull,
        [&](uint32_t count, bool isRle, uint32_t value, size_t byteOffset) {
            uint64_t runBitBase =
                (uint64_t(byteOffset) - out.valueBytesOffset) * 8;
            uint32_t emitted = 0;
            while (emitted < count) {
                uint32_t chunk = count - emitted < kDecodeChunk ? count - emitted : kDecodeChunk;
                if (isRle) {
                    out.items.push_back({valueStart + emitted, chunk, 0, value});
                } else {
                    uint64_t bit = runBitBase + uint64_t(emitted) * out.bitWidth;
                    if (bit > UINT32_MAX) return false;
                    out.items.push_back({valueStart + emitted, chunk, 1, uint32_t(bit)});
                }
                if (chunk > out.maxItemCount) out.maxItemCount = chunk;
                emitted += chunk;
            }
            valueStart += count;
            return true;
        });
}

}  // namespace sparkmetal
