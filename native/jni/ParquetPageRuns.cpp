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
                     bool hasDefLevels, PageRuns &out) {
    out = PageRuns{};
    ByteCursor cursor{page, length};

    // Definition levels: 4-byte LE length, then a bitWidth-1 hybrid stream.
    // Collected first as (count, defined) pairs, then folded into segments.
    std::vector<std::pair<uint32_t, bool>> defRuns;
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
                    // Bit-packed definition levels: expand bit by bit (rare for
                    // maxDef==1 data, but valid). Coalesce equal neighbors.
                    const uint8_t *bits = def.data + byteOffset;
                    for (uint32_t i = 0; i < count; ++i) {
                        bool defined = (bits[i >> 3] >> (i & 7)) & 1;
                        if (!defRuns.empty() && defRuns.back().second == defined) {
                            defRuns.back().first += 1;
                        } else {
                            defRuns.emplace_back(1, defined);
                        }
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
                row += chunk;
                if (run.second) value += chunk;
                remaining -= chunk;
            }
        }
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
                emitted += chunk;
            }
            valueStart += count;
            return true;
        });
}

}  // namespace sparkmetal
