package io.github.mohitpatil.sparkmetal;

import java.nio.ByteBuffer;

public final class NativeBridge {
    private NativeBridge() {}

    public static native void initialize(String metallibPath);

    public static native ByteBuffer allocateSharedInt32(int count);

    public static native ByteBuffer allocateSharedBytes(int count);

    public static native void releaseShared(ByteBuffer buffer);

    public static native void clearSharedBytes(ByteBuffer buffer, int count);

    public static native long fusedFilterProjectSum(
            ByteBuffer input,
            ByteBuffer validity,
            boolean hasNulls,
            int count,
            int threshold,
            int multiplier,
            int addend);

    public static native long fusedFilterProjectSumAddress(
            long inputAddress,
            long nullAddress,
            boolean hasNulls,
            int count,
            int threshold,
            int multiplier,
            int addend);

    public static native long prepareMembershipCount3(
            int[] keys0,
            int[] keys1,
            int[] keys2);

    public static native void releaseMembershipCount3(long preparedHandle);

    public static native long membershipCount3CopyFallbacks(long preparedHandle);

    public static native long membershipCount3PreparedAddress(
            long input0Address,
            long null0Address,
            boolean hasNull0,
            long input1Address,
            long null1Address,
            boolean hasNull1,
            long input2Address,
            long null2Address,
            boolean hasNull2,
            int count,
            long preparedHandle);

    public static native long membershipCount3PreparedBatches(
            long[] input0Addresses,
            long[] null0Addresses,
            boolean[] hasNull0,
            long[] input1Addresses,
            long[] null1Addresses,
            boolean[] hasNull1,
            long[] input2Addresses,
            long[] null2Addresses,
            boolean[] hasNull2,
            int[] counts,
            long preparedHandle);

    public static native long membershipCount3StreamBegin(long preparedHandle);

    /**
     * Submits one batch to the stream without waiting for the GPU. Per column
     * the input address holds either decoded int32 values (both dictionary
     * tables null) or Parquet dictionary ids, in which case exactly one of
     * dictPresence (unique build keys) or dictMultiplicity (duplicate build
     * keys) maps every dictionary id to its membership.
     */
    public static native void membershipCount3StreamSubmit(
            long streamHandle,
            long input0Address,
            long null0Address,
            boolean hasNull0,
            byte[] dictPresence0,
            int[] dictMultiplicity0,
            long input1Address,
            long null1Address,
            boolean hasNull1,
            byte[] dictPresence1,
            int[] dictMultiplicity1,
            long input2Address,
            long null2Address,
            boolean hasNull2,
            byte[] dictPresence2,
            int[] dictMultiplicity2,
            int count);

    public static native long membershipCount3StreamFinish(long streamHandle);

    public static native void membershipCount3StreamAbort(long streamHandle);

    // Allocates row-group ids (int32) and validity (uchar) planes for 3 columns
    // inside the stream, zero-fills validity via a blit, returns a handle.
    public static native long parquetRowGroupBegin(long streamHandle, int rowCount);

    // Generalization of parquetRowGroupBegin (Task 2): allocates keyCount
    // id/validity plane pairs (join-key columns, same layout as
    // parquetRowGroupBegin) plus measureCount value/validity plane pairs
    // (measure columns, decoded by parquetDecodeMeasurePage). All planes are
    // zero/blit-initialized. parquetRowGroupBegin(stream, rowCount) is
    // exactly parquetRowGroupBeginAggregate(stream, rowCount, 3, 0).
    public static native long parquetRowGroupBeginAggregate(
            long streamHandle, int rowCount, int keyCount, int measureCount);

    // Parses one decompressed V1 data page on the CPU and encodes its GPU
    // expansion into the stream (no wait). rowOffset is the first row of this
    // page within the row group for this column.
    public static native void parquetDecodePage(
            long streamHandle, long rowGroupHandle, int column,
            byte[] pageBytes, int pageLength, int valueCount, int rowOffset,
            boolean hasDefLevels);

    // Stages a measure column's dictionary (the VALUES the dictionary ids
    // point at) onto the row group; a null array means the column is
    // PLAIN-encoded. Set at most once per (row group, slot), before that
    // slot's first parquetDecodeMeasurePage call; the dictionary persists on
    // the row group and is reused across that column's pages.
    public static native void parquetSetMeasureDictionary(
            long rowGroupHandle, int measureSlot, int[] dictionary);

    // Decodes one V1 measure page into the row group's measure plane `slot`
    // (allocated by parquetRowGroupBeginAggregate). PLAIN pages memcpy the
    // packed values into value-space staging (all-valid: directly into the
    // plane at rowOffset); dictionary pages expand ids and materialize
    // through the slot's dictionary (parquetSetMeasureDictionary). Validity
    // is written exactly as for key columns.
    public static native void parquetDecodeMeasurePage(
            long streamHandle, long rowGroupHandle, int measureSlot,
            byte[] pageBytes, int pageLength, int valueCount, int rowOffset,
            boolean hasDefLevels);

    // Debug/verification: blocks, then copies the decoded planes out.
    public static native void parquetRowGroupRead(
            long streamHandle, long rowGroupHandle, int column,
            int[] idsOut, byte[] validityOut);

    // Debug/verification: blocks, then copies one decoded measure plane out.
    public static native void parquetRowGroupReadMeasure(
            long streamHandle, long rowGroupHandle, int measureSlot,
            int[] valuesOut, byte[] validityOut);

    // Releases a row-group handle without running membership (used by the smoke
    // test and error paths). Task 4 adds the membership variant.
    public static native void parquetRowGroupRelease(long rowGroupHandle);

    /**
     * Encodes the membership kernel over the row group's planes with per-column
     * dictionary tables (same contract as membershipCount3StreamSubmit: exactly
     * one of presence/multiplicity per column, matching the prepared kernel),
     * commits without waiting, and releases the row-group handle. The result is
     * accumulated by the existing membershipCount3StreamFinish.
     */
    public static native void parquetRowGroupCount(
            long streamHandle, long rowGroupHandle,
            byte[] dictPresence0, int[] dictMultiplicity0,
            byte[] dictPresence1, int[] dictMultiplicity1,
            byte[] dictPresence2, int[] dictMultiplicity2);

    /**
     * Encodes the grouped-aggregation kernel over one row group's decoded key
     * and measure planes and consumes the row-group handle (same lifecycle as
     * parquetRowGroupCount: commits without waiting, then releases the handle
     * -- the caller must not touch it again).
     *
     * <p>codes[k] maps column k's decoded int32 (dictionary-id space for a
     * dictionary-decoded column, value space otherwise) to either -1, meaning
     * the key is not a member and the row is dropped, or that column's
     * PREMULTIPLIED group component, so the row's group id is the sum of the
     * per-column codes. factors[k] may be null (every key unique, multiplier
     * 1) or an equally indexed table of duplicate-key multiplicities; the
     * row's weight is the product across columns.
     *
     * <p>aggKinds[a] is 0 for count(*), 1 for sum(measure), 2 for
     * count(measure); aggMeasureSlots[a] names the measure slot for kinds 1
     * and 2 and is ignored for kind 0. At most 4 key columns, 4 measure slots
     * and 8 aggregates are supported.
     *
     * <p>Results do NOT flow into membershipCount3StreamFinish. They
     * accumulate into a per-STREAM partial table allocated on the first call
     * (so every row group of the partition adds into the same table) and are
     * read back by parquetAggregateStreamFinish. Every call on one stream must
     * pass the same groupCount and the same number of aggregates.
     */
    public static native void parquetRowGroupAggregate(
            long streamHandle, long rowGroupHandle,
            int[][] codes, int[][] factors, int groupCount,
            int[] aggMeasureSlots, int[] aggKinds);

    /**
     * Waits for every command buffer on the stream, folds the aggregate
     * partial table into long[groupCount * aggCount] (row-major: group-major,
     * aggregate-minor) and destroys the stream. Returns a zero-length array if
     * no row group was ever aggregated on this stream.
     *
     * <p>The stream is destroyed even when this throws, exactly like
     * membershipCount3StreamFinish: callers must set their
     * "stream already finished" flag BEFORE calling, so a finally-block Abort
     * can never run against an already-deleted stream.
     */
    public static native long[] parquetAggregateStreamFinish(long streamHandle);

    /** Waits, reclaims and destroys the stream without producing a result. */
    public static native void parquetAggregateStreamAbort(long streamHandle);

}
