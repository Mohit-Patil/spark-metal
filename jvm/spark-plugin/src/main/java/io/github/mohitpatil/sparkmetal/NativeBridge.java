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

    // Parses one decompressed V1 data page on the CPU and encodes its GPU
    // expansion into the stream (no wait). rowOffset is the first row of this
    // page within the row group for this column.
    public static native void parquetDecodePage(
            long streamHandle, long rowGroupHandle, int column,
            byte[] pageBytes, int pageLength, int valueCount, int rowOffset,
            boolean hasDefLevels);

    // Debug/verification: blocks, then copies the decoded planes out.
    public static native void parquetRowGroupRead(
            long streamHandle, long rowGroupHandle, int column,
            int[] idsOut, byte[] validityOut);

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

}
