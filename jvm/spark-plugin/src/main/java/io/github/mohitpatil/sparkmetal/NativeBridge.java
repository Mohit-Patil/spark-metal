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

}
