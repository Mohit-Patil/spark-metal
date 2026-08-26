package io.github.mohitpatil.sparkmetal;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.file.Path;

public final class NativeBridgeSmoke {
    private NativeBridgeSmoke() {}

    public static void main(String[] arguments) {
        if (arguments.length != 3) {
            throw new IllegalArgumentException(
                    "Usage: NativeBridgeSmoke LIBSPARKMETAL METALLIB ROW_COUNT");
        }
        System.load(Path.of(arguments[0]).toAbsolutePath().toString());
        NativeBridge.initialize(Path.of(arguments[1]).toAbsolutePath().toString());
        int count = Integer.parseInt(arguments[2]);
        int threshold = 100;
        int multiplier = 3;
        int addend = 7;
        ByteBuffer bytes = NativeBridge.allocateSharedInt32(count).order(ByteOrder.nativeOrder());
        ByteBuffer validity = NativeBridge.allocateSharedBytes(count);
        try {
            IntBuffer values = bytes.asIntBuffer();
            int state = 0x5eed1234;
            long expected = 0;
            for (int index = 0; index < count; index++) {
                state = state * 1_664_525 + 1_013_904_223;
                int value = Integer.remainderUnsigned(state, 2_001) - 1_000;
                values.put(index, value);
                boolean valid = index % 17 != 0;
                validity.put(index, valid ? (byte) 0 : (byte) 1);
                if (valid && value > threshold) {
                    expected += (long) (value * multiplier + addend);
                }
            }
            long actual = NativeBridge.fusedFilterProjectSum(
                    bytes, validity, true, count, threshold, multiplier, addend);
            if (actual != expected) {
                throw new AssertionError("CPU=" + expected + ", Metal=" + actual);
            }
            System.out.printf(
                    "{\"rows\":%d,\"cpuSum\":%d,\"metalSum\":%d,\"match\":true}%n",
                    count, expected, actual);
        } finally {
            NativeBridge.releaseShared(validity);
            NativeBridge.releaseShared(bytes);
        }
    }
}
