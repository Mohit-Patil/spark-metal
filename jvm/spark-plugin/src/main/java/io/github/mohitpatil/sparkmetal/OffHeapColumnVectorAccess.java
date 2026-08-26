package io.github.mohitpatil.sparkmetal;

import java.lang.reflect.Field;

import org.apache.spark.sql.execution.vectorized.OffHeapColumnVector;
import org.apache.spark.sql.vectorized.ColumnVector;

public final class OffHeapColumnVectorAccess {
    private static final Field NULLS = field("nulls");

    private OffHeapColumnVectorAccess() {}

    public static boolean supportsIntAddress(ColumnVector vector) {
        return vector instanceof OffHeapColumnVector
                && !((OffHeapColumnVector) vector).hasDictionary();
    }

    public static long valueAddress(ColumnVector vector) {
        return ((OffHeapColumnVector) vector).valuesNativeAddress();
    }

    public static long nullAddress(ColumnVector vector) {
        try {
            return NULLS.getLong(vector);
        } catch (IllegalAccessException exception) {
            throw new IllegalStateException("Cannot access Spark off-heap null storage", exception);
        }
    }

    private static Field field(String name) {
        try {
            Field field = OffHeapColumnVector.class.getDeclaredField(name);
            field.setAccessible(true);
            return field;
        } catch (ReflectiveOperationException exception) {
            throw new ExceptionInInitializerError(exception);
        }
    }
}
