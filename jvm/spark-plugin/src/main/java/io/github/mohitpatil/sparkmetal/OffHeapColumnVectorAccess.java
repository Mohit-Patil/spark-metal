package io.github.mohitpatil.sparkmetal;

import java.lang.reflect.Field;

import org.apache.spark.sql.execution.datasources.parquet.ParquetDictionary;
import org.apache.spark.sql.execution.vectorized.Dictionary;
import org.apache.spark.sql.execution.vectorized.OffHeapColumnVector;
import org.apache.spark.sql.execution.vectorized.WritableColumnVector;
import org.apache.spark.sql.vectorized.ColumnVector;

public final class OffHeapColumnVectorAccess {
    private static final Field NULLS = field(OffHeapColumnVector.class, "nulls");
    private static final Field DICTIONARY = field(WritableColumnVector.class, "dictionary");
    private static final Field PARQUET_DICTIONARY = field(ParquetDictionary.class, "dictionary");

    private OffHeapColumnVectorAccess() {}

    public static boolean supportsIntAddress(ColumnVector vector) {
        return vector instanceof OffHeapColumnVector
                && !((OffHeapColumnVector) vector).hasDictionary();
    }

    /**
     * True when the vector keeps Parquet dictionary ids off-heap, so membership
     * can be tested on the GPU against a per-dictionary presence table without
     * materializing the decoded values.
     */
    public static boolean supportsDictionaryIntAddress(ColumnVector vector) {
        if (!(vector instanceof OffHeapColumnVector)) {
            return false;
        }
        OffHeapColumnVector offHeap = (OffHeapColumnVector) vector;
        return offHeap.hasDictionary()
                && dictionaryOf(offHeap) instanceof ParquetDictionary
                && offHeap.getDictionaryIds() instanceof OffHeapColumnVector;
    }

    public static long valueAddress(ColumnVector vector) {
        return ((OffHeapColumnVector) vector).valuesNativeAddress();
    }

    public static long dictionaryIdsAddress(ColumnVector vector) {
        return ((OffHeapColumnVector) ((OffHeapColumnVector) vector).getDictionaryIds())
                .valuesNativeAddress();
    }

    public static Object dictionary(ColumnVector vector) {
        return dictionaryOf((OffHeapColumnVector) vector);
    }

    public static int dictionaryMaxId(ColumnVector vector) {
        try {
            Object parquetDictionary = PARQUET_DICTIONARY.get(dictionaryOf((OffHeapColumnVector) vector));
            return ((org.apache.parquet.column.Dictionary) parquetDictionary).getMaxId();
        } catch (IllegalAccessException exception) {
            throw new IllegalStateException("Cannot access the Parquet dictionary", exception);
        }
    }

    public static int decodeDictionaryInt(ColumnVector vector, int id) {
        return ((Dictionary) dictionaryOf((OffHeapColumnVector) vector)).decodeToInt(id);
    }

    public static long nullAddress(ColumnVector vector) {
        try {
            return NULLS.getLong(vector);
        } catch (IllegalAccessException exception) {
            throw new IllegalStateException("Cannot access Spark off-heap null storage", exception);
        }
    }

    private static Object dictionaryOf(OffHeapColumnVector vector) {
        try {
            return DICTIONARY.get(vector);
        } catch (IllegalAccessException exception) {
            throw new IllegalStateException("Cannot access the Spark column dictionary", exception);
        }
    }

    private static Field field(Class<?> owner, String name) {
        try {
            Field field = owner.getDeclaredField(name);
            field.setAccessible(true);
            return field;
        } catch (ReflectiveOperationException exception) {
            throw new ExceptionInInitializerError(exception);
        }
    }
}
