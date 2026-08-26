package io.github.mohitpatil.sparkmetal

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.execution.{SparkPlan, UnaryExecNode}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.execution.vectorized.OnHeapColumnVector
import org.apache.spark.sql.types.LongType
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

case class MetalFusedSumExec(
    outputAttribute: Attribute,
    inputOrdinal: Int,
    threshold: Int,
    multiplier: Int,
    addend: Int,
    nativeLibrary: String,
    metalLibrary: String,
    child: SparkPlan) extends UnaryExecNode {

  override def output: Seq[Attribute] = Seq(outputAttribute)

  override def supportsColumnar: Boolean = true

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "numInputRows" -> SQLMetrics.createMetric(sparkContext, "number of input rows"),
    "numInputBatches" -> SQLMetrics.createMetric(sparkContext, "number of input batches"),
    "metalTime" -> SQLMetrics.createTimingMetric(sparkContext, "Metal execution and bridge time"))

  override protected def doExecute(): RDD[InternalRow] =
    throw new UnsupportedOperationException("MetalFusedSumExec is columnar-only")

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val inputRows = longMetric("numInputRows")
    val inputBatches = longMetric("numInputBatches")
    val metalTime = longMetric("metalTime")
    child.executeColumnar().mapPartitions { batches =>
      SparkMetalNative.ensureInitialized(nativeLibrary, metalLibrary)
      batches.map { batch =>
        val rowCount = batch.numRows()
        inputRows += rowCount
        inputBatches += 1
        try {
          val input = batch.column(inputOrdinal)
          val started = System.nanoTime()
          val partialSum = executeBatch(input, rowCount)
          metalTime += (System.nanoTime() - started) / 1000000
          val vector = new OnHeapColumnVector(1, LongType)
          vector.putLong(0, partialSum)
          new ColumnarBatch(Array[ColumnVector](vector), 1)
        } finally {
          batch.closeIfFreeable()
        }
      }
    }
  }

  private def executeBatch(input: ColumnVector, rowCount: Int): Long = {
    if (OffHeapColumnVectorAccess.supportsIntAddress(input)) {
      return NativeBridge.fusedFilterProjectSumAddress(
        OffHeapColumnVectorAccess.valueAddress(input),
        if (input.hasNull()) OffHeapColumnVectorAccess.nullAddress(input) else 0L,
        input.hasNull(),
        rowCount,
        threshold,
        multiplier,
        addend)
    }
    val shared = SharedBufferPool.acquire(rowCount)
    val integers = shared.values.asIntBuffer()
    var row = 0
    while (row < rowCount) {
      integers.put(row, input.getInt(row))
      row += 1
    }
    val hasNulls = input.hasNull()
    if (hasNulls) {
      NativeBridge.clearSharedBytes(shared.nulls, rowCount)
      row = 0
      while (row < rowCount) {
        if (input.isNullAt(row)) {
          shared.nulls.put(row, 1.toByte)
        }
        row += 1
      }
    }
    NativeBridge.fusedFilterProjectSum(
      shared.values,
      shared.nulls,
      hasNulls,
      rowCount,
      threshold,
      multiplier,
      addend)
  }

  override protected def withNewChildInternal(newChild: SparkPlan): MetalFusedSumExec =
    copy(child = newChild)
}
