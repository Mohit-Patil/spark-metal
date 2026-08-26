package io.github.mohitpatil.sparkmetal

import java.util.concurrent.Executors

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.execution.vectorized.OnHeapColumnVector
import org.apache.spark.sql.types.LongType
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

case class MetalFusedMembershipCountExec(
    outputAttribute: Attribute,
    factOrdinals: Seq[Int],
    keyPlans: Seq[SparkPlan],
    nativeLibrary: String,
    metalLibrary: String,
    factPlan: SparkPlan) extends SparkPlan {

  require(factOrdinals.length == 3 && keyPlans.length == 3)

  override def children: Seq[SparkPlan] = factPlan +: keyPlans

  override def output: Seq[Attribute] = Seq(outputAttribute)

  override def supportsColumnar: Boolean = true

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "numInputRows" -> SQLMetrics.createMetric(sparkContext, "number of fact rows"),
    "numInputBatches" -> SQLMetrics.createMetric(sparkContext, "number of fact batches"),
    "inputCopyFallbacks" -> SQLMetrics.createMetric(sparkContext, "Metal input copy fallbacks"),
    "dimensionTime" -> SQLMetrics.createTimingMetric(sparkContext, "dimension key collection time"),
    "membershipBuildTime" -> SQLMetrics.createTimingMetric(sparkContext, "membership map build time"),
    "metalTime" -> SQLMetrics.createTimingMetric(sparkContext, "Metal membership and count time"))

  override protected def doExecute(): RDD[InternalRow] =
    throw new UnsupportedOperationException("MetalFusedMembershipCountExec is columnar-only")

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val dimensionTime = longMetric("dimensionTime")
    val dimensionStarted = System.nanoTime()
    val keyExecutor = Executors.newFixedThreadPool(keyPlans.length)
    implicit val keyExecutionContext: ExecutionContext =
      ExecutionContext.fromExecutor(keyExecutor)
    val keys = try {
      Await.result(Future.traverse(keyPlans) { plan =>
        Future(plan.executeCollect().filterNot(_.isNullAt(0)).map(_.getInt(0)).sorted)
      }, Duration.Inf)
    } finally {
      keyExecutor.shutdown()
    }
    dimensionTime += (System.nanoTime() - dimensionStarted) / 1000000
    val inputRows = longMetric("numInputRows")
    val inputBatches = longMetric("numInputBatches")
    val inputCopyFallbacks = longMetric("inputCopyFallbacks")
    val membershipBuildTime = longMetric("membershipBuildTime")
    val metalTime = longMetric("metalTime")
    val multiplicities = keys.map(_.groupMapReduce(identity)(_ => 1L)(_ + _))
    val denseDomains = keys.forall { values =>
      values.nonEmpty && values.last.toLong - values.head.toLong + 1L <= 16L * 1024 * 1024
    }
    factPlan.executeColumnar().mapPartitions { batches =>
      SparkMetalNative.ensureInitialized(nativeLibrary, metalLibrary)
      var partitionCount = 0L
      val preparedHandle = if (denseDomains) {
        val started = System.nanoTime()
        val handle = NativeBridge.prepareMembershipCount3(keys(0), keys(1), keys(2))
        membershipBuildTime += (System.nanoTime() - started) / 1000000
        handle
      } else 0L
      try {
        while (batches.hasNext) {
          val batch = batches.next()
          val rowCount = batch.numRows()
          inputRows += rowCount
          inputBatches += 1
          try {
            val columns = factOrdinals.map(batch.column)
            val started = System.nanoTime()
            partitionCount += executeBatch(
              columns, multiplicities, rowCount, preparedHandle)
            metalTime += (System.nanoTime() - started) / 1000000
          } finally {
            batch.closeIfFreeable()
          }
        }
      } finally {
        if (preparedHandle != 0L) {
          inputCopyFallbacks += NativeBridge.membershipCount3CopyFallbacks(preparedHandle)
          NativeBridge.releaseMembershipCount3(preparedHandle)
        }
      }
      val vector = new OnHeapColumnVector(1, LongType)
      vector.putLong(0, partitionCount)
      Iterator.single(new ColumnarBatch(Array[ColumnVector](vector), 1))
    }
  }

  private def executeBatch(
      columns: Seq[ColumnVector],
      multiplicities: Seq[Map[Int, Long]],
      rowCount: Int,
      preparedHandle: Long): Long = {
    if (multiplicities.exists(_.isEmpty)) {
      return 0L
    }
    if (preparedHandle != 0L && columns.forall(OffHeapColumnVectorAccess.supportsIntAddress)) {
      return NativeBridge.membershipCount3PreparedAddress(
        OffHeapColumnVectorAccess.valueAddress(columns(0)),
        if (columns(0).hasNull()) OffHeapColumnVectorAccess.nullAddress(columns(0)) else 0L,
        columns(0).hasNull(),
        OffHeapColumnVectorAccess.valueAddress(columns(1)),
        if (columns(1).hasNull()) OffHeapColumnVectorAccess.nullAddress(columns(1)) else 0L,
        columns(1).hasNull(),
        OffHeapColumnVectorAccess.valueAddress(columns(2)),
        if (columns(2).hasNull()) OffHeapColumnVectorAccess.nullAddress(columns(2)) else 0L,
        columns(2).hasNull(),
        rowCount,
        preparedHandle)
    }
    var count = 0L
    var row = 0
    while (row < rowCount) {
      if (!columns(0).isNullAt(row) && !columns(1).isNullAt(row) && !columns(2).isNullAt(row) &&
          multiplicities(0).contains(columns(0).getInt(row)) &&
          multiplicities(1).contains(columns(1).getInt(row)) &&
          multiplicities(2).contains(columns(2).getInt(row))) {
        count += multiplicities(0)(columns(0).getInt(row)) *
          multiplicities(1)(columns(1).getInt(row)) *
          multiplicities(2)(columns(2).getInt(row))
      }
      row += 1
    }
    count
  }

  override protected def withNewChildrenInternal(newChildren: IndexedSeq[SparkPlan]): SparkPlan =
    copy(factPlan = newChildren.head, keyPlans = newChildren.tail)
}
