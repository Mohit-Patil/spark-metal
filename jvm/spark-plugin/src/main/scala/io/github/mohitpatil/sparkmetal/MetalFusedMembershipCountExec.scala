package io.github.mohitpatil.sparkmetal

import java.util.UUID
import java.util.concurrent.Executors

import scala.collection.mutable
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

/**
 * Builds the native dense membership maps once per query execution and shares
 * the prepared handle across every concurrently running partition task in this
 * JVM. The last task to release the token frees the native maps and reports
 * the accumulated copy fallbacks.
 */
private[sparkmetal] object PreparedMembershipCache {
  private final class Entry(val handle: Long, var remaining: Int)
  private val entries = mutable.HashMap.empty[String, Entry]

  def acquire(
      token: String,
      keys: Seq[Array[Int]],
      expectedUses: Int,
      buildTime: SQLMetric): Long = synchronized {
    val entry = entries.getOrElseUpdate(token, {
      val started = System.nanoTime()
      val handle = NativeBridge.prepareMembershipCount3(keys(0), keys(1), keys(2))
      buildTime += (System.nanoTime() - started) / 1000000
      new Entry(handle, expectedUses)
    })
    entry.handle
  }

  def release(token: String): Long = synchronized {
    entries.get(token) match {
      case Some(entry) =>
        entry.remaining -= 1
        if (entry.remaining <= 0) {
          entries.remove(token)
          val fallbacks = NativeBridge.membershipCount3CopyFallbacks(entry.handle)
          NativeBridge.releaseMembershipCount3(entry.handle)
          fallbacks
        } else 0L
      case None => 0L
    }
  }
}

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
    "numMetalCommands" -> SQLMetrics.createMetric(sparkContext, "number of Metal command buffers"),
    "inputCopyFallbacks" -> SQLMetrics.createMetric(sparkContext, "Metal input copy fallbacks"),
    "dimensionTime" -> SQLMetrics.createTimingMetric(sparkContext, "dimension key collection time"),
    "membershipBuildTime" -> SQLMetrics.createTimingMetric(sparkContext, "membership map build time"),
    "metalTime" -> SQLMetrics.createTimingMetric(sparkContext, "Metal submit and final wait time"))

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
        Future(plan.executeCollect().filterNot(_.isNullAt(0)).map(_.getInt(0)))
      }, Duration.Inf)
    } finally {
      keyExecutor.shutdown()
    }
    dimensionTime += (System.nanoTime() - dimensionStarted) / 1000000
    val inputRows = longMetric("numInputRows")
    val inputBatches = longMetric("numInputBatches")
    val metalCommands = longMetric("numMetalCommands")
    val inputCopyFallbacks = longMetric("inputCopyFallbacks")
    val membershipBuildTime = longMetric("membershipBuildTime")
    val metalTime = longMetric("metalTime")
    val denseDomains = keys.forall { values =>
      values.nonEmpty && {
        var minimum = Int.MaxValue
        var maximum = Int.MinValue
        values.foreach { value =>
          if (value < minimum) minimum = value
          if (value > maximum) maximum = value
        }
        maximum.toLong - minimum.toLong + 1L <= 16L * 1024 * 1024
      }
    }
    val factBatches = factPlan.executeColumnar()
    val prepareToken = UUID.randomUUID().toString
    val prepareUses = factBatches.getNumPartitions
    factBatches.mapPartitions { batches =>
      SparkMetalNative.ensureInitialized(nativeLibrary, metalLibrary)
      lazy val multiplicities = keys.map(_.groupMapReduce(identity)(_ => 1L)(_ + _))
      lazy val allKeysUnique = multiplicities.forall(_.valuesIterator.forall(_ == 1L))
      val presenceTables = new java.util.IdentityHashMap[AnyRef, Array[Byte]]()
      val multiplicityTables = new java.util.IdentityHashMap[AnyRef, Array[Int]]()
      val preparedHandle = if (denseDomains) {
        PreparedMembershipCache.acquire(prepareToken, keys, prepareUses, membershipBuildTime)
      } else 0L
      var partitionCount = 0L
      var metalNanos = 0L
      try {
        val streamHandle =
          if (preparedHandle != 0L) NativeBridge.membershipCount3StreamBegin(preparedHandle)
          else 0L
        var streamFinished = streamHandle == 0L
        try {
          while (batches.hasNext) {
            val batch = batches.next()
            val rowCount = batch.numRows()
            inputRows += rowCount
            inputBatches += 1
            val columns = factOrdinals.map(batch.column)
            if (rowCount == 0) {
              batch.closeIfFreeable()
            } else if (streamHandle != 0L && columns.forall { column =>
                OffHeapColumnVectorAccess.supportsIntAddress(column) ||
                OffHeapColumnVectorAccess.supportsDictionaryIntAddress(column) }) {
              val started = System.nanoTime()
              def inputAddress(ordinal: Int): Long = {
                val column = columns(ordinal)
                if (OffHeapColumnVectorAccess.supportsIntAddress(column)) {
                  OffHeapColumnVectorAccess.valueAddress(column)
                } else {
                  OffHeapColumnVectorAccess.dictionaryIdsAddress(column)
                }
              }
              def dictionaryArray(column: ColumnVector): Array[Int] = {
                val maxId = OffHeapColumnVectorAccess.dictionaryMaxId(column)
                Array.tabulate(maxId + 1)(id => OffHeapColumnVectorAccess.decodeDictionaryInt(column, id))
              }
              def presenceTable(ordinal: Int): Array[Byte] = {
                val column = columns(ordinal)
                if (OffHeapColumnVectorAccess.supportsIntAddress(column) || !allKeysUnique) null
                else presenceTables.computeIfAbsent(
                  OffHeapColumnVectorAccess.dictionary(column),
                  _ => MembershipTables.presence(dictionaryArray(column), multiplicities(ordinal)))
              }
              def multiplicityTable(ordinal: Int): Array[Int] = {
                val column = columns(ordinal)
                if (OffHeapColumnVectorAccess.supportsIntAddress(column) || allKeysUnique) null
                else multiplicityTables.computeIfAbsent(
                  OffHeapColumnVectorAccess.dictionary(column),
                  _ => MembershipTables.multiplicity(dictionaryArray(column), multiplicities(ordinal)))
              }
              def nullAddress(ordinal: Int): Long =
                if (columns(ordinal).hasNull()) OffHeapColumnVectorAccess.nullAddress(columns(ordinal))
                else 0L
              try {
                NativeBridge.membershipCount3StreamSubmit(
                  streamHandle,
                  inputAddress(0), nullAddress(0), columns(0).hasNull(),
                  presenceTable(0), multiplicityTable(0),
                  inputAddress(1), nullAddress(1), columns(1).hasNull(),
                  presenceTable(1), multiplicityTable(1),
                  inputAddress(2), nullAddress(2), columns(2).hasNull(),
                  presenceTable(2), multiplicityTable(2),
                  rowCount)
              } finally {
                batch.closeIfFreeable()
              }
              metalNanos += System.nanoTime() - started
              metalCommands += 1
            } else {
              try {
                partitionCount += countOnCpu(columns, multiplicities, rowCount)
              } finally {
                batch.closeIfFreeable()
              }
            }
          }
          if (streamHandle != 0L) {
            val started = System.nanoTime()
            // Set the flag before calling Finish: Finish deletes the native
            // stream even when it throws (e.g. a GPU command failure), so
            // the finally-block's Abort must never fire afterward -- it
            // would touch an already-deleted stream.
            streamFinished = true
            partitionCount += NativeBridge.membershipCount3StreamFinish(streamHandle)
            metalNanos += System.nanoTime() - started
          }
        } finally {
          if (!streamFinished) {
            NativeBridge.membershipCount3StreamAbort(streamHandle)
          }
        }
      } finally {
        metalTime += metalNanos / 1000000
        if (preparedHandle != 0L) {
          inputCopyFallbacks += PreparedMembershipCache.release(prepareToken)
        }
      }
      val vector = new OnHeapColumnVector(1, LongType)
      vector.putLong(0, partitionCount)
      Iterator.single(new ColumnarBatch(Array[ColumnVector](vector), 1))
    }
  }

  private def countOnCpu(
      columns: Seq[ColumnVector],
      multiplicities: Seq[Map[Int, Long]],
      rowCount: Int): Long = {
    if (multiplicities.exists(_.isEmpty)) {
      return 0L
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
