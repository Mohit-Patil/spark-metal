package io.github.mohitpatil.sparkmetal

import java.nio.{ByteBuffer, ByteOrder}
import java.util.concurrent.Executors

import scala.collection.mutable
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.jdk.CollectionConverters._

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.parquet.column.impl.ColumnReadStoreImpl
import org.apache.parquet.column.page.{DataPageV1, DictionaryPage, PageReadStore}
import org.apache.parquet.column.{ColumnDescriptor, Encoding}
import org.apache.parquet.example.DummyRecordConverter
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.hadoop.util.HadoopInputFile
import org.apache.parquet.schema.MessageType

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, BoundReference, GenericInternalRow, UnsafeProjection}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.types.{
  BooleanType, ByteType, DataType, DateType, Decimal, DecimalType, DoubleType, IntegerType, LongType,
  ShortType, StringType}

/** One (file, row-group) unit of work, planned on the driver. */
private[sparkmetal] case class ParquetGroupedAggregateSplit(
    file: String, rowGroupIndex: Int, rowCount: Long)

/**
 * One user-visible aggregate's internal-slot wiring. `sumSlot`/`countSlot`
 * index into the flat internal aggregate arrays (`aggKinds`/`aggMeasureSlots`)
 * built by [[MetalParquetGroupedAggregateExec.buildInternalAggPlan]]:
 *
 *  - "sum": `sumSlot` is a kind-1 (sum) internal slot over the measure
 *    column; `countSlot` is a paired kind-2 (count non-null) slot over the
 *    SAME measure column, used only to decide whether the user-visible sum
 *    is null (zero non-null contributions).
 *  - "avg": identical wiring to "sum" -- the partial Average buffer is
 *    exactly (sum, count), unevaluated (division happens in the FINAL
 *    aggregate downstream of this operator, not here).
 *  - "count": `sumSlot` is -1 (unused); `countSlot` is a kind-0 (count-star)
 *    or kind-2 (count non-null column) slot, whichever `input` calls for.
 */
private[sparkmetal] case class AggSlotMapping(function: String, sumSlot: Int, countSlot: Int)

/**
 * Drives the GPU Parquet key+measure decode (Task 2) and the grouped
 * aggregation kernel (Task 3) per (file, row group), accumulating into one
 * per-partition partial table per [[GroupSpace]]-assigned group. Key columns
 * decode either through a dictionary (dictionary-id-space code tables) or,
 * since Task 6b, PLAIN (dense value-space code tables) -- per chunk, not per
 * operator (see `decodeKeyColumn`). Falls back to a CPU parquet-mr
 * aggregation for any row group the native decoder rejects, and to a
 * whole-operator CPU hash-join + hash-aggregate when [[GroupSpace.build]]
 * itself cannot represent the dimensions' cross product (an oversized group
 * space, or a duplicate join key in an attributed dimension), or when a
 * PLAIN-eligible dimension's join-key domain does not fit a value-space
 * code table (`MaxValueSpaceKey`) -- see `doExecute`'s domain guard and
 * `executeWholeOperatorCpuFallback`. Planned by
 * [[SparkMetalColumnarRule]] in place of a partial [[org.apache.spark.sql.execution.aggregate.HashAggregateExec]]
 * (Task 6); also driven directly by ParquetDecodeSmoke's "agg-exec" mode for
 * operator-level testing.
 *
 * This operator is row-based (not columnar): its output feeds Spark's own
 * shuffle + final aggregate, which expect [[InternalRow]]s.
 */
case class MetalParquetGroupedAggregateExec(
    outputAttributes: Seq[Attribute],
    files: Seq[String],
    keyColumnNames: Seq[String],
    measureColumnNames: Seq[String],
    aggSpecs: Seq[GroupedAggregateShape.AggSpec],
    groupKeyDimensionIndex: Seq[(Int, Int)],
    keyPlans: Seq[SparkPlan],
    nativeLibrary: String,
    metalLibrary: String) extends SparkPlan {

  require(keyColumnNames.nonEmpty && keyColumnNames.length <= 4,
    s"MetalParquetGroupedAggregateExec requires 1-4 key columns, got ${keyColumnNames.length}")
  require(keyColumnNames.length == keyPlans.length,
    s"keyColumnNames (${keyColumnNames.length}) and keyPlans (${keyPlans.length}) must have the same length")
  require(measureColumnNames.length <= 4,
    s"MetalParquetGroupedAggregateExec supports at most 4 measure columns, got ${measureColumnNames.length}")
  require(aggSpecs.nonEmpty, "MetalParquetGroupedAggregateExec requires at least one aggregate")
  groupKeyDimensionIndex.foreach { case (dimensionIndex, attributeIndex) =>
    require(dimensionIndex >= 0 && dimensionIndex < keyColumnNames.length,
      s"groupKeyDimensionIndex entry ($dimensionIndex, $attributeIndex): dimension index out of range " +
        s"[0, ${keyColumnNames.length})")
    // keyPlans' output schema is resolved at construction time (before any
    // execution), so the attribute index can be validated up front too --
    // ordinal 0 of a keyPlan's output is the join key, ordinals 1.. are its
    // group-key attributes (see collectDimensions).
    val attributeCount = keyPlans(dimensionIndex).output.length - 1
    require(attributeIndex >= 0 && attributeIndex < attributeCount,
      s"groupKeyDimensionIndex entry ($dimensionIndex, $attributeIndex): attribute index out of range " +
        s"[0, $attributeCount) for dimension $dimensionIndex")
  }

  private val DictionaryEncodings: Set[Encoding] = Set(Encoding.PLAIN_DICTIONARY, Encoding.RLE_DICTIONARY)

  override def children: Seq[SparkPlan] = keyPlans

  override def output: Seq[Attribute] = outputAttributes

  override def supportsColumnar: Boolean = false

  override protected def doExecuteColumnar(): RDD[org.apache.spark.sql.vectorized.ColumnarBatch] =
    throw new UnsupportedOperationException("MetalParquetGroupedAggregateExec is row-based only")

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "numRowGroups" -> SQLMetrics.createMetric(sparkContext, "number of Parquet row groups"),
    "numPagesDecoded" -> SQLMetrics.createMetric(sparkContext, "number of Parquet data pages decoded"),
    "cpuFallbackRowGroups" -> SQLMetrics.createMetric(sparkContext, "row groups aggregated on CPU"),
    "decodeParseTime" -> SQLMetrics.createTimingMetric(sparkContext, "page parse and GPU-encode time"),
    "metalTime" -> SQLMetrics.createTimingMetric(sparkContext, "Metal aggregate and final wait time"),
    "dimensionTime" -> SQLMetrics.createTimingMetric(sparkContext, "dimension key collection time"),
    "numGroups" -> SQLMetrics.createMetric(sparkContext, "group space size"),
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of partial rows emitted"))

  override protected def doExecute(): RDD[InternalRow] = {
    val dimensionTime = longMetric("dimensionTime")
    val (dimensions, dimensionNanos) = collectDimensions()
    dimensionTime += dimensionNanos / 1000000
    // Captured once, on the driver, for BOTH execution paths: a dimension's
    // OWN attribute types (not outputAttributes' declared types) are what a
    // dimension's attribute row was actually encoded with (UnsafeProjection
    // in collectDimensions), so reading it back must use the SAME types --
    // outputAttributes is validated for arity only (see the require below).
    val dimensionAttributeTypes: Seq[Seq[DataType]] = dimensions.map(_.attributeTypes)

    val (aggKinds, aggMeasureSlots, aggSlotMappings, occupancySlot) =
      MetalParquetGroupedAggregateExec.buildInternalAggPlan(aggSpecs, measureColumnNames)
    val internalAggCount = aggKinds.length
    val expectedOutputColumns =
      groupKeyDimensionIndex.length + aggSpecs.map(spec => if (spec.function == "avg") 2 else 1).sum
    require(outputAttributes.length == expectedOutputColumns,
      s"outputAttributes has ${outputAttributes.length} columns, expected $expectedOutputColumns " +
        s"(${groupKeyDimensionIndex.length} group keys + ${aggSpecs.length} aggregates)")

    // Task 6b: a PLAIN-decoded key chunk decodes straight into VALUE space
    // (decodeKeyColumn) -- the JVM builds that dimension's code/factor tables
    // sized `dimMaxKey + 1` and indexed by the raw join-key value itself, with
    // NO min offset (see NativeBridge.parquetRowGroupAggregate's Javadoc: a
    // fact value below the table's populated range simply reads one of the
    // -1-filled low entries). That is only safe/bounded when every dimension's
    // join-key domain is non-negative and small enough to keep the table
    // within the kernel's practical size budget -- checked here, once, from
    // the SAME collected rows `GroupSpace.build` is about to consume, for
    // EVERY dimension (not just ones a footer inspection would show as
    // PLAIN today: a column can be dictionary-encoded in some files and
    // PLAIN in others -- see decodeKeyColumn -- so the bound must hold
    // whenever a dimension's chunk MIGHT decode through the PLAIN path).
    // Violating it is a runtime data condition, not a planning defect (the
    // rule's eligibility gate has no access to these rows), so -- exactly
    // like GroupSpace.build itself failing -- it routes the whole operator
    // through the CPU hash-join fallback below rather than throwing.
    val keyDomains: Seq[Option[(Int, Int)]] = dimensions.map { dimension =>
      if (dimension.rows.isEmpty) {
        None
      } else {
        var min = Int.MaxValue
        var max = Int.MinValue
        dimension.rows.foreach { case (key, _) =>
          if (key < min) min = key
          if (key > max) max = key
        }
        Some((min, max))
      }
    }
    val domainViolation: Option[String] = keyDomains.zipWithIndex.collectFirst {
      case (Some((min, max)), index)
          if min < 0 || max > MetalParquetGroupedAggregateExec.MaxValueSpaceKey =>
        s"value-space domain violation: key column ${keyColumnNames(index)}'s join-key domain " +
          s"[$min, $max] falls outside the value-space code table's supported range " +
          s"[0, ${MetalParquetGroupedAggregateExec.MaxValueSpaceKey}]"
    }
    // dimMaxKeys(k) sizes dimension k's PLAIN value-space table lazily, the
    // first time decodeKeyColumn actually encounters a PLAIN chunk for
    // column k (executeWithGroupSpace); -1 for an empty dimension (never
    // dereferenced -- GroupSpace.build already fails a truly empty dimension
    // before this would matter).
    val dimMaxKeys: Array[Int] = keyDomains.map(_.map(_._2).getOrElse(-1)).toArray

    val builtEither: Either[String, GroupSpace.Built] = domainViolation match {
      case Some(reason) => Left(reason)
      case None =>
        GroupSpace.build(dimensions, maxGroups = 1 << 20).flatMap { built =>
          val estimatedBytes = built.groupCount.toLong * internalAggCount.toLong * 16L
          if (estimatedBytes > 64L * 1024 * 1024) {
            Left(s"group space memory estimate $estimatedBytes bytes (groups=${built.groupCount}, " +
              s"internalAggs=$internalAggCount) exceeds the 64MB budget")
          } else {
            Right(built)
          }
        }
    }

    val numGroups = longMetric("numGroups")

    builtEither match {
      case Right(built) =>
        numGroups += built.groupCount
        executeWithGroupSpace(
          built, dimensionAttributeTypes, dimMaxKeys, aggKinds, aggMeasureSlots, aggSlotMappings,
          occupancySlot, internalAggCount)
      // An unsupported dimension attribute type is a planning defect, not a
      // runtime data condition: GroupedAggregateShape/Task 6's matcher is
      // expected to reject any such region before this operator is ever
      // constructed, so this throws driver-side (loudly, at doExecute time)
      // rather than routing into the whole-operator CPU fallback, which
      // would otherwise crash INSIDE an executor task the first time
      // attributeValue() hit the same unsupported type (IllegalStateException
      // from deep inside a Spark task is a much worse failure mode than a
      // clear driver-side throw naming the type).
      case Left(reason) if reason.contains("unsupported attribute type") =>
        throw new IllegalStateException(
          s"MetalParquetGroupedAggregateExec: GroupSpace.build rejected a dimension attribute type " +
            s"($reason) -- this should have been rejected at planning time, not reached construction")
      case Left(reason) =>
        // reason may come from three independent sources -- the value-space
        // domain guard above (GroupSpace.build never even runs), the 64MB
        // group-space memory estimate (GroupSpace.build DID succeed; the
        // estimate check on its result rejected it), or GroupSpace.build
        // itself (a duplicate key or oversized cross product) -- each
        // already carries its own distinguishing prefix/detail, so this
        // message states only what is true of all three: dense group-space
        // execution is unavailable for this reason and this call is falling
        // back.
        logWarning(s"MetalParquetGroupedAggregateExec: cannot use the dense group-space GPU path " +
          s"($reason); falling back to a whole-operator CPU hash-join + hash-aggregate")
        executeWholeOperatorCpuFallback(
          dimensions, dimensionAttributeTypes, aggKinds, aggMeasureSlots, aggSlotMappings, occupancySlot)
    }
  }

  // -------------------------------------------------------------------
  // Dimension collection
  // -------------------------------------------------------------------

  /**
   * Collects every keyPlan's `(joinKey, attrs...)` rows in parallel (mirrors
   * [[MetalParquetMembershipCountExec]]'s dimension fan-out) and builds one
   * [[GroupSpace.Dimension]] per keyPlan. Ordinal 0 of each row is the join
   * key (an int); ordinals 1.. are the dimension's group-key attributes, if
   * any. `attrs` is projected off with an [[UnsafeProjection]] over
   * [[BoundReference]]s shifted by one -- this is what actually exercises a
   * real `executeCollect()`-sourced `UnsafeRow` (as opposed to Task 4's
   * directly-constructed `GenericInternalRow`s) against
   * [[GroupSpace.attributeValue]]'s type-driven getters (`getDecimal` in
   * particular, whose `UnsafeRow` and `GenericInternalRow` implementations
   * genuinely differ). `.copy()` on the projection's output is required
   * (not just defensive): `UnsafeProjection.apply` reuses its own output
   * buffer across calls.
   */
  private def collectDimensions(): (Seq[GroupSpace.Dimension], Long) = {
    val started = System.nanoTime()
    val executor = Executors.newFixedThreadPool(math.max(1, keyPlans.length))
    val dimensions = try {
      implicit val executionContext: ExecutionContext = ExecutionContext.fromExecutor(executor)
      Await.result(Future.traverse(keyPlans) { plan =>
        Future {
          val schema = plan.output
          val attributeCount = schema.length - 1
          val attributeTypes = schema.drop(1).map(_.dataType)
          val projection: UnsafeProjection =
            if (attributeCount > 0) {
              val boundExprs = schema.drop(1).zipWithIndex.map { case (attribute, index) =>
                BoundReference(index + 1, attribute.dataType, attribute.nullable)
              }
              UnsafeProjection.create(boundExprs)
            } else {
              null
            }
          val rows = plan.executeCollect().flatMap { row =>
            if (row.isNullAt(0)) {
              None
            } else {
              val key = row.getInt(0)
              val attributeRow: InternalRow =
                if (attributeCount > 0) projection(row).copy() else InternalRow()
              Some((key, attributeRow))
            }
          }
          GroupSpace.Dimension(rows, attributeCount, attributeTypes)
        }
      }, Duration.Inf)
    } finally {
      executor.shutdown()
    }
    (dimensions, System.nanoTime() - started)
  }

  // -------------------------------------------------------------------
  // Main path: a valid GroupSpace was built
  // -------------------------------------------------------------------

  private def executeWithGroupSpace(
      built: GroupSpace.Built,
      dimensionAttributeTypes: Seq[Seq[DataType]],
      dimMaxKeys: Array[Int],
      aggKinds: Array[Int],
      aggMeasureSlots: Array[Int],
      aggSlotMappings: Seq[AggSlotMapping],
      occupancySlot: Int,
      internalAggCount: Int): RDD[InternalRow] = {
    val numRowGroups = longMetric("numRowGroups")
    val numPagesDecoded = longMetric("numPagesDecoded")
    val cpuFallbackRowGroups = longMetric("cpuFallbackRowGroups")
    val decodeParseTime = longMetric("decodeParseTime")
    val metalTime = longMetric("metalTime")
    val numOutputRows = longMetric("numOutputRows")

    val splits = enumerateSplits()
    val numPartitions = math.max(1, math.min(splits.length, sparkContext.defaultParallelism))
    val splitRDD = sparkContext.parallelize(splits, numPartitions)
    val groupCount = built.groupCount
    val totalCells = groupCount * internalAggCount

    splitRDD.mapPartitions { splitIterator =>
      SparkMetalNative.ensureInitialized(nativeLibrary, metalLibrary)
      val splitsInPartition = splitIterator.toArray
      val readers = mutable.Map.empty[String, ParquetFileReader]

      def readerFor(file: String): ParquetFileReader =
        readers.getOrElseUpdate(file, {
          val opened = ParquetFileReader.open(HadoopInputFile.fromPath(
            new Path(file), MetalParquetGroupedAggregateExec.sharedConfiguration))
          opened.setRequestedSchema(
            allColumnDescriptors(file, opened.getFooter.getFileMetaData.getSchema).asJava)
          opened
        })

      // PLAIN value-space code/factor tables (Task 6b), one slot per key
      // column, built lazily by decodeKeyColumn the first time this
      // partition actually decodes a PLAIN chunk for that column, and reused
      // for every subsequent row group: unlike a dictionary chunk's
      // per-row-group translation (which depends on THAT file's dictionary),
      // a PLAIN column's value-space table depends only on the dimension's
      // own data (built.codesByKey(k)/factorsByKey(k), fixed for the whole
      // operator), so building it once per partition avoids repeating a
      // dimMaxKeys(k)-sized allocation and fill for every row group.
      val plainValueSpaceTables = new Array[(Array[Int], Array[Int])](keyColumnNames.length)

      var localRowGroups = 0L
      var localPages = 0L
      var localFallbacks = 0L
      var parseNanos = 0L
      var metalNanos = 0L
      // Allocated lazily -- only once this partition actually needs a
      // per-row-group CPU fallback -- since eagerly allocating it up front
      // would cost up to groupCount * internalAggCount * 8 bytes (as much
      // as 32MB, half the 64MB budget check's own ceiling) on the common
      // path where every row group decodes cleanly on the GPU.
      var cpuFallbackBuffer: Array[Long] = null
      def cpuFallback(): Array[Long] = {
        if (cpuFallbackBuffer == null) cpuFallbackBuffer = new Array[Long](totalCells)
        cpuFallbackBuffer
      }

      var preparedHandle = 0L
      var streamHandle = 0L
      var streamFinished = false
      var gpuBuffer: Array[Long] = null
      try {
        // preparedHandle/streamHandle acquisition lives INSIDE this try (not
        // before it) so that a throw from membershipCount3StreamBegin --
        // after prepareMembershipCount3 already succeeded -- still reaches
        // the outer finally's release rather than leaking the prepared
        // handle.
        preparedHandle = NativeBridge.prepareMembershipCount3(Array(1), Array(1), Array(1))
        streamHandle = NativeBridge.membershipCount3StreamBegin(preparedHandle)
        try {
          splitsInPartition.foreach { split =>
            val reader = readerFor(split.file)
            val schema = reader.getFooter.getFileMetaData.getSchema
            val keyDescriptors = keyColumnNames.map(name => descriptorFor(schema, split.file, name))
            val measureDescriptors = measureColumnNames.map(name => descriptorFor(schema, split.file, name))

            var rowGroupHandle = 0L
            try {
              val pageReadStore = reader.readRowGroup(split.rowGroupIndex)
              try {
                val parseStarted = System.nanoTime()
                rowGroupHandle = NativeBridge.parquetRowGroupBeginAggregate(
                  streamHandle, split.rowCount.toInt, keyColumnNames.length, measureColumnNames.length)

                val codeTables = new Array[Array[Int]](keyColumnNames.length)
                val factorTables = new Array[Array[Int]](keyColumnNames.length)
                for (k <- keyColumnNames.indices) {
                  val (codes, factors) = decodeKeyColumn(
                    streamHandle, rowGroupHandle, k, split, pageReadStore, keyDescriptors(k),
                    built.codesByKey(k), built.factorsByKey(k), dimMaxKeys(k), plainValueSpaceTables,
                    () => localPages += 1)
                  codeTables(k) = codes
                  factorTables(k) = factors
                }
                for (m <- measureColumnNames.indices) {
                  decodeMeasureColumn(
                    streamHandle, rowGroupHandle, m, split, pageReadStore, measureDescriptors(m),
                    () => localPages += 1)
                }
                parseNanos += System.nanoTime() - parseStarted

                val metalStarted = System.nanoTime()
                NativeBridge.parquetRowGroupAggregate(
                  streamHandle, rowGroupHandle, codeTables, factorTables, groupCount,
                  aggMeasureSlots, aggKinds)
                metalNanos += System.nanoTime() - metalStarted
                // The handle is consumed by a successful Aggregate call --
                // clear it immediately so that if pageReadStore.close()
                // (below, in this same finally) throws, the catch block
                // does not release an already-consumed handle (a
                // use-after-free) AND does not double-count this row group
                // by recomputing it on the CPU on top of its already-landed
                // GPU contribution.
                rowGroupHandle = 0L
                localRowGroups += 1
              } finally {
                pageReadStore.close()
              }
            } catch {
              case e: RuntimeException =>
                // The native decoder rejected a page (or some other runtime
                // surprise) partway through this row group. The handle is
                // consumed by parquetRowGroupAggregate ONLY on success, so a
                // handle that was actually opened here but not yet consumed
                // (rowGroupHandle != 0L) must be released explicitly; pages
                // already fed to the failed GPU pass cannot be re-read, so
                // the CPU recompute uses a fresh PageReadStore.
                logWarning(s"Row group ${split.file}#${split.rowGroupIndex} fell back to CPU", e)
                if (rowGroupHandle != 0L) {
                  NativeBridge.parquetRowGroupRelease(rowGroupHandle)
                }
                localFallbacks += 1
                localRowGroups += 1
                val freshStore = reader.readRowGroup(split.rowGroupIndex)
                try {
                  aggregateRowGroupOnCpu(
                    freshStore, schema, keyDescriptors, measureDescriptors,
                    built.codesByKey, built.factorsByKey, aggKinds, aggMeasureSlots, cpuFallback())
                } finally {
                  freshStore.close()
                }
            }
          }

          // Set the flag BEFORE Finish: Finish destroys the native stream
          // even when it throws, so the finally-block Abort must never fire
          // afterward -- it would touch an already-deleted stream.
          streamFinished = true
          val finishStarted = System.nanoTime()
          gpuBuffer = NativeBridge.parquetAggregateStreamFinish(streamHandle)
          metalNanos += System.nanoTime() - finishStarted
        } finally {
          if (streamHandle != 0L && !streamFinished) {
            NativeBridge.parquetAggregateStreamAbort(streamHandle)
          }
        }
      } finally {
        readers.values.foreach(_.close())
        if (preparedHandle != 0L) {
          NativeBridge.releaseMembershipCount3(preparedHandle)
        }
        metalTime += metalNanos / 1000000
        decodeParseTime += parseNanos / 1000000
        numRowGroups += localRowGroups
        numPagesDecoded += localPages
        cpuFallbackRowGroups += localFallbacks
      }

      // parquetAggregateStreamFinish's contract: zero-length iff no row
      // group ever reached a successful Aggregate call on this stream,
      // otherwise exactly groupCount * internalAggCount. Anything else
      // would mean the native/JVM group-space size agreement was violated
      // -- a real corruption bug, not a legitimate runtime condition, so it
      // is a hard `require` rather than a silently-tolerated branch.
      require(gpuBuffer != null, "parquetAggregateStreamFinish returned null unexpectedly")
      val resultBuffer: Array[Long] =
        if (gpuBuffer.length == totalCells) {
          // Combine in place into gpuBuffer (a fresh array this task
          // exclusively owns) instead of allocating a third same-sized
          // array -- cpuFallbackBuffer is the only other one, and it is
          // itself allocated only when actually needed (see cpuFallback()).
          if (cpuFallbackBuffer != null) {
            var i = 0
            while (i < totalCells) {
              gpuBuffer(i) += cpuFallbackBuffer(i)
              i += 1
            }
          }
          gpuBuffer
        } else {
          require(gpuBuffer.isEmpty,
            s"parquetAggregateStreamFinish returned length ${gpuBuffer.length}, expected 0 or $totalCells")
          if (cpuFallbackBuffer != null) cpuFallbackBuffer else new Array[Long](totalCells)
        }

      val toUnsafeRow = outputProjection()
      val outputRows = mutable.ArrayBuffer.empty[InternalRow]
      var g = 0
      while (g < groupCount) {
        val base = g * internalAggCount
        if (resultBuffer(base + occupancySlot) > 0L) {
          outputRows += toUnsafeRow(buildOutputRow(
            built.groupTuples(g), dimensionAttributeTypes, resultBuffer, base, aggSlotMappings)).copy()
        }
        g += 1
      }
      numOutputRows += outputRows.size
      outputRows.iterator
    }
  }

  /**
   * Per-(row group, key column): dispatches on whether this row group's
   * chunk has a dictionary page (Task 6b -- a column may be dictionary in
   * one file and PLAIN in another, so this is decided per chunk, not once
   * for the whole operator):
   *
   *  - Dictionary chunk (unchanged from before Task 6b): translates
   *    [[GroupSpace.Built]]'s VALUE-keyed code/factor maps into
   *    dictionary-id-indexed tables for THIS row group's own dictionary,
   *    then decodes every page via `NativeBridge.parquetDecodePage`'s
   *    dictionary path (`isPlain = false`).
   *  - PLAIN chunk (no dictionary page at all): reuses (building once per
   *    partition, in `plainValueSpaceTables(ordinal)`) a dense VALUE-space
   *    table sized `dimMaxKey + 1` and indexed directly by raw key value --
   *    [[GroupSpace.Built]]'s code/factor maps are ALREADY value-keyed, so
   *    this is a direct array-fill, not a translation -- then decodes every
   *    page via `parquetDecodePage`'s PLAIN path (`isPlain = true`), which
   *    writes literal packed int32 values straight into the key plane (see
   *    `NativeBridge.parquetRowGroupAggregate`'s Javadoc for why the fact
   *    side needs no min offset or extra bounds handling here).
   *
   * `dimMaxKey` and `plainValueSpaceTables` are only ever touched on the
   * PLAIN path; a dimension that always turns out to be dictionary-encoded
   * across every file this partition reads never allocates one.
   */
  private def decodeKeyColumn(
      streamHandle: Long,
      rowGroupHandle: Long,
      ordinal: Int,
      split: ParquetGroupedAggregateSplit,
      pageReadStore: PageReadStore,
      descriptor: ColumnDescriptor,
      codesByKey: Map[Int, Int],
      factorsByKey: Map[Int, Int],
      dimMaxKey: Int,
      plainValueSpaceTables: Array[(Array[Int], Array[Int])],
      onPageDecoded: () => Unit): (Array[Int], Array[Int]) = {
    val pageReader = pageReadStore.getPageReader(descriptor)
    val dictionaryPage = pageReader.readDictionaryPage()
    val isPlain = dictionaryPage == null
    val (codeTable, factorTable) =
      if (!isPlain) {
        val dictionaryValues = decodeDictionary(dictionaryPage)
        val codes = dictionaryValues.map(value => codesByKey.getOrElse(value, -1))
        val factors =
          if (factorsByKey.isEmpty) null else dictionaryValues.map(value => factorsByKey.getOrElse(value, 1))
        (codes, factors)
      } else {
        if (plainValueSpaceTables(ordinal) == null) {
          plainValueSpaceTables(ordinal) =
            MetalParquetGroupedAggregateExec.buildValueSpaceTables(dimMaxKey, codesByKey, factorsByKey)
        }
        plainValueSpaceTables(ordinal)
      }

    val hasDefLevels = descriptor.getMaxDefinitionLevel > 0
    var rowOffset = 0
    var rawPage = pageReader.readPage()
    while (rawPage != null) {
      val dataPage = rawPage match {
        case v1: DataPageV1 => v1
        case other =>
          throw new RuntimeException(
            s"${split.file}: ${keyColumnNames(ordinal)} unsupported Parquet page type ${other.getClass}")
      }
      val encoding = dataPage.getValueEncoding
      if (isPlain) {
        if (encoding != Encoding.PLAIN) {
          throw new RuntimeException(
            s"${split.file}: ${keyColumnNames(ordinal)} expected PLAIN encoding, got $encoding")
        }
      } else if (!DictionaryEncodings.contains(encoding)) {
        throw new RuntimeException(
          s"${split.file}: ${keyColumnNames(ordinal)} unsupported value encoding $encoding")
      }
      val pageBytes = dataPage.getBytes.toByteArray
      val valueCount = dataPage.getValueCount
      NativeBridge.parquetDecodePage(
        streamHandle, rowGroupHandle, ordinal, pageBytes, pageBytes.length, valueCount, rowOffset, hasDefLevels,
        isPlain)
      onPageDecoded()
      rowOffset += valueCount
      rawPage = pageReader.readPage()
    }
    if (rowOffset != split.rowCount) {
      throw new RuntimeException(
        s"${split.file}: ${keyColumnNames(ordinal)} pages covered $rowOffset rows, expected ${split.rowCount}")
    }
    (codeTable, factorTable)
  }

  /**
   * Per-(row group, measure slot): stages the dictionary if present (a null
   * dictionary means PLAIN -- both are supported for measures, unlike key
   * columns), then walks and decodes every V1 data page into the measure
   * plane.
   */
  private def decodeMeasureColumn(
      streamHandle: Long,
      rowGroupHandle: Long,
      slot: Int,
      split: ParquetGroupedAggregateSplit,
      pageReadStore: PageReadStore,
      descriptor: ColumnDescriptor,
      onPageDecoded: () => Unit): Unit = {
    val pageReader = pageReadStore.getPageReader(descriptor)
    val dictionaryPage = pageReader.readDictionaryPage()
    if (dictionaryPage != null) {
      NativeBridge.parquetSetMeasureDictionary(rowGroupHandle, slot, decodeDictionary(dictionaryPage))
    }

    val hasDefLevels = descriptor.getMaxDefinitionLevel > 0
    var rowOffset = 0
    var rawPage = pageReader.readPage()
    while (rawPage != null) {
      val dataPage = rawPage match {
        case v1: DataPageV1 => v1
        case other =>
          throw new RuntimeException(
            s"${split.file}: ${measureColumnNames(slot)} unsupported Parquet page type ${other.getClass}")
      }
      val encoding = dataPage.getValueEncoding
      if (dictionaryPage != null) {
        if (!DictionaryEncodings.contains(encoding)) {
          throw new RuntimeException(
            s"${split.file}: ${measureColumnNames(slot)} expected dictionary encoding, got $encoding")
        }
      } else if (encoding != Encoding.PLAIN) {
        throw new RuntimeException(
          s"${split.file}: ${measureColumnNames(slot)} expected PLAIN encoding, got $encoding")
      }
      val pageBytes = dataPage.getBytes.toByteArray
      val valueCount = dataPage.getValueCount
      NativeBridge.parquetDecodeMeasurePage(
        streamHandle, rowGroupHandle, slot, pageBytes, pageBytes.length, valueCount, rowOffset, hasDefLevels)
      onPageDecoded()
      rowOffset += valueCount
      rawPage = pageReader.readPage()
    }
    if (rowOffset != split.rowCount) {
      throw new RuntimeException(
        s"${split.file}: ${measureColumnNames(slot)} pages covered $rowOffset rows, expected ${split.rowCount}")
    }
  }

  private def decodeDictionary(dictionaryPage: DictionaryPage): Array[Int] = {
    val bytes = dictionaryPage.getBytes.toByteArray
    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    Array.tabulate(dictionaryPage.getDictionarySize)(entry => buffer.getInt(entry * 4))
  }

  /**
   * CPU fallback for one row group: parquet-mr `ColumnReadStoreImpl` over
   * BOTH the key and measure columns, matching the GPU kernel's semantics
   * exactly (group id = sum of premultiplied per-column codes, factor =
   * product of per-column multiplicities, kind 0/1/2 = count-star/sum/
   * count-column) and accumulating into `buffer` -- the SAME dense
   * `groupCount * internalAggCount` layout the GPU path uses, so a
   * partition's GPU and CPU-fallback contributions can simply be added
   * element-wise. `codesByKey`/`factorsByKey` are [[GroupSpace.Built]]'s
   * VALUE-keyed maps used directly: `getInteger()` already returns the
   * decoded value regardless of encoding, so no dictionary-id translation
   * is needed here (unlike the GPU path, which decodes to dictionary ids
   * and needs a per-row-group translation table).
   */
  private def aggregateRowGroupOnCpu(
      store: PageReadStore,
      schema: MessageType,
      keyDescriptors: Seq[ColumnDescriptor],
      measureDescriptors: Seq[ColumnDescriptor],
      codesByKey: Seq[Map[Int, Int]],
      factorsByKey: Seq[Map[Int, Int]],
      aggKinds: Array[Int],
      aggMeasureSlots: Array[Int],
      buffer: Array[Long]): Unit = {
    val readStore = new ColumnReadStoreImpl(
      store, new DummyRecordConverter(schema).getRootConverter, schema, "")
    val keyReaders = keyDescriptors.map(readStore.getColumnReader)
    val measureReaders = measureDescriptors.map(readStore.getColumnReader)
    val keyMaxDefinitionLevels = keyDescriptors.map(_.getMaxDefinitionLevel)
    val measureMaxDefinitionLevels = measureDescriptors.map(_.getMaxDefinitionLevel)
    val internalAggCount = aggKinds.length

    var row = 0L
    val rows = store.getRowCount
    while (row < rows) {
      var member = true
      var groupId = 0
      var factor = 1L
      for (k <- keyReaders.indices) {
        val reader = keyReaders(k)
        // See countRowGroupOnCpu's comment in MetalParquetMembershipCountExec:
        // "current < max" is the correct null test for BOTH optional and
        // required columns; getInteger() must be called for every defined
        // value even once `member` has gone false, to keep this reader's
        // value stream in sync for the remaining rows.
        if (reader.getCurrentDefinitionLevel < keyMaxDefinitionLevels(k)) {
          member = false
        } else {
          val value = reader.getInteger
          if (member) {
            codesByKey(k).get(value) match {
              case Some(code) => groupId += code
              case None => member = false
            }
            if (member) {
              factorsByKey(k).get(value).foreach(f => factor *= f)
            }
          }
        }
        reader.consume()
      }

      val measureValues = new Array[Long](measureReaders.length)
      val measureIsNull = new Array[Boolean](measureReaders.length)
      for (m <- measureReaders.indices) {
        val reader = measureReaders(m)
        if (reader.getCurrentDefinitionLevel < measureMaxDefinitionLevels(m)) {
          measureIsNull(m) = true
        } else {
          measureValues(m) = reader.getInteger.toLong
        }
        reader.consume()
      }

      if (member) {
        val base = groupId * internalAggCount
        var a = 0
        while (a < internalAggCount) {
          aggKinds(a) match {
            case 0 => buffer(base + a) += factor
            case 1 =>
              val slot = aggMeasureSlots(a)
              if (!measureIsNull(slot)) buffer(base + a) += measureValues(slot) * factor
            case 2 =>
              val slot = aggMeasureSlots(a)
              if (!measureIsNull(slot)) buffer(base + a) += factor
          }
          a += 1
        }
      }
      row += 1
    }
  }

  // -------------------------------------------------------------------
  // Whole-operator CPU fallback: GroupSpace.build itself failed
  // -------------------------------------------------------------------

  /**
   * `GroupSpace.build` failing at runtime (an oversized cross product, or a
   * duplicate join key discovered in an attributed dimension -- the
   * eligibility check in [[GroupedAggregateShape]] is a plan-time estimate,
   * not a guarantee) leaves no dense group-id space to decode into at all.
   * Correctness comes first: every split in every partition is recomputed
   * with a genuine CPU hash-join + hash-aggregate instead, which naturally
   * handles the case `GroupSpace` cannot represent (a duplicate join key
   * fans a fact row out into SEVERAL groups, rather than folding into one
   * dense component).
   *
   * Per dimension, `dimension.rows` (join key -> attribute row, NOT
   * deduplicated) becomes a multimap `key -> Seq[attribute row]`; a fact
   * row's group membership is the Cartesian product of every dimension's
   * matching attribute rows, and each combination's group identity is the
   * VALUE tuple of its attribute columns (matching
   * [[GroupSpace.attributeValue]]'s semantics, so two dimension rows with
   * equal attribute values still collapse into the same output group).
   */
  private def executeWholeOperatorCpuFallback(
      dimensions: Seq[GroupSpace.Dimension],
      dimensionAttributeTypes: Seq[Seq[DataType]],
      aggKinds: Array[Int],
      aggMeasureSlots: Array[Int],
      aggSlotMappings: Seq[AggSlotMapping],
      occupancySlot: Int): RDD[InternalRow] = {
    val numRowGroups = longMetric("numRowGroups")
    val cpuFallbackRowGroups = longMetric("cpuFallbackRowGroups")
    val numOutputRows = longMetric("numOutputRows")
    val internalAggCount = aggKinds.length

    val dimensionMultimaps: Seq[Map[Int, IndexedSeq[InternalRow]]] = dimensions.map { dimension =>
      dimension.rows.groupBy(_._1).view.mapValues(_.map(_._2).toIndexedSeq).toMap
    }

    val splits = enumerateSplits()
    val numPartitions = math.max(1, math.min(splits.length, sparkContext.defaultParallelism))
    val splitRDD = sparkContext.parallelize(splits, numPartitions)

    splitRDD.mapPartitions { splitIterator =>
      // No native/Metal call happens on this path at all -- SparkMetalNative
      // is deliberately NOT initialized here, unlike executeWithGroupSpace.
      val splitsInPartition = splitIterator.toArray
      val readers = mutable.Map.empty[String, ParquetFileReader]

      def readerFor(file: String): ParquetFileReader =
        readers.getOrElseUpdate(file, {
          val opened = ParquetFileReader.open(HadoopInputFile.fromPath(
            new Path(file), MetalParquetGroupedAggregateExec.sharedConfiguration))
          opened.setRequestedSchema(
            allColumnDescriptors(file, opened.getFooter.getFileMetaData.getSchema).asJava)
          opened
        })

      val groupMap = mutable.LinkedHashMap.empty[Seq[Any], (Array[InternalRow], Array[Long])]
      var localRowGroups = 0L
      try {
        splitsInPartition.foreach { split =>
          val reader = readerFor(split.file)
          val schema = reader.getFooter.getFileMetaData.getSchema
          val keyDescriptors = keyColumnNames.map(name => descriptorFor(schema, split.file, name))
          val measureDescriptors = measureColumnNames.map(name => descriptorFor(schema, split.file, name))
          val pageReadStore = reader.readRowGroup(split.rowGroupIndex)
          try {
            aggregateRowGroupWithJoin(
              pageReadStore, schema, keyDescriptors, measureDescriptors,
              dimensionMultimaps, dimensionAttributeTypes, aggKinds, aggMeasureSlots, groupMap)
          } finally {
            pageReadStore.close()
          }
          localRowGroups += 1
        }
      } finally {
        readers.values.foreach(_.close())
        numRowGroups += localRowGroups
        cpuFallbackRowGroups += localRowGroups
      }

      // Every entry in groupMap was created by at least one matching combo,
      // and every combo unconditionally increments the (always kind-0,
      // always-last) occupancy slot -- so every group here is occupied by
      // construction; no separate occupancy filter is needed (unlike the
      // GPU path's dense, pre-sized array, which can hold never-touched
      // groups).
      val toUnsafeRow = outputProjection()
      val outputRows = groupMap.values.map { case (attributeRows, buffer) =>
        toUnsafeRow(buildOutputRow(attributeRows, dimensionAttributeTypes, buffer, 0, aggSlotMappings)).copy()
      }.toArray
      numOutputRows += outputRows.length
      outputRows.iterator
    }
  }

  /**
   * One row group's worth of real hash-join + hash-aggregate: for every
   * row, looks up each dimension's join key in its multimap (a miss or a
   * null key drops the row), takes the Cartesian product of the matching
   * attribute rows across dimensions, and accumulates each combination's
   * internal aggregate slots into `groupMap`, keyed by the VALUE tuple of
   * every dimension's attribute columns concatenated together.
   */
  private def aggregateRowGroupWithJoin(
      store: PageReadStore,
      schema: MessageType,
      keyDescriptors: Seq[ColumnDescriptor],
      measureDescriptors: Seq[ColumnDescriptor],
      dimensionMultimaps: Seq[Map[Int, IndexedSeq[InternalRow]]],
      dimensionAttributeTypes: Seq[Seq[DataType]],
      aggKinds: Array[Int],
      aggMeasureSlots: Array[Int],
      groupMap: mutable.LinkedHashMap[Seq[Any], (Array[InternalRow], Array[Long])]): Unit = {
    val internalAggCount = aggKinds.length
    val readStore = new ColumnReadStoreImpl(
      store, new DummyRecordConverter(schema).getRootConverter, schema, "")
    val keyReaders = keyDescriptors.map(readStore.getColumnReader)
    val measureReaders = measureDescriptors.map(readStore.getColumnReader)
    val keyMaxDefinitionLevels = keyDescriptors.map(_.getMaxDefinitionLevel)
    val measureMaxDefinitionLevels = measureDescriptors.map(_.getMaxDefinitionLevel)
    val dimensionCount = keyReaders.length

    var row = 0L
    val rows = store.getRowCount
    while (row < rows) {
      var member = true
      val candidates = new Array[IndexedSeq[InternalRow]](dimensionCount)
      for (d <- 0 until dimensionCount) {
        val reader = keyReaders(d)
        if (reader.getCurrentDefinitionLevel < keyMaxDefinitionLevels(d)) {
          member = false
        } else {
          val value = reader.getInteger
          if (member) {
            dimensionMultimaps(d).get(value) match {
              case Some(matches) => candidates(d) = matches
              case None => member = false
            }
          }
        }
        reader.consume()
      }

      val measureValues = new Array[Long](measureReaders.length)
      val measureIsNull = new Array[Boolean](measureReaders.length)
      for (m <- measureReaders.indices) {
        val reader = measureReaders(m)
        if (reader.getCurrentDefinitionLevel < measureMaxDefinitionLevels(m)) {
          measureIsNull(m) = true
        } else {
          measureValues(m) = reader.getInteger.toLong
        }
        reader.consume()
      }

      if (member) {
        cartesianProduct(candidates.toIndexedSeq).foreach { combo =>
          val key = combo.indices.flatMap(d => attributeKey(combo(d), dimensionAttributeTypes(d)))
          val entry = groupMap.getOrElseUpdate(key, (combo.toArray, new Array[Long](internalAggCount)))
          val buffer = entry._2
          var a = 0
          while (a < internalAggCount) {
            aggKinds(a) match {
              case 0 => buffer(a) += 1L
              case 1 =>
                val slot = aggMeasureSlots(a)
                if (!measureIsNull(slot)) buffer(a) += measureValues(slot)
              case 2 =>
                val slot = aggMeasureSlots(a)
                if (!measureIsNull(slot)) buffer(a) += 1L
            }
            a += 1
          }
        }
      }
      row += 1
    }
  }

  private def cartesianProduct(lists: IndexedSeq[IndexedSeq[InternalRow]]): Seq[Seq[InternalRow]] =
    lists.foldLeft(Seq(Vector.empty[InternalRow])) { (accumulated, list) =>
      for (prefix <- accumulated; item <- list) yield prefix :+ item
    }

  // -------------------------------------------------------------------
  // Output row construction (shared by both execution paths)
  // -------------------------------------------------------------------

  /**
   * Extracts field `ordinal`'s value out of a dimension's attribute row as
   * a plain, independently comparable value -- `null` for a null field,
   * otherwise the type-appropriate getter. Deliberately mirrors
   * [[GroupSpace.attributeValue]] (which is private to that object): this
   * is the SAME dispatch, needed here both to key the whole-operator CPU
   * fallback's hash map and to read a dimension attribute back out for
   * output-row construction -- ALWAYS with the dimension's OWN attribute
   * type (`dimensionAttributeTypes`), never `outputAttributes`' declared
   * type (see `buildOutputRow`'s doc for why the two must not be
   * conflated).
   */
  private def attributeValue(row: InternalRow, ordinal: Int, dataType: DataType): Any = {
    if (row.isNullAt(ordinal)) {
      null
    } else {
      dataType match {
        case IntegerType => row.getInt(ordinal)
        case LongType => row.getLong(ordinal)
        case ShortType => row.getShort(ordinal)
        case ByteType => row.getByte(ordinal)
        case BooleanType => row.getBoolean(ordinal)
        case DateType => row.getInt(ordinal)
        case StringType => row.getUTF8String(ordinal)
        case decimalType: DecimalType => row.getDecimal(ordinal, decimalType.precision, decimalType.scale)
        case other => throw new IllegalStateException(s"unsupported attribute type reached attributeValue: $other")
      }
    }
  }

  private def attributeKey(row: InternalRow, attributeTypes: Seq[DataType]): Seq[Any] =
    attributeTypes.zipWithIndex.map { case (dataType, ordinal) => attributeValue(row, ordinal, dataType) }

  /**
   * Converts an internal (exact int64) sum accumulator into the value its
   * user-visible output attribute expects. Every eligible sum/avg measure
   * is INT32-native (a plain int32 column, or `UnscaledValue` of a
   * decimal(<=9) column -- see `GroupedAggregateShape.resolveMeasure`), and
   * `DecimalAggregates` rewrites a decimal sum's buffer down to a bare
   * `LongType` (the unscaled value) before this operator would ever see it
   * -- Task 1's probe confirmed "sum-int and sum-decimal partials are
   * bigint" for every eligible TPC-DS query, and Average's own partial sum
   * buffer is always DOUBLE (also observed by Task 1's probe, exact and
   * within double's 2^53 safe range for the magnitudes this tier reaches).
   * `DecimalType` is handled too, defensively, in case a future caller ever
   * passes a genuinely Decimal-typed sum output.
   */
  private def convertSum(value: Long, dataType: DataType): Any = dataType match {
    case LongType => value
    case DoubleType => value.toDouble
    case decimalType: DecimalType => Decimal(value, decimalType.precision, decimalType.scale)
    case other => throw new IllegalStateException(s"unsupported sum output type: $other")
  }

  /**
   * Converts `buildOutputRow`'s `GenericInternalRow` into a genuine
   * `UnsafeRow`. This operator sits directly below a real
   * `ShuffleExchangeExec` once planned into an actual query (Task 6):
   * `UnsafeRowSerializerInstance.writeValue` hard-casts every row it
   * serializes to `UnsafeRow`, so a bare `GenericInternalRow` reaching that
   * boundary throws a `ClassCastException` deep inside the shuffle write
   * path -- invisible to Task 5's direct-construction smoke, which never
   * routed this operator's output through a real shuffle. One projection is
   * built per partition (not per row) and reused; callers must `.copy()`
   * its result before retaining more than one row, since `UnsafeProjection`
   * reuses its own output buffer across calls.
   */
  private def outputProjection(): UnsafeProjection =
    UnsafeProjection.create(outputAttributes.map(_.dataType).toArray)

  /**
   * Builds one partial output row for an occupied group: group-key values
   * (per `groupKeyDimensionIndex`, reading the DIMENSION's own attribute
   * type -- `dimensionAttributeTypes`, NOT `outputAttributes`' declared
   * type -- off the matching dimension's attribute row; a dimension's
   * attribute row was encoded with its own type in `collectDimensions`, and
   * `UnsafeRow.getDecimal(ordinal, precision, scale)` decodes differently
   * depending on the precision/scale it is told, so reading it back with a
   * mismatched type would silently corrupt the value. `outputAttributes` is
   * used for arity only (validated in `doExecute`), never for how to READ a
   * dimension attribute) followed by, per `aggSlotMappings` in order, each
   * aggregate's user-visible buffer value(s) -- one column for sum/count,
   * two (sum, count) for avg's unevaluated partial buffer.
   *
   * CRITICAL: avg's sum component must NEVER be null. Spark's
   * `Average.initialValues` is `(0, 0L)` (zero, not null) and its
   * `mergeExpressions` is a plain, NON-coalescing `Add` -- so a null
   * avg-sum emitted by one partition would poison the merged sum to null
   * across every other partition's real contribution once Spark's own
   * downstream final aggregate merges them, even though this group's true
   * answer is non-null. `sum`'s buffer, by contrast, genuinely starts null
   * and its merge DOES coalesce, so only `sum` keeps the
   * null-iff-non-null-count-is-zero gate; `avg` always emits its exact
   * int64 sum (0 when its paired count is 0, matching `Average`'s initial
   * value exactly).
   *
   * `buffer`/`base` let the same function serve both the GPU path's dense
   * `groupCount * internalAggCount` array (one group's window starts at
   * `base`) and the whole-operator CPU fallback's per-group `Array[Long]`
   * (`base == 0`).
   */
  private def buildOutputRow(
      dimensionAttributeRows: Array[InternalRow],
      dimensionAttributeTypes: Seq[Seq[DataType]],
      buffer: Array[Long],
      base: Int,
      aggSlotMappings: Seq[AggSlotMapping]): InternalRow = {
    val values = new Array[Any](outputAttributes.length)
    var k = 0
    while (k < groupKeyDimensionIndex.length) {
      val (dimensionIndex, attributeIndex) = groupKeyDimensionIndex(k)
      values(k) = attributeValue(
        dimensionAttributeRows(dimensionIndex), attributeIndex,
        dimensionAttributeTypes(dimensionIndex)(attributeIndex))
      k += 1
    }
    var outputIndex = groupKeyDimensionIndex.length
    aggSlotMappings.foreach { mapping =>
      mapping.function match {
        case "count" =>
          values(outputIndex) = buffer(base + mapping.countSlot)
          outputIndex += 1
        case "sum" =>
          val nonNull = buffer(base + mapping.countSlot) > 0L
          values(outputIndex) =
            if (nonNull) convertSum(buffer(base + mapping.sumSlot), outputAttributes(outputIndex).dataType) else null
          outputIndex += 1
        case "avg" =>
          // Unconditional -- see the CRITICAL note above. buffer(sumSlot)
          // is already exactly 0 when the paired count is 0 (an
          // accumulator that was never added to), matching
          // Average.initialValues's (0, 0L) precisely.
          values(outputIndex) = convertSum(buffer(base + mapping.sumSlot), outputAttributes(outputIndex).dataType)
          outputIndex += 1
          values(outputIndex) = buffer(base + mapping.countSlot)
          outputIndex += 1
      }
    }
    new GenericInternalRow(values)
  }

  // -------------------------------------------------------------------
  // Split enumeration and schema plumbing
  // -------------------------------------------------------------------

  private def descriptorFor(schema: MessageType, file: String, name: String): ColumnDescriptor =
    schema.getColumns.asScala.find(_.getPath()(0) == name)
      .getOrElse(throw new RuntimeException(s"$file: missing column $name"))

  private def allColumnDescriptors(file: String, schema: MessageType): Seq[ColumnDescriptor] =
    (keyColumnNames ++ measureColumnNames).distinct.map(name => descriptorFor(schema, file, name))

  private def enumerateSplits(): Seq[ParquetGroupedAggregateSplit] = {
    val configuration = MetalParquetGroupedAggregateExec.sharedConfiguration
    if (files.length <= 1) {
      return files.flatMap(footerSplits(_, configuration))
    }
    val executor = Executors.newFixedThreadPool(math.min(files.length, Runtime.getRuntime.availableProcessors))
    try {
      implicit val context: ExecutionContext = ExecutionContext.fromExecutor(executor)
      Await.result(
        Future.traverse(files)(file => Future(footerSplits(file, configuration))),
        Duration.Inf).flatten
    } finally {
      executor.shutdown()
    }
  }

  private def footerSplits(
      file: String, configuration: Configuration): Seq[ParquetGroupedAggregateSplit] = {
    def read(): Seq[ParquetGroupedAggregateSplit] = {
      val reader = ParquetFileReader.open(HadoopInputFile.fromPath(new Path(file), configuration))
      try {
        reader.getFooter.getBlocks.asScala.zipWithIndex.map { case (block, index) =>
          ParquetGroupedAggregateSplit(file, index, block.getRowCount)
        }.toSeq
      } finally {
        reader.close()
      }
    }
    ParquetEligibility.fileVersion(file) match {
      case None => read()
      case Some(version) =>
        MetalParquetGroupedAggregateExec.splitsByFile.get(version) match {
          case Some(splits) => splits
          case None =>
            val splits = read()
            MetalParquetGroupedAggregateExec.splitsByFile.put(version, splits)
            splits
        }
    }
  }

  override protected def withNewChildrenInternal(newChildren: IndexedSeq[SparkPlan]): SparkPlan =
    copy(keyPlans = newChildren)
}

private[sparkmetal] object MetalParquetGroupedAggregateExec {
  def sharedConfiguration: Configuration = ParquetEligibility.sharedConfiguration

  private[sparkmetal] val splitsByFile =
    new BoundedCache[FileVersion, Seq[ParquetGroupedAggregateSplit]](ParquetEligibility.MaxCachedFiles)

  /**
   * Task 6b: the largest join-key value a PLAIN key column's value-space
   * code table may need to hold, 4,194,303 (4M - 1) -- chosen so ONE table
   * (`(dimMaxKey + 1)` int32s) stays at or under 16MB; the codes table alone
   * is at most 16MB, and the codes+factors PAIR together (a factor table is
   * the SAME `dimMaxKey + 1` size, built only when the dimension has
   * duplicate keys -- see `buildValueSpaceTables`) is at most 32MB per key
   * column. `doExecute`'s domain guard rejects (routes to the
   * whole-operator CPU fallback) any dimension whose actual join-key domain
   * exceeds this, before `GroupSpace.build` even runs.
   */
  private[sparkmetal] val MaxValueSpaceKey: Int = 4194303

  /**
   * Builds one dimension's PLAIN value-space code/factor tables: size
   * `dimMaxKey + 1`, `-1`-filled (code) / `1`-filled (factor) by default, then
   * overwritten at every key value `codesByKey`/`factorsByKey` actually maps
   * -- no min offset (see `NativeBridge.parquetRowGroupAggregate`'s Javadoc):
   * every index below the dimension's smallest join key is left `-1`, exactly
   * like any other non-member slot, rather than treated specially. Called at
   * most once per (partition, key column) by `decodeKeyColumn`, since the
   * result depends only on the dimension's own (fixed, driver-collected)
   * data, not on any particular row group.
   */
  private[sparkmetal] def buildValueSpaceTables(
      dimMaxKey: Int, codesByKey: Map[Int, Int], factorsByKey: Map[Int, Int]): (Array[Int], Array[Int]) = {
    val size = dimMaxKey + 1
    val codes = Array.fill(size)(-1)
    codesByKey.foreach { case (value, code) => if (value >= 0 && value < size) codes(value) = code }
    val factors =
      if (factorsByKey.isEmpty) {
        null
      } else {
        val table = Array.fill(size)(1)
        factorsByKey.foreach { case (value, factor) => if (value >= 0 && value < size) table(value) = factor }
        table
      }
    (codes, factors)
  }

  /**
   * Expands `aggSpecs` into the kernel's flat internal aggregate arrays
   * (binding, per the task brief):
   *  - sum(col)  -> (kind 1 sum(col), kind 2 count(col))
   *  - avg(col)  -> (kind 1 sum(col), kind 2 count(col)) -- same shape as
   *    sum; the FINAL aggregate (downstream of this operator) divides.
   *  - count(1)/count(*) -> kind 0 count-star
   *  - count(col) -> kind 2 count(col)
   *  - PLUS one occupancy counter (kind 0), always appended last.
   *
   * Throws if the total exceeds the kernel's 8-aggregate cap -- a safety
   * net; Task 6 is expected to pre-check this at planning time.
   */
  private[sparkmetal] def buildInternalAggPlan(
      aggSpecs: Seq[GroupedAggregateShape.AggSpec],
      measureColumnNames: Seq[String]): (Array[Int], Array[Int], Seq[AggSlotMapping], Int) = {
    val kinds = mutable.ArrayBuffer.empty[Int]
    val measureSlots = mutable.ArrayBuffer.empty[Int]

    def addSlot(kind: Int, measureSlot: Int): Int = {
      kinds += kind
      measureSlots += measureSlot
      kinds.length - 1
    }

    def measureSlotIndex(attribute: Attribute): Int = {
      val index = measureColumnNames.indexOf(attribute.name)
      require(index >= 0,
        s"measure column ${attribute.name} not found in measureColumnNames $measureColumnNames")
      index
    }

    val mappings = aggSpecs.map { spec =>
      spec.function match {
        case "sum" | "avg" =>
          val measureIndex = spec.input match {
            case GroupedAggregateShape.FactColumn(attribute) => measureSlotIndex(attribute)
            case GroupedAggregateShape.CountStar =>
              throw new IllegalArgumentException(s"${spec.function} aggregate must have a FactColumn input")
          }
          val sumSlot = addSlot(1, measureIndex)
          val countSlot = addSlot(2, measureIndex)
          AggSlotMapping(spec.function, sumSlot, countSlot)
        case "count" =>
          spec.input match {
            case GroupedAggregateShape.CountStar =>
              AggSlotMapping("count", -1, addSlot(0, 0))
            case GroupedAggregateShape.FactColumn(attribute) =>
              AggSlotMapping("count", -1, addSlot(2, measureSlotIndex(attribute)))
          }
        case other =>
          throw new IllegalArgumentException(s"unsupported aggregate function: $other")
      }
    }

    val occupancySlot = addSlot(0, 0)
    require(kinds.length <= 8,
      s"grouped aggregate needs ${kinds.length} internal slots but the kernel caps at 8 -- " +
        "this should have been rejected at planning time")

    (kinds.toArray, measureSlots.toArray, mappings, occupancySlot)
  }
}
