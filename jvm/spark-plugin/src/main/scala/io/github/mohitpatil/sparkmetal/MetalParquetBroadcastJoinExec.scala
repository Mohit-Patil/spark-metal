package io.github.mohitpatil.sparkmetal

import java.nio.{ByteBuffer, ByteOrder}
import java.util.concurrent.Executors

import scala.collection.mutable
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.jdk.CollectionConverters._

import org.apache.hadoop.fs.Path
import org.apache.parquet.column.{ColumnDescriptor, Encoding}
import org.apache.parquet.column.impl.ColumnReadStoreImpl
import org.apache.parquet.column.page.{DataPageV1, DictionaryPage}
import org.apache.parquet.example.DummyRecordConverter
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.hadoop.util.HadoopInputFile
import org.apache.parquet.schema.MessageType

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, BoundReference, UnsafeProjection}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.execution.vectorized.OnHeapColumnVector
import org.apache.spark.sql.types.{
  BooleanType, ByteType, DateType, DecimalType, IntegerType, LongType, ShortType, StringType}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * Columnar-output broadcast-join tier (docs/GPU_BROADCAST_JOIN_SPEC.md):
 * replaces "[Project]? <- inner BroadcastHashJoin x N <- eligible Parquet
 * fact scan" and emits one [[ColumnarBatch]] per Parquet row group -- the
 * surviving fact rows' int32-backed columns plus the matched build-side
 * rows' attributes -- to whatever operator consumes the region.
 *
 * v1 (this class, CPU decode path): per (file, row group), parquet-mr reads
 * the key + fact output columns row by row; a row survives iff every join
 * key is non-null and matches its dimension. Unique-keyed, in-domain
 * dimensions probe a dense value-space `key -> build-row index` table (the
 * same table the GPU path will consume); a dimension with duplicate keys or
 * an out-of-domain key domain routes the whole operator through a multimap
 * probe with Cartesian fan-out (an inner join with duplicate build keys
 * multiplies rows), exactly mirroring the grouped tier's whole-operator CPU
 * fallback philosophy: correctness first, the dense/GPU path only when its
 * preconditions provably hold.
 *
 * Dimension collection reuses the execution-scoped subplan cache (Fix 2,
 * 2026-08-29), and everything derived from this region's keyPlans is cached
 * per (execution id, canonicalized keyPlans) so UNION siblings share it.
 */
case class MetalParquetBroadcastJoinExec(
    outputAttributes: Seq[Attribute],
    columnSources: Seq[BroadcastJoinShape.ColumnSource],
    files: Seq[String],
    keyColumnNames: Seq[String],
    factColumnNames: Seq[String],
    keyPlans: Seq[SparkPlan],
    nativeLibrary: String,
    metalLibrary: String) extends SparkPlan {

  require(keyColumnNames.nonEmpty && keyColumnNames.length <= 4,
    s"MetalParquetBroadcastJoinExec requires 1-4 key columns, got ${keyColumnNames.length}")
  require(keyColumnNames.length == keyPlans.length,
    s"keyColumnNames (${keyColumnNames.length}) and keyPlans (${keyPlans.length}) must match")
  require(factColumnNames.length <= 4,
    s"MetalParquetBroadcastJoinExec supports at most 4 fact output columns, got ${factColumnNames.length}")
  require(columnSources.length == outputAttributes.length,
    s"columnSources (${columnSources.length}) and outputAttributes (${outputAttributes.length}) must match")

  override def children: Seq[SparkPlan] = keyPlans

  override def output: Seq[Attribute] = outputAttributes

  override def supportsColumnar: Boolean = true

  override protected def doExecute(): RDD[InternalRow] =
    throw new UnsupportedOperationException("MetalParquetBroadcastJoinExec is columnar-only")

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "numRowGroups" -> SQLMetrics.createMetric(sparkContext, "number of Parquet row groups"),
    "numPagesDecoded" -> SQLMetrics.createMetric(sparkContext, "number of Parquet data pages decoded"),
    "cpuFallbackRowGroups" -> SQLMetrics.createMetric(sparkContext, "row groups joined on CPU"),
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "numOutputBatches" -> SQLMetrics.createMetric(sparkContext, "number of output batches"),
    "rowGroupReadTime" -> SQLMetrics.createTimingMetric(sparkContext, "row group read time"),
    "decodeParseTime" -> SQLMetrics.createTimingMetric(sparkContext, "page parse and GPU-encode time"),
    "pageSubmitTime" -> SQLMetrics.createTimingMetric(sparkContext, "native page submit time"),
    "metalTime" -> SQLMetrics.createTimingMetric(sparkContext, "Metal decode and wait time"),
    "dimensionTime" -> SQLMetrics.createTimingMetric(sparkContext, "dimension collection time"),
    "splitPlanTime" -> SQLMetrics.createTimingMetric(sparkContext, "split planning time"),
    "outputBuildTime" -> SQLMetrics.createTimingMetric(sparkContext, "output batch build time"),
    "nativeStagingTime" -> SQLMetrics.createTimingMetric(sparkContext, "native staging copy time"),
    "nativeParseTime" -> SQLMetrics.createTimingMetric(sparkContext, "native page parse time"),
    "nativeEncodeTime" -> SQLMetrics.createTimingMetric(sparkContext, "native GPU encode time"))

  // Fact output column index (into factColumnNames) per output position,
  // -1 for dimension-sourced outputs; resolved once, by name.
  private val factIndexByOutput: Array[Int] = columnSources.map {
    case BroadcastJoinShape.FactColumn(attribute) => factColumnNames.indexOf(attribute.name)
    case _: BroadcastJoinShape.DimensionColumn => -1
  }.toArray
  require(columnSources.zip(factIndexByOutput).forall {
    case (BroadcastJoinShape.FactColumn(_), index) => index >= 0
    case _ => true
  }, s"every FactColumn source must appear in factColumnNames $factColumnNames")

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val dimensionTime = longMetric("dimensionTime")
    val splitPlanTime = longMetric("splitPlanTime")
    val numRowGroups = longMetric("numRowGroups")
    val cpuFallbackRowGroups = longMetric("cpuFallbackRowGroups")
    val numOutputRows = longMetric("numOutputRows")
    val numOutputBatches = longMetric("numOutputBatches")
    val rowGroupReadTime = longMetric("rowGroupReadTime")
    val outputBuildTime = longMetric("outputBuildTime")
    val decodeParseTime = longMetric("decodeParseTime")
    val pageSubmitTime = longMetric("pageSubmitTime")
    val metalTime = longMetric("metalTime")
    val numPagesDecoded = longMetric("numPagesDecoded")
    val nativeStagingTime = longMetric("nativeStagingTime")
    val nativeParseTime = longMetric("nativeParseTime")
    val nativeEncodeTime = longMetric("nativeEncodeTime")

    val executionId = Option(sparkContext.getLocalProperty(
      org.apache.spark.sql.execution.SQLExecution.EXECUTION_ID_KEY))
    val regionCacheKey = executionId.map(id => (id, keyPlans.map(_.canonicalized)))
    val cachedRegion = regionCacheKey.flatMap(MetalParquetBroadcastJoinExec.preparedJoinCache.get)
    val prepared = cachedRegion.getOrElse(prepareRegion(executionId))
    if (cachedRegion.isEmpty) {
      dimensionTime += prepared.collectNanos / 1000000
      regionCacheKey.foreach(key =>
        MetalParquetBroadcastJoinExec.preparedJoinCache.put(key, prepared))
    }
    prepared.denseUnavailableReason.foreach { reason =>
      logWarning(s"MetalParquetBroadcastJoinExec: dense probe unavailable ($reason); " +
        "every row group will use the multimap CPU probe")
    }

    val splitPlanStarted = System.nanoTime()
    val splits = enumerateSplits()
    splitPlanTime += (System.nanoTime() - splitPlanStarted) / 1000000
    val numPartitions = math.max(1, math.min(splits.length, sparkContext.defaultParallelism))
    val splitRDD = sparkContext.parallelize(splits, numPartitions)

    val payloadBroadcast = prepared.payload
    val localColumnSources = columnSources
    val localFactIndex = factIndexByOutput
    val localOutput = outputAttributes
    val localKeyNames = keyColumnNames
    val localFactNames = factColumnNames

    val localNativeLibrary = nativeLibrary
    val localMetalLibrary = metalLibrary

    splitRDD.mapPartitions { splitIterator =>
      val payload = payloadBroadcast.value
      val allColumnNames = (localKeyNames ++ localFactNames).distinct
      val columnIndexByName = allColumnNames.zipWithIndex.toMap
      // A key column that is also a fact output column decodes ONCE, through
      // the fact (measure) plane -- which materializes raw values -- and the
      // key probes that same plane by value; keyFactSlot(k) >= 0 names it.
      val keyFactSlot: Array[Int] = localKeyNames.map(localFactNames.indexOf(_)).toArray
      val keyCount = localKeyNames.length
      val factCount = localFactNames.length

      SparkMetalNative.ensureInitialized(localNativeLibrary, localMetalLibrary)
      val preparedHandle = NativeBridge.prepareMembershipCount3(Array(1), Array(1), Array(1))
      val streamHandle = NativeBridge.membershipCount3StreamBegin(preparedHandle)
      var streamClosed = false
      def closeStream(): Unit = if (!streamClosed) {
        streamClosed = true
        try {
          val timers = NativeBridge.parquetStreamTimers(streamHandle)
          nativeStagingTime += timers(0) / 1000000
          nativeParseTime += timers(1) / 1000000
          nativeEncodeTime += timers(2) / 1000000
        } finally {
          NativeBridge.parquetAggregateStreamAbort(streamHandle)
          NativeBridge.releaseMembershipCount3(preparedHandle)
        }
      }
      Option(org.apache.spark.TaskContext.get())
        .foreach(_.addTaskCompletionListener[Unit](_ => closeStream()))

      val DictionaryEncodings: Set[Encoding] = Set(Encoding.PLAIN_DICTIONARY, Encoding.RLE_DICTIONARY)

      // Per-partition compact-output scratch, reused across row groups:
      // allocating (and therefore zeroing) fresh rowCount-sized arrays per
      // row group cost ~60-80MB of memset per row group for outputs that
      // hold only survivors. Safe to reuse because flatMap fully drains one
      // split's iterator before opening the next, and emitted batches copy
      // into their own vectors. Grown to the largest row group seen.
      var scratchRows = 0
      var scratchDimIndices: Array[Array[Int]] = null
      var scratchFactValues: Array[Array[Int]] = null
      var scratchFactNulls: Array[Byte] = null
      def ensureScratch(rowCount: Int): Unit = if (rowCount > scratchRows) {
        scratchRows = rowCount
        scratchDimIndices = Array.tabulate(keyCount)(_ => new Array[Int](rowCount))
        scratchFactValues = Array.tabulate(factCount)(_ => new Array[Int](rowCount))
        scratchFactNulls = new Array[Byte](math.max(1, factCount * rowCount))
      }

      splitIterator.flatMap { split =>
        val readStarted = System.nanoTime()
        val reader = ParquetFileReader.open(HadoopInputFile.fromPath(
          new Path(split.file), MetalParquetGroupedAggregateExec.sharedConfiguration))
        try {
          val schema = reader.getFooter.getFileMetaData.getSchema
          val descriptors = allColumnNames.map(name =>
            schema.getColumns.asScala.find(_.getPath()(0) == name)
              .getOrElse(throw new RuntimeException(s"${split.file}: missing column $name")))
          val descriptorByName = allColumnNames.zip(descriptors).toMap
          reader.setRequestedSchema(descriptors.asJava)
          val store = reader.readRowGroup(split.rowGroupIndex)
          rowGroupReadTime += (System.nanoTime() - readStarted) / 1000000
          numRowGroups += 1

          val rowCount = split.rowCount.toInt
          var rowGroupHandle = 0L
          try {
            // ---- GPU decode: pages into planes on the stream ----
            val parseStarted = System.nanoTime()
            rowGroupHandle = NativeBridge.parquetRowGroupBeginAggregate(
              streamHandle, rowCount, keyCount, factCount)

            // One page walk per DISTINCT column: a column serving as both a
            // join key and a fact output submits each page to BOTH plane
            // types (the pages are read once; only the GPU expansion runs
            // twice). Key planes keep dictionary IDS (the probe/kernel
            // translates via the chunk's own dictionary); fact (measure)
            // planes MATERIALIZE raw values through their staged dictionary.
            val keyDictionaries = new Array[Array[Int]](keyCount)
            for (columnName <- allColumnNames) {
              val keyIndex = localKeyNames.indexOf(columnName)
              val factSlot = localFactNames.indexOf(columnName)
              val descriptor = descriptorByName(columnName)
              val pageReader = store.getPageReader(descriptor)
              val dictionaryPage = pageReader.readDictionaryPage()
              val isPlain = dictionaryPage == null
              val dictionaryValues =
                if (isPlain) null else MetalParquetBroadcastJoinExec.decodeDictionary(dictionaryPage)
              if (keyIndex >= 0) keyDictionaries(keyIndex) = dictionaryValues
              if (factSlot >= 0 && dictionaryValues != null) {
                NativeBridge.parquetSetMeasureDictionary(rowGroupHandle, factSlot, dictionaryValues)
              }
              val hasDefLevels = descriptor.getMaxDefinitionLevel > 0
              var rowOffset = 0
              var rawPage = pageReader.readPage()
              while (rawPage != null) {
                val dataPage = rawPage match {
                  case v1: DataPageV1 => v1
                  case other => throw new RuntimeException(
                    s"${split.file}: $columnName unsupported Parquet page type ${other.getClass}")
                }
                val encoding = dataPage.getValueEncoding
                if (isPlain) {
                  if (encoding != Encoding.PLAIN) throw new RuntimeException(
                    s"${split.file}: $columnName expected PLAIN encoding, got $encoding")
                } else if (!DictionaryEncodings.contains(encoding)) {
                  throw new RuntimeException(
                    s"${split.file}: $columnName unsupported value encoding $encoding")
                }
                val pageBytes = dataPage.getBytes.toByteArray
                val valueCount = dataPage.getValueCount
                val submitStarted = System.nanoTime()
                if (keyIndex >= 0) {
                  NativeBridge.parquetDecodePage(
                    streamHandle, rowGroupHandle, keyIndex, pageBytes, pageBytes.length, valueCount,
                    rowOffset, hasDefLevels, isPlain)
                }
                if (factSlot >= 0) {
                  NativeBridge.parquetDecodeMeasurePage(
                    streamHandle, rowGroupHandle, factSlot, pageBytes, pageBytes.length, valueCount,
                    rowOffset, hasDefLevels)
                }
                pageSubmitTime += (System.nanoTime() - submitStarted) / 1000000
                numPagesDecoded += 1
                rowOffset += valueCount
                rawPage = pageReader.readPage()
              }
              if (rowOffset != rowCount) throw new RuntimeException(
                s"${split.file}: $columnName pages covered $rowOffset rows, expected $rowCount")
            }
            decodeParseTime += (System.nanoTime() - parseStarted) / 1000000

            payload.denseTables match {
              case Some(tables) =>
                // ---- GPU probe + compaction (v1.1): only SURVIVORS come
                // back. Per-chunk translation exactly as the grouped tier:
                // a dictionary chunk's table is re-indexed into that
                // chunk's dictionary-id space; a PLAIN chunk uses the
                // value-space table directly.
                val codesTables: Array[Array[Int]] = Array.tabulate(keyCount) { k =>
                  val table = tables(k)
                  val dictionary = keyDictionaries(k)
                  if (dictionary == null) table
                  else dictionary.map(value =>
                    if (value >= 0 && value < table.length) table(value) else -1)
                }
                val metalStarted = System.nanoTime()
                ensureScratch(rowCount)
                val outDimIndices = scratchDimIndices
                val outFactValues = scratchFactValues
                val outFactNulls = scratchFactNulls
                val survivors = NativeBridge.parquetRowGroupJoinCompact(
                  streamHandle, rowGroupHandle, codesTables,
                  outDimIndices, outFactValues, outFactNulls)
                NativeBridge.parquetRowGroupRelease(rowGroupHandle)
                rowGroupHandle = 0L
                metalTime += (System.nanoTime() - metalStarted) / 1000000

                store.close()
                reader.close()
                new MetalParquetBroadcastJoinExec.CompactedBatchIterator(
                  survivors, rowCount, outDimIndices, outFactValues, outFactNulls,
                  payload.dimensions, localColumnSources, localFactIndex, localOutput,
                  numOutputRows, numOutputBatches, outputBuildTime)

              case None =>
                // ---- Duplicate-key/multimap region: full-plane readback and
                // JVM probe with Cartesian fan-out (the kernel cannot fan
                // out; these regions are rare and stay on this path).
                val metalStarted = System.nanoTime()
                val columnValues = new Array[Array[Int]](allColumnNames.length)
                val columnValidity = new Array[Array[Byte]](allColumnNames.length)
                val columnDictionaries = new Array[Array[Int]](allColumnNames.length)
                for (m <- 0 until factCount) {
                  val position = columnIndexByName(localFactNames(m))
                  val values = new Array[Int](rowCount)
                  val validity = new Array[Byte](rowCount)
                  NativeBridge.parquetRowGroupReadMeasure(
                    streamHandle, rowGroupHandle, m, values, validity)
                  columnValues(position) = values
                  columnValidity(position) = validity
                }
                for (k <- 0 until keyCount if keyFactSlot(k) < 0) {
                  val position = columnIndexByName(localKeyNames(k))
                  val values = new Array[Int](rowCount)
                  val validity = new Array[Byte](rowCount)
                  NativeBridge.parquetRowGroupRead(
                    streamHandle, rowGroupHandle, k, values, validity)
                  columnValues(position) = values
                  columnValidity(position) = validity
                  columnDictionaries(position) = keyDictionaries(k)
                }
                NativeBridge.parquetRowGroupRelease(rowGroupHandle)
                rowGroupHandle = 0L
                metalTime += (System.nanoTime() - metalStarted) / 1000000

                store.close()
                reader.close()
                new MetalParquetBroadcastJoinExec.PlaneProbeBatchIterator(
                  rowCount, allColumnNames.length,
                  localKeyNames.map(columnIndexByName).toArray,
                  localFactNames.map(columnIndexByName).toArray,
                  columnValues, columnValidity, columnDictionaries,
                  payload.dimensions, payload.denseTables, payload.multimaps,
                  localColumnSources, localFactIndex, localOutput,
                  numOutputRows, numOutputBatches, outputBuildTime)
            }
          } catch {
            case e: RuntimeException =>
              // The native decoder rejected a page (or some other runtime
              // surprise) partway through this row group: release the
              // still-alive handle, and recompute the row group on the CPU
              // from a FRESH PageReadStore (pages already consumed by the
              // failed decode cannot be re-read).
              logWarning(s"Row group ${split.file}#${split.rowGroupIndex} fell back to CPU", e)
              if (rowGroupHandle != 0L) {
                NativeBridge.parquetRowGroupRelease(rowGroupHandle)
              }
              store.close()
              cpuFallbackRowGroups += 1
              val freshStore = reader.readRowGroup(split.rowGroupIndex)
              new MetalParquetBroadcastJoinExec.CpuProbeBatchIterator(
                freshStore, reader, schema, allColumnNames, descriptors,
                localKeyNames, localFactNames,
                payload.dimensions, payload.denseTables, payload.multimaps,
                localColumnSources, localFactIndex, localOutput,
                numOutputRows, numOutputBatches, outputBuildTime)
          }
        } catch {
          case error: Throwable =>
            reader.close()
            throw error
        }
      }
    }
  }

  /**
   * Everything derived purely from this region's keyPlans: the collected
   * dimensions, the dense `key -> build-row index` tables (broadcast) when
   * every dimension has unique, in-domain keys, and the generic multimap
   * probe state otherwise. Cached per (execution, canonical keyPlans).
   */
  private def prepareRegion(executionId: Option[String]): MetalParquetBroadcastJoinExec.PreparedJoin = {
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
          val collected: Array[InternalRow] = executionId match {
            case Some(id) =>
              val cacheKey = (id, plan.canonicalized)
              MetalParquetGroupedAggregateExec.collectedSubplanCache.get(cacheKey) match {
                case Some(rows) => rows
                case None =>
                  val rows: Array[InternalRow] = plan.executeCollect().toArray
                  MetalParquetGroupedAggregateExec.collectedSubplanCache.put(cacheKey, rows)
                  rows
              }
            case None => plan.executeCollect().toArray
          }
          val rows = collected.flatMap { row =>
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
    val collectNanos = System.nanoTime() - started

    // Dense tables demand unique keys in [0, MaxValueSpaceKey] for EVERY
    // dimension; the first violation switches the whole region to the
    // multimap probe (an inner join with duplicate build keys must fan out,
    // which the dense table cannot express).
    var denseUnavailable: Option[String] = None
    dimensions.zipWithIndex.foreach { case (dimension, index) =>
      if (denseUnavailable.isEmpty) {
        val seen = mutable.HashSet.empty[Int]
        dimension.rows.foreach { case (key, _) =>
          if (denseUnavailable.isEmpty) {
            if (key < 0 || key > MetalParquetGroupedAggregateExec.MaxValueSpaceKey) {
              denseUnavailable = Some(
                s"dimension ${keyColumnNames(index)} key $key outside " +
                  s"[0, ${MetalParquetGroupedAggregateExec.MaxValueSpaceKey}]")
            } else if (!seen.add(key)) {
              denseUnavailable = Some(s"dimension ${keyColumnNames(index)} has duplicate key $key")
            }
          }
        }
        if (dimension.rows.isEmpty && denseUnavailable.isEmpty) {
          denseUnavailable = Some(s"dimension ${keyColumnNames(index)} is empty")
        }
      }
    }

    val denseTables: Option[Array[Array[Int]]] =
      if (denseUnavailable.isDefined) None
      else Some(dimensions.map { dimension =>
        val maxKey = dimension.rows.iterator.map(_._1).max
        val table = Array.fill(maxKey + 1)(-1)
        dimension.rows.iterator.zipWithIndex.foreach { case ((key, _), rowIndex) =>
          table(key) = rowIndex
        }
        table
      }.toArray)
    val multimaps: Seq[Map[Int, Array[Int]]] =
      if (denseUnavailable.isEmpty) Seq.empty
      else dimensions.map { dimension =>
        val grouped = mutable.HashMap.empty[Int, mutable.ArrayBuffer[Int]]
        dimension.rows.iterator.zipWithIndex.foreach { case ((key, _), rowIndex) =>
          grouped.getOrElseUpdate(key, mutable.ArrayBuffer.empty) += rowIndex
        }
        grouped.view.mapValues(_.toArray).toMap
      }

    val payload = sparkContext.broadcast(
      MetalParquetBroadcastJoinExec.JoinProbePayload(dimensions, denseTables, multimaps))
    MetalParquetBroadcastJoinExec.PreparedJoin(payload, collectNanos, denseUnavailable)
  }

  private[sparkmetal] def enumerateSplits(): Seq[ParquetGroupedAggregateSplit] = {
    val configuration = MetalParquetGroupedAggregateExec.sharedConfiguration
    def footerSplits(file: String): Seq[ParquetGroupedAggregateSplit] = {
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
    if (files.length <= 1) {
      return files.flatMap(footerSplits)
    }
    val executor = Executors.newFixedThreadPool(
      math.min(files.length, Runtime.getRuntime.availableProcessors))
    try {
      implicit val context: ExecutionContext = ExecutionContext.fromExecutor(executor)
      Await.result(Future.traverse(files)(file => Future(footerSplits(file))), Duration.Inf).flatten
    } finally {
      executor.shutdown()
    }
  }

  override protected def withNewChildrenInternal(newChildren: IndexedSeq[SparkPlan]): SparkPlan =
    copy(keyPlans = newChildren)
}

private[sparkmetal] object MetalParquetBroadcastJoinExec {

  /**
   * Everything the executor-side probe needs, shipped as ONE broadcast so
   * concurrent tasks share a single deserialized copy -- closure-captured
   * dimensions were deserialized once PER TASK, and eight copies of a
   * string-heavy dimension (item: ~204k rows) stacked on an already-heavy
   * query's own buffers OOMed the 6g heap (q23b).
   */
  private[sparkmetal] case class JoinProbePayload(
      dimensions: Seq[GroupSpace.Dimension],
      denseTables: Option[Array[Array[Int]]],
      multimaps: Seq[Map[Int, Array[Int]]])

  private[sparkmetal] case class PreparedJoin(
      payload: Broadcast[JoinProbePayload],
      collectNanos: Long,
      denseUnavailableReason: Option[String])

  private[sparkmetal] val preparedJoinCache =
    new BoundedCache[(String, Seq[SparkPlan]), PreparedJoin](32)

  /**
   * Gathers one dimension-attribute output column: for batch rows
   * 0..count-1, reads `indices(sourceOffset + i)`'s build row and writes its
   * attribute `ordinal` (or a null) into the vector — one type dispatch per
   * COLUMN, tight monomorphic loops per case. Shared by the probe iterators
   * and the compacted-survivor iterator.
   */
  private def gatherDimensionColumn(
      vector: OnHeapColumnVector,
      dimension: GroupSpace.Dimension,
      ordinal: Int,
      indices: Array[Int],
      sourceOffset: Int,
      count: Int): Unit = {
    val dimensionRows = dimension.rows
    def loop(write: (Int, InternalRow) => Unit): Unit = {
      var i = 0
      while (i < count) {
        val attributeRow = dimensionRows(indices(sourceOffset + i))._2
        if (attributeRow.isNullAt(ordinal)) vector.putNull(i) else write(i, attributeRow)
        i += 1
      }
    }
    dimension.attributeTypes(ordinal) match {
      case IntegerType | DateType => loop((i, r) => vector.putInt(i, r.getInt(ordinal)))
      case LongType => loop((i, r) => vector.putLong(i, r.getLong(ordinal)))
      case ShortType => loop((i, r) => vector.putShort(i, r.getShort(ordinal)))
      case ByteType => loop((i, r) => vector.putByte(i, r.getByte(ordinal)))
      case BooleanType => loop((i, r) => vector.putBoolean(i, r.getBoolean(ordinal)))
      case StringType => loop { (i, r) =>
        val bytes = r.getUTF8String(ordinal).getBytes
        vector.putByteArray(i, bytes, 0, bytes.length)
      }
      case decimal: DecimalType => loop((i, r) =>
        vector.putDecimal(i, r.getDecimal(ordinal, decimal.precision, decimal.scale), decimal.precision))
      case other =>
        throw new IllegalStateException(s"unsupported dimension attribute type $other")
    }
  }

  /**
   * Emits bounded batches from fused_join_compact's survivor arrays: fact
   * values/null flags come straight from the compacted readback
   * (`factNulls` is fact-major with `nullStride` — the row-group row count
   * — matching the kernel's out_fact_nulls layout); dimension attributes
   * gather by compacted build-row index. No probing happens here — the GPU
   * already did it.
   */
  private[sparkmetal] final class CompactedBatchIterator(
      survivors: Int,
      nullStride: Int,
      dimIndices: Array[Array[Int]],
      factValues: Array[Array[Int]],
      factNulls: Array[Byte],
      dimensions: Seq[GroupSpace.Dimension],
      columnSources: Seq[BroadcastJoinShape.ColumnSource],
      factIndexByOutput: Array[Int],
      outputAttributes: Seq[Attribute],
      numOutputRows: SQLMetric,
      numOutputBatches: SQLMetric,
      outputBuildTime: SQLMetric) extends Iterator[ColumnarBatch] {

    private val BatchRows = 32768
    private var offset = 0

    override def hasNext: Boolean = offset < survivors

    override def next(): ColumnarBatch = {
      val buildStarted = System.nanoTime()
      val count = math.min(BatchRows, survivors - offset)
      val vectors: Array[OnHeapColumnVector] = outputAttributes.map { attribute =>
        new OnHeapColumnVector(math.max(1, count), attribute.dataType)
      }.toArray
      var position = 0
      while (position < vectors.length) {
        val factSlot = factIndexByOutput(position)
        if (factSlot >= 0) {
          val values = factValues(factSlot)
          val nullBase = factSlot * nullStride + offset
          val vector = vectors(position)
          var i = 0
          while (i < count) {
            if (factNulls(nullBase + i) != 0) vector.putNull(i)
            else vector.putInt(i, values(offset + i))
            i += 1
          }
        } else {
          columnSources(position) match {
            case BroadcastJoinShape.DimensionColumn(dimensionIndex, ordinal) =>
              gatherDimensionColumn(
                vectors(position), dimensions(dimensionIndex), ordinal,
                dimIndices(dimensionIndex), offset, count)
            case other =>
              throw new IllegalStateException(s"unexpected column source $other")
          }
        }
        position += 1
      }
      offset += count
      numOutputRows += count
      numOutputBatches += 1
      outputBuildTime += (System.nanoTime() - buildStarted) / 1000000
      new ColumnarBatch(vectors.map(vector => vector: ColumnVector), count)
    }
  }

  private[sparkmetal] def decodeDictionary(dictionaryPage: DictionaryPage): Array[Int] = {
    val bytes = dictionaryPage.getBytes.toByteArray
    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    Array.tabulate(dictionaryPage.getDictionarySize)(entry => buffer.getInt(entry * 4))
  }

  /**
   * Probe of one row group as an iterator of BOUNDED batches (~32k output
   * rows each): per fact row, either the dense tables (unique keys: at most
   * one match per dimension) or the multimaps (duplicate keys: Cartesian
   * fan-out across dimensions -- an inner join with duplicate build keys
   * multiplies rows) decide survival; survivors append into
   * [[OnHeapColumnVector]]s -- fact values by int, dimension attributes by
   * type from the matched build row. Subclasses supply the per-row column
   * values (`readRow`): parquet-mr readers (CPU decode) or decoded GPU
   * planes.
   *
   * Bounding matters: a whole-row-group batch of a low-selectivity join
   * over a multi-million-row row group (inventory joined to item's string
   * attributes, q22) OOMed a 6g heap across 8 concurrent tasks. The
   * iterator closes its resources when the row group is exhausted; a
   * task-completion listener covers early termination (LIMIT).
   */
  private[sparkmetal] abstract class ProbeBatchIterator(
      totalRows: Long,
      allColumnCount: Int,
      keyReaderIndex: Array[Int],
      factReaderIndex: Array[Int],
      dimensions: Seq[GroupSpace.Dimension],
      denseTables: Option[Array[Array[Int]]],
      multimaps: Seq[Map[Int, Array[Int]]],
      columnSources: Seq[BroadcastJoinShape.ColumnSource],
      factIndexByOutput: Array[Int],
      outputAttributes: Seq[Attribute],
      numOutputRows: SQLMetric,
      numOutputBatches: SQLMetric,
      outputBuildTime: SQLMetric) extends Iterator[ColumnarBatch] {

    private val BatchRows = 32768

    /** Fill currentValues/currentNull for row `rowIndex` (RAW column values). */
    protected def readRow(rowIndex: Long): Unit

    /** Release the subclass's resources; called exactly once. */
    protected def onClose(): Unit

    protected val currentValues = new Array[Int](allColumnCount)
    protected val currentNull = new Array[Boolean](allColumnCount)
    private val keyCount = keyReaderIndex.length
    private val factCount = factReaderIndex.length
    private val matchedRows = new Array[Int](keyCount)
    private var row = 0L
    private var closed = false

    // Per-output-column emit plan, resolved ONCE: emitting through a
    // per-value pattern match on columnSources/attributeTypes made
    // outputBuildTime the dominant cost of every low-selectivity region
    // (q22: 7.2s of a 10.7s query for 26.5M rows). outFactSlot(p) >= 0
    // names a fact slot; otherwise outDimIndex/outDimOrdinal name the
    // matched dimension attribute.
    private val outFactSlot: Array[Int] = factIndexByOutput
    private val outDimIndex = new Array[Int](outputAttributes.length)
    private val outDimOrdinal = new Array[Int](outputAttributes.length)
    columnSources.zipWithIndex.foreach {
      case (BroadcastJoinShape.DimensionColumn(dimensionIndex, attributeOrdinal), position) =>
        outDimIndex(position) = dimensionIndex
        outDimOrdinal(position) = attributeOrdinal
      case _ => ()
    }

    // Survivor capture buffers (column-major): the probe writes fact values
    // and matched dimension row indices here; emission then runs one tight,
    // monomorphic loop per OUTPUT column. Grown geometrically for multimap
    // fan-out overshoot.
    private var captureCapacity = BatchRows + 1024
    private var capturedFactValues = Array.ofDim[Int](math.max(1, factCount), captureCapacity)
    private var capturedFactNulls = Array.ofDim[Boolean](math.max(1, factCount), captureCapacity)
    private var capturedDimRows = Array.ofDim[Int](keyCount, captureCapacity)
    private var captured = 0

    private def capture(): Unit = {
      if (captured == captureCapacity) {
        val grown = captureCapacity * 2
        capturedFactValues = capturedFactValues.map(java.util.Arrays.copyOf(_, grown))
        capturedFactNulls = capturedFactNulls.map(java.util.Arrays.copyOf(_, grown))
        capturedDimRows = capturedDimRows.map(java.util.Arrays.copyOf(_, grown))
        captureCapacity = grown
      }
      var f = 0
      while (f < factCount) {
        val readerIndex = factReaderIndex(f)
        capturedFactValues(f)(captured) = currentValues(readerIndex)
        capturedFactNulls(f)(captured) = currentNull(readerIndex)
        f += 1
      }
      var k = 0
      while (k < keyCount) {
        capturedDimRows(k)(captured) = matchedRows(k)
        k += 1
      }
      captured += 1
    }

    private def emitColumn(position: Int, vector: OnHeapColumnVector, count: Int): Unit = {
      val factSlot = outFactSlot(position)
      if (factSlot >= 0) {
        val values = capturedFactValues(factSlot)
        val nulls = capturedFactNulls(factSlot)
        var i = 0
        while (i < count) {
          if (nulls(i)) vector.putNull(i) else vector.putInt(i, values(i))
          i += 1
        }
      } else {
        gatherDimensionColumn(
          vector, dimensions(outDimIndex(position)), outDimOrdinal(position),
          capturedDimRows(outDimIndex(position)), 0, count)
      }
    }

    Option(org.apache.spark.TaskContext.get())
      .foreach(_.addTaskCompletionListener[Unit](_ => close()))

    // Cannot run in this constructor -- the subclass's fields (readers,
    // planes) are not initialized yet when this body executes. Subclasses
    // call it at the end of their own constructor.
    protected def closeIfEmpty(): Unit = if (totalRows == 0) close()

    override def hasNext: Boolean = !closed && row < totalRows

    override def next(): ColumnarBatch = {
      val buildStarted = System.nanoTime()
      captured = 0
      while (row < totalRows && captured < BatchRows) {
        readRow(row)

        denseTables match {
          case Some(tables) =>
            var member = true
            var k = 0
            while (member && k < keyCount) {
              val readerIndex = keyReaderIndex(k)
              if (currentNull(readerIndex)) {
                member = false
              } else {
                val value = currentValues(readerIndex)
                val table = tables(k)
                val matched = if (value >= 0 && value < table.length) table(value) else -1
                if (matched < 0) member = false else matchedRows(k) = matched
              }
              k += 1
            }
            if (member) capture()
          case None =>
            // Multimap probe. The whole fan-out of one fact row lands in one
            // batch (the odometer is not interruptible), so a batch may
            // overshoot BatchRows by one row's fan-out -- bounded by the
            // dimensions' duplicate multiplicities, not the row group size.
            var member = true
            val matches = new Array[Array[Int]](keyCount)
            var k = 0
            while (member && k < keyCount) {
              val readerIndex = keyReaderIndex(k)
              if (currentNull(readerIndex)) {
                member = false
              } else {
                multimaps(k).get(currentValues(readerIndex)) match {
                  case Some(rowIndices) => matches(k) = rowIndices
                  case None => member = false
                }
              }
              k += 1
            }
            if (member) {
              val counters = new Array[Int](keyCount)
              var exhausted = false
              while (!exhausted) {
                var i = 0
                while (i < keyCount) {
                  matchedRows(i) = matches(i)(counters(i))
                  i += 1
                }
                capture()
                var position = keyCount - 1
                var carrying = true
                while (carrying && position >= 0) {
                  counters(position) += 1
                  if (counters(position) < matches(position).length) {
                    carrying = false
                  } else {
                    counters(position) = 0
                    position -= 1
                  }
                }
                exhausted = carrying
              }
            }
        }
        row += 1
      }
      if (row >= totalRows) close()

      val outputRow = captured
      val vectors: Array[OnHeapColumnVector] = outputAttributes.map { attribute =>
        new OnHeapColumnVector(math.max(1, outputRow), attribute.dataType)
      }.toArray
      var position = 0
      while (position < vectors.length) {
        emitColumn(position, vectors(position), outputRow)
        position += 1
      }
      numOutputRows += outputRow
      numOutputBatches += 1
      outputBuildTime += (System.nanoTime() - buildStarted) / 1000000
      new ColumnarBatch(vectors.map(vector => vector: ColumnVector), outputRow)
    }

    private def close(): Unit = if (!closed) {
      closed = true
      onClose()
    }
  }

  /** CPU decode: parquet-mr column readers over the distinct key + fact columns. */
  private[sparkmetal] final class CpuProbeBatchIterator(
      store: org.apache.parquet.column.page.PageReadStore,
      reader: ParquetFileReader,
      schema: MessageType,
      allColumnNames: Seq[String],
      descriptors: Seq[ColumnDescriptor],
      keyColumnNames: Seq[String],
      factColumnNames: Seq[String],
      dimensions: Seq[GroupSpace.Dimension],
      denseTables: Option[Array[Array[Int]]],
      multimaps: Seq[Map[Int, Array[Int]]],
      columnSources: Seq[BroadcastJoinShape.ColumnSource],
      factIndexByOutput: Array[Int],
      outputAttributes: Seq[Attribute],
      numOutputRows: SQLMetric,
      numOutputBatches: SQLMetric,
      outputBuildTime: SQLMetric)
    extends ProbeBatchIterator(
      store.getRowCount, allColumnNames.length,
      keyColumnNames.map(allColumnNames.zipWithIndex.toMap).toArray,
      factColumnNames.map(allColumnNames.zipWithIndex.toMap).toArray,
      dimensions, denseTables, multimaps, columnSources, factIndexByOutput, outputAttributes,
      numOutputRows, numOutputBatches, outputBuildTime) {

    private val readStore = new ColumnReadStoreImpl(
      store, new DummyRecordConverter(schema).getRootConverter, schema, "")
    private val readers = descriptors.map(readStore.getColumnReader).toArray
    private val maxDefinitionLevels = descriptors.map(_.getMaxDefinitionLevel).toArray

    closeIfEmpty()

    override protected def readRow(rowIndex: Long): Unit = {
      var column = 0
      while (column < readers.length) {
        val columnReader = readers(column)
        if (columnReader.getCurrentDefinitionLevel < maxDefinitionLevels(column)) {
          currentNull(column) = true
          currentValues(column) = 0
        } else {
          currentNull(column) = false
          currentValues(column) = columnReader.getInteger
        }
        columnReader.consume()
        column += 1
      }
    }

    override protected def onClose(): Unit = {
      store.close()
      reader.close()
    }
  }

  /**
   * GPU decode: reads the row group's decoded planes back from the stream
   * (blocking) and probes them. `columnValues(c)` holds column c's plane in
   * `allColumnNames` order; `columnDictionaries(c)` translates a
   * dictionary-decoded KEY plane's ids to raw values (null when the plane
   * already holds raw values -- PLAIN key chunks and all fact planes, which
   * materialize through their dictionary natively).
   */
  private[sparkmetal] final class PlaneProbeBatchIterator(
      totalRows: Int,
      allColumnCount: Int,
      keyReaderIndex: Array[Int],
      factReaderIndex: Array[Int],
      columnValues: Array[Array[Int]],
      columnValidity: Array[Array[Byte]],
      columnDictionaries: Array[Array[Int]],
      dimensions: Seq[GroupSpace.Dimension],
      denseTables: Option[Array[Array[Int]]],
      multimaps: Seq[Map[Int, Array[Int]]],
      columnSources: Seq[BroadcastJoinShape.ColumnSource],
      factIndexByOutput: Array[Int],
      outputAttributes: Seq[Attribute],
      numOutputRows: SQLMetric,
      numOutputBatches: SQLMetric,
      outputBuildTime: SQLMetric)
    extends ProbeBatchIterator(
      totalRows, allColumnCount, keyReaderIndex, factReaderIndex,
      dimensions, denseTables, multimaps, columnSources, factIndexByOutput, outputAttributes,
      numOutputRows, numOutputBatches, outputBuildTime) {

    closeIfEmpty()

    override protected def readRow(rowIndex: Long): Unit = {
      val index = rowIndex.toInt
      var column = 0
      while (column < allColumnCount) {
        // Plane validity convention (see scatter_segments and
        // ParquetDecodeSmoke): the zero-filled plane means VALID; the decode
        // writes 1 to mark a NULL row.
        if (columnValidity(column)(index) != 0) {
          currentNull(column) = true
          currentValues(column) = 0
        } else {
          currentNull(column) = false
          val raw = columnValues(column)(index)
          val dictionary = columnDictionaries(column)
          currentValues(column) =
            if (dictionary == null) raw
            else if (raw >= 0 && raw < dictionary.length) dictionary(raw)
            // A dictionary id outside its own dictionary means a corrupt
            // plane; surface it rather than fabricate a value.
            else throw new IllegalStateException(
              s"dictionary id $raw outside dictionary of ${dictionary.length}")
        }
        column += 1
      }
    }

    override protected def onClose(): Unit = ()
  }

}
