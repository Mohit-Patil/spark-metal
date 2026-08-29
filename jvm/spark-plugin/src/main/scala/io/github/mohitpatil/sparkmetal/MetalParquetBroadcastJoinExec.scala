package io.github.mohitpatil.sparkmetal

import java.util.concurrent.Executors

import scala.collection.mutable
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.jdk.CollectionConverters._

import org.apache.hadoop.fs.Path
import org.apache.parquet.column.ColumnDescriptor
import org.apache.parquet.column.impl.ColumnReadStoreImpl
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

    val dimensions = prepared.dimensions
    val denseTables = prepared.denseTables // Option[Broadcast[...]]
    val multimaps = prepared.multimaps
    val localColumnSources = columnSources
    val localFactIndex = factIndexByOutput
    val localOutput = outputAttributes
    val localKeyNames = keyColumnNames
    val localFactNames = factColumnNames

    splitRDD.mapPartitions { splitIterator =>
      splitIterator.flatMap { split =>
        val readStarted = System.nanoTime()
        val reader = ParquetFileReader.open(HadoopInputFile.fromPath(
          new Path(split.file), MetalParquetGroupedAggregateExec.sharedConfiguration))
        try {
          val schema = reader.getFooter.getFileMetaData.getSchema
          val allColumnNames = (localKeyNames ++ localFactNames).distinct
          val descriptors = allColumnNames.map(name =>
            schema.getColumns.asScala.find(_.getPath()(0) == name)
              .getOrElse(throw new RuntimeException(s"${split.file}: missing column $name")))
          reader.setRequestedSchema(descriptors.asJava)
          val store = reader.readRowGroup(split.rowGroupIndex)
          rowGroupReadTime += (System.nanoTime() - readStarted) / 1000000
          numRowGroups += 1
          cpuFallbackRowGroups += 1
          // Bounded batches: the iterator owns store+reader and closes them
          // when the row group is exhausted (a task-completion listener
          // covers early termination, e.g. a LIMIT upstream).
          new MetalParquetBroadcastJoinExec.CpuProbeBatchIterator(
            store, reader, schema, allColumnNames, descriptors, localKeyNames, localFactNames,
            dimensions, denseTables.map(_.value), multimaps,
            localColumnSources, localFactIndex, localOutput,
            numOutputRows, numOutputBatches, outputBuildTime)
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

    val denseTables: Option[Broadcast[Array[Array[Int]]]] =
      if (denseUnavailable.isDefined) None
      else Some(sparkContext.broadcast(dimensions.map { dimension =>
        val maxKey = dimension.rows.iterator.map(_._1).max
        val table = Array.fill(maxKey + 1)(-1)
        dimension.rows.iterator.zipWithIndex.foreach { case ((key, _), rowIndex) =>
          table(key) = rowIndex
        }
        table
      }.toArray))
    val multimaps: Seq[Map[Int, Array[Int]]] =
      if (denseUnavailable.isEmpty) Seq.empty
      else dimensions.map { dimension =>
        val grouped = mutable.HashMap.empty[Int, mutable.ArrayBuffer[Int]]
        dimension.rows.iterator.zipWithIndex.foreach { case ((key, _), rowIndex) =>
          grouped.getOrElseUpdate(key, mutable.ArrayBuffer.empty) += rowIndex
        }
        grouped.view.mapValues(_.toArray).toMap
      }

    MetalParquetBroadcastJoinExec.PreparedJoin(
      dimensions, collectNanos, denseTables, multimaps, denseUnavailable)
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

  private[sparkmetal] case class PreparedJoin(
      dimensions: Seq[GroupSpace.Dimension],
      collectNanos: Long,
      denseTables: Option[Broadcast[Array[Array[Int]]]],
      multimaps: Seq[Map[Int, Array[Int]]],
      denseUnavailableReason: Option[String])

  private[sparkmetal] val preparedJoinCache =
    new BoundedCache[(String, Seq[SparkPlan]), PreparedJoin](32)

  /**
   * CPU probe of one row group as an iterator of BOUNDED batches (~128k
   * output rows each): parquet-mr readers over the DISTINCT key + fact
   * columns (a column serving as both key and output is read once); per
   * row, either the dense tables (unique keys: at most one match per
   * dimension) or the multimaps (duplicate keys: Cartesian fan-out across
   * dimensions) decide survival, and survivors append into
   * [[OnHeapColumnVector]]s -- fact values by int, dimension attributes by
   * type from the matched build row.
   *
   * Bounding matters: a whole-row-group batch of a low-selectivity join
   * over a multi-million-row row group (inventory joined to item's string
   * attributes, q22) OOMed a 6g heap across 8 concurrent tasks. The
   * iterator owns `store` and `reader`, closing both when the row group is
   * exhausted; a task-completion listener covers early termination (LIMIT).
   */
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
      outputBuildTime: SQLMetric) extends Iterator[ColumnarBatch] {

    private val BatchRows = 131072

    private val readStore = new ColumnReadStoreImpl(
      store, new DummyRecordConverter(schema).getRootConverter, schema, "")
    private val readers = descriptors.map(readStore.getColumnReader).toArray
    private val maxDefinitionLevels = descriptors.map(_.getMaxDefinitionLevel).toArray
    private val columnIndexByName = allColumnNames.zipWithIndex.toMap
    private val keyReaderIndex = keyColumnNames.map(columnIndexByName).toArray
    private val factReaderIndex = factColumnNames.map(columnIndexByName).toArray
    private val keyCount = keyColumnNames.length

    private val currentValues = new Array[Int](allColumnNames.length)
    private val currentNull = new Array[Boolean](allColumnNames.length)
    private val matchedRows = new Array[Int](keyCount)
    private var row = 0L
    private val rows = store.getRowCount
    private var closed = false

    Option(org.apache.spark.TaskContext.get())
      .foreach(_.addTaskCompletionListener[Unit](_ => close()))
    if (rows == 0) close()

    override def hasNext: Boolean = !closed && row < rows

    override def next(): ColumnarBatch = {
      val buildStarted = System.nanoTime()
      val vectors: Array[OnHeapColumnVector] = outputAttributes.map { attribute =>
        new OnHeapColumnVector(math.min(BatchRows, 65536), attribute.dataType)
      }.toArray
      var outputRow = 0
      while (row < rows && outputRow < BatchRows) {
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
            if (member) {
              reserveCapacity(vectors, outputRow + 1)
              writeOutputRow(
                vectors, outputRow, columnSources, factIndexByOutput, factReaderIndex,
                currentValues, currentNull, dimensions, matchedRows, outputAttributes)
              outputRow += 1
            }
          case None =>
            // Multimap probe with Cartesian fan-out across dimensions. The
            // whole fan-out of one fact row lands in one batch (the odometer
            // is not interruptible), so a batch may overshoot BatchRows by
            // one row's fan-out -- bounded by the dimensions' duplicate
            // multiplicities, not by the row group size.
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
                reserveCapacity(vectors, outputRow + 1)
                writeOutputRow(
                  vectors, outputRow, columnSources, factIndexByOutput, factReaderIndex,
                  currentValues, currentNull, dimensions, matchedRows, outputAttributes)
                outputRow += 1
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
      if (row >= rows) close()
      numOutputRows += outputRow
      numOutputBatches += 1
      outputBuildTime += (System.nanoTime() - buildStarted) / 1000000
      new ColumnarBatch(vectors.map(vector => vector: ColumnVector), outputRow)
    }

    private def close(): Unit = if (!closed) {
      closed = true
      store.close()
      reader.close()
    }
  }

  private def reserveCapacity(vectors: Array[OnHeapColumnVector], needed: Int): Unit = {
    var i = 0
    while (i < vectors.length) {
      vectors(i).reserve(needed)
      i += 1
    }
  }

  private def writeOutputRow(
      vectors: Array[OnHeapColumnVector],
      outputRow: Int,
      columnSources: Seq[BroadcastJoinShape.ColumnSource],
      factIndexByOutput: Array[Int],
      factReaderIndex: Array[Int],
      currentValues: Array[Int],
      currentNull: Array[Boolean],
      dimensions: Seq[GroupSpace.Dimension],
      matchedRows: Array[Int],
      outputAttributes: Seq[Attribute]): Unit = {
    var position = 0
    while (position < columnSources.length) {
      val vector = vectors(position)
      columnSources(position) match {
        case BroadcastJoinShape.FactColumn(_) =>
          val readerIndex = factReaderIndex(factIndexByOutput(position))
          if (currentNull(readerIndex)) vector.putNull(outputRow)
          else vector.putInt(outputRow, currentValues(readerIndex))
        case BroadcastJoinShape.DimensionColumn(dimensionIndex, attributeOrdinal) =>
          val dimension = dimensions(dimensionIndex)
          val attributeRow = dimension.rows(matchedRows(dimensionIndex))._2
          if (attributeRow.isNullAt(attributeOrdinal)) {
            vector.putNull(outputRow)
          } else {
            dimension.attributeTypes(attributeOrdinal) match {
              case IntegerType | DateType => vector.putInt(outputRow, attributeRow.getInt(attributeOrdinal))
              case LongType => vector.putLong(outputRow, attributeRow.getLong(attributeOrdinal))
              case ShortType => vector.putShort(outputRow, attributeRow.getShort(attributeOrdinal))
              case ByteType => vector.putByte(outputRow, attributeRow.getByte(attributeOrdinal))
              case BooleanType => vector.putBoolean(outputRow, attributeRow.getBoolean(attributeOrdinal))
              case StringType =>
                val bytes = attributeRow.getUTF8String(attributeOrdinal).getBytes
                vector.putByteArray(outputRow, bytes, 0, bytes.length)
              case decimal: DecimalType =>
                vector.putDecimal(outputRow,
                  attributeRow.getDecimal(attributeOrdinal, decimal.precision, decimal.scale),
                  decimal.precision)
              case other =>
                throw new IllegalStateException(s"unsupported dimension attribute type $other")
            }
          }
      }
      position += 1
    }
  }
}
