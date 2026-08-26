package io.github.mohitpatil.sparkmetal

import java.nio.{ByteBuffer, ByteOrder}
import java.util.UUID
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
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.execution.vectorized.OnHeapColumnVector
import org.apache.spark.sql.types.LongType
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/** One (file, row-group) unit of work, planned on the driver. */
private[sparkmetal] case class ParquetMembershipSplit(
    file: String, rowGroupIndex: Int, rowCount: Long)

/**
 * Drives the GPU Parquet decode + membership count (Tasks 3/4) per
 * (file, row group), falling back to a CPU parquet-mr count for any row
 * group whose pages the native decoder rejects. Planned by
 * [[SparkMetalColumnarRule]] in place of [[MetalFusedMembershipCountExec]]
 * whenever the fact-side scan is an eligible bucketless, unpartitioned
 * Parquet [[org.apache.spark.sql.execution.FileSourceScanExec]] (Task 6);
 * also driven directly by ParquetDecodeSmoke's "exec" mode for
 * operator-level testing.
 */
case class MetalParquetMembershipCountExec(
    outputAttribute: Attribute,
    files: Seq[String],
    columnNames: Seq[String],
    keyPlans: Seq[SparkPlan],
    nativeLibrary: String,
    metalLibrary: String) extends SparkPlan {

  require(columnNames.length == 3, "MetalParquetMembershipCountExec requires exactly 3 key columns")
  require(keyPlans.length == 3, "MetalParquetMembershipCountExec requires exactly 3 key plans")

  private val DictionaryEncodings: Set[Encoding] = Set(Encoding.PLAIN_DICTIONARY, Encoding.RLE_DICTIONARY)

  override def children: Seq[SparkPlan] = keyPlans

  override def output: Seq[Attribute] = Seq(outputAttribute)

  override def supportsColumnar: Boolean = true

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "numRowGroups" -> SQLMetrics.createMetric(sparkContext, "number of Parquet row groups"),
    "numPagesDecoded" -> SQLMetrics.createMetric(sparkContext, "number of Parquet data pages decoded"),
    "cpuFallbackRowGroups" -> SQLMetrics.createMetric(sparkContext, "row groups counted on CPU"),
    "decodeParseTime" -> SQLMetrics.createTimingMetric(sparkContext, "page parse and GPU-encode time"),
    "rowGroupReadTime" -> SQLMetrics.createTimingMetric(sparkContext, "Parquet row-group read and decompress time"),
    "partitionTime" -> SQLMetrics.createTimingMetric(sparkContext, "total in-task time"),
    "splitPlanTime" -> SQLMetrics.createTimingMetric(sparkContext, "driver-side split enumeration time"),
    "pageSubmitTime" -> SQLMetrics.createTimingMetric(sparkContext, "native page-submit time"),
    "metalTime" -> SQLMetrics.createTimingMetric(sparkContext, "Metal count and final wait time"),
    "dimensionTime" -> SQLMetrics.createTimingMetric(sparkContext, "dimension key collection time"),
    "membershipBuildTime" -> SQLMetrics.createTimingMetric(sparkContext, "membership map build time"))

  override protected def doExecute(): RDD[InternalRow] =
    throw new UnsupportedOperationException("MetalParquetMembershipCountExec is columnar-only")

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

    val numRowGroups = longMetric("numRowGroups")
    val numPagesDecoded = longMetric("numPagesDecoded")
    val cpuFallbackRowGroups = longMetric("cpuFallbackRowGroups")
    val decodeParseTime = longMetric("decodeParseTime")
    val rowGroupReadTime = longMetric("rowGroupReadTime")
    val pageSubmitTime = longMetric("pageSubmitTime")
    val metalTime = longMetric("metalTime")
    val membershipBuildTime = longMetric("membershipBuildTime")

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

    val splitPlanTime = longMetric("splitPlanTime")
    val partitionTime = longMetric("partitionTime")
    val splitPlanStarted = System.nanoTime()
    val splits = enumerateSplits()
    splitPlanTime += (System.nanoTime() - splitPlanStarted) / 1000000
    val numPartitions = math.max(1, math.min(splits.length, sparkContext.defaultParallelism))
    val splitRDD = sparkContext.parallelize(splits, numPartitions)
    val prepareToken = UUID.randomUUID().toString
    val prepareUses = splitRDD.getNumPartitions

    splitRDD.mapPartitions { splitIterator =>
      val partitionStarted = System.nanoTime()
      SparkMetalNative.ensureInitialized(nativeLibrary, metalLibrary)
      val splitsInPartition = splitIterator.toArray
      val multiplicities = keys.map(_.groupMapReduce(identity)(_ => 1L)(_ + _))
      val allKeysUnique = multiplicities.forall(_.valuesIterator.forall(_ == 1L))
      val preparedHandle = if (denseDomains) {
        PreparedMembershipCache.acquire(prepareToken, keys, prepareUses, membershipBuildTime)
      } else 0L

      var partitionCount = 0L
      var metalNanos = 0L
      var parseNanos = 0L
      var readNanos = 0L
      val submitNanos = new Array[Long](1)
      var localRowGroups = 0L
      var localPages = 0L
      var localFallbacks = 0L
      val readers = mutable.Map.empty[String, ParquetFileReader]

      def readerFor(file: String): ParquetFileReader =
        readers.getOrElseUpdate(file, {
          val opened = ParquetFileReader.open(HadoopInputFile.fromPath(
            new Path(file), MetalParquetMembershipCountExec.sharedConfiguration))
          // Without this, readRowGroup pulls EVERY column chunk of the row
          // group off disk -- all 23 columns of store_sales when this operator
          // reads 3 -- because a reader opened without a projection requests
          // the whole file schema. On TPC-DS SF10 that was ~88ms per task of
          // pure wasted I/O against a ~200ms CPU query.
          opened.setRequestedSchema(columnDescriptors(
            file, opened.getFooter.getFileMetaData.getSchema).asJava)
          opened
        })

      try {
        val streamHandle =
          if (preparedHandle != 0L) NativeBridge.membershipCount3StreamBegin(preparedHandle) else 0L
        var streamFinished = streamHandle == 0L
        try {
          splitsInPartition.foreach { split =>
            val reader = readerFor(split.file)
            val schema = reader.getFooter.getFileMetaData.getSchema
            val descriptors = columnDescriptors(split.file, schema)

            if (streamHandle != 0L) {
              var rowGroupHandle = 0L
              try {
                val parseStarted = System.nanoTime()
                val pageReadStore = reader.readRowGroup(split.rowGroupIndex)
                readNanos += System.nanoTime() - parseStarted
                try {
                  rowGroupHandle = NativeBridge.parquetRowGroupBegin(streamHandle, split.rowCount.toInt)
                  val tables = descriptors.zipWithIndex.map { case (descriptor, ordinal) =>
                    decodeRowGroupColumn(
                      streamHandle, rowGroupHandle, ordinal, split, pageReadStore, descriptor,
                      multiplicities(ordinal), allKeysUnique, () => localPages += 1, submitNanos)
                  }
                  parseNanos += System.nanoTime() - parseStarted

                  val metalStarted = System.nanoTime()
                  NativeBridge.parquetRowGroupCount(
                    streamHandle, rowGroupHandle,
                    tables(0)._1, tables(0)._2,
                    tables(1)._1, tables(1)._2,
                    tables(2)._1, tables(2)._2)
                  metalNanos += System.nanoTime() - metalStarted
                  localRowGroups += 1
                } finally {
                  pageReadStore.close()
                }
              } catch {
                case e: RuntimeException =>
                  // The native decoder rejected a page (or some other
                  // runtime surprise occurred) partway through this row
                  // group. Release whatever handle exists -- never Release
                  // after a successful Count, never touch pages already
                  // consumed by the failed GPU pass -- and recompute this
                  // row group's contribution on the CPU from a fresh
                  // PageReadStore.
                  logWarning(s"Row group ${split.file}#${split.rowGroupIndex} fell back to CPU", e)
                  if (rowGroupHandle != 0L) {
                    NativeBridge.parquetRowGroupRelease(rowGroupHandle)
                  }
                  localFallbacks += 1
                  localRowGroups += 1
                  val freshStore = reader.readRowGroup(split.rowGroupIndex)
                  try {
                    partitionCount += countRowGroupOnCpu(freshStore, schema, descriptors, multiplicities)
                  } finally {
                    freshStore.close()
                  }
              }
            } else {
              // Dimension keys are not dense enough for the prepared GPU
              // path at all: count every row group on the CPU.
              val pageReadStore = reader.readRowGroup(split.rowGroupIndex)
              localFallbacks += 1
              localRowGroups += 1
              try {
                partitionCount += countRowGroupOnCpu(pageReadStore, schema, descriptors, multiplicities)
              } finally {
                pageReadStore.close()
              }
            }
          }

          if (streamHandle != 0L) {
            val started = System.nanoTime()
            // Set the flag before calling Finish: Finish deletes the native
            // stream even when it throws, so the finally block's Abort must
            // never fire afterward -- it would touch an already-deleted
            // stream.
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
        readers.values.foreach(_.close())
        // Recorded here with the rest, so a task that fails partway still
        // reports the time it spent rather than silently contributing nothing.
        partitionTime += (System.nanoTime() - partitionStarted) / 1000000
        metalTime += metalNanos / 1000000
        decodeParseTime += parseNanos / 1000000
        rowGroupReadTime += readNanos / 1000000
        pageSubmitTime += submitNanos(0) / 1000000
        numRowGroups += localRowGroups
        numPagesDecoded += localPages
        cpuFallbackRowGroups += localFallbacks
        if (preparedHandle != 0L) {
          PreparedMembershipCache.release(prepareToken)
        }
      }

      val vector = new OnHeapColumnVector(1, LongType)
      vector.putLong(0, partitionCount)
      Iterator.single(new ColumnarBatch(Array[ColumnVector](vector), 1))
    }
  }

  /**
   * Reads one column-chunk's dictionary page, builds its presence or
   * multiplicity table, then walks and decodes every V1 data page for this
   * (row group, column) into the native stream. Returns the (presence,
   * multiplicity) pair for parquetRowGroupCount -- exactly one side
   * populated, matching the native contract.
   */
  private def decodeRowGroupColumn(
      streamHandle: Long,
      rowGroupHandle: Long,
      ordinal: Int,
      split: ParquetMembershipSplit,
      pageReadStore: PageReadStore,
      descriptor: ColumnDescriptor,
      keys: Map[Int, Long],
      allKeysUnique: Boolean,
      onPageDecoded: () => Unit,
      submitNanos: Array[Long]): (Array[Byte], Array[Int]) = {
    val pageReader = pageReadStore.getPageReader(descriptor)
    val dictionaryPage = pageReader.readDictionaryPage()
    if (dictionaryPage == null) {
      throw new RuntimeException(s"${split.file}: ${columnNames(ordinal)} has no dictionary page")
    }
    val dictionaryValues = decodeDictionary(dictionaryPage)
    val hasDefLevels = descriptor.getMaxDefinitionLevel > 0

    var rowOffset = 0
    var rawPage = pageReader.readPage()
    while (rawPage != null) {
      val dataPage = rawPage match {
        case v1: DataPageV1 => v1
        case other =>
          throw new RuntimeException(
            s"${split.file}: ${columnNames(ordinal)} unsupported Parquet page type ${other.getClass}")
      }
      if (!DictionaryEncodings.contains(dataPage.getValueEncoding)) {
        throw new RuntimeException(
          s"${split.file}: ${columnNames(ordinal)} unsupported value encoding ${dataPage.getValueEncoding}")
      }
      val pageBytes = dataPage.getBytes.toByteArray
      val valueCount = dataPage.getValueCount
      val submitStarted = System.nanoTime()
      NativeBridge.parquetDecodePage(
        streamHandle, rowGroupHandle, ordinal,
        pageBytes, pageBytes.length, valueCount, rowOffset, hasDefLevels)
      submitNanos(0) += System.nanoTime() - submitStarted
      onPageDecoded()
      rowOffset += valueCount
      rawPage = pageReader.readPage()
    }
    if (rowOffset != split.rowCount) {
      throw new RuntimeException(
        s"${split.file}: ${columnNames(ordinal)} pages covered $rowOffset rows, expected ${split.rowCount}")
    }

    if (allKeysUnique) {
      (MembershipTables.presence(dictionaryValues, keys), null)
    } else {
      (null, MembershipTables.multiplicity(dictionaryValues, keys))
    }
  }

  private def decodeDictionary(dictionaryPage: DictionaryPage): Array[Int] = {
    val bytes = dictionaryPage.getBytes.toByteArray
    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    val values = new Array[Int](dictionaryPage.getDictionarySize)
    var entry = 0
    while (entry < values.length) {
      values(entry) = buffer.getInt(entry * 4)
      entry += 1
    }
    values
  }

  /**
   * CPU fallback: per-row triple definition-level check + multiplicity
   * product, matching the GPU kernel's semantics exactly. store must be a
   * fresh PageReadStore -- pages already consumed by a failed GPU pass
   * cannot be re-read.
   */
  private def countRowGroupOnCpu(
      store: PageReadStore,
      schema: MessageType,
      descriptors: Seq[ColumnDescriptor],
      multiplicities: Seq[Map[Int, Long]]): Long = {
    val readStore = new ColumnReadStoreImpl(
      store, new DummyRecordConverter(schema).getRootConverter, schema, "")
    val readers = descriptors.map(readStore.getColumnReader)
    // A `required` column (maxDefinitionLevel == 0 -- eligibility admits
    // this, not just optional maxDef == 1 columns) has NO definition-level
    // stream at all: getCurrentDefinitionLevel() always returns 0 for it,
    // which is also the "null" sentinel for an *optional* column. Comparing
    // against the hardcoded literal 0 therefore misclassifies every row of
    // a required column as null, zeroing the count silently. The correct
    // null test is "current < max": for an optional column (max 1) that is
    // still exactly the 0-vs-1 check; for a required column (max 0) it is
    // never true, since getCurrentDefinitionLevel() can't exceed 0 either.
    val maxDefinitionLevels = descriptors.map(_.getMaxDefinitionLevel)
    var count = 0L
    var row = 0L
    val rows = store.getRowCount
    while (row < rows) {
      var product = 1L
      var member = true
      readers.zipWithIndex.foreach { case (reader, ordinal) =>
        // getInteger() must be called for every defined (non-null) value on
        // every column, even once `member` has already gone false for this
        // row: it is what advances that column's dictionary-values reader,
        // not consume() (which only advances the repetition/definition
        // level state). Skipping it once member is false -- as the task
        // brief's own sketch does -- leaves that reader's value stream
        // desynced by one slot for every remaining row in the row group,
        // producing silently wrong values (and wrong counts) from that
        // point on. Caught by the sparse-cpu-fallback TDD case in
        // ParquetDecodeSmoke's exec mode: 386 vs. an expected 329.
        if (reader.getCurrentDefinitionLevel < maxDefinitionLevels(ordinal)) {
          member = false
        } else {
          val value = reader.getInteger
          if (member) {
            val m = multiplicities(ordinal).getOrElse(value, 0L)
            if (m == 0L) member = false else product *= m
          }
        }
        reader.consume()
      }
      if (member) count += product
      row += 1
    }
    count
  }

  /**
   * Reads every input file's footer to enumerate (file, row group) work units.
   * Both the sharing of one [[Configuration]] and the parallel fan-out matter:
   * a fresh `new Configuration()` per file re-walks Hadoop's default resource
   * list, and the footer reads are pure I/O latency. Serially with a
   * per-file Configuration this cost 35ms of driver time per execution on the
   * 33.5M-row synthetic benchmark, against a ~190ms total query.
   */
  private def enumerateSplits(): Seq[ParquetMembershipSplit] = {
    val configuration = MetalParquetMembershipCountExec.sharedConfiguration
    if (files.length <= 1) {
      return files.flatMap(footerSplits(_, configuration))
    }
    val executor = Executors.newFixedThreadPool(
      math.min(files.length, Runtime.getRuntime.availableProcessors))
    try {
      implicit val context: ExecutionContext = ExecutionContext.fromExecutor(executor)
      Await.result(
        Future.traverse(files)(file => Future(footerSplits(file, configuration))),
        Duration.Inf).flatten
    } finally {
      executor.shutdown()
    }
  }

  /** Resolves this operator's three key columns in one file's schema. */
  private def columnDescriptors(file: String, schema: MessageType): Seq[ColumnDescriptor] =
    columnNames.map { name =>
      schema.getColumns.asScala.find(_.getPath()(0) == name)
        .getOrElse(throw new RuntimeException(s"$file: missing column $name"))
    }

  /**
   * One file's row groups, memoised across executions and invalidated the same
   * way as the eligibility verdict -- planning re-reads these footers for every
   * execution of every query over the same files. Keyed by file version alone,
   * unlike the eligibility verdict: a block's index and row count are properties
   * of the row group, not of any column, so they are the same whichever columns
   * a query asks for.
   */
  private def footerSplits(
      file: String, configuration: Configuration): Seq[ParquetMembershipSplit] = {
    def read(): Seq[ParquetMembershipSplit] = {
      val reader = ParquetFileReader.open(HadoopInputFile.fromPath(new Path(file), configuration))
      try {
        reader.getFooter.getBlocks.asScala.zipWithIndex.map { case (block, index) =>
          ParquetMembershipSplit(file, index, block.getRowCount)
        }.toSeq
      } finally {
        reader.close()
      }
    }
    ParquetEligibility.fileVersion(file) match {
      case None => read()
      case Some(version) =>
        MetalParquetMembershipCountExec.splitsByFile.get(version) match {
          case Some(splits) => splits
          case None =>
            val splits = read()
            MetalParquetMembershipCountExec.splitsByFile.put(version, splits)
            splits
        }
    }
  }

  override protected def withNewChildrenInternal(newChildren: IndexedSeq[SparkPlan]): SparkPlan =
    copy(keyPlans = newChildren)
}

private[sparkmetal] object MetalParquetMembershipCountExec {
  /**
   * One Hadoop [[Configuration]] per JVM, for the driver's footer scan and for
   * each task's [[ParquetFileReader]]s. Constructing one is not free -- it
   * re-reads Hadoop's default resource list -- and this operator would
   * otherwise build a fresh one per file per execution. Configuration is not
   * thread-safe for mutation, but nothing here mutates it, and parquet-mr only
   * reads from it.
   */
  def sharedConfiguration: Configuration = ParquetEligibility.sharedConfiguration

  /**
   * Row-group enumeration per file version, bounded and typed. See
   * [[MetalParquetMembershipCountExec.footerSplits]] for why the file version
   * alone is a sufficient key here.
   */
  private[sparkmetal] val splitsByFile =
    new BoundedCache[FileVersion, Seq[ParquetMembershipSplit]](ParquetEligibility.MaxCachedFiles)
}
