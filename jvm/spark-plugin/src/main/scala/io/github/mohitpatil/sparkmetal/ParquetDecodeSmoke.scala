package io.github.mohitpatil.sparkmetal

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.Files

import scala.jdk.CollectionConverters._

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.parquet.column.Encoding
import org.apache.parquet.column.page.DataPageV1
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.hadoop.util.HadoopInputFile

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.types.LongType

// Round-trips a dictionary-encoded Parquet file through the CPU page parser
// and the GPU expansion kernels, then checks the decoded (dictionary id,
// validity) planes against the same file read through Spark's own Parquet
// reader. See task-3-brief.md Step 1.
object ParquetDecodeSmoke {
  private val Columns = Seq("k0", "k1", "k2")
  private val DictionaryEncodings = Set(Encoding.PLAIN_DICTIONARY, Encoding.RLE_DICTIONARY)

  def main(arguments: Array[String]): Unit = {
    if (arguments.nonEmpty && arguments(0) == "exec") {
      require(arguments.length >= 3,
        "Usage: ParquetDecodeSmoke exec NATIVE_LIB METAL_LIB [ROWS]")
      val rows = if (arguments.length > 3) arguments(3).toInt else 100003
      runExecMode(arguments(1), arguments(2), rows)
    } else {
      runDecodeAndCountMode(arguments)
    }
  }

  private def runDecodeAndCountMode(arguments: Array[String]): Unit = {
    require(arguments.length >= 2, "Usage: ParquetDecodeSmoke NATIVE_LIB METAL_LIB [ROWS]")
    val nativeLibrary = arguments(0)
    val metalLibrary = arguments(1)
    val rows = if (arguments.length > 2) arguments(2).toInt else 100003

    val spark = SparkSession.builder().appName("spark-metal-parquet-decode-smoke").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    val tempDir = Files.createTempDirectory("spark-metal-parquet-decode-smoke")
    val path = tempDir.resolve("data").toString
    try {
      writeDataset(spark, path, rows)
      val partFile = Files.list(java.nio.file.Path.of(path)).iterator().asScala
        .map(_.toString)
        .find(_.endsWith(".parquet"))
        .getOrElse(throw new RuntimeException(s"No .parquet part file found under $path"))

      SparkMetalNative.ensureInitialized(nativeLibrary, metalLibrary)
      val preparedHandle = NativeBridge.prepareMembershipCount3(Array(1), Array(1), Array(1))

      val decodedValues = Array.fill(Columns.length)(new Array[Int](rows))
      val decodedIsNull = Array.fill(Columns.length)(new Array[Boolean](rows))

      var coveredRows = 0
      var fullyNullPages = 0
      val reader = ParquetFileReader.open(HadoopInputFile.fromPath(new Path(partFile), new Configuration()))
      try {
        val descriptors = Columns.map { name =>
          reader.getFooter.getFileMetaData.getSchema.getColumns.asScala
            .find(_.getPath()(0) == name)
            .getOrElse(throw new RuntimeException(s"Missing column $name"))
        }

        var pageReadStore = reader.readNextRowGroup()
        while (pageReadStore != null) {
          val rowGroupRows = pageReadStore.getRowCount.toInt
          val streamHandle = NativeBridge.membershipCount3StreamBegin(preparedHandle)
          val rowGroupHandle = NativeBridge.parquetRowGroupBegin(streamHandle, rowGroupRows)
          try {
            for (columnIndex <- Columns.indices) {
              val descriptor = descriptors(columnIndex)
              val pageReader = pageReadStore.getPageReader(descriptor)
              val dictionaryPage = pageReader.readDictionaryPage()
              if (dictionaryPage == null) {
                throw new RuntimeException(
                  s"${Columns(columnIndex)}: expected a dictionary page but found none")
              }
              val dictionaryBytes = dictionaryPage.getBytes.toByteArray
              val dictionaryBuffer = ByteBuffer.wrap(dictionaryBytes).order(ByteOrder.LITTLE_ENDIAN)
              val dictionaryValues = new Array[Int](dictionaryPage.getDictionarySize)
              for (entry <- dictionaryValues.indices) {
                dictionaryValues(entry) = dictionaryBuffer.getInt(entry * 4)
              }

              val hasDefLevels = descriptor.getMaxDefinitionLevel > 0

              // Tracks each page's (rowStart, rowCount) within the row group
              // so we can later detect a page that decoded to all-null rows
              // (exercising the entirely-null-page path in parquetDecodePage).
              val pageRanges = scala.collection.mutable.ArrayBuffer.empty[(Int, Int)]

              var rowOffset = 0
              var rawPage = pageReader.readPage()
              while (rawPage != null) {
                val dataPage = rawPage match {
                  case v1: DataPageV1 => v1
                  case other =>
                    throw new RuntimeException(
                      s"${Columns(columnIndex)}: unsupported Parquet page type ${other.getClass}")
                }
                if (!DictionaryEncodings.contains(dataPage.getValueEncoding)) {
                  throw new RuntimeException(
                    s"${Columns(columnIndex)}: unsupported value encoding ${dataPage.getValueEncoding}")
                }
                val pageBytes = dataPage.getBytes.toByteArray
                val valueCount = dataPage.getValueCount
                pageRanges += ((rowOffset, valueCount))

                NativeBridge.parquetDecodePage(
                  streamHandle, rowGroupHandle, columnIndex,
                  pageBytes, pageBytes.length, valueCount, rowOffset, hasDefLevels)

                rowOffset += valueCount
                rawPage = pageReader.readPage()
              }
              if (rowOffset != rowGroupRows) {
                throw new RuntimeException(
                  s"${Columns(columnIndex)}: pages covered $rowOffset rows, expected $rowGroupRows")
              }

              val idsOut = new Array[Int](rowGroupRows)
              val validityOut = new Array[Byte](rowGroupRows)
              NativeBridge.parquetRowGroupRead(streamHandle, rowGroupHandle, columnIndex, idsOut, validityOut)

              for (row <- 0 until rowGroupRows) {
                val globalRow = coveredRows + row
                if (validityOut(row) == 0) {
                  decodedValues(columnIndex)(globalRow) = dictionaryValues(idsOut(row))
                } else {
                  decodedIsNull(columnIndex)(globalRow) = true
                }
              }

              for ((pageStart, pageCount) <- pageRanges) {
                if (pageCount > 0 &&
                    (pageStart until pageStart + pageCount).forall(row => validityOut(row) != 0)) {
                  fullyNullPages += 1
                }
              }
            }
          } finally {
            NativeBridge.parquetRowGroupRelease(rowGroupHandle)
            NativeBridge.membershipCount3StreamAbort(streamHandle)
          }
          coveredRows += rowGroupRows
          pageReadStore = reader.readNextRowGroup()
        }
      } finally {
        reader.close()
      }
      NativeBridge.releaseMembershipCount3(preparedHandle)
      require(coveredRows == rows, s"Row groups covered $coveredRows rows, expected $rows")

      val referenceRows = spark.read.parquet(path).select("k0", "k1", "k2").collect()
      require(referenceRows.length == rows,
        s"Reference row count ${referenceRows.length} did not match $rows")

      var mismatches = 0
      for (row <- 0 until rows) {
        val sparkRow = referenceRows(row)
        for (columnIndex <- Columns.indices) {
          val expectedNull = sparkRow.isNullAt(columnIndex)
          val actualNull = decodedIsNull(columnIndex)(row)
          if (expectedNull != actualNull) {
            mismatches += 1
          } else if (!expectedNull && sparkRow.getInt(columnIndex) != decodedValues(columnIndex)(row)) {
            mismatches += 1
          }
        }
      }

      val matched = mismatches == 0
      val coveredFullyNullPage = fullyNullPages > 0
      println(
        s"""{"rows":$rows,"mismatches":$mismatches,"match":$matched,""" +
          s""""fullyNullPages":$fullyNullPages}""")
      if (!matched) sys.exit(1)
      if (!coveredFullyNullPage) {
        throw new RuntimeException(
          "No fully-null Parquet data page was observed; the entirely-null-page " +
            "path in parquetDecodePage was not exercised by this run")
      }

      // Task 4: fuse the row-group decode with the GPU membership count.
      // Member set is values 50..199 (inclusive) in every column; a row
      // qualifies iff all three keys are non-null and within that range.
      // The Task 3 comparison pass above already Read+Released its row
      // groups, so verifying the fused count requires a second decode pass
      // into fresh row-group handles.
      //
      // memberLow is deliberately 50, not 100: dictionary entry 0 (the id
      // that zero-filled null rows carry on the ids plane -- see
      // parquetRowGroupBegin) decodes to 51 in every column here, since the
      // first non-null value each column's writer encounters is id=1's
      // (id=0 is always null: id % 17 == 0). A member set starting above 51
      // would make presence[0] == 0, so a null row leaking past a broken
      // nullMask would be silently excluded by the presence table anyway --
      // masking exactly the bug this check exists to catch. With 51 in the
      // member set, presence[0] == 1, so leaked null rows inflate
      // actualCount above expectedCount (which excludes them via
      // sparkRow.isNullAt) and the check below fails.
      val memberLow = 50
      val memberHigh = 199
      var expectedCount = 0L
      for (row <- 0 until rows) {
        val sparkRow = referenceRows(row)
        val qualifies = Columns.indices.forall { columnIndex =>
          !sparkRow.isNullAt(columnIndex) && {
            val value = sparkRow.getInt(columnIndex)
            value >= memberLow && value <= memberHigh
          }
        }
        if (qualifies) expectedCount += 1
      }

      val countPreparedHandle = NativeBridge.prepareMembershipCount3(Array(1), Array(1), Array(1))
      val countStreamHandle = NativeBridge.membershipCount3StreamBegin(countPreparedHandle)
      var actualCount = 0L
      var countStreamFinished = false
      try {
        val countReader = ParquetFileReader.open(HadoopInputFile.fromPath(new Path(partFile), new Configuration()))
        try {
          val descriptors = Columns.map { name =>
            countReader.getFooter.getFileMetaData.getSchema.getColumns.asScala
              .find(_.getPath()(0) == name)
              .getOrElse(throw new RuntimeException(s"Missing column $name"))
          }
          var pageReadStore = countReader.readNextRowGroup()
          while (pageReadStore != null) {
            val rowGroupRows = pageReadStore.getRowCount.toInt
            val rowGroupHandle = NativeBridge.parquetRowGroupBegin(countStreamHandle, rowGroupRows)
            val presenceTables = new Array[Array[Byte]](Columns.length)
            for (columnIndex <- Columns.indices) {
              val descriptor = descriptors(columnIndex)
              val pageReader = pageReadStore.getPageReader(descriptor)
              val dictionaryPage = pageReader.readDictionaryPage()
              if (dictionaryPage == null) {
                throw new RuntimeException(
                  s"${Columns(columnIndex)}: expected a dictionary page but found none")
              }
              val dictionaryBytes = dictionaryPage.getBytes.toByteArray
              val dictionaryBuffer = ByteBuffer.wrap(dictionaryBytes).order(ByteOrder.LITTLE_ENDIAN)
              val dictionarySize = dictionaryPage.getDictionarySize
              val presence = new Array[Byte](dictionarySize)
              for (entry <- 0 until dictionarySize) {
                val value = dictionaryBuffer.getInt(entry * 4)
                presence(entry) = if (value >= memberLow && value <= memberHigh) 1.toByte else 0.toByte
              }
              presenceTables(columnIndex) = presence

              val hasDefLevels = descriptor.getMaxDefinitionLevel > 0
              var rowOffset = 0
              var rawPage = pageReader.readPage()
              while (rawPage != null) {
                val dataPage = rawPage match {
                  case v1: DataPageV1 => v1
                  case other =>
                    throw new RuntimeException(
                      s"${Columns(columnIndex)}: unsupported Parquet page type ${other.getClass}")
                }
                if (!DictionaryEncodings.contains(dataPage.getValueEncoding)) {
                  throw new RuntimeException(
                    s"${Columns(columnIndex)}: unsupported value encoding ${dataPage.getValueEncoding}")
                }
                val pageBytes = dataPage.getBytes.toByteArray
                val valueCount = dataPage.getValueCount
                NativeBridge.parquetDecodePage(
                  countStreamHandle, rowGroupHandle, columnIndex,
                  pageBytes, pageBytes.length, valueCount, rowOffset, hasDefLevels)
                rowOffset += valueCount
                rawPage = pageReader.readPage()
              }
              if (rowOffset != rowGroupRows) {
                throw new RuntimeException(
                  s"${Columns(columnIndex)}: pages covered $rowOffset rows, expected $rowGroupRows")
              }
            }
            NativeBridge.parquetRowGroupCount(
              countStreamHandle, rowGroupHandle,
              presenceTables(0), null, presenceTables(1), null, presenceTables(2), null)
            pageReadStore = countReader.readNextRowGroup()
          }
        } finally {
          countReader.close()
        }
        actualCount = NativeBridge.membershipCount3StreamFinish(countStreamHandle)
        countStreamFinished = true
      } finally {
        // Mirror the first pass's finally-abort: on any failure above (a
        // thrown exception before StreamFinish), abort the stream instead
        // of leaking it. On success StreamFinish already tore the stream
        // down, so there is nothing left to abort.
        if (!countStreamFinished) {
          NativeBridge.membershipCount3StreamAbort(countStreamHandle)
        }
      }
      NativeBridge.releaseMembershipCount3(countPreparedHandle)

      println(s"""{"expectedCount":$expectedCount,"actualCount":$actualCount}""")
      require(actualCount == expectedCount,
        s"GPU row-group membership count $actualCount did not match expected $expectedCount")
    } finally {
      spark.stop()
    }
  }

  // Task 5: drive MetalParquetMembershipCountExec directly (it is not yet
  // reachable from the planner -- that's Task 6) over the same Task 3/4
  // dataset, and compare its count against an equivalent DataFrame join
  // count computed entirely by Spark's own engine.
  private def runExecMode(nativeLibrary: String, metalLibrary: String, rows: Int): Unit = {
    val spark = SparkSession.builder().appName("spark-metal-parquet-decode-smoke-exec").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    val tempDir = Files.createTempDirectory("spark-metal-parquet-decode-smoke-exec")
    try {
      // Member set: values 50..199 in every column. writeDataset's k0/k1/k2
      // all start their non-null range at +50 (see the memberLow comment in
      // runDecodeAndCountMode), so this range is non-empty for every
      // column, giving a non-zero expected count; and every column also
      // carries id % 17 == 0 (plus k0's wide contiguous range) nulls, so a
      // key set alone does not make the join count -- null fact rows must
      // be correctly excluded for the counts to match.
      val memberLow = 50
      val memberHigh = 199
      import spark.implicits._

      def check(
          caseName: String,
          files: Seq[String],
          keyValues: Seq[Int],
          minRowGroups: Option[Int] = None,
          minFiles: Option[Int] = None): Unit = {
        val dim0 = keyValues.toDF("k0j")
        val dim1 = keyValues.toDF("k1j")
        val dim2 = keyValues.toDF("k2j")

        val outputAttribute = AttributeReference("count", LongType, nullable = false)()
        val exec = MetalParquetMembershipCountExec(
          outputAttribute,
          files,
          Columns,
          Seq(
            dim0.queryExecution.executedPlan,
            dim1.queryExecution.executedPlan,
            dim2.queryExecution.executedPlan),
          nativeLibrary,
          metalLibrary)

        val partitionCounts = exec.executeColumnar().mapPartitions { batches =>
          batches.map { batch =>
            val value = batch.column(0).getLong(0)
            batch.close()
            value
          }
        }.collect()
        val actualCount = partitionCounts.sum
        // Read after the action above completes: task-level accumulator
        // updates are merged into the driver-side SQLMetric synchronously
        // as part of that action, so this reflects the total across every
        // partition/row-group this run touched.
        val observedRowGroups = exec.metrics("numRowGroups").value

        val fact = spark.read.parquet(files: _*).select("k0", "k1", "k2")
        val expectedCount = fact
          .join(dim0, fact("k0") === dim0("k0j"))
          .join(dim1, fact("k1") === dim1("k1j"))
          .join(dim2, fact("k2") === dim2("k2j"))
          .count()

        println(
          s"""{"mode":"exec","case":"$caseName","rows":$rows,"files":${files.length},""" +
            s""""numRowGroups":$observedRowGroups,"expectedCount":$expectedCount,""" +
            s""""actualCount":$actualCount}""")
        require(expectedCount > 0, s"$caseName: expected join count must be non-zero")
        require(actualCount == expectedCount,
          s"$caseName: MetalParquetMembershipCountExec count $actualCount did not match " +
            s"expected $expectedCount")
        minRowGroups.foreach { min =>
          require(observedRowGroups >= min,
            s"$caseName: expected numRowGroups >= $min, observed $observedRowGroups")
        }
        minFiles.foreach { min =>
          require(files.length >= min, s"$caseName: expected files >= $min, got ${files.length}")
        }
      }

      // --- Layout A: a single file, a single row group -------------------
      val pathSingle = tempDir.resolve("data-single").toString
      writeDataset(spark, pathSingle, rows)
      val singleFile = listParquetFiles(pathSingle)
      require(singleFile.length == 1, s"expected exactly 1 part file under $pathSingle, got $singleFile")

      // Unique keys 50..199: every column's non-null range starts at +50
      // (see the memberLow comment in runDecodeAndCountMode), so this range
      // is non-empty for every column; and every column also carries
      // id % 17 == 0 (plus k0's wide contiguous range) nulls, so a key
      // match alone does not make the count -- null fact rows must be
      // correctly excluded. All keys are unique, so this exercises the
      // presence-table path and, with dense small-integer domains, the GPU
      // stream (parquetRowGroupBegin/parquetDecodePage/parquetRowGroupCount).
      check("unique-dense", singleFile, memberLow to memberHigh)

      // Duplicate every key twice in all three dimensions: each matching
      // fact row now has build-side multiplicity 2 per column, so a
      // qualifying row contributes 2*2*2=8 to the count instead of 1. This
      // exercises the multiplicity-table path (MembershipTables.multiplicity)
      // instead of the presence-table path, on the same GPU stream.
      check("duplicate-dense", singleFile, (memberLow to memberHigh) ++ (memberLow to memberHigh))

      // One key far outside the fact table's actual value range (max ~950)
      // blows the dense-domain span (max - min + 1 <= 16M) without changing
      // which fact rows match, forcing denseDomains = false for all three
      // columns and driving every row group through the CPU
      // (ColumnReadStoreImpl) fallback path instead of the GPU stream.
      check("sparse-cpu-fallback", singleFile, (memberLow to memberHigh) :+ 20000000)

      // --- Layout B: one file, multiple row groups ------------------------
      // parquet-mr only re-checks buffered row-group size at record-count
      // checkpoints it extrapolates from the average buffered bytes/record
      // seen so far (capped at +10,000 records between checks); for this
      // dictionary-coded, highly-compressible synthetic data that estimate
      // starts tiny, so a "realistic" 256KB threshold was empirically found
      // to never trip a checkpoint before end-of-file (single ~331KB row
      // group for all 100003 rows). A much smaller 4KB threshold forces
      // frequent checkpoints instead, splitting this dataset into ~200 row
      // groups within the single part file -- exercising the per-split
      // loop's multiple-splits-per-file path (same cached ParquetFileReader,
      // several readRowGroup(index) calls in sequence) the way SF10-shaped
      // data from other writers could.
      val pathMultiRowGroup = tempDir.resolve("data-multi-rowgroup").toString
      writeDataset(spark, pathMultiRowGroup, rows, rowGroupBytes = Some(4096))
      val multiRowGroupFile = listParquetFiles(pathMultiRowGroup)
      require(multiRowGroupFile.length == 1,
        s"expected exactly 1 part file under $pathMultiRowGroup, got $multiRowGroupFile")
      check("multi-rowgroup-single-file", multiRowGroupFile, memberLow to memberHigh, minRowGroups = Some(2))

      // --- Layout C: multiple files, splits spanning files ----------------
      val pathMultiFile = tempDir.resolve("data-multi-file").toString
      writeDataset(spark, pathMultiFile, rows, partitions = 3)
      val multiFiles = listParquetFiles(pathMultiFile)
      check("multi-file", multiFiles, memberLow to memberHigh, minFiles = Some(2))
    } finally {
      spark.stop()
    }
  }

  /**
   * @param partitions   number of output part files (1 = coalesce to a
   *                     single file, preserving row order; >1 repartitions
   *                     -- with a shuffle -- into that many files, which the
   *                     row/null formulas below tolerate fine since they
   *                     only depend on the `id` value, not row order).
   * @param rowGroupBytes when set, forces a small Parquet row-group (block)
   *                     size so a single part file spans multiple row
   *                     groups. Set both as a DataFrameWriter option and on
   *                     the Hadoop configuration directly (belt and
   *                     suspenders -- some parquet-mr/Spark version
   *                     combinations only honor one of the two), and
   *                     restored afterward so it does not leak into other
   *                     datasets written by the same SparkSession.
   */
  private def writeDataset(
      spark: SparkSession,
      path: String,
      rows: Int,
      partitions: Int = 1,
      rowGroupBytes: Option[Int] = None): Unit = {
    // k0 additionally goes null for a long contiguous id range (on top of the
    // modulo pattern), guaranteeing at least one data page that is 100% null
    // -- exercising the entirely-null-page path in
    // parquetDecodePage/scatter_segments. k1/k2 keep the plain modulo-null
    // pattern so the mixed-null-page path stays covered too.
    //
    // A dictionary-encoded int column packs to only a few bits/value, so the
    // default 1MB parquet.page.size can fit all 100k+ rows in a single page.
    // Forcing small (8KB) pages isn't enough on its own either: parquet-mr
    // additionally caps every page at 20,000 rows (DEFAULT_PAGE_ROW_COUNT_LIMIT),
    // and RLE-encodes a long null run so cheaply that a page straddling the
    // start of the null run rides that 20,000-row cap deep into the run
    // before cutting. The null range here (70,001 rows) is wide enough that,
    // even after that first straddling page, at least one subsequent
    // 20,000-row-capped page starts and ends fully inside the null run.
    val hadoopConf = spark.sparkContext.hadoopConfiguration
    val previousBlockSize = hadoopConf.get("parquet.block.size")
    rowGroupBytes.foreach(bytes => hadoopConf.setInt("parquet.block.size", bytes))
    try {
      val base = spark.range(rows)
        .selectExpr(
          "CASE WHEN id BETWEEN 20000 AND 90000 THEN CAST(NULL AS INT) " +
            "WHEN id % 17 = 0 THEN CAST(NULL AS INT) " +
            "ELSE CAST(pmod(id, 900) + 50 AS INT) END AS k0",
          "CASE WHEN id % 17 = 0 THEN CAST(NULL AS INT) ELSE CAST(pmod(id, 850) + 50 AS INT) END AS k1",
          "CASE WHEN id % 17 = 0 THEN CAST(NULL AS INT) ELSE CAST(pmod(id, 700) + 50 AS INT) END AS k2")
      val laidOut = if (partitions <= 1) base.coalesce(1) else base.repartition(partitions)
      var writer = laidOut.write.option("parquet.page.size", "8192")
      rowGroupBytes.foreach(bytes => writer = writer.option("parquet.block.size", bytes.toString))
      writer.mode("errorifexists").parquet(path)
    } finally {
      rowGroupBytes.foreach { _ =>
        if (previousBlockSize != null) hadoopConf.set("parquet.block.size", previousBlockSize)
        else hadoopConf.unset("parquet.block.size")
      }
    }
  }

  private def listParquetFiles(path: String): Seq[String] =
    Files.list(java.nio.file.Path.of(path)).iterator().asScala
      .map(_.toString)
      .filter(_.endsWith(".parquet"))
      .toSeq
      .sorted
}
