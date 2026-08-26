package io.github.mohitpatil.sparkmetal

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.Files

import scala.collection.mutable
import scala.jdk.CollectionConverters._

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.parquet.column.Encoding
import org.apache.parquet.column.page.DataPageV1
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.hadoop.util.HadoopInputFile

import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.catalyst.expressions.aggregate.Partial
import org.apache.spark.sql.execution.aggregate.HashAggregateExec
import org.apache.spark.sql.functions.{avg, col, count, expr, lit, sum}
import org.apache.spark.sql.types.{
  DateType, Decimal, DecimalType, DoubleType, IntegerType, LongType, StringType, StructField, StructType}
import org.apache.spark.unsafe.types.UTF8String

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
    } else if (arguments.nonEmpty && arguments(0) == "measure") {
      require(arguments.length >= 3,
        "Usage: ParquetDecodeSmoke measure NATIVE_LIB METAL_LIB [ROWS]")
      val rows = if (arguments.length > 3) arguments(3).toInt else 100003
      runMeasureMode(arguments(1), arguments(2), rows)
    } else if (arguments.nonEmpty && arguments(0) == "aggregate") {
      require(arguments.length >= 3,
        "Usage: ParquetDecodeSmoke aggregate NATIVE_LIB METAL_LIB [ROWS]")
      val rows = if (arguments.length > 3) arguments(3).toInt else 100003
      runAggregateMode(arguments(1), arguments(2), rows)
    } else if (arguments.nonEmpty && arguments(0) == "agg-exec") {
      require(arguments.length >= 3,
        "Usage: ParquetDecodeSmoke agg-exec NATIVE_LIB METAL_LIB [ROWS]")
      val rows = if (arguments.length > 3) arguments(3).toInt else 100003
      runAggExecMode(arguments(1), arguments(2), rows)
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
                  pageBytes, pageBytes.length, valueCount, rowOffset, hasDefLevels, /* isPlain = */ false)

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
                  pageBytes, pageBytes.length, valueCount, rowOffset, hasDefLevels, /* isPlain = */ false)
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
        // Set the flag before calling Finish: Finish deletes the native
        // stream even when it throws, so the finally block's Abort must
        // never fire afterward -- it would touch an already-deleted stream.
        countStreamFinished = true
        actualCount = NativeBridge.membershipCount3StreamFinish(countStreamHandle)
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

  // Task 2: decode measure columns (PLAIN pages and dictionary pages whose
  // VALUES -- not just ids -- must be materialized) into GPU value planes,
  // via parquetRowGroupBeginAggregate/parquetSetMeasureDictionary/
  // parquetDecodeMeasurePage. Two datasets exercise both value-section
  // layouts for both an int32 column (quantity) and a decimal(7,2) column
  // (price, compared against Spark's own read as its unscaled int): one
  // low-cardinality (parquet-mr keeps the dictionary), one high-cardinality
  // with a tiny parquet.dictionary.page.size (parquet-mr abandons the
  // dictionary and falls back to PLAIN, same trick as
  // writeDictionaryOverflowDataset above).
  // line_total has NO null pattern at all (every row non-null): the only
  // column exercising the all-valid decode path for both value-section
  // layouts -- PLAIN all-valid (a GPU blit copy, not a CPU memcpy -- see the
  // CRITICAL fix in parquetDecodeMeasurePage) and Dictionary all-valid
  // (expand_value_runs with materialize=1 against a real staged dictionary).
  // quantity/price null on every 13th/11th row, so neither ever lands a
  // fully-non-null PAGE in practice (the periodic null pattern touches every
  // page) -- line_total is what actually exercises those two paths.
  //
  // Ordered FIRST (slot 0), decoded before quantity/price: the row group's
  // shared pagesSinceCommit counter (kPagesPerCommit = 32) triggers an
  // automatic mid-decode commit, and once ANY commit has happened the
  // row-group-begin zero-fill blit that the CRITICAL fix guards against has
  // already executed -- a CPU-memcpy bug decoding line_total AFTER that
  // point would go undetected by coincidence, not because it was fixed.
  // Decoding line_total's ~10 pages first (rows / dictionary size puts it
  // well under the 32-page threshold) guarantees its all-valid pages are
  // decoded while that very first zero-fill blit is still pending, so the
  // bug reproduces deterministically rather than by luck of column order.
  private val MeasureColumns = Seq("line_total", "quantity", "price")
  private val MeasureIsDecimal = Seq(false, false, true)

  private def runMeasureMode(nativeLibrary: String, metalLibrary: String, rows: Int): Unit = {
    val spark = SparkSession.builder().appName("spark-metal-parquet-measure-smoke").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    val tempDir = Files.createTempDirectory("spark-metal-parquet-measure-smoke")
    try {
      SparkMetalNative.ensureInitialized(nativeLibrary, metalLibrary)

      val pathDictFriendly = tempDir.resolve("data-measure-dict").toString
      writeMeasureDataset(spark, pathDictFriendly, rows, highCardinality = false)
      val dictFriendlyFile = singlePartFile(pathDictFriendly)

      val pathForcedPlain = tempDir.resolve("data-measure-plain").toString
      writeMeasureDataset(spark, pathForcedPlain, rows, highCardinality = true)
      val forcedPlainFile = singlePartFile(pathForcedPlain)

      checkMeasureDecode(spark, "dictionary-friendly", pathDictFriendly, dictFriendlyFile, rows,
        expectDictionary = true)
      checkMeasureDecode(spark, "forced-plain", pathForcedPlain, forcedPlainFile, rows,
        expectDictionary = false)
    } finally {
      spark.stop()
    }
  }

  private def checkMeasureDecode(
      spark: SparkSession,
      caseName: String,
      path: String,
      partFile: String,
      rows: Int,
      expectDictionary: Boolean): Unit = {
    val decodedValues = Array.fill(MeasureColumns.length)(new Array[Int](rows))
    val decodedIsNull = Array.fill(MeasureColumns.length)(new Array[Boolean](rows))
    var coveredRows = 0
    var sawDictionaryPage = false
    var sawPlainPage = false

    val preparedHandle = NativeBridge.prepareMembershipCount3(Array(1), Array(1), Array(1))
    val reader = ParquetFileReader.open(HadoopInputFile.fromPath(new Path(partFile), new Configuration()))
    try {
      val descriptors = MeasureColumns.map { name =>
        reader.getFooter.getFileMetaData.getSchema.getColumns.asScala
          .find(_.getPath()(0) == name)
          .getOrElse(throw new RuntimeException(s"Missing column $name"))
      }
      var pageReadStore = reader.readNextRowGroup()
      while (pageReadStore != null) {
        val rowGroupRows = pageReadStore.getRowCount.toInt
        val streamHandle = NativeBridge.membershipCount3StreamBegin(preparedHandle)
        val rowGroupHandle = NativeBridge.parquetRowGroupBeginAggregate(
          streamHandle, rowGroupRows, 0, MeasureColumns.length)
        try {
          for (slot <- MeasureColumns.indices) {
            val descriptor = descriptors(slot)
            val pageReader = pageReadStore.getPageReader(descriptor)
            val dictionaryPage = pageReader.readDictionaryPage()
            if (dictionaryPage != null) {
              sawDictionaryPage = true
              val dictionaryBytes = dictionaryPage.getBytes.toByteArray
              val dictionaryBuffer = ByteBuffer.wrap(dictionaryBytes).order(ByteOrder.LITTLE_ENDIAN)
              val dictionaryValues = new Array[Int](dictionaryPage.getDictionarySize)
              for (entry <- dictionaryValues.indices) {
                dictionaryValues(entry) = dictionaryBuffer.getInt(entry * 4)
              }
              NativeBridge.parquetSetMeasureDictionary(rowGroupHandle, slot, dictionaryValues)
            } else {
              sawPlainPage = true
            }

            val hasDefLevels = descriptor.getMaxDefinitionLevel > 0
            var rowOffset = 0
            var rawPage = pageReader.readPage()
            while (rawPage != null) {
              val dataPage = rawPage match {
                case v1: DataPageV1 => v1
                case other => throw new RuntimeException(
                  s"${MeasureColumns(slot)}: unsupported Parquet page type ${other.getClass}")
              }
              val encoding = dataPage.getValueEncoding
              if (dictionaryPage != null) {
                require(DictionaryEncodings.contains(encoding),
                  s"${MeasureColumns(slot)}: expected dictionary encoding, got $encoding")
              } else {
                require(encoding == Encoding.PLAIN,
                  s"${MeasureColumns(slot)}: expected PLAIN encoding, got $encoding")
              }
              val pageBytes = dataPage.getBytes.toByteArray
              val valueCount = dataPage.getValueCount
              NativeBridge.parquetDecodeMeasurePage(
                streamHandle, rowGroupHandle, slot,
                pageBytes, pageBytes.length, valueCount, rowOffset, hasDefLevels)
              rowOffset += valueCount
              rawPage = pageReader.readPage()
            }
            require(rowOffset == rowGroupRows,
              s"${MeasureColumns(slot)}: pages covered $rowOffset rows, expected $rowGroupRows")
          }

          // Read back only AFTER every slot's pages are decoded, not
          // interleaved per-slot: parquetRowGroupReadMeasure forces a commit
          // (see its native implementation), and reading a slot back
          // immediately after that slot's own pages would commit -- and so
          // flush the row-group-begin zero-fill blit -- before a LATER
          // slot's pages are ever decoded, silently sidestepping the
          // CRITICAL fix's hazard window instead of exercising it. This
          // mirrors how a real consumer (Task 3's aggregation kernel) reads
          // the planes too: after every measure column for the row group has
          // been decoded, not column-by-column.
          for (slot <- MeasureColumns.indices) {
            val valuesOut = new Array[Int](rowGroupRows)
            val validityOut = new Array[Byte](rowGroupRows)
            NativeBridge.parquetRowGroupReadMeasure(
              streamHandle, rowGroupHandle, slot, valuesOut, validityOut)
            for (row <- 0 until rowGroupRows) {
              val globalRow = coveredRows + row
              if (validityOut(row) == 0) {
                decodedValues(slot)(globalRow) = valuesOut(row)
              } else {
                decodedIsNull(slot)(globalRow) = true
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
    require(coveredRows == rows, s"$caseName: row groups covered $coveredRows rows, expected $rows")

    val referenceRows =
      spark.read.parquet(path).select(MeasureColumns.head, MeasureColumns.tail: _*).collect()
    require(referenceRows.length == rows,
      s"$caseName: reference row count ${referenceRows.length} did not match $rows")

    var mismatches = 0
    for (row <- 0 until rows) {
      val sparkRow = referenceRows(row)
      for (columnIndex <- MeasureColumns.indices) {
        val expectedNull = sparkRow.isNullAt(columnIndex)
        if (expectedNull != decodedIsNull(columnIndex)(row)) {
          mismatches += 1
        } else if (!expectedNull) {
          // decimal(7,2) is backed by an INT32 physical type; compare the raw
          // unscaled integer both sides agree on rather than the decimal
          // value, per the brief.
          val expectedValue =
            if (MeasureIsDecimal(columnIndex)) sparkRow.getDecimal(columnIndex).unscaledValue().intValue()
            else sparkRow.getInt(columnIndex)
          if (expectedValue != decodedValues(columnIndex)(row)) mismatches += 1
        }
      }
    }

    val matched = mismatches == 0
    println(
      s"""{"mode":"measure","case":"$caseName","rows":$rows,"mismatches":$mismatches,""" +
        s""""match":$matched,"sawDictionaryPage":$sawDictionaryPage,"sawPlainPage":$sawPlainPage}""")
    if (!matched) sys.exit(1)
    // Guard against the test silently degrading: each dataset is written to
    // deterministically exercise one specific value-section layout for BOTH
    // measure columns, so a run that never actually saw that layout would
    // pass the match check above without having tested anything.
    if (expectDictionary) {
      require(sawDictionaryPage,
        s"$caseName: expected at least one dictionary-encoded measure page, saw none")
    } else {
      require(sawPlainPage,
        s"$caseName: expected at least one PLAIN-encoded measure page, saw none")
      require(!sawDictionaryPage,
        s"$caseName: expected every measure page to be PLAIN, but saw a dictionary page too")
    }
  }

  /**
   * @param highCardinality when false, quantity/price are low-cardinality
   *                        (parquet-mr keeps the dictionary for both). When
   *                        true, quantity is the strictly-increasing `id`
   *                        and price is near-unique across `rows`, combined
   *                        with a small parquet.dictionary.page.size (same
   *                        trick as writeDictionaryOverflowDataset) so
   *                        parquet-mr abandons the dictionary and both
   *                        columns are written PLAIN.
   */
  private def writeMeasureDataset(
      spark: SparkSession, path: String, rows: Int, highCardinality: Boolean): Unit = {
    val hadoopConf = spark.sparkContext.hadoopConfiguration
    val previousDictionarySize = hadoopConf.get("parquet.dictionary.page.size")
    if (highCardinality) hadoopConf.setInt("parquet.dictionary.page.size", 4096)
    try {
      // quantity nulls every 13th row (per the brief); price nulls on a
      // different modulus (11th row) so the two columns' segment boundaries
      // don't line up, exercising more distinct null-gap shapes.
      val quantityExpr =
        if (highCardinality)
          "CASE WHEN id % 13 = 0 THEN CAST(NULL AS INT) ELSE CAST(id AS INT) END AS quantity"
        else
          "CASE WHEN id % 13 = 0 THEN CAST(NULL AS INT) ELSE CAST(pmod(id, 50) + 1 AS INT) END AS quantity"
      val priceExpr =
        if (highCardinality)
          "CASE WHEN id % 11 = 0 THEN CAST(NULL AS DECIMAL(7,2)) " +
            "ELSE CAST(pmod(id, 99999) AS DECIMAL(7,2)) END AS price"
        else
          "CASE WHEN id % 11 = 0 THEN CAST(NULL AS DECIMAL(7,2)) " +
            "ELSE CAST(pmod(id, 300) + 100 AS DECIMAL(7,2)) END AS price"
      // line_total: no CASE WHEN NULL branch at all -- every row non-null, so
      // every page of this column is all-valid. Low-cardinality for the
      // dictionary-friendly write (dictionary all-valid path); the
      // strictly-increasing id for the forced-PLAIN write (PLAIN all-valid
      // path, the same overflow trick as quantity/price above).
      val lineTotalExpr =
        if (highCardinality) "CAST(id AS INT) AS line_total"
        else "CAST(pmod(id, 40) + 1 AS INT) AS line_total"
      var writer = spark.range(rows)
        .selectExpr(quantityExpr, priceExpr, lineTotalExpr)
        .coalesce(1)
        .write
        .option("parquet.page.size", "8192")
      if (highCardinality) writer = writer.option("parquet.dictionary.page.size", "4096")
      writer.mode("errorifexists").parquet(path)
    } finally {
      if (highCardinality) {
        if (previousDictionarySize != null) hadoopConf.set("parquet.dictionary.page.size", previousDictionarySize)
        else hadoopConf.unset("parquet.dictionary.page.size")
      }
    }
  }

  // --- Task 4: driver-side group-space builder + grouped aggregation on the
  // GPU --------------------------------------------------------------------
  //
  // Two dimensions, joined against a synthetic fact table exactly like a
  // real broadcast-side build: `region` is attribute-free (it contributes
  // NO group component, only a membership gate and a duplicate-build-key
  // multiplicity factor -- region key 2 appears twice on the build side) and
  // `category` carries 4 attributes spanning FOUR DIFFERENT TYPES: major
  // (int), brand (string), listPrice (decimal(7,2)), launchDate (date).
  // TPC-DS group keys are frequently strings (q3/q42/q52/q55 group by
  // i_brand/i_category) and decimals/dates are common dimension attributes
  // too, so GroupSpace's positive path needs to actually exercise all of
  // those -- not just int32 -- and DecimalType in particular is the only
  // parameterized extractor (getDecimal(ordinal, precision, scale)): a wrong
  // precision/scale against the row's real layout would silently produce a
  // wrong value rather than fail loudly, so it needs its own coverage, not
  // an assumption that "it's just like the other types."
  //
  // Distinct join key 13 deliberately shares its FULL attribute tuple
  // (major=1, brand="acme", listPrice=19.99, launchDate=2020-01-15) with key
  // 10 -- proving GroupSpace.build groups by the whole ATTRIBUTE TUPLE
  // (value-compared across every type in it), not by join-key identity: a
  // fact row joining on either key 10 or key 13 lands in the very same
  // accumulator (and, since key 10 is quantity-positive and key 13
  // quantity-negative -- see writeAggregateDataset -- their contributions
  // actually offset within that one group). Crucially, listPrice/launchDate
  // are NOT unique per (major, brand): key 11 shares key 10's exact
  // listPrice AND launchDate but has a different brand ("globex"), and key
  // 14 shares key 12's exact listPrice AND launchDate but has a different
  // major/brand -- both prove that a decimal/date match alone is NOT enough
  // to collapse two keys into one group; every field in the tuple must
  // match. Key 14 is a dimension row with no matching fact rows at all, so
  // its group must come back all zeros; key 15 is a fact-side value with no
  // matching dimension row, exercising the code-gate (as opposed to the
  // null-gate) on the attributed dimension.
  //
  // GroupSpace.build assigns the dense group ids (in VALUE space); this test
  // then does by hand what Task 5 will do generically -- translate the
  // resulting value-keyed maps into dict-id-indexed tables for the kernel --
  // and treats an ACTUAL Spark join + groupBy over the same data as the
  // reference, rather than a hand-derived expectation: that is what proves
  // GroupSpace's code assignment, radix packing, and factor semantics match
  // real join semantics.
  private val AggregateKeyColumns = Seq("region_key", "category_key")
  private val AggregateMeasureColumns = Seq("quantity", "price", "amount")
  // count(*), sum(quantity), sum(price_unscaled), sum(amount), count(quantity)
  // -- all three aggregate kinds (0 = count-star, 1 = sum, 2 = count-col).
  private val AggregateKinds = Array(0, 1, 1, 1, 2)
  private val AggregateSlots = Array(0, 0, 1, 2, 0)

  // Region dimension (attribute-free): key 2 appears twice, so its factor is
  // 2. Fact-side region_key also carries a value (4) with no dimension row
  // at all, exercising the code-gate on the attribute-free dimension too.
  private val RegionDimensionKeys = Seq(1, 2, 2, 3)

  // Category dimension: (joinKey, major: Int, brand: String,
  // listPriceCents: unscaled decimal(7,2), launchDate: ISO date string) --
  // a single source of truth both the GroupSpace-side InternalRows and the
  // Spark reference DataFrame below are derived from, so the two can never
  // silently drift apart. Distinct attribute tuples: (1,"acme",19.99,
  // 2020-01-15) [keys 10, 13 -- collapse], (1,"globex",19.99,2020-01-15)
  // [key 11], (2,"acme",29.99,2021-06-01) [key 12], (3,"initech",29.99,
  // 2021-06-01) [key 14, no fact rows] -- 4 distinct tuples, same as before
  // GroupSpace was extended, so the group count (and every downstream
  // group-count-dependent assertion/kernel limit) is unchanged.
  private val CategoryDimensionData: Seq[(Int, Int, String, Long, String)] = Seq(
    (10, 1, "acme", 1999L, "2020-01-15"),
    (11, 1, "globex", 1999L, "2020-01-15"),
    (12, 2, "acme", 2999L, "2021-06-01"),
    (13, 1, "acme", 1999L, "2020-01-15"),
    (14, 3, "initech", 2999L, "2021-06-01"))

  private def runAggregateMode(nativeLibrary: String, metalLibrary: String, rows: Int): Unit = {
    val spark = SparkSession.builder().appName("spark-metal-parquet-aggregate-smoke").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    val tempDir = Files.createTempDirectory("spark-metal-parquet-aggregate-smoke")
    try {
      SparkMetalNative.ensureInitialized(nativeLibrary, metalLibrary)

      // Step 2 (build-reject checks): pure GroupSpace.build unit assertions,
      // no I/O -- run first since they don't need the dataset below.
      checkGroupSpaceBuildRejections()

      val path = tempDir.resolve("data-aggregate").toString
      writeAggregateDataset(spark, path, rows)
      val partFile = singlePartFile(path)

      val aggregateCount = AggregateKinds.length

      // The group space, built from VALUE-keyed dimension rows (not
      // dictionary ids -- see GroupSpace's own doc comment).
      val regionDimensionRows: Array[(Int, InternalRow)] =
        RegionDimensionKeys.map(key => (key, InternalRow())).toArray
      val categoryDimensionRows: Array[(Int, InternalRow)] =
        CategoryDimensionData.map { case (key, major, brand, listPriceCents, launchDateIso) =>
          val launchDateEpochDay = java.sql.Date.valueOf(launchDateIso).toLocalDate.toEpochDay.toInt
          (key, InternalRow(
            major, UTF8String.fromString(brand), Decimal(listPriceCents, 7, 2), launchDateEpochDay))
        }.toArray
      // Dimension order here (region, then category) MUST match
      // AggregateKeyColumns' order (region_key, then category_key) below:
      // built.codesByKey(i)/built.factorsByKey(i) is dimension i's map, and
      // it is used to translate AggregateKeyColumns(i)'s dictionary values
      // into the i-th code/factor table passed to
      // NativeBridge.parquetRowGroupAggregate -- dimension i and key column
      // i are the SAME column, just at two different stages of translation.
      val built = GroupSpace.build(
        Seq(
          GroupSpace.Dimension(regionDimensionRows, attributeCount = 0),
          GroupSpace.Dimension(
            categoryDimensionRows,
            attributeCount = 4,
            attributeTypes = Seq(IntegerType, StringType, DecimalType(7, 2), DateType))),
        maxGroups = 1024) match {
        case Right(b) => b
        case Left(reason) => throw new RuntimeException(s"GroupSpace.build unexpectedly rejected: $reason")
      }
      require(built.codesByKey.length == AggregateKeyColumns.length,
        s"GroupSpace dimension count ${built.codesByKey.length} must match " +
          s"AggregateKeyColumns.length ${AggregateKeyColumns.length} (dimension i == key column i)")
      // region contributes 1 component (attribute-free) x category's 4
      // distinct attribute tuples (10 and 13 collapse together) = 4 groups.
      require(built.groupCount == 4,
        s"expected 4 groups (1 region component x 4 category tuples), got ${built.groupCount}")

      // Reference: an ACTUAL Spark join (region and category dimension
      // tables against the fact file) + groupBy(major, brand, listPrice,
      // launchDate), not a hand-derived formula. Spark's own equi-join
      // already implements the null-gate (a null key never matches) and the
      // code-gate (region_key = 4 / category_key = 15 match no dimension
      // row); its groupBy already implements the ignore-nulls sum/
      // count(column) policy the kernel is expected to match, and groups by
      // the string/decimal/date attributes exactly the same way real
      // TPC-DS-shaped queries (q3/q42/q52/q55 group by i_brand/i_category)
      // would.
      import spark.implicits._
      val fact = spark.read.parquet(path)
      val regionDim = RegionDimensionKeys.toDF("region_key_b")
      val categoryDim = CategoryDimensionData.map { case (key, major, brand, listPriceCents, launchDateIso) =>
        (key, major, brand, BigDecimal(listPriceCents, 2), java.sql.Date.valueOf(launchDateIso))
      }.toDF("category_key_b", "major", "brand", "listPrice", "launchDate")
      val enriched = fact
        .join(regionDim, fact("region_key") === regionDim("region_key_b"))
        .join(categoryDim, fact("category_key") === categoryDim("category_key_b"))
        .withColumn("price_unscaled", expr("CAST(price * 100 AS INT)"))
      val referenceAgg = enriched.groupBy("major", "brand", "listPrice", "launchDate").agg(
        count(lit(1)).as("cnt"),
        sum(col("quantity")).as("sumQuantity"),
        sum(col("price_unscaled")).as("sumPriceUnscaled"),
        sum(col("amount")).as("sumAmount"),
        count(col("quantity")).as("countQuantity")
      ).collect()
      // listPrice/launchDate are folded into the map key as exact,
      // scale/timezone-independent canonical values (unscaled cents,
      // epoch-day) so they compare equal to GroupSpace's own catalyst-level
      // representations below regardless of the DecimalType precision/scale
      // Spark inferred for this DataFrame (which need not match the
      // DecimalType(7,2) GroupSpace was told about -- only the numeric
      // value has to agree).
      def decimalCents(decimal: java.math.BigDecimal): Long =
        decimal.movePointRight(2).setScale(0).longValueExact()
      val referenceMap: Map[(Int, String, Long, Int), Array[Long]] = referenceAgg.map { row =>
        def longOrZero(index: Int): Long = if (row.isNullAt(index)) 0L else row.getLong(index)
        val key = (
          row.getInt(0),
          row.getString(1),
          decimalCents(row.getDecimal(2)),
          row.getDate(3).toLocalDate.toEpochDay.toInt)
        key -> Array(longOrZero(4), longOrZero(5), longOrZero(6), longOrZero(7), longOrZero(8))
      }.toMap

      // Fold the reference map into a group-major array via groupTuples --
      // this is what actually exercises groupTuples as the output contract,
      // not just codesByKey/factorsByKey.
      val expected = new Array[Long](built.groupCount * aggregateCount)
      for (group <- 0 until built.groupCount) {
        val categoryTuple = built.groupTuples(group)(1)
        val key = (
          categoryTuple.getInt(0),
          categoryTuple.getUTF8String(1).toString,
          decimalCents(categoryTuple.getDecimal(2, 7, 2).toJavaBigDecimal),
          categoryTuple.getInt(3))
        val values = referenceMap.getOrElse(key, new Array[Long](5))
        for (aggregate <- 0 until aggregateCount) {
          expected(group * aggregateCount + aggregate) = values(aggregate)
        }
      }

      // The dataset only earns its keep if it actually reaches the cases
      // this task is about; assert that on the reference totals, before
      // ever comparing them to the GPU's.
      val emptyGroup = (0 until built.groupCount).find { g =>
        val tuple = built.groupTuples(g)(1)
        tuple.getInt(0) == 3 && tuple.getUTF8String(1).toString == "initech"
      }.getOrElse(throw new RuntimeException("expected a group for category tuple (major=3, brand=\"initech\")"))
      require((0 until aggregateCount).forall(a => expected(emptyGroup * aggregateCount + a) == 0L),
        s"group $emptyGroup (category key 14, tuple (3,\"initech\")) was expected to receive no rows, got " +
          (0 until aggregateCount).map(a => expected(emptyGroup * aggregateCount + a)).mkString(","))
      val quantitySums = (0 until built.groupCount).map(g => expected(g * aggregateCount + 1))
      require(quantitySums.exists(_ < 0L),
        s"expected at least one negative per-group sum(quantity), got $quantitySums")
      val amountSums = (0 until built.groupCount).map(g => expected(g * aggregateCount + 3))
      require(amountSums.exists(_ > 4294967296L),
        s"expected a per-group sum(amount) above 2^32, got $amountSums")
      require(amountSums.exists(_ < -4294967296L),
        s"expected a per-group sum(amount) below -2^32, got $amountSums")

      var rowGroups = 0
      var coveredRows = 0
      val preparedHandle = NativeBridge.prepareMembershipCount3(Array(1), Array(1), Array(1))
      // ONE stream for the whole file: the aggregate partial table is
      // per-stream, so every row group below accumulates into the same device
      // buffer across separate command buffers.
      val streamHandle = NativeBridge.membershipCount3StreamBegin(preparedHandle)
      var streamFinished = false
      var actual: Array[Long] = null
      try {
        val reader = ParquetFileReader.open(HadoopInputFile.fromPath(new Path(partFile), new Configuration()))
        try {
          val schemaColumns = reader.getFooter.getFileMetaData.getSchema.getColumns.asScala
          def descriptorFor(name: String) = schemaColumns
            .find(_.getPath()(0) == name)
            .getOrElse(throw new RuntimeException(s"Missing column $name"))
          val keyDescriptors = AggregateKeyColumns.map(descriptorFor)
          val measureDescriptors = AggregateMeasureColumns.map(descriptorFor)

          var pageReadStore = reader.readNextRowGroup()
          while (pageReadStore != null) {
            val rowGroupRows = pageReadStore.getRowCount.toInt
            val rowGroupHandle = NativeBridge.parquetRowGroupBeginAggregate(
              streamHandle, rowGroupRows, AggregateKeyColumns.length, AggregateMeasureColumns.length)
            val codeTables = new Array[Array[Int]](AggregateKeyColumns.length)
            val factorTables = new Array[Array[Int]](AggregateKeyColumns.length)
            for (columnIndex <- AggregateKeyColumns.indices) {
              val descriptor = keyDescriptors(columnIndex)
              val pageReader = pageReadStore.getPageReader(descriptor)
              val dictionaryPage = pageReader.readDictionaryPage()
              if (dictionaryPage == null) {
                throw new RuntimeException(
                  s"${AggregateKeyColumns(columnIndex)}: expected a dictionary page but found none")
              }
              val dictionaryBuffer =
                ByteBuffer.wrap(dictionaryPage.getBytes.toByteArray).order(ByteOrder.LITTLE_ENDIAN)
              val dictionaryValues =
                Array.tabulate(dictionaryPage.getDictionarySize)(entry => dictionaryBuffer.getInt(entry * 4))
              // Key columns decode to dictionary IDS, so the code/factor
              // tables handed to the kernel must be indexed in dict-id space
              // -- rebuilt per row group, since each column chunk has its
              // own dictionary. GroupSpace's maps are VALUE-keyed, so this
              // is exactly the translation Task 5 will make generic: a
              // dict id with no entry in codesByKey is not a member (-1);
              // one with no entry in factorsByKey has multiplicity 1.
              val codeMap = built.codesByKey(columnIndex)
              codeTables(columnIndex) = dictionaryValues.map(value => codeMap.getOrElse(value, -1))
              val factorMap = built.factorsByKey(columnIndex)
              factorTables(columnIndex) =
                if (factorMap.isEmpty) null else dictionaryValues.map(value => factorMap.getOrElse(value, 1))

              val hasDefLevels = descriptor.getMaxDefinitionLevel > 0
              var rowOffset = 0
              var rawPage = pageReader.readPage()
              while (rawPage != null) {
                val dataPage = rawPage match {
                  case v1: DataPageV1 => v1
                  case other => throw new RuntimeException(
                    s"${AggregateKeyColumns(columnIndex)}: unsupported page type ${other.getClass}")
                }
                require(DictionaryEncodings.contains(dataPage.getValueEncoding),
                  s"${AggregateKeyColumns(columnIndex)}: unsupported value encoding " +
                    dataPage.getValueEncoding)
                val pageBytes = dataPage.getBytes.toByteArray
                NativeBridge.parquetDecodePage(
                  streamHandle, rowGroupHandle, columnIndex,
                  pageBytes, pageBytes.length, dataPage.getValueCount, rowOffset, hasDefLevels,
                  /* isPlain = */ false)
                rowOffset += dataPage.getValueCount
                rawPage = pageReader.readPage()
              }
              require(rowOffset == rowGroupRows,
                s"${AggregateKeyColumns(columnIndex)}: pages covered $rowOffset rows, expected $rowGroupRows")
            }

            for (slot <- AggregateMeasureColumns.indices) {
              val descriptor = measureDescriptors(slot)
              val pageReader = pageReadStore.getPageReader(descriptor)
              val dictionaryPage = pageReader.readDictionaryPage()
              if (dictionaryPage != null) {
                val dictionaryBuffer =
                  ByteBuffer.wrap(dictionaryPage.getBytes.toByteArray).order(ByteOrder.LITTLE_ENDIAN)
                val dictionaryValues =
                  Array.tabulate(dictionaryPage.getDictionarySize)(entry => dictionaryBuffer.getInt(entry * 4))
                NativeBridge.parquetSetMeasureDictionary(rowGroupHandle, slot, dictionaryValues)
              }
              val hasDefLevels = descriptor.getMaxDefinitionLevel > 0
              var rowOffset = 0
              var rawPage = pageReader.readPage()
              while (rawPage != null) {
                val dataPage = rawPage match {
                  case v1: DataPageV1 => v1
                  case other => throw new RuntimeException(
                    s"${AggregateMeasureColumns(slot)}: unsupported page type ${other.getClass}")
                }
                val encoding = dataPage.getValueEncoding
                if (dictionaryPage != null) {
                  require(DictionaryEncodings.contains(encoding),
                    s"${AggregateMeasureColumns(slot)}: expected dictionary encoding, got $encoding")
                } else {
                  require(encoding == Encoding.PLAIN,
                    s"${AggregateMeasureColumns(slot)}: expected PLAIN encoding, got $encoding")
                }
                val pageBytes = dataPage.getBytes.toByteArray
                NativeBridge.parquetDecodeMeasurePage(
                  streamHandle, rowGroupHandle, slot,
                  pageBytes, pageBytes.length, dataPage.getValueCount, rowOffset, hasDefLevels)
                rowOffset += dataPage.getValueCount
                rawPage = pageReader.readPage()
              }
              require(rowOffset == rowGroupRows,
                s"${AggregateMeasureColumns(slot)}: pages covered $rowOffset rows, expected $rowGroupRows")
            }

            // Consumes rowGroupHandle (like parquetRowGroupCount): no
            // Release, and the handle must not be touched again.
            NativeBridge.parquetRowGroupAggregate(
              streamHandle, rowGroupHandle, codeTables, factorTables,
              built.groupCount, AggregateSlots, AggregateKinds)
            rowGroups += 1
            coveredRows += rowGroupRows
            pageReadStore = reader.readNextRowGroup()
          }
        } finally {
          reader.close()
        }
        // Set the flag BEFORE Finish: Finish destroys the native stream even
        // when it throws, so the finally-block Abort must never fire after it.
        streamFinished = true
        actual = NativeBridge.parquetAggregateStreamFinish(streamHandle)
      } finally {
        if (!streamFinished) NativeBridge.parquetAggregateStreamAbort(streamHandle)
        // Released here, not after the comparisons below: the prepared handle
        // outlives the stream and must come back on the failure paths too.
        NativeBridge.releaseMembershipCount3(preparedHandle)
      }

      require(coveredRows == rows, s"row groups covered $coveredRows rows, expected $rows")
      require(actual.length == expected.length,
        s"GPU returned ${actual.length} accumulators, expected ${expected.length}")
      val mismatches = expected.indices.count(index => actual(index) != expected(index))
      val matched = mismatches == 0
      println(
        s"""{"mode":"aggregate","rows":$rows,"rowGroups":$rowGroups,""" +
          s""""groups":${built.groupCount},"aggregates":$aggregateCount,""" +
          s""""mismatches":$mismatches,"match":$matched,""" +
          s""""maxAmountSum":${amountSums.max},"minAmountSum":${amountSums.min}}""")
      if (!matched) {
        val first = expected.indices.find(index => actual(index) != expected(index)).get
        println(
          s"""{"mode":"aggregate","firstMismatchIndex":$first,""" +
            s""""group":${first / aggregateCount},"aggregate":${first % aggregateCount},""" +
            s""""expected":${expected(first)},"actual":${actual(first)}}""")
        sys.exit(1)
      }
      // Cross-command-buffer accumulation into the one per-stream partial
      // table is the whole point of the per-STREAM (not per-row-group) table;
      // a single row group would never exercise it.
      require(rowGroups >= 2,
        s"expected the aggregate dataset to span at least 2 row groups, got $rowGroups")
    } finally {
      spark.stop()
    }
  }

  /**
   * Step 2 of the brief: GroupSpace.build build-reject unit checks, run
   * inside the aggregate smoke mode (no separate test runner in this
   * project -- see run-parquet-decode-smoke-test.sh).
   */
  private def checkGroupSpaceBuildRejections(): Unit = {
    // A duplicate join key in an ATTRIBUTED dimension (attributeCount > 0)
    // must be rejected: unlike an attribute-free dimension, there is no
    // factor to fold a repeated key into once it carries attributes.
    val duplicateKeyDimension = GroupSpace.Dimension(
      Array((1, InternalRow(10)), (1, InternalRow(20))), attributeCount = 1, attributeTypes = Seq(IntegerType))
    val duplicateKeyResult = GroupSpace.build(Seq(duplicateKeyDimension), maxGroups = 1000)
    require(duplicateKeyResult.isLeft,
      s"expected a duplicate join key in an attributed dimension to be rejected, got $duplicateKeyResult")

    // An oversized group space (the mixed-radix product exceeds maxGroups)
    // must be rejected: 50 x 50 = 2500 groups against a cap of 100.
    val bigDimension0 = GroupSpace.Dimension(
      Array.tabulate(50)(i => (i, InternalRow(i))), attributeCount = 1, attributeTypes = Seq(IntegerType))
    val bigDimension1 = GroupSpace.Dimension(
      Array.tabulate(50)(i => (i + 1000, InternalRow(i))), attributeCount = 1, attributeTypes = Seq(IntegerType))
    val oversizedResult = GroupSpace.build(Seq(bigDimension0, bigDimension1), maxGroups = 100)
    require(oversizedResult.isLeft,
      s"expected a 50x50=2500-group space to be rejected by maxGroups=100, got $oversizedResult")

    // An empty dimension (zero collected rows) must be rejected.
    val emptyDimension = GroupSpace.Dimension(Array.empty, attributeCount = 0)
    val emptyResult = GroupSpace.build(Seq(emptyDimension), maxGroups = 1000)
    require(emptyResult.isLeft, s"expected an empty dimension to be rejected, got $emptyResult")

    // An attribute type GroupSpace.build does not support (e.g. DoubleType)
    // must be rejected, naming the type -- it never even gets to reading
    // any row's value.
    val unsupportedTypeDimension = GroupSpace.Dimension(
      Array((1, InternalRow(1.5d))), attributeCount = 1, attributeTypes = Seq(DoubleType))
    val unsupportedTypeResult = GroupSpace.build(Seq(unsupportedTypeDimension), maxGroups = 1000)
    require(unsupportedTypeResult.isLeft,
      s"expected an unsupported attribute type (DoubleType) to be rejected, got $unsupportedTypeResult")
    val unsupportedTypeReason = unsupportedTypeResult match {
      case Left(reason) => reason
      case Right(_) => ""
    }
    require(unsupportedTypeReason.contains("DoubleType"),
      s"expected the rejection reason to name DoubleType, got $unsupportedTypeResult")

    println(
      s"""{"mode":"aggregate","case":"group-space-build-rejections",""" +
        s""""duplicateKeyRejected":${duplicateKeyResult.isLeft},""" +
        s""""oversizedRejected":${oversizedResult.isLeft},""" +
        s""""emptyDimensionRejected":${emptyResult.isLeft},""" +
        s""""unsupportedTypeRejected":${unsupportedTypeResult.isLeft}}""")
  }

  /**
   * region_key in {1,2,3,4} (4 is a non-member -- no region dimension row --
   * so the code-gate is exercised); category_key in {10,11,12,13,14,15} (14
   * is a dimension row with no fact rows at all -- one group is guaranteed
   * to come back all zeros; 15 is a fact value with no dimension row, a
   * second, independent exercise of the code-gate). Nulls fall on two
   * different moduli (43, 47) so both the null-gate and the code-gate drop
   * rows independently.
   *
   * amount is ~1e6 per row and negative for category_key = 12, so with
   * ~rows/5 rows per category the per-group totals run past +/-2^32 in both
   * directions -- the hi-word/carry path the split 32-bit atomics have to
   * get right. quantity is negative for category_key IN (11, 13) (13 is one
   * of the two keys that collapse into the (major=1, brand="acme") group
   * with key 10, so that group's quantity sum mixes positive and negative
   * contributions from two distinct join keys) and null every 13th row, so
   * the sum/count(col) null policy is exercised; price is null every 11th
   * row on a different modulus; amount is never null, so its measure plane
   * also covers the no-nulls (mask bit clear) path.
   *
   * A small parquet.block.size splits the file into several row groups so
   * the per-stream partial table accumulates across command buffers.
   */
  private def writeAggregateDataset(spark: SparkSession, path: String, rows: Int): Unit = {
    val hadoopConf = spark.sparkContext.hadoopConfiguration
    val previousBlockSize = hadoopConf.get("parquet.block.size")
    hadoopConf.setInt("parquet.block.size", 65536)
    try {
      val base = spark.range(rows)
        .selectExpr(
          "id",
          "CASE WHEN id % 43 = 0 THEN CAST(NULL AS INT) ELSE CAST(pmod(id, 4) + 1 AS INT) END AS region_key",
          "CASE WHEN id % 47 = 0 THEN CAST(NULL AS INT) " +
            "ELSE CASE pmod(id, 5) " +
            "WHEN 0 THEN 10 WHEN 1 THEN 11 WHEN 2 THEN 12 WHEN 3 THEN 13 ELSE 15 END " +
            "END AS category_key")
      base
        .withColumn("quantity",
          expr("CASE WHEN id % 13 = 0 THEN CAST(NULL AS INT) " +
            "WHEN category_key IN (11, 13) THEN CAST(-(pmod(id, 50) + 1) AS INT) " +
            "ELSE CAST(pmod(id, 50) + 1 AS INT) END"))
        .withColumn("price",
          expr("CASE WHEN id % 11 = 0 THEN CAST(NULL AS DECIMAL(7,2)) " +
            "ELSE CAST(pmod(id, 300) + 100 AS DECIMAL(7,2)) END"))
        .withColumn("amount",
          expr("CASE WHEN category_key = 12 THEN CAST(-(1000000 + pmod(id, 7) * 13) AS INT) " +
            "ELSE CAST(1000000 + pmod(id, 7) * 13 AS INT) END"))
        .drop("id")
        .coalesce(1)
        .write
        .option("parquet.page.size", "8192")
        .option("parquet.block.size", "65536")
        .mode("errorifexists").parquet(path)
    } finally {
      if (previousBlockSize != null) hadoopConf.set("parquet.block.size", previousBlockSize)
      else hadoopConf.unset("parquet.block.size")
    }
  }

  private def singlePartFile(path: String): String = {
    val files = listParquetFiles(path)
    require(files.length == 1, s"expected exactly 1 part file under $path, got $files")
    files.head
  }

  /**
   * Test-fixture sanity check (mirrors runDecodeAndCountMode's
   * `fullyNullPages` assertion for the dictionary path, generalized to any
   * column/predicate): walks `file`'s raw Parquet pages for `columnName` and
   * requires at least one page whose entire row range satisfies
   * `isNullRow` -- proving, from the file's own physical page layout rather
   * than an assumption about how the writer chose to split pages, that the
   * fixture actually contains the "a whole page decodes to zero non-null
   * values" shape a test claims to exercise. `isNullRow` is the SUITE'S OWN
   * ground truth (the exact predicate used to construct the column), not a
   * decode of the file, so this needs no native call at all.
   */
  private def assertHasFullyNullPage(file: String, columnName: String, isNullRow: Int => Boolean): Unit = {
    val reader = ParquetFileReader.open(HadoopInputFile.fromPath(new Path(file), new Configuration()))
    try {
      val descriptor = reader.getFooter.getFileMetaData.getSchema.getColumns.asScala
        .find(_.getPath()(0) == columnName)
        .getOrElse(throw new RuntimeException(s"$file: missing column $columnName"))
      var fullyNullPages = 0
      var globalRowOffset = 0
      var pageReadStore = reader.readNextRowGroup()
      while (pageReadStore != null) {
        val pageReader = pageReadStore.getPageReader(descriptor)
        pageReader.readDictionaryPage() // consumed if present; a PLAIN chunk has none.
        var rowOffset = globalRowOffset
        var rawPage = pageReader.readPage()
        while (rawPage != null) {
          val valueCount = rawPage.getValueCount
          if (valueCount > 0 && (rowOffset until rowOffset + valueCount).forall(isNullRow)) {
            fullyNullPages += 1
          }
          rowOffset += valueCount
          rawPage = pageReader.readPage()
        }
        globalRowOffset += pageReadStore.getRowCount.toInt
        pageReadStore = reader.readNextRowGroup()
      }
      require(fullyNullPages > 0,
        s"$file: expected at least one fully-null $columnName page (observed 0) -- this test fixture " +
          "does not actually exercise the all-null-page shape it claims to")
    } finally {
      reader.close()
    }
  }

  private val GroupedExecKeyColumns = Seq("cat_key", "region_key")
  private val GroupedExecMeasureColumns = Seq("quantity", "amount")

  /**
   * Task 5: drive MetalParquetGroupedAggregateExec directly (it is not yet
   * reachable from the planner -- that's Task 6) over a synthetic star: a
   * fact table with 2 join keys (cat_key -> an ATTRIBUTED dimension with a
   * string and a decimal attribute, region_key -> an attribute-free
   * dimension with a DUPLICATE key) and 2 measures (quantity, amount) that
   * both carry nulls and negatives. Four user aggregates cover every
   * function/kind combination this operator supports: sum(quantity),
   * count(quantity), avg(amount), count(1).
   *
   * Each case compares the operator's emitted PARTIAL rows -- merged per
   * group on the JVM (sum ignoring None+None==None, count/occupancy summed
   * plainly, avg finalized as mergedSum/mergedCount) -- against Spark's own
   * end-to-end join + groupBy + agg answer over the SAME files/dimensions,
   * per the brief (comparing partials against a plan-substituted final
   * aggregate is not attempted; the actual join+groupBy is the ground
   * truth).
   */
  private def runAggExecMode(nativeLibrary: String, metalLibrary: String, rows: Int): Unit = {
    val spark = SparkSession.builder().appName("spark-metal-grouped-aggregate-exec-smoke").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    val tempDir = Files.createTempDirectory("spark-metal-grouped-aggregate-exec-smoke")
    try {
      SparkMetalNative.ensureInitialized(nativeLibrary, metalLibrary)
      import spark.implicits._

      // Final-review fix wave, fix 1: an IsNotNull filter on a non-key
      // column (here, the SUM's own measure input) must never let
      // GroupedAggregateShape.matchRegion accept the region -- checked FIRST
      // and independently of the exec-level dataset setup below, which
      // exercises a different (already-shape-matched) path entirely.
      checkIsNotNullOnMeasureRejected(spark, tempDir)

      val quantityAttribute = AttributeReference("quantity", IntegerType)()
      val amountAttribute = AttributeReference("amount", IntegerType)()
      val aggSpecs = Seq(
        GroupedAggregateShape.AggSpec(
          "sum", GroupedAggregateShape.FactColumn(quantityAttribute), unscaled = false, sumDataType = LongType),
        GroupedAggregateShape.AggSpec(
          "count", GroupedAggregateShape.FactColumn(quantityAttribute), unscaled = false, sumDataType = LongType),
        GroupedAggregateShape.AggSpec(
          "avg", GroupedAggregateShape.FactColumn(amountAttribute), unscaled = false, sumDataType = DoubleType),
        GroupedAggregateShape.AggSpec(
          "count", GroupedAggregateShape.CountStar, unscaled = false, sumDataType = LongType))

      // Group-key output columns come from dimension 0 (cat_key) only --
      // dimension 1 (region_key) is attribute-free and contributes nothing
      // but a membership gate + multiplicity factor.
      val outputAttributes = Seq(
        AttributeReference("major", IntegerType, nullable = true)(),
        AttributeReference("brand", StringType, nullable = true)(),
        AttributeReference("listPrice", DecimalType(7, 2), nullable = true)(),
        AttributeReference("sumQuantity", LongType, nullable = true)(),
        AttributeReference("countQuantity", LongType, nullable = false)(),
        AttributeReference("avgAmountSum", DoubleType, nullable = true)(),
        AttributeReference("avgAmountCount", LongType, nullable = false)(),
        AttributeReference("countStar", LongType, nullable = false)())
      val groupKeyDimensionIndex = Seq((0, 0), (0, 1), (0, 2))

      // cat_key in {1,2,3,4}: key 4 has no fact rows (an all-zero group is
      // guaranteed); a fact-side cat_key of 5 matches no dimension row
      // (code-gate). listPrice is explicitly cast to DecimalType(7,2) --
      // the SAME precision/scale outputAttributes declares -- since
      // UnsafeRow's getDecimal(ordinal, precision, scale) decodes
      // differently depending on the precision it is told, and the two
      // must agree for a real UnsafeRow-backed dimension row (unlike
      // GenericInternalRow, which ignores precision entirely).
      // cat_key = 6 is a dedicated "all-null-quantity" group (see the id < 50
      // reserved band in writeFactDataset below): it proves the per-aggregate
      // null gate (sum is null iff its paired non-null count is zero) fires
      // independently per aggregate -- that group's sum(quantity) is null
      // while its avg(amount) is populated, in the SAME emitted row. (cat_key
      // = 5 is deliberately left with NO dimension row -- the main dataset's
      // cat_key formula below ranges over 1..5, so 5 remains the existing
      // "fact value matches no dimension row" code-gate case.)
      val catRows = Seq(
        (1, 1, "acme", BigDecimal(19.99)),
        (2, 1, "globex", BigDecimal(19.99)),
        (3, 2, "acme", BigDecimal(29.99)),
        (4, 3, "initech", BigDecimal(29.99)),
        (6, 99, "nullq", BigDecimal(9.99)))
      val catDim = catRows.toDF("cat_key", "major", "brand", "listPrice")
        .withColumn("listPrice", col("listPrice").cast(DecimalType(7, 2)))

      // region_key in {10,20,20,30}: 20 is DUPLICATED in this
      // attribute-free dimension, folding into a multiplicity factor of 2
      // (not rejected, unlike a duplicate in an attributed dimension); a
      // fact-side region_key of 40 matches no dimension row.
      val regionDim = Seq(10, 20, 20, 30).toDF("region_key")

      def writeFactDataset(catKeyExpr: String, path: String): Unit = {
        spark.range(rows)
          .selectExpr(
            "id",
            // A narrow id < 50 band is reserved for cat_key = 6 with
            // region_key always = 10 (a guaranteed full match) and quantity
            // always null (see below) -- every other row uses the caller's
            // own cat_key formula untouched.
            s"CASE WHEN id < 50 THEN 6 ELSE ($catKeyExpr) END AS cat_key",
            "CASE WHEN id < 50 THEN 10 WHEN id % 7 = 0 THEN CAST(NULL AS INT) " +
              "ELSE CASE pmod(id, 4) WHEN 0 THEN 10 WHEN 1 THEN 20 WHEN 2 THEN 30 ELSE 40 END END AS region_key")
          .withColumn("quantity",
            expr("CASE WHEN id < 50 THEN CAST(NULL AS INT) " +
              "WHEN id % 11 = 0 THEN CAST(NULL AS INT) " +
              "WHEN pmod(id, 5) = 0 THEN CAST(-(pmod(id, 20) + 1) AS INT) " +
              "ELSE CAST(pmod(id, 20) + 1 AS INT) END"))
          .withColumn("amount",
            expr("CASE WHEN id % 13 = 0 THEN CAST(NULL AS INT) " +
              "WHEN pmod(id, 6) = 0 THEN CAST(-(1000 + pmod(id, 50)) AS INT) " +
              "ELSE CAST(1000 + pmod(id, 50) AS INT) END"))
          .drop("id")
          .coalesce(1)
          .write.mode("errorifexists").parquet(path)
      }

      val mainPath = tempDir.resolve("data-main").toString
      writeFactDataset(
        "CASE WHEN id % 17 = 0 THEN CAST(NULL AS INT) ELSE CAST(pmod(id, 5) + 1 AS INT) END", mainPath)
      val mainFile = singlePartFile(mainPath)

      // A second file whose cat_key is near-unique (raw id, offset well
      // past the dimension's real key range) for ~99% of rows -- same
      // trick as writeDictionaryOverflowDataset -- so parquet-mr abandons
      // the dictionary for it. Before Task 6b, decodeKeyColumn's
      // dictionaryPage == null check threw for every row group of this
      // file, driving it through the per-row-group CPU fallback; Task 6b's
      // PLAIN key path now decodes it on the GPU instead (see Case B below).
      // Every 100th row is deliberately a REAL match (cat_key/region_key
      // both drawn from the dimensions' actual key sets) so the GPU's
      // value-space code table -- both its populated entries AND its
      // -1-filled non-member entries for the other ~99% of (out-of-range)
      // ids -- is genuinely exercised, not just a file that matches nothing
      // at all.
      val overflowPath = tempDir.resolve("data-overflow").toString
      val hadoopConf = spark.sparkContext.hadoopConfiguration
      val previousDictionarySize = hadoopConf.get("parquet.dictionary.page.size")
      hadoopConf.setInt("parquet.dictionary.page.size", 4096)
      try {
        spark.range(rows)
          .selectExpr(
            "id",
            "CASE WHEN id % 100 = 0 THEN CAST(pmod(id, 4) + 1 AS INT) " +
              "ELSE CAST(id + 1000000 AS INT) END AS cat_key",
            "CASE WHEN id % 100 = 0 THEN CAST(pmod(id, 3) * 10 + 10 AS INT) " +
              "ELSE CAST(pmod(id, 4) * 10 + 10 AS INT) END AS region_key")
          .withColumn("quantity",
            expr("CASE WHEN id % 9 = 0 THEN CAST(NULL AS INT) ELSE CAST(pmod(id, 20) + 1 AS INT) END"))
          .withColumn("amount",
            expr("CASE WHEN id % 15 = 0 THEN CAST(NULL AS INT) ELSE CAST(1000 + pmod(id, 50) AS INT) END"))
          .drop("id")
          .coalesce(1)
          .write
          .option("parquet.dictionary.page.size", "4096")
          .mode("errorifexists").parquet(overflowPath)
      } finally {
        if (previousDictionarySize != null) hadoopConf.set("parquet.dictionary.page.size", previousDictionarySize)
        else hadoopConf.unset("parquet.dictionary.page.size")
      }
      val overflowFile = singlePartFile(overflowPath)

      // A third file, purpose-built to restore coverage of
      // aggregateRowGroupOnCpu (the per-row-group CPU fallback) for a
      // genuinely mixed-encoding key CHUNK: one row group whose cat_key
      // column chunk carries a REAL dictionary page (decodeKeyColumn sees
      // isPlain = false for the whole chunk) followed, mid-chunk, by a page
      // that parquet-mr itself wrote as PLAIN (the dictionary overflowed
      // partway through). decodeKeyColumn's per-page encoding assert then
      // throws ("expected dictionary encoding, got PLAIN") on that later
      // page -- exactly the check Task 6b added alongside its PLAIN-chunk
      // path (see decodeKeyColumn) -- driving this row group through the
      // per-row-group CPU fallback for a reason INDEPENDENT of Task 6b's own
      // PLAIN-chunk recovery (that recovery is instead covered by
      // "dictionary-overflow-plain-recovered-on-gpu" above, whose whole
      // point is that a chunk with NO dictionary page at all now succeeds on
      // the GPU).
      //
      // Getting a genuine dictionary-page-then-PLAIN-page split within ONE
      // row group needs care: a column that is unique-valued from row 0 (the
      // trick overflowFile/writeDictionaryOverflowDataset use) empirically
      // abandons its dictionary before ever completing a first page, so no
      // dictionary page is ever written at all -- which is why THIS dataset
      // instead runs in two phases: the first 2,000 rows draw cat_key from
      // the dimension's real, tiny key set (so a page checkpoint -- forced
      // early by small parquet.page.size/parquet.page.row.count.limit
      // overrides -- flushes at least one genuine dictionary-encoded page
      // before anything can overflow), and the remaining rows are
      // near-unique large values that reliably blow the (deliberately small)
      // parquet.dictionary.page.size budget well before the row group ends.
      // Verified empirically (see task-6b-report.md's fix-round addendum)
      // that this produces ColumnChunkMetaData.getEncodingStats() with both
      // hasDictionaryPages() and hasNonDictionaryEncodedPages() true, in
      // exactly one row group, for this file's actual row count.
      val midChunkFallbackPath = tempDir.resolve("data-mid-chunk-fallback").toString
      spark.range(rows)
        .selectExpr(
          "id",
          "CASE WHEN id < 2000 THEN CAST(pmod(id, 4) + 1 AS INT) " +
            "ELSE CAST(id + 1000000 AS INT) END AS cat_key",
          "CASE pmod(id, 4) WHEN 0 THEN 10 WHEN 1 THEN 20 WHEN 2 THEN 30 ELSE 40 END AS region_key")
        .withColumn("quantity",
          expr("CASE WHEN id % 11 = 0 THEN CAST(NULL AS INT) ELSE CAST(pmod(id, 20) + 1 AS INT) END"))
        .withColumn("amount",
          expr("CASE WHEN id % 13 = 0 THEN CAST(NULL AS INT) ELSE CAST(1000 + pmod(id, 50) AS INT) END"))
        .drop("id")
        .coalesce(1)
        .write
        .option("parquet.dictionary.page.size", "8000")
        .option("parquet.page.size", "256")
        .option("parquet.page.size.row.check.min", "10")
        .option("parquet.page.row.count.limit", "100")
        .option("parquet.block.size", (128L * 1024 * 1024).toString)
        .mode("errorifexists").parquet(midChunkFallbackPath)
      val midChunkFallbackFile = singlePartFile(midChunkFallbackPath)

      // A fourth file: a NULLABLE PLAIN key column with a genuine,
      // verified-present fully-null data page (Task 6b fix-round Important
      // #3 -- the PLAIN-with-nulls key path had ZERO coverage before this,
      // which is exactly why the Critical all-null-page bug (Important #1:
      // the early return on a PLAIN page skipped the with-nulls branch
      // whenever nonNullCount == 0, including a genuinely all-null page,
      // leaving those rows at the row group's zero-fill -- ids = 0,
      // validity = 0, i.e. silently "defined, key value 0") survived a
      // fully-green suite). parquet.enable.dictionary=false forces every
      // column PLAIN (deterministic -- no dictionary-overflow heuristics to
      // get right, unlike midChunkFallbackFile above); region_key is null
      // for the first `nullBand` rows and parquet.page.row.count.limit=50
      // forces small, row-count-aligned pages, so `assertHasFullyNullPage`
      // below can PROVE (from the file's own raw pages, not just an
      // assumption) that at least one emitted page is entirely null before
      // the exec ever touches it.
      //
      // regionDimZero (below, NOT the shared `regionDim`) deliberately
      // includes region_key = 0 as a REAL member: this is what makes the
      // bug actually observable as a wrong AGGREGATE, not just silently
      // absorbed. The shared catDim/regionDim never contain key 0, so a
      // leaked "key 0" row would coincidentally still be dropped as a
      // non-member (codesByKey.getOrElse(0, -1) = -1) even with the bug
      // present -- a false-negative test. With regionDimZero's key 0 a
      // genuine member, a leaked null-as-0 row instead joins group 0 for
      // real, inflating that group's sum/count -- a mismatch
      // sparkReference's true inner join (which correctly drops null keys)
      // will catch. cat_key is left ordinary (cycling the dimensions' real
      // keys) so only region_key's nulls are under test.
      val nullBand = 500
      val nullablePlainKeyPath = tempDir.resolve("data-nullable-plain-key").toString
      spark.range(rows)
        .selectExpr(
          "id",
          "CAST(pmod(id, 4) + 1 AS INT) AS cat_key",
          s"CASE WHEN id < $nullBand THEN CAST(NULL AS INT) " +
            "ELSE CASE pmod(id, 4) WHEN 0 THEN 0 WHEN 1 THEN 20 WHEN 2 THEN 30 ELSE 40 END " +
            "END AS region_key")
        .withColumn("quantity",
          expr("CASE WHEN id % 11 = 0 THEN CAST(NULL AS INT) ELSE CAST(pmod(id, 20) + 1 AS INT) END"))
        .withColumn("amount",
          expr("CASE WHEN id % 13 = 0 THEN CAST(NULL AS INT) ELSE CAST(1000 + pmod(id, 50) AS INT) END"))
        .drop("id")
        .coalesce(1)
        .write
        .option("parquet.enable.dictionary", "false")
        .option("parquet.page.row.count.limit", "50")
        .option("parquet.page.size.row.check.min", "10")
        .mode("errorifexists").parquet(nullablePlainKeyPath)
      val nullablePlainKeyFile = singlePartFile(nullablePlainKeyPath)
      assertHasFullyNullPage(nullablePlainKeyFile, "region_key", row => row < nullBand)
      val regionDimZero = Seq(0, 20, 20, 30).toDF("region_key")

      def buildExec(
          files: Seq[String],
          catDimDF: org.apache.spark.sql.DataFrame,
          regionDimDF: org.apache.spark.sql.DataFrame): MetalParquetGroupedAggregateExec =
        MetalParquetGroupedAggregateExec(
          outputAttributes, files, GroupedExecKeyColumns, GroupedExecMeasureColumns, aggSpecs,
          groupKeyDimensionIndex,
          Seq(catDimDF.queryExecution.executedPlan, regionDimDF.queryExecution.executedPlan),
          nativeLibrary, metalLibrary)

      def decimalCents(decimal: java.math.BigDecimal): Long = decimal.movePointRight(2).setScale(0).longValueExact()

      type GroupKey = (Int, String, Long)
      // avgSum is a plain (never-null) Double, NOT Option[Double]: per
      // Average.initialValues == (0, 0L) and its non-coalescing merge, the
      // partial avg-sum buffer this operator emits must never be null (see
      // buildOutputRow's CRITICAL comment) -- modeling it as an Option here
      // would let a reintroduced null slip through a lenient
      // getOrElse(0.0)-style merge instead of failing loudly. sumQuantity
      // stays Option[Long]: Sum's buffer genuinely starts null and its
      // merge DOES coalesce.
      type MergedTuple = (Option[Long], Long, Double, Long, Long) // sumQ, countQ, avgSum, avgCount, countStar
      type FinalTuple = (Option[Long], Long, Option[Double], Long) // sumQ, countQ, avg, countStar

      def sparkReference(
          files: Seq[String],
          catDimDF: org.apache.spark.sql.DataFrame,
          regionDimDF: org.apache.spark.sql.DataFrame): Map[GroupKey, FinalTuple] = {
        val fact = spark.read.parquet(files: _*)
        val enriched = fact
          .join(catDimDF, fact("cat_key") === catDimDF("cat_key"))
          .join(regionDimDF, fact("region_key") === regionDimDF("region_key"))
        val referenceAgg = enriched.groupBy("major", "brand", "listPrice").agg(
          sum(col("quantity")).as("sumQuantity"),
          count(col("quantity")).as("countQuantity"),
          avg(col("amount")).as("avgAmount"),
          count(lit(1)).as("countStar")
        ).collect()
        referenceAgg.map { row =>
          val key: GroupKey = (row.getInt(0), row.getString(1), decimalCents(row.getDecimal(2)))
          val sumQuantity = if (row.isNullAt(3)) None else Some(row.getLong(3))
          val countQuantity = row.getLong(4)
          val avgAmount = if (row.isNullAt(5)) None else Some(row.getDouble(5))
          val countStar = row.getLong(6)
          key -> (sumQuantity, countQuantity, avgAmount, countStar)
        }.toMap
      }

      def mergeAndFinalize(caseName: String, collected: Array[InternalRow]): Map[GroupKey, FinalTuple] = {
        val merged = mutable.LinkedHashMap.empty[GroupKey, MergedTuple]
        collected.foreach { row =>
          val major = row.getInt(0)
          val brand = row.getUTF8String(1).toString
          val listPriceCents = decimalCents(row.getDecimal(2, 7, 2).toJavaBigDecimal)
          val key: GroupKey = (major, brand, listPriceCents)
          // Sum's partial buffer genuinely starts null (coalescing merge,
          // per Sum.mergeExpressions): null iff this partial's non-null
          // count was zero.
          val sumQuantity = if (row.isNullAt(3)) None else Some(row.getLong(3))
          val countQuantity = row.getLong(4)
          // Avg's partial sum must NEVER be null (Average.initialValues is
          // (0, 0L), and its mergeExpressions is a plain, NON-coalescing
          // Add) -- assert that contract here, at the point every partial
          // row is actually read, rather than letting a violation surface
          // several steps later as a confusing final-answer mismatch.
          require(!row.isNullAt(5),
            s"$caseName: avg sum partial buffer was null -- Average.initialValues is (0, 0L), never null")
          val avgSum = row.getDouble(5)
          val avgCount = row.getLong(6)
          val countStar = row.getLong(7)
          val existing = merged.getOrElse(key, (None, 0L, 0.0, 0L, 0L))
          val combinedSum = (existing._1, sumQuantity) match {
            case (None, None) => None
            case (a, b) => Some(a.getOrElse(0L) + b.getOrElse(0L))
          }
          // Plain (non-coalescing) add, exactly matching Average's real
          // mergeExpressions -- both operands are guaranteed non-null by
          // the require above, so there is nothing to coalesce; if a future
          // regression ever DID let a null slip through as a garbage 0.0,
          // this is the same arithmetic Spark's own merge would apply, not
          // a more lenient stand-in for it.
          val combinedAvgSum = existing._3 + avgSum
          merged(key) = (combinedSum, existing._2 + countQuantity, combinedAvgSum, existing._4 + avgCount, existing._5 + countStar)
        }
        merged.view.mapValues { case (sumQ, countQ, avgSum, avgCount, countStar) =>
          val finalAvg = if (avgCount > 0) Some(avgSum / avgCount) else None
          (sumQ, countQ, finalAvg, countStar)
        }.toMap
      }

      def compare(
          caseName: String,
          files: Seq[String],
          catDimDF: org.apache.spark.sql.DataFrame,
          regionDimDF: org.apache.spark.sql.DataFrame,
          minCpuFallback: Option[Long] = None,
          exactCpuFallback: Option[Long] = None,
          requireGpuCpuMix: Boolean = false): Unit = {
        val exec = buildExec(files, catDimDF, regionDimDF)
        val collected = exec.execute().collect()
        val actual = mergeAndFinalize(caseName, collected)
        val reference = sparkReference(files, catDimDF, regionDimDF)

        val cpuFallback = exec.metrics("cpuFallbackRowGroups").value
        val numRowGroups = exec.metrics("numRowGroups").value
        println(
          s"""{"mode":"agg-exec","case":"$caseName","groupsActual":${actual.size},""" +
            s""""groupsExpected":${reference.size},"numRowGroups":$numRowGroups,""" +
            s""""cpuFallbackRowGroups":$cpuFallback}""")

        require(reference.nonEmpty, s"$caseName: expected the reference join+groupBy to produce at least one group")
        require(actual.keySet == reference.keySet,
          s"$caseName: group key sets differ. actualOnly=${actual.keySet -- reference.keySet} " +
            s"expectedOnly=${reference.keySet -- actual.keySet}")
        var mismatches = 0
        actual.foreach { case (key, value) =>
          if (value != reference(key)) {
            mismatches += 1
            println(
              s"""{"mode":"agg-exec","case":"$caseName","mismatchKey":"$key",""" +
                s""""actual":"$value","expected":"${reference(key)}"}""")
          }
        }
        println(s"""{"mode":"agg-exec","case":"$caseName","mismatches":$mismatches,"match":${mismatches == 0}}""")
        if (mismatches != 0) sys.exit(1)
        minCpuFallback.foreach { min =>
          require(cpuFallback >= min, s"$caseName: expected cpuFallbackRowGroups >= $min, got $cpuFallback")
        }
        exactCpuFallback.foreach { expected =>
          require(cpuFallback == expected,
            s"$caseName: expected cpuFallbackRowGroups == $expected (pure GPU path), got $cpuFallback")
        }
        if (requireGpuCpuMix) {
          require(cpuFallback > 0 && cpuFallback < numRowGroups,
            s"$caseName: expected a genuine MIX of GPU-successful and CPU-fallback row groups, " +
              s"got cpuFallbackRowGroups=$cpuFallback of numRowGroups=$numRowGroups")
        }
      }

      // Case A: a single, dictionary-friendly file -- the normal (dense)
      // GroupSpace path. exactCpuFallback = 0 proves the GPU path actually
      // ran (not just that the final answer happened to match).
      compare("main", Seq(mainFile), catDim, regionDim, exactCpuFallback = Some(0))

      // Case B: main + a dictionary-overflow file. Before Task 6b, cat_key's
      // dictionary-less chunk in overflowFile made decodeKeyColumn throw
      // ("has no dictionary page"), driving that file's row group through
      // the per-row-group CPU fallback while mainFile's stayed GPU-
      // successful (a genuine mix). Task 6b's whole point is that this no
      // longer happens: decodeKeyColumn now decodes a key column with no
      // dictionary page via the PLAIN path into a dense VALUE-space code
      // table, so overflowFile's row group ALSO succeeds on the GPU --
      // exactCpuFallback = 0 proves exactly that (this is the operator-level
      // analogue of the real q3/ss_item_sk recovery the TPC-DS smoke
      // exercises). The combined result is still exact either way, but this
      // now demonstrates the value-space table's own membership filtering
      // genuinely running on the GPU: only every 100th row of overflowFile
      // (id % 100 == 0) carries a cat_key/region_key drawn from the
      // dimensions' real key range -- the other ~99% are non-members (raw
      // ids offset past dimMaxKey, or simply absent from codesByKey), and
      // must fall out via the kernel's existing bounds/sign checks, not a
      // CPU recompute. (This dataset no longer forces the per-row-group CPU
      // fallback for a KEY column via "whole chunk is PLAIN, no dictionary
      // page at all" -- Case B2 below restores that coverage via a
      // different, still-genuine trigger: a chunk that legitimately mixes a
      // real dictionary page with a later PLAIN page. Coverage of the
      // dictionary-required version of THIS case's original trick lives on
      // in "exec" mode's identically-named case, which drives
      // MetalParquetMembershipCountExec -- a different operator whose key
      // decode Task 6b did not touch.)
      compare(
        "dictionary-overflow-plain-recovered-on-gpu", Seq(mainFile, overflowFile), catDim, regionDim,
        exactCpuFallback = Some(0))

      // Case B2: main + midChunkFallbackFile (see its construction above).
      // midChunkFallbackFile's cat_key chunk carries a REAL dictionary page,
      // so decodeKeyColumn takes the (unchanged) dictionary path for the
      // whole chunk -- but a later page in that SAME chunk is actually
      // PLAIN (parquet-mr's own dictionary overflowed mid-chunk), which
      // decodeKeyColumn's per-page encoding assert rejects
      // ("expected dictionary encoding, got PLAIN"), throwing and driving
      // this row group through aggregateRowGroupOnCpu -- restoring this
      // suite's only coverage of the grouped-aggregate operator's
      // per-row-group CPU fallback (lost when Case B above stopped
      // triggering it). requireGpuCpuMix proves this is a genuine mix
      // (mainFile's row group still GPU-successful, not every row group
      // silently falling back); the combined result must still be exact.
      compare(
        "mid-chunk-dictionary-to-plain-cpu-fallback", Seq(mainFile, midChunkFallbackFile), catDim, regionDim,
        minCpuFallback = Some(1), requireGpuCpuMix = true)

      // Case B3: nullablePlainKeyFile alone, joined against regionDimZero
      // (NOT the shared regionDim -- see the file's own construction
      // comment above for why key 0 must be a genuine member here).
      // region_key is entirely PLAIN (parquet.enable.dictionary=false) and
      // includes a verified (assertHasFullyNullPage, above) fully-null page
      // among its nulls. exactCpuFallback = 0 both proves the PLAIN-with-
      // nulls path ran on the GPU (not a fallback masking the bug) and,
      // combined with the exact result_match check inside compare(),
      // exercises the Critical fix (Important #1): before it, this case's
      // leaked null rows would present as region_key = 0, a real member of
      // regionDimZero, silently inflating that group's sum/count/avg --
      // wrong without ever throwing or falling back, i.e. undetectable by
      // any plan-shape or fallback-count check, only by comparing the exact
      // aggregate values against sparkReference's true inner join. See
      // task-6b-report.md's fix-round addendum for the RED evidence
      // (git-stash ablation back to the pre-fix early return) that this
      // case's mismatch check actually fails without the fix.
      compare(
        "nullable-plain-key-fully-null-page", Seq(nullablePlainKeyFile), catDim, regionDimZero,
        exactCpuFallback = Some(0))

      // Case C: cat_key = 1 appears TWICE in the (attributed) catDim with
      // DIFFERENT attribute tuples -- GroupSpace.build rejects this
      // (Left), forcing the whole-operator CPU hash-join + hash-aggregate
      // fallback. Real join fan-out means fact rows with cat_key = 1 now
      // belong to TWO groups; the Spark reference join naturally produces
      // that fan-out too, so an exact match here proves the fallback's
      // semantics are correct, not just non-crashing.
      val catRowsDuplicated = catRows :+ ((1, 9, "other", BigDecimal(50.00)))
      val catDimDuplicated = catRowsDuplicated.toDF("cat_key", "major", "brand", "listPrice")
        .withColumn("listPrice", col("listPrice").cast(DecimalType(7, 2)))
      compare("whole-operator-cpu-duplicate-key", Seq(mainFile), catDimDuplicated, regionDim)

      // Case D (regression for the CRITICAL avg-null-partial bug): cat_key
      // = 7 ("avgnull" dimension row) receives fact rows from TWO SEPARATE
      // FILES -- forcing >= 2 splits/partitions (numPartitions =
      // min(splits.length, defaultParallelism), and a 2-element Seq sliced
      // into 2 partitions gets exactly one element each) so this ONE group
      // genuinely spans a partition boundary. One file's rows all have a
      // NULL amount (that partition's avg(amount) internal (sum, count) is
      // (0, 0) for this group -- exactly the case that used to emit a null
      // avg-sum); the other file's rows have a real, non-null amount. If
      // buildOutputRow ever regresses to gating avg's sum on
      // non-null-count again, the all-null-amount partition's partial row
      // emits a null avg-sum, mergeAndFinalize's non-coalescing add
      // propagates that null forever, and the final avg comes out None --
      // but Spark's OWN real end-to-end answer for this group is
      // non-null (the OTHER file's rows have real amounts), so `compare`
      // fails on a group-value mismatch instead of silently passing.
      val catRowsWithAvgNull = catRows :+ ((7, 77, "avgnull", BigDecimal(1.23)))
      val catDimWithAvgNull = catRowsWithAvgNull.toDF("cat_key", "major", "brand", "listPrice")
        .withColumn("listPrice", col("listPrice").cast(DecimalType(7, 2)))
      val avgNullPath = tempDir.resolve("data-avg-null-part").toString
      writeAvgSplitDataset(spark, avgNullPath, rowsInPart = 200, amountNull = true)
      val avgNullFile = singlePartFile(avgNullPath)
      val avgValuePath = tempDir.resolve("data-avg-value-part").toString
      writeAvgSplitDataset(spark, avgValuePath, rowsInPart = 200, amountNull = false)
      val avgValueFile = singlePartFile(avgValuePath)
      compare("cross-partition-avg-null", Seq(avgNullFile, avgValueFile), catDimWithAvgNull, regionDim)

      // Case E: an unsupported dimension attribute type (DoubleType) must
      // throw driver-side, at doExecute(), NOT route into the
      // whole-operator CPU fallback -- that fallback's attributeValue()
      // would otherwise crash with an unhandled IllegalStateException deep
      // inside an executor task the first time it tried to key its hash
      // map by that value, which is a much worse failure mode than a clear
      // driver-side throw naming the offending type.
      val badAttributeDim = Seq((1, 1.5d)).toDF("cat_key", "badAttr")
      val badOutputAttributes = Seq(
        AttributeReference("badAttr", DoubleType, nullable = true)(),
        AttributeReference("countStar", LongType, nullable = false)())
      val badAggSpecs = Seq(GroupedAggregateShape.AggSpec(
        "count", GroupedAggregateShape.CountStar, unscaled = false, sumDataType = LongType))
      val badExec = MetalParquetGroupedAggregateExec(
        badOutputAttributes, Seq(mainFile), Seq("cat_key"), Seq.empty, badAggSpecs, Seq((0, 0)),
        Seq(badAttributeDim.queryExecution.executedPlan), nativeLibrary, metalLibrary)
      val threwIllegalState = try {
        badExec.execute()
        false
      } catch {
        case _: IllegalStateException => true
      }
      println(s"""{"mode":"agg-exec","case":"unsupported-attribute-type-throws","threw":$threwIllegalState}""")
      require(threwIllegalState,
        "expected an unsupported dimension attribute type to throw IllegalStateException at doExecute(), " +
          "not fall back or silently succeed")
    } finally {
      spark.stop()
    }
  }

  /**
   * Final-review fix wave, fix 1 (RED first): builds a broadcast-joined
   * dimension/fact query whose partial HashAggregateExec would otherwise
   * match the v1 grouped-aggregate region shape, but adds an explicit
   * `WHERE f.measure IS NOT NULL` -- a filter on the SUM's own measure
   * column, not a join key. Before the fix, GroupedAggregateShape's
   * `isOnlyNotNullPredicate` accepted IsNotNull over ANY attribute purely by
   * predicate shape, so this query's region matched (Right(region)) and the
   * operator would have silently summed past the null-filtering the CPU
   * plan performs. `matchRegion` must instead return Left, naming the
   * measure column, not the region.
   */
  private def checkIsNotNullOnMeasureRejected(spark: SparkSession, tempDir: java.nio.file.Path): Unit = {
    import spark.implicits._
    val dimPath = tempDir.resolve("isnotnull-dim").toString
    val factPath = tempDir.resolve("isnotnull-fact").toString
    Seq((1, "acme"), (2, "globex")).toDF("cat_key", "name")
      .coalesce(1).write.mode("errorifexists").parquet(dimPath)
    Seq((1, Some(10)), (2, Some(20)), (1, None: Option[Int]))
      .toDF("cat_key", "measure")
      .coalesce(1).write.mode("errorifexists").parquet(factPath)

    spark.read.parquet(dimPath).createOrReplaceTempView("isnotnull_dim")
    spark.read.parquet(factPath).createOrReplaceTempView("isnotnull_fact")
    try {
      val plan = spark.sql(
        "SELECT d.name, sum(f.measure) FROM isnotnull_fact f " +
          "JOIN isnotnull_dim d ON f.cat_key = d.cat_key " +
          "WHERE f.measure IS NOT NULL GROUP BY d.name"
      ).queryExecution.executedPlan

      val outcomes = plan.collect {
        case aggregate: HashAggregateExec
            if aggregate.aggregateExpressions.nonEmpty && aggregate.aggregateExpressions.forall(_.mode == Partial) =>
          GroupedAggregateShape.matchRegion(aggregate)
      }
      require(outcomes.nonEmpty,
        s"expected at least one partial HashAggregateExec in the plan for the IsNotNull-on-measure probe:\n$plan")
      val reasons = outcomes.map {
        case Left(reason) => reason
        case Right(region) =>
          throw new IllegalStateException(
            s"expected matchRegion to reject an IsNotNull filter on the measure column, " +
              s"but it matched: $region")
      }
      require(reasons.forall(reason => reason.contains("IsNotNull") && reason.contains("non-key")),
        s"expected the rejection reason to name a non-key IsNotNull filter, got: $reasons")
      println(s"""{"mode":"agg-exec","case":"isnotnull-on-measure-rejected","reasons":${reasons.mkString("[\"", "\", \"", "\"]")}}""")
    } finally {
      spark.catalog.dropTempView("isnotnull_dim")
      spark.catalog.dropTempView("isnotnull_fact")
    }
  }

  /**
   * One partition-sized file for the cross-partition avg-null regression
   * case: every row is cat_key = 7 / region_key = 10 (both guaranteed
   * matches), quantity always non-null, and amount either ALWAYS null
   * (`amountNull = true`) or always non-null (`amountNull = false`).
   */
  private def writeAvgSplitDataset(
      spark: SparkSession, path: String, rowsInPart: Int, amountNull: Boolean): Unit = {
    val base = spark.range(rowsInPart)
      .selectExpr("id", "CAST(7 AS INT) AS cat_key", "CAST(10 AS INT) AS region_key")
      .withColumn("quantity", expr("CAST(pmod(id, 10) + 1 AS INT)"))
    val withAmount =
      if (amountNull) base.withColumn("amount", expr("CAST(NULL AS INT)"))
      else base.withColumn("amount", expr("CAST(500 + pmod(id, 20) AS INT)"))
    withAmount.drop("id").coalesce(1).write.mode("errorifexists").parquet(path)
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
          minFiles: Option[Int] = None,
          minCpuFallbackRowGroups: Option[Int] = None,
          requirePartialFallback: Boolean = false): Unit = {
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
        val observedCpuFallbacks = exec.metrics("cpuFallbackRowGroups").value

        val fact = spark.read.parquet(files: _*).select("k0", "k1", "k2")
        val expectedCount = fact
          .join(dim0, fact("k0") === dim0("k0j"))
          .join(dim1, fact("k1") === dim1("k1j"))
          .join(dim2, fact("k2") === dim2("k2j"))
          .count()

        println(
          s"""{"mode":"exec","case":"$caseName","rows":$rows,"files":${files.length},""" +
            s""""numRowGroups":$observedRowGroups,"cpuFallbackRowGroups":$observedCpuFallbacks,""" +
            s""""expectedCount":$expectedCount,"actualCount":$actualCount}""")
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
        minCpuFallbackRowGroups.foreach { min =>
          require(observedCpuFallbacks >= min,
            s"$caseName: expected cpuFallbackRowGroups >= $min, observed $observedCpuFallbacks")
        }
        if (requirePartialFallback) {
          require(observedCpuFallbacks > 0 && observedCpuFallbacks < observedRowGroups,
            s"$caseName: expected a MIX of CPU fallback and GPU-successful row groups, " +
              s"observed $observedCpuFallbacks of $observedRowGroups on CPU")
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
      // NOTE: this dataset's row groups turn out to fall back to the CPU
      // path themselves (parquet-mr abandons the dictionary for most/all
      // chunks once parquet.block.size is this small -- see
      // writeDictionaryOverflowDataset below for a much more direct,
      // deterministic way to force that). That's fine here: this layout's
      // job is only to prove the multi-row-group-in-one-file split-planning
      // and reader-reuse path works and the total still matches, regardless
      // of which path each row group takes.
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

      // --- Layout D: required (non-nullable) columns, routed to the CPU
      // fallback ------------------------------------------------------------
      // countRowGroupOnCpu classified every row of a `required` column
      // (maxDefinitionLevel == 0 -- planning-time eligibility admits this,
      // not just optional maxDef == 1 columns) as null: it compared
      // getCurrentDefinitionLevel() against the literal 0, but 0 is ALSO
      // what a required column's reader always returns (it has no
      // definition-level stream at all), so every row failed the "is this
      // row null" check and the count silently came out 0. Writing this
      // dataset via an explicit non-nullable StructType (rather than
      // relying on Spark's CAST/pmod nullability inference, which is not
      // guaranteed to propagate non-nullability all the way to the Parquet
      // writer) and verifying the footer directly locks in that the
      // dataset actually exercises maxDefinitionLevel == 0. The sparse key
      // domain (the same trick as sparse-cpu-fallback) forces every row
      // group through countRowGroupOnCpu -- the exact function that had the
      // bug.
      val pathRequired = tempDir.resolve("data-required").toString
      writeRequiredColumnsDataset(spark, pathRequired, rows)
      val requiredFile = listParquetFiles(pathRequired)
      require(requiredFile.length == 1, s"expected exactly 1 part file under $pathRequired, got $requiredFile")
      locally {
        val verifyReader = ParquetFileReader.open(
          HadoopInputFile.fromPath(new Path(requiredFile.head), new Configuration()))
        try {
          val columns = verifyReader.getFooter.getFileMetaData.getSchema.getColumns.asScala
          val maxDefinitionLevels = Columns.map(name =>
            columns.find(_.getPath()(0) == name).get.getMaxDefinitionLevel)
          println(
            s"""{"mode":"exec","case":"required-columns-schema-check",""" +
              s""""maxDefinitionLevels":${maxDefinitionLevels.mkString("[", ",", "]")}}""")
          require(maxDefinitionLevels.forall(_ == 0),
            s"expected all 3 columns required (maxDefinitionLevel == 0), got $maxDefinitionLevels")
        } finally {
          verifyReader.close()
        }
      }
      check(
        "required-columns-cpu-fallback",
        requiredFile,
        (memberLow to memberHigh) :+ 20000000,
        minCpuFallbackRowGroups = Some(1))

      // --- Layout E: mid-run native GPU-decode failure, forcing the
      // per-row-group catch -> parquetRowGroupRelease -> fresh readRowGroup
      // -> countRowGroupOnCpu branch, with OTHER row groups in the same run
      // still succeeding on the GPU -------------------------------------
      // First attempt: make k0 low-cardinality everywhere except one narrow
      // id band (raw, high-cardinality id values there), combined with a
      // small parquet.block.size + parquet.dictionary.page.size, hoping
      // parquet-mr would abandon the dictionary only for row groups
      // overlapping that band. Empirically this was extremely sensitive to
      // exact row counts, block-size checkpoint timing, and where the band
      // landed relative to row-group boundaries: probing many
      // (blockSize, dictSize) combinations swung between 0% and 100% of row
      // groups losing their dictionary with no stable partial-overflow
      // region, and shifting the row count or band position moved the
      // cliff unpredictably. Not a reliable foundation for a regression
      // test.
      //
      // Instead: two separate FILES. singleFile (already written and
      // GPU-validated by the three checks above) contributes a row group
      // that is guaranteed to decode on the GPU. A second file's k0 is the
      // raw, strictly increasing `id` -- every value unique -- so its
      // buffered dictionary size grows linearly and predictably with
      // record count; parquet-mr's periodic size-check heuristic (which
      // extrapolates the next checkpoint from the observed bytes/record)
      // converges reliably onto the true growth rate for a genuinely
      // unique-valued column, unlike the highly-compressible low-cardinality
      // data elsewhere in this file, and so reliably catches and abandons
      // the dictionary partway through -- deterministic regardless of exact
      // row count, block size, or default parquet.block.size checkpoint
      // timing. decodeRowGroupColumn's dictionaryPage == null check then
      // throws for that file's row group (k0 is ordinal 0, so the failure
      // happens before k1/k2 are ever touched), driving it through the CPU
      // fallback while singleFile's row group stays on the GPU stream in
      // the very same run -- unlike sparse-cpu-fallback/
      // required-columns-cpu-fallback above, which force *every* row group
      // to skip the GPU stream entirely via denseDomains = false.
      val pathDictionaryOverflow = tempDir.resolve("data-dictionary-overflow").toString
      writeDictionaryOverflowDataset(spark, pathDictionaryOverflow, rows)
      val dictionaryOverflowFile = listParquetFiles(pathDictionaryOverflow)
      require(dictionaryOverflowFile.length == 1,
        s"expected exactly 1 part file under $pathDictionaryOverflow, got $dictionaryOverflowFile")
      check(
        "dictionary-overflow-cpu-fallback",
        singleFile ++ dictionaryOverflowFile,
        memberLow to memberHigh,
        minRowGroups = Some(2),
        minFiles = Some(2),
        minCpuFallbackRowGroups = Some(1),
        requirePartialFallback = true)

      checkEligibilityCacheIsColumnKeyed(spark, tempDir.toString)
    } finally {
      spark.stop()
    }
  }

  /**
   * ParquetEligibility.check memoises its per-file verdict so that planning does
   * not re-read every footer on every execution. Every check it performs is
   * specific to the requested columns -- existence, INT32-ness, nesting, and the
   * codec/encodings of *those* column chunks -- so the cache has to be keyed by
   * (file version, column names) and not by the file alone.
   *
   * Both directions of getting that wrong are unsafe, and both are asserted
   * here on freshly written files so each starts with a cold cache:
   *
   *  - Ask for eligible columns, then for a column list containing a missing
   *    one. Keyed by file alone, the second call returns the cached Right and
   *    an ineligible plan walks through the safety gate, only to throw later
   *    from outside the per-row-group CPU-fallback catch.
   *  - Ask for the missing column first, then for the eligible columns. Keyed
   *    by file alone, the second call returns the cached Left and a perfectly
   *    eligible plan is silently refused acceleration.
   */
  private def checkEligibilityCacheIsColumnKeyed(spark: SparkSession, tempDir: String): Unit = {
    def partFileUnder(path: String): String =
      Files.list(java.nio.file.Path.of(path)).iterator().asScala
        .map(_.toString)
        .find(_.endsWith(".parquet"))
        .getOrElse(throw new RuntimeException(s"No .parquet part file found under $path"))

    val eligible = Columns
    val withMissingColumn = Columns.dropRight(1) :+ "not_a_column"

    val rightFirstPath = s"$tempDir/eligibility-cache-right-first"
    writeDataset(spark, rightFirstPath, 20000)
    val rightFirst = partFileUnder(rightFirstPath)
    val warmRight = ParquetEligibility.check(Seq(rightFirst), eligible)
    require(warmRight.isRight, s"expected the eligible column list to pass, got $warmRight")
    val afterWarmRight = ParquetEligibility.check(Seq(rightFirst), withMissingColumn)
    require(afterWarmRight.isLeft,
      "a column list naming a missing column must be rejected even after the same file " +
        s"passed for different columns, got $afterWarmRight")

    val leftFirstPath = s"$tempDir/eligibility-cache-left-first"
    writeDataset(spark, leftFirstPath, 20000)
    val leftFirst = partFileUnder(leftFirstPath)
    val warmLeft = ParquetEligibility.check(Seq(leftFirst), withMissingColumn)
    require(warmLeft.isLeft, s"expected the missing column to be rejected, got $warmLeft")
    val afterWarmLeft = ParquetEligibility.check(Seq(leftFirst), eligible)
    require(afterWarmLeft.isRight,
      "an eligible column list must pass even after the same file was rejected for " +
        s"different columns, got $afterWarmLeft")

    // And the cache must still answer a repeat of the same question the same way.
    require(ParquetEligibility.check(Seq(rightFirst), eligible).isRight,
      "repeating an eligible check must stay eligible")
    require(ParquetEligibility.check(Seq(leftFirst), withMissingColumn).isLeft,
      "repeating a rejected check must stay rejected")

    println(
      s"""{"mode":"exec","case":"eligibility-cache-column-keyed",""" +
        s""""eligibleThenMissing":"Right,Left","missingThenEligible":"Left,Right"}""")
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

  /**
   * All three columns `required` (non-nullable), with no null CAST anywhere
   * in the row formula. Built from an explicit non-nullable StructType
   * rather than relying on Spark's CAST/pmod nullability inference to
   * propagate all the way to the Parquet writer's schema -- this guarantees
   * maxDefinitionLevel == 0 for every column, which the exec-mode caller
   * verifies directly against the written footer.
   */
  private def writeRequiredColumnsDataset(spark: SparkSession, path: String, rows: Int): Unit = {
    val rowRdd = spark.sparkContext.range(0, rows).map { id =>
      Row(
        (((id % 900) + 50).toInt),
        (((id % 850) + 50).toInt),
        (((id % 700) + 50).toInt))
    }
    val schema = StructType(Seq(
      StructField("k0", IntegerType, nullable = false),
      StructField("k1", IntegerType, nullable = false),
      StructField("k2", IntegerType, nullable = false)))
    spark.createDataFrame(rowRdd, schema)
      .coalesce(1)
      .write
      .option("parquet.page.size", "8192")
      .mode("errorifexists").parquet(path)
  }

  /**
   * k0 is the raw `id` value -- strictly increasing, so every row's value
   * is unique. A genuinely unique-valued column's buffered dictionary size
   * grows linearly and predictably with record count (unlike this file's
   * suite's usual highly-compressible low-cardinality columns, where the
   * bytes/record estimate used by parquet-mr's periodic size-check
   * heuristic starts small and is a poor predictor of when a threshold
   * will actually be crossed). That predictability is what makes this
   * reliable: with a small parquet.dictionary.page.size, the periodic
   * checkpoint the writer schedules converges quickly onto the true growth
   * rate and is guaranteed to catch the threshold being crossed within a
   * few thousand rows, so parquet-mr abandons the dictionary (falls back
   * to PLAIN, writing no dictionary page at all) for k0's column chunk --
   * deterministically, regardless of exact row count or block-size
   * checkpoint timing. k1/k2 keep normal low-cardinality formulas and stay
   * dictionary-encoded, but that doesn't matter for whether this file's row
   * group succeeds: decodeRowGroupColumn processes k0 first (ordinal 0)
   * and throws immediately once its dictionary page is missing, before k1/
   * k2 are ever touched.
   */
  private def writeDictionaryOverflowDataset(spark: SparkSession, path: String, rows: Int): Unit = {
    val hadoopConf = spark.sparkContext.hadoopConfiguration
    val previousDictionarySize = hadoopConf.get("parquet.dictionary.page.size")
    hadoopConf.setInt("parquet.dictionary.page.size", 4096)
    try {
      spark.range(rows)
        .selectExpr(
          "CAST(id AS INT) AS k0",
          "CASE WHEN id % 17 = 0 THEN CAST(NULL AS INT) ELSE CAST(pmod(id, 850) + 50 AS INT) END AS k1",
          "CASE WHEN id % 17 = 0 THEN CAST(NULL AS INT) ELSE CAST(pmod(id, 700) + 50 AS INT) END AS k2")
        .coalesce(1)
        .write
        .option("parquet.page.size", "8192")
        .option("parquet.dictionary.page.size", "4096")
        .mode("errorifexists").parquet(path)
    } finally {
      if (previousDictionarySize != null) hadoopConf.set("parquet.dictionary.page.size", previousDictionarySize)
      else hadoopConf.unset("parquet.dictionary.page.size")
    }
  }

  private def listParquetFiles(path: String): Seq[String] =
    Files.list(java.nio.file.Path.of(path)).iterator().asScala
      .map(_.toString)
      .filter(_.endsWith(".parquet"))
      .toSeq
      .sorted
}
