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

import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.types.{IntegerType, LongType, StructField, StructType}

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

  private def singlePartFile(path: String): String = {
    val files = listParquetFiles(path)
    require(files.length == 1, s"expected exactly 1 part file under $path, got $files")
    files.head
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
