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

// Round-trips a dictionary-encoded Parquet file through the CPU page parser
// and the GPU expansion kernels, then checks the decoded (dictionary id,
// validity) planes against the same file read through Spark's own Parquet
// reader. See task-3-brief.md Step 1.
object ParquetDecodeSmoke {
  private val Columns = Seq("k0", "k1", "k2")
  private val DictionaryEncodings = Set(Encoding.PLAIN_DICTIONARY, Encoding.RLE_DICTIONARY)

  def main(arguments: Array[String]): Unit = {
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

  private def writeDataset(spark: SparkSession, path: String, rows: Int): Unit = {
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
    spark.range(rows)
      .selectExpr(
        "CASE WHEN id BETWEEN 20000 AND 90000 THEN CAST(NULL AS INT) " +
          "WHEN id % 17 = 0 THEN CAST(NULL AS INT) " +
          "ELSE CAST(pmod(id, 900) + 50 AS INT) END AS k0",
        "CASE WHEN id % 17 = 0 THEN CAST(NULL AS INT) ELSE CAST(pmod(id, 850) + 50 AS INT) END AS k1",
        "CASE WHEN id % 17 = 0 THEN CAST(NULL AS INT) ELSE CAST(pmod(id, 700) + 50 AS INT) END AS k2")
      .coalesce(1)
      .write
      .option("parquet.page.size", "8192")
      .mode("errorifexists").parquet(path)
  }
}
