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
