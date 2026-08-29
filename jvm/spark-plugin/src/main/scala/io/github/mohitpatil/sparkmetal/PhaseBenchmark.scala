package io.github.mohitpatil.sparkmetal

import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import scala.collection.mutable
import scala.jdk.CollectionConverters._

import org.apache.hadoop.fs.Path
import org.apache.parquet.column.Encoding
import org.apache.parquet.column.page.DataPageV1
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.hadoop.util.HadoopInputFile

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.SparkPlan

/**
 * Standalone per-phase benchmark: prices each stage of the Metal
 * grouped-aggregate pipeline IN ISOLATION over the same inputs the real
 * operator reads, so the umbrella timers (decodeParseTime above all) can be
 * attributed to specific stages instead of guessed at. Stages measured per
 * accelerated region, each as its own repeated, medianed measurement:
 *
 *   - planning: catalyst analysis+planning cost (first plan = cold, the
 *     rest = memoised eligibility/footers)
 *   - split_enum: Parquet footer reads, fresh vs memoised
 *   - dimension_collect: the region's dimension executeCollect fan-out
 *   - group_space_build: GroupSpace.build over those dimensions
 *   - rowgroup_read: parquet-mr row-group read + page walk + page byte
 *     copies, NO parse and NO GPU (pure JVM read side)
 *   - parse_only: rowgroup_read plus the native CPU page parse
 *     (NativeBridge.parquetParsePageBenchmark), still no GPU
 *   - spark_scan: Spark's own vectorized reader over the same columns, as
 *     the CPU-side baseline for the read+decode boundary
 *   - end_to_end: the full query, with every SQL metric of every Metal
 *     operator in the executed plan harvested from the final run
 *
 * All stage timings are single-process sums (like the SQL metrics, which sum
 * across tasks), so a stage number is directly comparable to a metric sum,
 * not to wall clock.
 */
object PhaseBenchmark {

  def main(arguments: Array[String]): Unit = arguments.headOption match {
    case Some("run") if arguments.length == 7 =>
      run(arguments(1), arguments(2), arguments(3), arguments(4).toInt, arguments(5).toInt,
        arguments(6).split(",").map(_.trim).filter(_.nonEmpty).toSeq)
    case _ =>
      throw new IllegalArgumentException(
        "Usage: PhaseBenchmark run DATA_ROOT QUERIES_DIR OUTPUT_JSON WARMUPS RUNS q1,q2,...")
  }

  private def run(
      dataRoot: String,
      queriesDir: String,
      output: String,
      warmups: Int,
      runs: Int,
      queryNames: Seq[String]): Unit = {
    require(warmups >= 0 && runs > 0 && queryNames.nonEmpty)
    val spark = SparkSession.builder().appName("spark-metal-phase-benchmark").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try {
      val tables = new File(dataRoot).listFiles().filter(_.isDirectory).sortBy(_.getName)
      require(tables.nonEmpty, s"No Parquet table directories under $dataRoot")
      tables.foreach(table => spark.read.parquet(table.getPath).createOrReplaceTempView(table.getName))

      val queryReports = queryNames.map { name =>
        println(s"Phase benchmark: $name")
        val sql = Files.readString(Paths.get(queriesDir, s"$name.sql"), StandardCharsets.UTF_8)
        name -> benchmarkQuery(spark, sql, warmups, runs)
      }

      val json = jsonObject(queryReports.map { case (name, report) => name -> report }, indent = 2)
      Files.writeString(Paths.get(output), json + "\n", StandardCharsets.UTF_8)
      println(json)
    } finally {
      spark.stop()
    }
  }

  private def benchmarkQuery(
      spark: SparkSession, sql: String, warmups: Int, runs: Int): String = {
    // Phase: planning. The first plan of a fresh statement pays cold
    // eligibility/footers (unless an earlier query already warmed the same
    // files -- reported as-is, in query order).
    val planningTimes = (0 to runs).map(_ => timeMs(spark.sql(sql).queryExecution.executedPlan))
    val plan = spark.sql(sql).queryExecution.executedPlan
    val groupedNodes = plan.collect { case node: MetalParquetGroupedAggregateExec => node }
    val membershipNodes = plan.collect { case node: MetalParquetMembershipCountExec => node }
    val joinNodes = plan.collect { case node: MetalParquetBroadcastJoinExec => node }

    val regionReports = groupedNodes.zipWithIndex.map { case (node, index) =>
      s"region_$index" -> benchmarkGroupedRegion(spark, node, runs)
    } ++ joinNodes.zipWithIndex.map { case (node, index) =>
      s"join_region_$index" -> benchmarkJoinRegion(spark, node, runs)
    }

    // Phase: end to end, with the final run's Metal metrics.
    (0 until warmups).foreach(_ => spark.sql(sql).collect())
    val endToEnd = mutable.ArrayBuffer.empty[Double]
    var finalPlan: SparkPlan = null
    for (_ <- 0 until runs) {
      val query = spark.sql(sql)
      endToEnd += timeMs(query.collect())._1
      finalPlan = query.queryExecution.executedPlan
    }
    val metalMetrics = finalPlan.collect {
      case node: SparkPlan if node.nodeName.startsWith("Metal") =>
        node.nodeName -> jsonObject(
          node.metrics.toSeq.sortBy(_._1).map { case (metricName, metric) =>
            metricName -> metric.value.toString
          }, indent = 8)
    }

    jsonObject(Seq(
      "planning_cold_ms" -> planningTimes.head._1.toString,
      "planning_warm_ms" -> median(planningTimes.tail.map(_._1)).toString,
      "grouped_regions" -> groupedNodes.length.toString,
      "membership_regions" -> membershipNodes.length.toString,
      "join_regions" -> joinNodes.length.toString,
      "end_to_end_ms" -> jsonList(endToEnd.toSeq),
      "end_to_end_median_ms" -> median(endToEnd.toSeq).toString) ++
      regionReports ++
      metalMetrics.zipWithIndex.map { case ((nodeName, metricsJson), index) =>
        s"metrics_${index}_$nodeName" -> metricsJson
      }, indent = 4)
  }

  private def benchmarkGroupedRegion(
      spark: SparkSession, node: MetalParquetGroupedAggregateExec, runs: Int): String = {
    val columns = (node.keyColumnNames ++ node.measureColumnNames).distinct

    // Phase: split enumeration -- fresh footer reads vs the memoised path
    // the operator actually takes.
    val splitEnumFresh = (0 until runs).map(_ => timeMs(freshFooterSplits(node.files))._1)
    val splitEnumMemo = (0 until runs).map(_ => timeMs(node.enumerateSplits())._1)
    val splits = node.enumerateSplits()

    // Phase: dimension collect (the operator's own parallel fan-out) and
    // GroupSpace.build over the last collected dimensions.
    val dimensionRuns = (0 until runs).map(_ => node.collectDimensions())
    val dimensionTimes = dimensionRuns.map(_._2 / 1e6)
    val dimensions = dimensionRuns.last._1
    val groupSpaceTimes = (0 until runs).map(_ => timeMs(GroupSpace.build(dimensions, 1 << 20))._1)

    // Phase: row-group read (page walk, byte copies, no parse), then the
    // same walk plus the native CPU parse. Their difference prices
    // parseDataPageV1 + the JNI byte transfer alone.
    SparkMetalNative.ensureInitialized(node.nativeLibrary, node.metalLibrary)
    val readRuns = (0 until runs).map(_ => timeMs(walkRowGroups(node.files, columns, parse = false)))
    val parseRuns = (0 until runs).map(_ => timeMs(walkRowGroups(node.files, columns, parse = true)))
    val (pages, bytes) = readRuns.last._2

    // Phase: Spark's own vectorized scan of the same columns (CPU baseline
    // for the read+decode boundary; count(col) forces every column read).
    val scanAggregates = columns.map(column => s"count($column)")
    val scanTimes = (0 until runs).map { _ =>
      timeMs(spark.read.parquet(node.files: _*).selectExpr(scanAggregates: _*).collect())._1
    }

    jsonObject(Seq(
      "files" -> node.files.length.toString,
      "row_groups" -> splits.length.toString,
      "pages" -> pages.toString,
      "page_bytes" -> bytes.toString,
      "key_columns" -> jsonStringList(node.keyColumnNames),
      "measure_columns" -> jsonStringList(node.measureColumnNames),
      "split_enum_fresh_ms" -> median(splitEnumFresh).toString,
      "split_enum_memoised_ms" -> median(splitEnumMemo).toString,
      "dimension_collect_ms" -> median(dimensionTimes).toString,
      "group_space_build_ms" -> median(groupSpaceTimes).toString,
      "rowgroup_read_ms" -> median(readRuns.map(_._1)).toString,
      "read_plus_parse_ms" -> median(parseRuns.map(_._1)).toString,
      "spark_scan_ms" -> median(scanTimes).toString), indent = 6)
  }

  /**
   * Join-region stages: split enumeration, dimension collect (the region's
   * keyPlans), row-group read of the DISTINCT key+fact columns without
   * parse, the same walk plus the native CPU parse, and Spark's own scan of
   * those columns -- the group-space stages do not apply.
   */
  private def benchmarkJoinRegion(
      spark: SparkSession, node: MetalParquetBroadcastJoinExec, runs: Int): String = {
    val columns = (node.keyColumnNames ++ node.factColumnNames).distinct

    val splitEnumFresh = (0 until runs).map(_ => timeMs(freshFooterSplits(node.files))._1)
    val splits = node.enumerateSplits()

    val dimensionTimes = (0 until runs).map { _ =>
      timeMs(node.keyPlans.foreach(_.executeCollect()))._1
    }

    SparkMetalNative.ensureInitialized(node.nativeLibrary, node.metalLibrary)
    val readRuns = (0 until runs).map(_ => timeMs(walkRowGroups(node.files, columns, parse = false)))
    val parseRuns = (0 until runs).map(_ => timeMs(walkRowGroups(node.files, columns, parse = true)))
    val (pages, bytes) = readRuns.last._2

    val scanAggregates = columns.map(column => s"count($column)")
    val scanTimes = (0 until runs).map { _ =>
      timeMs(spark.read.parquet(node.files: _*).selectExpr(scanAggregates: _*).collect())._1
    }

    jsonObject(Seq(
      "files" -> node.files.length.toString,
      "row_groups" -> splits.length.toString,
      "pages" -> pages.toString,
      "page_bytes" -> bytes.toString,
      "key_columns" -> jsonStringList(node.keyColumnNames),
      "fact_columns" -> jsonStringList(node.factColumnNames),
      "split_enum_fresh_ms" -> median(splitEnumFresh).toString,
      "dimension_collect_ms" -> median(dimensionTimes).toString,
      "rowgroup_read_ms" -> median(readRuns.map(_._1)).toString,
      "read_plus_parse_ms" -> median(parseRuns.map(_._1)).toString,
      "spark_scan_ms" -> median(scanTimes).toString), indent = 6)
  }

  private def freshFooterSplits(files: Seq[String]): Long = {
    var rowGroups = 0L
    files.foreach { file =>
      val reader = ParquetFileReader.open(
        HadoopInputFile.fromPath(new Path(file), ParquetEligibility.sharedConfiguration))
      try {
        rowGroups += reader.getFooter.getBlocks.size()
      } finally {
        reader.close()
      }
    }
    rowGroups
  }

  /**
   * Replicates the operator's read side: per file, a projected reader; per
   * row group, a PageReadStore; per column, the dictionary page plus every
   * V1 data page's decompressed bytes. With `parse = true` each page also
   * runs the native CPU parse (no stream, no GPU). Single-threaded, so the
   * result is a sum-of-work figure comparable to the summed SQL metrics.
   */
  private def walkRowGroups(
      files: Seq[String], columns: Seq[String], parse: Boolean): (Long, Long) = {
    var pages = 0L
    var bytes = 0L
    files.foreach { file =>
      val reader = ParquetFileReader.open(
        HadoopInputFile.fromPath(new Path(file), ParquetEligibility.sharedConfiguration))
      try {
        val schema = reader.getFooter.getFileMetaData.getSchema
        val descriptors = columns.map { name =>
          schema.getColumns.asScala.find(_.getPath()(0) == name)
            .getOrElse(throw new RuntimeException(s"$file: missing column $name"))
        }
        reader.setRequestedSchema(descriptors.asJava)
        for (rowGroupIndex <- 0 until reader.getFooter.getBlocks.size()) {
          val store = reader.readRowGroup(rowGroupIndex)
          try {
            descriptors.foreach { descriptor =>
              val pageReader = store.getPageReader(descriptor)
              pageReader.readDictionaryPage()
              val hasDefLevels = descriptor.getMaxDefinitionLevel > 0
              var rawPage = pageReader.readPage()
              while (rawPage != null) {
                rawPage match {
                  case v1: DataPageV1 =>
                    val pageBytes = v1.getBytes.toByteArray
                    pages += 1
                    bytes += pageBytes.length
                    if (parse) {
                      NativeBridge.parquetParsePageBenchmark(
                        pageBytes, pageBytes.length, v1.getValueCount,
                        hasDefLevels, v1.getValueEncoding == Encoding.PLAIN)
                    }
                  case _ => ()
                }
                rawPage = pageReader.readPage()
              }
            }
          } finally {
            store.close()
          }
        }
      } finally {
        reader.close()
      }
    }
    (pages, bytes)
  }

  private def timeMs[T](work: => T): (Double, T) = {
    val started = System.nanoTime()
    val result = work
    ((System.nanoTime() - started) / 1e6, result)
  }

  private def median(values: Seq[Double]): Double = {
    val sorted = values.sorted
    if (sorted.length % 2 == 0) (sorted(sorted.length / 2 - 1) + sorted(sorted.length / 2)) / 2
    else sorted(sorted.length / 2)
  }

  // Values arriving here are either numbers rendered with toString or
  // already-rendered nested JSON (starting with '{' or '['); strings go
  // through jsonStringList only, so no general escaping is needed.
  private def jsonObject(fields: Seq[(String, String)], indent: Int): String = {
    val pad = " " * indent
    fields.map { case (key, value) => s"""$pad"$key": $value""" }
      .mkString("{\n", ",\n", s"\n${" " * (indent - 2)}}")
  }

  private def jsonList(values: Seq[Double]): String = values.mkString("[", ", ", "]")

  private def jsonStringList(values: Seq[String]): String =
    values.map(value => s""""$value"""").mkString("[", ", ", "]")
}
