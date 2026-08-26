package io.github.mohitpatil.sparkmetal

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.sql.SparkSession

object Q96SyntheticBenchmark {
  private val Query =
    """SELECT count(*)
      |FROM store_sales, household_demographics, time_dim, store
      |WHERE ss_sold_time_sk = time_dim.t_time_sk
      |  AND ss_hdemo_sk = household_demographics.hd_demo_sk
      |  AND ss_store_sk = s_store_sk
      |  AND time_dim.t_hour = 20
      |  AND time_dim.t_minute >= 30
      |  AND household_demographics.hd_dep_count = 7
      |  AND store.s_store_name = 'ese'
      |ORDER BY count(*)
      |LIMIT 100""".stripMargin

  def main(arguments: Array[String]): Unit = arguments.headOption match {
    case Some("generate") if arguments.length == 3 =>
      generate(arguments(1), arguments(2).toLong, edgeCases = false)
    case Some("generate-edge") if arguments.length == 3 =>
      generate(arguments(1), arguments(2).toLong, edgeCases = true)
    case Some("run") if arguments.length == 6 =>
      benchmark(
        arguments(1), arguments(2), arguments(3).toInt, arguments(4).toInt,
        arguments(5).toBoolean)
    case _ =>
      throw new IllegalArgumentException(
        "Usage: Q96SyntheticBenchmark generate|generate-edge ROOT_PATH ROWS | " +
          "run ROOT_PATH OUTPUT_JSON WARMUPS RUNS EXPECT_METAL")
  }

  private def generate(root: String, rows: Long, edgeCases: Boolean): Unit = {
    require(rows > 0)
    val spark = SparkSession.builder().appName("spark-metal-q96-synthetic-generate").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try {
      val timeExpression = if (edgeCases) {
        "CASE WHEN pmod(id, 97) = 0 THEN NULL ELSE CAST(pmod(id, 1440) AS INT) END AS ss_sold_time_sk"
      } else {
        "CAST(pmod(id, 1440) AS INT) AS ss_sold_time_sk"
      }
      spark.range(rows)
        .selectExpr(
          timeExpression,
          "CAST(pmod(floor(id / 1440), 1000) AS INT) AS ss_hdemo_sk",
          "CAST(pmod(floor(id / 14400), 100) AS INT) AS ss_store_sk")
        .write.mode("errorifexists").parquet(s"$root/store_sales")

      spark.range(1440)
        .selectExpr(
          "CAST(id AS INT) AS t_time_sk",
          "CAST(floor(id / 60) AS INT) AS t_hour",
          "CAST(pmod(id, 60) AS INT) AS t_minute")
        .write.mode("errorifexists").parquet(s"$root/time_dim")

      spark.range(1000)
        .selectExpr(
          "CAST(id AS INT) AS hd_demo_sk",
          "CAST(pmod(id, 10) AS INT) AS hd_dep_count")
        .write.mode("errorifexists").parquet(s"$root/household_demographics")

      spark.range(if (edgeCases) 101 else 100)
        .selectExpr(
          "CAST(CASE WHEN id = 100 THEN 17 ELSE id END AS INT) AS s_store_sk",
          "CASE WHEN id IN (17, 100) THEN 'ese' ELSE concat('store-', id) END AS s_store_name")
        .write.mode("errorifexists").parquet(s"$root/store")
      println(s"Generated q96-shaped data with $rows fact rows at $root (edgeCases=$edgeCases)")
    } finally {
      spark.stop()
    }
  }

  private def benchmark(
      root: String,
      output: String,
      warmups: Int,
      runs: Int,
      expectMetal: Boolean): Unit = {
    require(warmups >= 0 && runs > 0)
    val spark = SparkSession.builder().appName("spark-metal-q96-synthetic-benchmark").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try {
      Seq("store_sales", "household_demographics", "time_dim", "store").foreach { table =>
        spark.read.parquet(s"$root/$table").createOrReplaceTempView(table)
      }

      var expected: Option[Long] = None
      for (_ <- 0 until warmups) {
        val value = spark.sql(Query).collect().head.getLong(0)
        expected.foreach(reference => require(value == reference))
        expected = Some(value)
      }
      val timings = ArrayBuffer.empty[Double]
      var finalPlan = ""
      var acceleratorMetrics = Map.empty[String, Long]
      for (_ <- 0 until runs) {
        val started = System.nanoTime()
        val query = spark.sql(Query)
        val value = query.collect().head.getLong(0)
        val elapsed = (System.nanoTime() - started).toDouble / 1000000000.0
        expected.foreach(reference => require(value == reference))
        expected = Some(value)
        timings += elapsed
        finalPlan = query.queryExecution.executedPlan.toString()
        acceleratorMetrics = query.queryExecution.executedPlan.collectFirst {
          case node: MetalFusedMembershipCountExec =>
            node.metrics.map { case (name, metric) => name -> metric.value }
          case node: MetalParquetMembershipCountExec =>
            node.metrics.map { case (name, metric) => name -> metric.value }
        }.getOrElse(Map.empty)
      }
      val sorted = timings.sorted
      val median = if (sorted.length % 2 == 0) {
        (sorted(sorted.length / 2 - 1) + sorted(sorted.length / 2)) / 2
      } else {
        sorted(sorted.length / 2)
      }
      val accelerated = finalPlan.contains("MetalFusedMembershipCount") ||
        finalPlan.contains("MetalParquetMembershipCount")
      require(accelerated == expectMetal,
        s"Unexpected q96 accelerator state; expected=$expectMetal, plan=$finalPlan")
      val escapedPlan = finalPlan
        .replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
      val observations = timings.mkString("[", ",", "]")
      val metricsJson = acceleratorMetrics.toSeq.sortBy(_._1)
        .map { case (name, value) => s"\"$name\": $value" }
        .mkString("{", ", ", "}")
      val json =
        s"""{
           |  "accelerated": $accelerated,
           |  "result": ${expected.get},
           |  "warmups": $warmups,
           |  "runs": $runs,
           |  "seconds": $observations,
           |  "medianSeconds": $median,
           |  "acceleratorMetrics": $metricsJson,
           |  "plan": "$escapedPlan"
           |}
           |""".stripMargin
      Files.writeString(Path.of(output), json, StandardCharsets.UTF_8)
      println(json)
    } finally {
      spark.stop()
    }
  }
}
