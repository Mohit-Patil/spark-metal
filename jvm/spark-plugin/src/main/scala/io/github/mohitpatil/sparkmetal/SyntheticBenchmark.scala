package io.github.mohitpatil.sparkmetal

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.sql.SparkSession

object SyntheticBenchmark {
  def main(arguments: Array[String]): Unit = arguments.headOption match {
    case Some("generate") if arguments.length == 3 => generate(arguments(1), arguments(2).toLong)
    case Some("run") if arguments.length == 5 =>
      benchmark(arguments(1), arguments(2), arguments(3).toInt, arguments(4).toInt)
    case _ =>
      throw new IllegalArgumentException(
        "Usage: SyntheticBenchmark generate PARQUET_PATH ROWS | " +
          "run PARQUET_PATH OUTPUT_JSON WARMUPS RUNS")
  }

  private def generate(path: String, rows: Long): Unit = {
    require(rows > 101 && rows <= Int.MaxValue.toLong)
    val spark = SparkSession.builder().appName("spark-metal-synthetic-generate").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try {
      spark.range(rows)
        .selectExpr("CAST(id AS INT) AS value")
        .write
        .mode("errorifexists")
        .parquet(path)
      println(s"Generated $rows integer rows at $path")
    } finally {
      spark.stop()
    }
  }

  private def benchmark(path: String, output: String, warmups: Int, runs: Int): Unit = {
    require(warmups >= 0 && runs > 0)
    val spark = SparkSession.builder().appName("spark-metal-synthetic-benchmark").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try {
      spark.read.parquet(path).createOrReplaceTempView("synthetic_values")
      val sql =
        """SELECT sum(CASE WHEN value > 100 THEN value * 3 + 7 ELSE 0 END) AS total
          |FROM synthetic_values""".stripMargin
      var expected: Option[Long] = None
      for (_ <- 0 until warmups) {
        val value = spark.sql(sql).collect().head.getLong(0)
        expected.foreach(reference => require(value == reference))
        expected = Some(value)
      }
      val timings = ArrayBuffer.empty[Double]
      var finalPlan = ""
      for (_ <- 0 until runs) {
        val started = System.nanoTime()
        val query = spark.sql(sql)
        val value = query.collect().head.getLong(0)
        val elapsed = (System.nanoTime() - started).toDouble / 1000000000.0
        expected.foreach(reference => require(value == reference))
        expected = Some(value)
        timings += elapsed
        finalPlan = query.queryExecution.executedPlan.toString()
      }
      val sorted = timings.sorted
      val median = if (sorted.length % 2 == 0) {
        (sorted(sorted.length / 2 - 1) + sorted(sorted.length / 2)) / 2
      } else {
        sorted(sorted.length / 2)
      }
      val accelerated = finalPlan.contains("MetalFusedSum")
      val expectsMetal = spark.conf.getOption("spark.metal.nativeLibrary").nonEmpty
      require(accelerated == expectsMetal,
        s"Unexpected accelerator plan state; expected=$expectsMetal, plan=$finalPlan")
      val escapedPlan = finalPlan
        .replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
      val observations = timings.mkString("[", ",", "]")
      val json =
        s"""{
           |  "accelerated": $accelerated,
           |  "result": ${expected.get},
           |  "warmups": $warmups,
           |  "runs": $runs,
           |  "seconds": $observations,
           |  "medianSeconds": $median,
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
