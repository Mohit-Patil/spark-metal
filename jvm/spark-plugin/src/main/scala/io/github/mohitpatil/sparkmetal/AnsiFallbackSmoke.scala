package io.github.mohitpatil.sparkmetal

import org.apache.spark.sql.SparkSession

object AnsiFallbackSmoke {
  def main(arguments: Array[String]): Unit = {
    val rowCount = arguments.headOption.map(_.toInt).getOrElse(100003)
    val spark = SparkSession.builder().appName("spark-metal-ansi-fallback-smoke").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try {
      require(spark.conf.get("spark.sql.ansi.enabled").toBoolean,
        "ANSI fallback smoke test must run with spark.sql.ansi.enabled=true")
      val query = spark.sql(
        s"""SELECT sum(CASE WHEN value > 100 THEN value * 3 + 7 ELSE 0 END) AS total
           |FROM (SELECT CAST(id AS INT) AS value FROM range(0, $rowCount))""".stripMargin)
      val actual = query.collect().head.getLong(0)
      val first = 101L
      val last = rowCount.toLong - 1
      val selected = math.max(0L, last - first + 1)
      val valueSum = if (selected == 0) 0L else (first + last) * selected / 2
      val expected = 3L * valueSum + 7L * selected
      require(actual == expected, s"CPU reference $expected did not match Spark result $actual")
      val plan = query.queryExecution.executedPlan.toString()
      require(!plan.contains("MetalFusedSum"),
        s"ANSI query must fall back to CPU, but Metal operator was present:\n$plan")
      println(s"{\"ansiEnabled\":true,\"rows\":$rowCount,\"expected\":$expected," +
        s"\"actual\":$actual,\"metalOperator\":false,\"match\":true}")
    } finally {
      spark.stop()
    }
  }
}
