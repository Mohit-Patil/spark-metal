package io.github.mohitpatil.sparkmetal

import org.apache.spark.sql.SparkSession

object PluginSmoke {
  def main(arguments: Array[String]): Unit = {
    val rowCount = arguments.headOption.map(_.toInt).getOrElse(1000003)
    val spark = SparkSession.builder().appName("spark-metal-plugin-smoke").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try {
      val query = spark.sql(
        s"""SELECT sum(CASE WHEN value > 100 THEN value * 3 + 7 ELSE 0 END) AS total
           |FROM (SELECT CAST(id AS INT) AS value FROM range(0, $rowCount))""".stripMargin)
      println(query.queryExecution.executedPlan.toString())
      val actual = query.collect().head.getLong(0)
      val first = 101L
      val last = rowCount.toLong - 1
      val selected = math.max(0L, last - first + 1)
      val valueSum = if (selected == 0) 0L else (first + last) * selected / 2
      val expected = 3L * valueSum + 7L * selected
      require(actual == expected, s"CPU reference $expected did not match Metal result $actual")
      val plan = query.queryExecution.executedPlan.toString()
      require(plan.contains("MetalFusedSum"), s"Metal operator missing from executed plan:\n$plan")
      println(plan)
      println(s"{\"rows\":$rowCount,\"expected\":$expected,\"actual\":$actual,\"match\":true}")

      val nullableQuery = spark.sql(
        s"""SELECT sum(CASE WHEN value > 100 THEN value * 3 + 7 ELSE 0 END) AS total
           |FROM (
           |  SELECT CASE WHEN id % 17 = 0 THEN NULL ELSE CAST(id AS INT) END AS value
           |  FROM range(0, $rowCount)
           |)""".stripMargin)
      val nullableActual = nullableQuery.collect().head.getLong(0)
      var nullableExpected = 0L
      var value = 101
      while (value < rowCount) {
        if (value % 17 != 0) nullableExpected += value.toLong * 3 + 7
        value += 1
      }
      require(nullableActual == nullableExpected,
        s"Nullable CPU reference $nullableExpected did not match Metal result $nullableActual")
      val nullablePlan = nullableQuery.queryExecution.executedPlan.toString()
      require(nullablePlan.contains("MetalFusedSum"),
        s"Metal operator missing from nullable executed plan:\n$nullablePlan")
      println(s"{\"nullableRows\":$rowCount,\"expected\":$nullableExpected," +
        s"\"actual\":$nullableActual,\"match\":true}")
    } finally {
      spark.stop()
    }
  }
}
