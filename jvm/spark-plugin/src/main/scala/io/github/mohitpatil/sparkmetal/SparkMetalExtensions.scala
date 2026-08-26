package io.github.mohitpatil.sparkmetal

import org.apache.spark.sql.{SparkSession, SparkSessionExtensions}

final class SparkMetalExtensions extends (SparkSessionExtensions => Unit) {
  override def apply(extensions: SparkSessionExtensions): Unit = {
    extensions.injectColumnar { session: SparkSession =>
      new SparkMetalColumnarRule(
        session.conf.get("spark.metal.nativeLibrary"),
        session.conf.get("spark.metal.metalLibrary"),
        session.conf.get("spark.sql.ansi.enabled").toBoolean)
    }
  }
}
