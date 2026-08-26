package io.github.mohitpatil.sparkmetal

import scala.jdk.CollectionConverters._

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.parquet.column.Encoding
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.hadoop.metadata.CompressionCodecName
import org.apache.parquet.hadoop.util.HadoopInputFile
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName

object ParquetEligibility {
  private val SupportedEncodings: Set[Encoding] = Set(
    Encoding.RLE, Encoding.BIT_PACKED, Encoding.PLAIN_DICTIONARY, Encoding.RLE_DICTIONARY)
  private val SupportedCodecs: Set[CompressionCodecName] = Set(
    CompressionCodecName.SNAPPY, CompressionCodecName.UNCOMPRESSED)

  def check(paths: Seq[String], columnNames: Seq[String]): Either[String, Unit] = {
    paths.foldLeft[Either[String, Unit]](Right(())) { (state, path) =>
      state.flatMap(_ => checkFile(path, columnNames))
    }
  }

  private def checkFile(path: String, columnNames: Seq[String]): Either[String, Unit] = {
    val reader = ParquetFileReader.open(
      HadoopInputFile.fromPath(new Path(path), new Configuration()))
    try {
      val schema = reader.getFooter.getFileMetaData.getSchema
      for (name <- columnNames) {
        val descriptor = schema.getColumns.asScala.find(_.getPath()(0) == name)
          .getOrElse(return Left(s"$path: column $name is missing"))
        if (descriptor.getPrimitiveType.getPrimitiveTypeName != PrimitiveTypeName.INT32)
          return Left(s"$path: $name is not INT32")
        if (descriptor.getMaxRepetitionLevel != 0 || descriptor.getMaxDefinitionLevel > 1)
          return Left(s"$path: $name has unsupported nesting")
      }
      for (block <- reader.getFooter.getBlocks.asScala;
           chunk <- block.getColumns.asScala
           if columnNames.contains(chunk.getPath.toDotString)) {
        if (!SupportedCodecs.contains(chunk.getCodec))
          return Left(s"$path: ${chunk.getPath}: codec ${chunk.getCodec}")
        val unsupported = chunk.getEncodings.asScala.filterNot(SupportedEncodings.contains)
        if (unsupported.nonEmpty)
          return Left(s"$path: ${chunk.getPath}: encodings $unsupported")
        if (!chunk.getEncodings.asScala.exists(e =>
            e == Encoding.PLAIN_DICTIONARY || e == Encoding.RLE_DICTIONARY))
          return Left(s"$path: ${chunk.getPath}: values are not dictionary-encoded")
      }
      Right(())
    } finally reader.close()
  }

  def main(args: Array[String]): Unit = {
    val columns = args.head.split(",").toSeq
    var failed = false
    args.tail.foreach { path =>
      check(Seq(path), columns) match {
        case Right(()) => println(s"OK $path")
        case Left(reason) => println(s"INELIGIBLE $path $reason"); failed = true
      }
    }
    if (failed) sys.exit(1)
  }
}
