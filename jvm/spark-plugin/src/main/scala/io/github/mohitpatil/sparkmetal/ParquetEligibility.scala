package io.github.mohitpatil.sparkmetal

import java.util.concurrent.{ConcurrentHashMap, Executors}

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
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

  /**
   * One Hadoop [[Configuration]] per JVM, shared with
   * [[MetalParquetMembershipCountExec]]. Building one re-walks Hadoop's default
   * resource list, and this check would otherwise build one per file.
   */
  private[sparkmetal] lazy val sharedConfiguration: Configuration = new Configuration()

  /**
   * Verdict cache. check() runs on the driver during *planning*, so it repeats
   * for every execution of every query over the same files -- 30 serial footer
   * reads per plan for TPC-DS SF10 store_sales. The key carries the file's
   * length and modification time, so a rewritten file is re-checked rather than
   * answered from a stale verdict.
   */
  private final case class FileKey(path: String, length: Long, modificationTime: Long)
  private val verdicts = new ConcurrentHashMap[FileKey, Either[String, Unit]]()
  private val footerFacts = new ConcurrentHashMap[FileKey, Any]()

  /**
   * Memoises anything derived purely from one Parquet file's footer, keyed the
   * same way as the eligibility verdict so a rewritten file recomputes. Used by
   * [[MetalParquetMembershipCountExec]] for its row-group enumeration, which
   * planning would otherwise redo on every execution. Falls back to computing
   * without caching when the file's status cannot be read.
   */
  private[sparkmetal] def cachedByFileVersion[A](path: String)(compute: => A): A =
    fileKey(path) match {
      case None => compute
      case Some(key) =>
        val cached = footerFacts.get(key)
        if (cached != null) cached.asInstanceOf[A]
        else {
          val value = compute
          footerFacts.put(key, value)
          value
        }
    }

  private def fileKey(path: String): Option[FileKey] =
    try {
      val hadoopPath = new Path(path)
      val status = hadoopPath.getFileSystem(sharedConfiguration).getFileStatus(hadoopPath)
      Some(FileKey(path, status.getLen, status.getModificationTime))
    } catch {
      // An unreadable file is not cacheable; the caller recomputes, and its own
      // failure handling applies.
      case _: Exception => None
    }

  def check(paths: Seq[String], columnNames: Seq[String]): Either[String, Unit] = {
    val results = if (paths.length <= 1) {
      paths.map(cachedCheckFile(_, columnNames))
    } else {
      val executor = Executors.newFixedThreadPool(
        math.min(paths.length, Runtime.getRuntime.availableProcessors))
      try {
        implicit val context: ExecutionContext = ExecutionContext.fromExecutor(executor)
        Await.result(
          Future.traverse(paths)(path => Future(cachedCheckFile(path, columnNames))),
          Duration.Inf)
      } finally {
        executor.shutdown()
      }
    }
    // Report the first ineligible file, matching the previous fold's behaviour.
    results.collectFirst { case left @ Left(_) => left }.getOrElse(Right(()))
  }

  private def cachedCheckFile(path: String, columnNames: Seq[String]): Either[String, Unit] = {
    // An unreadable file is not cacheable; fall through to checkFile, whose
    // caller already treats a thrown failure as "not eligible".
    fileKey(path) match {
      case None => checkFile(path, columnNames)
      case Some(k) =>
        val cached = verdicts.get(k)
        if (cached != null) cached
        else {
          val verdict = checkFile(path, columnNames)
          verdicts.put(k, verdict)
          verdict
        }
    }
  }

  private def checkFile(path: String, columnNames: Seq[String]): Either[String, Unit] = {
    val reader = ParquetFileReader.open(
      HadoopInputFile.fromPath(new Path(path), sharedConfiguration))
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
