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

/**
 * Identity of one *version* of a Parquet file. Anything memoised against this
 * recomputes when the file is rewritten, since a rewrite changes its length,
 * its modification time, or both.
 */
private[sparkmetal] final case class FileVersion(
    path: String, length: Long, modificationTime: Long)

/**
 * Small bounded LRU behind a lock. The driver-side Parquet caches live for the
 * lifetime of the JVM and are keyed per file version, so an unbounded map grows
 * without limit on a long-lived driver that reads many distinct files. The
 * bound is far above any single query's file count -- TPC-DS SF10 store_sales
 * is 30 files -- so the caches still hit for the repeated-planning case they
 * exist for; past the bound the least-recently-used entry is evicted and simply
 * recomputed.
 *
 * java.util.LinkedHashMap in access order reorders on `get` as well as `put`,
 * so reads mutate and every access has to hold the lock.
 */
private[sparkmetal] final class BoundedCache[K, V](maxEntries: Int) {
  private val entries = new java.util.LinkedHashMap[K, V](16, 0.75f, true) {
    override def removeEldestEntry(eldest: java.util.Map.Entry[K, V]): Boolean =
      size() > maxEntries
  }

  def get(key: K): Option[V] = synchronized(Option(entries.get(key)))

  def put(key: K, value: V): Unit = synchronized(entries.put(key, value))
}

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

  /** Shared bound for every driver-side Parquet cache. See [[BoundedCache]]. */
  private[sparkmetal] val MaxCachedFiles = 512

  /**
   * Verdict cache. check() runs on the driver during *planning*, so it repeats
   * for every execution of every query over the same files -- 30 serial footer
   * reads per plan for TPC-DS SF10 store_sales.
   *
   * The key is (file version, column names), NOT the file alone. Every check in
   * checkFile is specific to the requested columns: whether they exist, whether
   * they are INT32, their nesting, and the codec and encodings of *their* column
   * chunks. Keyed by file alone, a second query over the same files with
   * different key columns is answered from the first query's verdict -- a
   * cached Right lets an ineligible plan past the safety gate, where a
   * missing-or-non-INT32 column then throws outside the CPU-fallback catch, and
   * a cached Left wrongly rejects an eligible plan.
   *
   * Column names are held as a List so the key's equality is by value.
   */
  private val verdicts =
    new BoundedCache[(FileVersion, List[String]), Either[String, Unit]](MaxCachedFiles)

  /**
   * The version of one file, or None when its status cannot be read -- an
   * unreadable file is not cacheable, and callers recompute and apply their own
   * failure handling.
   */
  private[sparkmetal] def fileVersion(path: String): Option[FileVersion] =
    try {
      val hadoopPath = new Path(path)
      val status = hadoopPath.getFileSystem(sharedConfiguration).getFileStatus(hadoopPath)
      Some(FileVersion(path, status.getLen, status.getModificationTime))
    } catch {
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

  private def cachedCheckFile(path: String, columnNames: Seq[String]): Either[String, Unit] =
    fileVersion(path) match {
      case None => checkFile(path, columnNames)
      case Some(version) =>
        val key = (version, columnNames.toList)
        verdicts.get(key) match {
          case Some(verdict) => verdict
          case None =>
            val verdict = checkFile(path, columnNames)
            verdicts.put(key, verdict)
            verdict
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
