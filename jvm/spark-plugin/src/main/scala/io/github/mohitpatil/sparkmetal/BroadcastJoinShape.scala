package io.github.mohitpatil.sparkmetal

import java.io.File

import scala.collection.mutable
import scala.io.{Codec, Source}
import scala.util.control.NonFatal

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.catalyst.optimizer.BuildLeft
import org.apache.spark.sql.execution.{
  ColumnarToRowExec, FileSourceScanExec, InputAdapter, ProjectExec, SparkPlan,
  WholeStageCodegenExec}
import org.apache.spark.sql.execution.joins.BroadcastHashJoinExec
import org.apache.spark.sql.types.IntegerType

/**
 * Classifies physical-plan subtrees against the v1 broadcast-join region
 * shape (docs/GPU_BROADCAST_JOIN_SPEC.md): `[Project]? <- Inner
 * BroadcastHashJoin x N (1..4, single int32 equi-key each) <- [IsNotNull on
 * join keys only]? <- bare Parquet FileSourceScanExec`, where every region
 * output column is either an int32 fact column or a dimension (build-side)
 * attribute of a GroupSpace-supported type.
 *
 * Reuses [[GroupedAggregateShape]]'s fact-side walker and lineage resolver
 * verbatim -- the region here is exactly a grouped-aggregate region MINUS
 * the aggregate on top, so the two matchers must never drift apart on what
 * a valid join chain is.
 *
 * Unlike the grouped tier, this shape may legitimately anchor BELOW
 * operators the plugin cannot replace (a window, a rollup Expand, a
 * sort-merge join): the tier accelerates the scan+join subregion and hands
 * columnar batches to whatever consumes them, so the probe deliberately
 * reports such subregions as eligible.
 *
 * This is a read-only probe; `SparkMetalColumnarRule` performs the actual
 * replacement. Also the `inspect-broadcast-joins.sh` CLI entry point.
 */
private[sparkmetal] object BroadcastJoinShape {

  /** Where one region output column's values come from. */
  sealed trait ColumnSource
  /** An int32 column read straight off the fact scan. */
  case class FactColumn(attribute: Attribute) extends ColumnSource
  /**
   * `dimensionAttributes(dimensionIndex)(attributeOrdinal)` of the matched
   * build-side row -- ordinals index the region's per-dimension consumed
   * attribute list (which is also the keyPlan projection order: keyPlan
   * ordinal 0 is the join key, 1 + attributeOrdinal is this attribute).
   */
  case class DimensionColumn(dimensionIndex: Int, attributeOrdinal: Int) extends ColumnSource

  case class Region(
      top: SparkPlan, // the node the rule replaces
      output: Seq[Attribute], // == top.output
      columnSources: Seq[ColumnSource], // one per output attribute, same order
      joins: Seq[BroadcastHashJoinExec], // outermost-first (walkFactSide order)
      factKeys: Seq[Attribute], // fact-side join key per join, same order
      dimensionKeys: Seq[AttributeReference], // build-side join key per join
      // Per join: the build-side attributes the region output consumes, in
      // build-side output order. May be empty for a filter-only dimension.
      dimensionAttributes: Seq[Seq[Attribute]],
      scan: FileSourceScanExec)

  def matchRegion(plan: SparkPlan): Either[String, Region] = {
    val anchor = unwrap(plan) match {
      case project: ProjectExec =>
        unwrap(project.child) match {
          case _: BroadcastHashJoinExec => Right(project)
          case other => Left(s"project is not directly over a broadcast join: ${other.nodeName}")
        }
      case join: BroadcastHashJoinExec => Right(join)
      case other => Left(s"not a project-over-join or join node: ${other.nodeName}")
    }
    for {
      anchored <- anchor
      walk <- GroupedAggregateShape.walkFactSide(anchored)
      _ <- GroupedAggregateShape.validateNotNullTargets(walk)
      _ <- if (walk.joins.nonEmpty) Right(()) else Left("no broadcast join in region")
      _ <- if (walk.joins.length <= 4) Right(())
           else Left(s"more than 4 joins (${walk.joins.length})")
      dimensionKeys <- resolveDimensionKeys(walk.joins)
      resolved <- resolveOutputs(plan, walk)
    } yield {
      val (columnSources, dimensionAttributes) = resolved
      Region(plan, plan.output, columnSources, walk.joins, walk.factKeys,
        dimensionKeys, dimensionAttributes, walk.scan)
    }
  }

  /**
   * Each join's build-side equi-key must be a bare int32 attribute (the
   * fact-side key is validated symmetrically inside walkFactSide).
   */
  private def resolveDimensionKeys(
      joins: Seq[BroadcastHashJoinExec]): Either[String, Seq[AttributeReference]] = {
    val keys = joins.map { join =>
      val key = if (join.buildSide == BuildLeft) join.leftKeys.head else join.rightKeys.head
      key match {
        case attribute: AttributeReference if attribute.dataType == IntegerType => Right(attribute)
        case other => Left(s"build-side join key is not a bare int32 attribute: $other")
      }
    }
    keys.collectFirst { case Left(reason) => reason }
      .toLeft(keys.collect { case Right(attribute) => attribute })
  }

  /**
   * Resolves every output attribute of `top` to its source: an int32 fact
   * column, or one build side's attribute. Dimension attribute ordinals are
   * assigned per dimension in build-side output order, so they line up with
   * the keyPlan projection `(joinKey, attrs...)` the planner builds.
   */
  private def resolveOutputs(
      top: SparkPlan,
      walk: GroupedAggregateShape.FactWalk): Either[String, (Seq[ColumnSource], Seq[Seq[Attribute]])] = {
    val buildOutputs: Seq[Seq[Attribute]] = walk.joins.map { join =>
      (if (join.buildSide == BuildLeft) join.left else join.right).output
    }
    val joinIndexByExprId: Map[Long, Int] = buildOutputs.zipWithIndex.flatMap {
      case (attributes, index) => attributes.map(attribute => attribute.exprId.id -> index)
    }.toMap

    // Pass 1: lineage per output position, collecting each dimension's
    // consumed attributes (deduplicated; ordered by first appearance, fixed
    // to build-side output order below).
    val consumed = Array.fill(walk.joins.length)(mutable.LinkedHashSet.empty[Attribute])
    val lineages = new Array[Either[Attribute, Attribute]](top.output.length) // Left=fact, Right=dim
    for ((attribute, position) <- top.output.zipWithIndex) {
      GroupedAggregateShape.resolve(attribute.exprId, top) match {
        case Left(reason) => return Left(reason)
        case Right(GroupedAggregateShape.OnFact(factAttribute)) =>
          if (!isInt32BackedFactType(factAttribute.dataType)) {
            return Left(s"fact output column ${factAttribute.name} is not int32-backed " +
              s"(${factAttribute.dataType.simpleString})")
          }
          lineages(position) = Left(factAttribute)
        case Right(GroupedAggregateShape.OnDimension(dimensionAttribute)) =>
          joinIndexByExprId.get(dimensionAttribute.exprId.id) match {
            case None =>
              return Left(s"dimension attribute ${dimensionAttribute.name} not on any build side")
            case Some(index) =>
              if (!GroupSpace.isSupportedAttributeType(dimensionAttribute.dataType)) {
                return Left(s"unsupported dimension attribute type " +
                  s"${dimensionAttribute.dataType.simpleString} (${dimensionAttribute.name})")
              }
              consumed(index) += dimensionAttribute
              lineages(position) = Right(dimensionAttribute)
          }
      }
    }

    // Canonical per-dimension attribute order = build-side output order.
    val dimensionAttributes: Seq[Seq[Attribute]] = buildOutputs.zipWithIndex.map {
      case (attributes, index) => attributes.filter(consumed(index).contains)
    }
    val ordinal: Map[Long, (Int, Int)] = dimensionAttributes.zipWithIndex.flatMap {
      case (attributes, dimensionIndex) => attributes.zipWithIndex.map {
        case (attribute, attributeOrdinal) => attribute.exprId.id -> (dimensionIndex, attributeOrdinal)
      }
    }.toMap

    val columnSources: Seq[ColumnSource] = lineages.toSeq.map {
      case Left(factAttribute) => FactColumn(factAttribute)
      case Right(dimensionAttribute) =>
        val (dimensionIndex, attributeOrdinal) = ordinal(dimensionAttribute.exprId.id)
        DimensionColumn(dimensionIndex, attributeOrdinal)
    }
    Right((columnSources, dimensionAttributes))
  }

  /**
   * Fact output column types the decoder can serve from an int32 plane:
   * plain int32, int-backed dates, and decimals of precision <= 9, which
   * Parquet stores as physical INT32 at these schemas and Spark's writable
   * column vectors store as unscaled ints (`putInt`). Whether a concrete
   * file really uses the INT32 physical type is ParquetEligibility's check,
   * exactly as for the grouped tier's measures.
   */
  private[sparkmetal] def isInt32BackedFactType(dataType: org.apache.spark.sql.types.DataType): Boolean =
    dataType match {
      case IntegerType | org.apache.spark.sql.types.DateType => true
      case decimal: org.apache.spark.sql.types.DecimalType => decimal.precision <= 9
      case _ => false
    }

  private def unwrap(plan: SparkPlan): SparkPlan = plan match {
    case codegen: WholeStageCodegenExec => unwrap(codegen.child)
    case adapter: InputAdapter => unwrap(adapter.child)
    case columnarToRow: ColumnarToRowExec => unwrap(columnarToRow.child)
    case other => other
  }

  // -------------------------------------------------------------------
  // CLI probe (mirrors GroupedAggregateShape.main)
  // -------------------------------------------------------------------

  def main(args: Array[String]): Unit = {
    val dataDir = args.lift(0).getOrElse("benchmark-data/tpcds-sf10-parquet")
    val queriesDir = args.lift(1).getOrElse(".tools/spark-assets/sql/core/src/test/resources/tpcds")
    val selection = args.lift(2).getOrElse("all")

    val spark = SparkSession.builder().appName("spark-metal-broadcast-join-shape-probe").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try {
      val tableDirs = Option(new File(dataDir).listFiles())
        .getOrElse(Array.empty[File])
        .filter(_.isDirectory)
        .sortBy(_.getName)
      require(tableDirs.nonEmpty, s"no table directories found under $dataDir")
      tableDirs.foreach(dir => spark.read.parquet(dir.getPath).createOrReplaceTempView(dir.getName))

      val queryFiles = Option(new File(queriesDir).listFiles((_, name) => name.startsWith("q") && name.endsWith(".sql")))
        .getOrElse(Array.empty[File])
        .sortBy(file => querySortKey(file.getName))
      require(queryFiles.nonEmpty, s"no TPC-DS query files found under $queriesDir")

      val chosen = if (selection == "all") {
        queryFiles.toSeq
      } else {
        val byName = queryFiles.map(file => file.getName.stripSuffix(".sql") -> file).toMap
        selection.split(",").map(_.trim).filter(_.nonEmpty).toSeq.map { name =>
          byName.getOrElse(name, throw new IllegalArgumentException(s"unknown query: $name"))
        }
      }

      val verdicts = mutable.LinkedHashMap.empty[String, Either[String, Region]]
      chosen.foreach { file =>
        val name = file.getName.stripSuffix(".sql")
        val outcomes = evaluateQuery(spark, file)
        outcomes.collectFirst { case Right(region) => region } match {
          case Some(region) =>
            verdicts(name) = Right(region)
            println(s"ELIGIBLE $name ${describe(region)}")
          case None =>
            val reason = outcomes.collectFirst { case Left(r) => r }
              .getOrElse("no project-over-join or join anchor in the plan")
            verdicts(name) = Left(reason)
            println(s"INELIGIBLE $name $reason")
        }
      }
      val eligibleCount = verdicts.valuesIterator.count(_.isRight)
      println(s"# $eligibleCount/${chosen.length} queries eligible")

      // Hard assertions, from the verified 2026-08-29 sweep: q21/q22/q62/
      // q97/q99 are clean star shapes; q43 exercises the decimal(7,2)
      // int32-backed fact column; q14a exercises a subregion feeding a
      // sort-merge join. (No query-level negative assertions -- valid
      // subregions legitimately exist under windows/rollups/sort-merge
      // joins, see the scaladoc. q13/q48/q19 do NOT match: residual
      // non-equi join conditions, a real v1 limit.)
      def assertMatches(query: String): Unit = verdicts.get(query).foreach { verdict =>
        require(verdict.isRight, s"$query must match the broadcast-join shape: $verdict")
      }
      Seq("q21", "q22", "q43", "q62", "q97", "q99", "q14a").foreach(assertMatches)
    } finally {
      spark.stop()
    }
  }

  private def evaluateQuery(spark: SparkSession, file: File): Seq[Either[String, Region]] = {
    val statements = readFile(file).split(";").map(_.trim).filter(_.nonEmpty)
    statements.toSeq.flatMap { statement =>
      try {
        val plan = spark.sql(statement).queryExecution.executedPlan
        // Pre-order: the first Right is the outermost matchable region.
        plan.collect {
          case node @ (_: ProjectExec | _: BroadcastHashJoinExec) => matchRegion(node)
        }
      } catch {
        case NonFatal(e) => Seq(Left(s"failed to plan statement: ${e.getMessage}"))
      }
    }
  }

  private def readFile(file: File): String = {
    val source = Source.fromFile(file)(Codec.UTF8)
    try source.mkString finally source.close()
  }

  private def querySortKey(fileName: String): (Int, String) = {
    val stem = fileName.stripSuffix(".sql")
    val digits = stem.drop(1).takeWhile(_.isDigit)
    (if (digits.nonEmpty) digits.toInt else 0, stem)
  }

  private def describe(region: Region): String = {
    val factOutputs = region.columnSources.count(_.isInstanceOf[FactColumn])
    s"joins=${region.joins.length} outputs=${region.output.length} " +
      s"(fact=$factOutputs dim=${region.output.length - factOutputs}) " +
      s"factKeys=${region.factKeys.map(_.name).mkString(",")}"
  }
}
