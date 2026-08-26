package io.github.mohitpatil.sparkmetal

import java.io.File

import scala.io.{Codec, Source}
import scala.util.control.NonFatal

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.{
  Alias, And, Attribute, AttributeReference, Expression, ExprId, IsNotNull, Literal,
  NamedExpression, UnscaledValue}
import org.apache.spark.sql.catalyst.expressions.aggregate.{
  AggregateExpression, AggregateFunction, Average, Count, Partial, Sum}
import org.apache.spark.sql.catalyst.optimizer.BuildLeft
import org.apache.spark.sql.execution.{
  ColumnarToRowExec, ExpandExec, FileSourceScanExec, FilterExec, InputAdapter, ProjectExec,
  SparkPlan, WholeStageCodegenExec}
import org.apache.spark.sql.execution.aggregate.HashAggregateExec
import org.apache.spark.sql.execution.datasources.parquet.ParquetFileFormat
import org.apache.spark.sql.execution.joins.BroadcastHashJoinExec
import org.apache.spark.sql.types.{DataType, DecimalType, IntegerType}

/**
 * Classifies TPC-DS physical-plan subtrees against the v1 grouped-aggregate
 * region shape: a partial Sum/Count/Average [[HashAggregateExec]] grouping on
 * broadcast-side (dimension) attributes, fed by a chain of Inner
 * [[BroadcastHashJoinExec]] nodes (single int32 equi-key, streamed side is
 * the fact lineage) down to a bare Parquet [[FileSourceScanExec]], with only
 * [[ProjectExec]] / [[FilterExec]] (`IsNotNull`-only) / [[ColumnarToRowExec]]
 * allowed in between.
 *
 * This is a read-only probe. It never rewrites a plan -- it only reports
 * whether Task 6's planner (a later task) would find a region here, and why
 * not when it would not. This object also serves as the
 * `inspect-grouped-aggregates.sh` CLI entry point.
 */
private[sparkmetal] object GroupedAggregateShape {

  sealed trait AggInput
  case object CountStar extends AggInput // count(1)
  case class FactColumn(attribute: Attribute) extends AggInput // sum/avg/count(col), col int32 on the scan
  // UnscaledValue(col) is represented as FactColumn with unscaled = true.

  case class AggSpec(function: String, input: AggInput, unscaled: Boolean, sumDataType: DataType)

  case class Region(
      aggregate: HashAggregateExec, // the partial aggregate
      groupKeys: Seq[Attribute], // dimension attributes, in output order
      aggs: Seq[AggSpec],
      joins: Seq[BroadcastHashJoinExec], // outermost-first
      factKeys: Seq[Attribute], // fact join key per join, same order
      scan: FileSourceScanExec,
      measureColumns: Seq[Attribute]) // distinct fact measure columns

  /** Where an attribute's lineage bottoms out while walking down from the aggregate. */
  private sealed trait Lineage
  private case class OnFact(attribute: Attribute) extends Lineage
  private case class OnDimension(attribute: Attribute) extends Lineage

  private case class FactWalk(
      scan: FileSourceScanExec,
      joins: Seq[BroadcastHashJoinExec],
      factKeys: Seq[Attribute])

  /**
   * Returns the matched region, or `Left(reason)` naming the first
   * disqualifier (expand-below-aggregate, non-broadcast join, expression
   * group key, unsupported aggregate form, non-int32 measure, filter beyond
   * IsNotNull, ...).
   */
  def matchRegion(plan: SparkPlan): Either[String, Region] = plan match {
    case aggregate: HashAggregateExec => matchAggregate(aggregate)
    case other => Left(s"not a hash aggregate node: ${other.nodeName}")
  }

  private def matchAggregate(aggregate: HashAggregateExec): Either[String, Region] = {
    // Controller ruling: Expand (rollup/cube) anywhere between the aggregate
    // and the scan disqualifies the region, regardless of where it sits.
    if (aggregate.child.collect { case e: ExpandExec => e }.nonEmpty) {
      return Left("Expand (rollup/cube) between the aggregate and the scan")
    }

    val aggregateExpressions = aggregate.aggregateExpressions
    if (aggregateExpressions.isEmpty) {
      return Left("no aggregate expressions")
    }
    if (aggregateExpressions.exists(_.mode != Partial)) {
      return Left("aggregate is not partial-mode")
    }
    if (aggregateExpressions.exists(e => e.isDistinct || e.filter.isDefined)) {
      return Left("distinct or filtered aggregate is unsupported")
    }
    if (aggregateExpressions.exists(e => !isSupportedFunction(e.aggregateFunction))) {
      return Left("unsupported aggregate form (only Sum/Count/Average partials)")
    }

    val groupingExpressions = aggregate.groupingExpressions
    val countOnly = aggregateExpressions.forall(_.aggregateFunction.isInstanceOf[Count])
    // Controller ruling: a region with zero group keys and only count
    // aggregates is the existing membership operators' business (q96/q88/
    // q90), not this tier's. The two rules must never compete for the same
    // region.
    if (groupingExpressions.isEmpty && countOnly) {
      return Left("count-only region with no group keys (handled by the membership operators)")
    }

    for {
      groupKeys <- resolveGroupKeys(groupingExpressions, aggregate.child)
      aggs <- resolveAggSpecs(aggregateExpressions, aggregate.child)
      walk <- walkFactSide(aggregate.child)
    } yield {
      Region(
        aggregate = aggregate,
        groupKeys = groupKeys,
        aggs = aggs,
        joins = walk.joins,
        factKeys = walk.factKeys,
        scan = walk.scan,
        measureColumns = aggs.collect { case AggSpec(_, FactColumn(attribute), _, _) => attribute }.distinct)
    }
  }

  private def isSupportedFunction(function: AggregateFunction): Boolean = function match {
    case _: Sum | _: Count | _: Average => true
    case _ => false
  }

  private def resolveGroupKeys(
      groupingExpressions: Seq[NamedExpression],
      child: SparkPlan): Either[String, Seq[Attribute]] = {
    groupingExpressions.foldLeft[Either[String, Vector[Attribute]]](Right(Vector.empty)) {
      case (Left(reason), _) => Left(reason)
      case (Right(acc), attribute: AttributeReference) =>
        resolve(attribute.exprId, child) match {
          case Right(OnDimension(resolved)) => Right(acc :+ resolved)
          case Right(OnFact(_)) =>
            Left("group key resolves to the fact side, not a broadcast dimension attribute")
          case Left(reason) => Left(reason)
        }
      case (Right(_), _) => Left("expression group key")
    }
  }

  private def resolveAggSpecs(
      aggregateExpressions: Seq[AggregateExpression],
      child: SparkPlan): Either[String, Seq[AggSpec]] = {
    aggregateExpressions.foldLeft[Either[String, Vector[AggSpec]]](Right(Vector.empty)) {
      case (Left(reason), _) => Left(reason)
      case (Right(acc), expression) => resolveAggSpec(expression, child).map(acc :+ _)
    }
  }

  private def resolveAggSpec(expression: AggregateExpression, child: SparkPlan): Either[String, AggSpec] = {
    val function = expression.aggregateFunction
    // sumDataType always comes from the function's own aggregate buffer
    // attribute -- never assumed. This is where Average's partial sum type
    // (Double for ints, wider Decimal for decimals) is actually observed.
    val sumDataType = function.aggBufferAttributes.head.dataType

    function match {
      case count: Count if count.children.length == 1 && isNonNullLiteral(count.children.head) =>
        Right(AggSpec("count", CountStar, unscaled = false, sumDataType))
      case count: Count if count.children.length == 1 =>
        resolveMeasure(count.children.head, child).map { case (attribute, unscaled) =>
          AggSpec("count", FactColumn(attribute), unscaled, sumDataType)
        }
      case sum: Sum =>
        resolveMeasure(sum.child, child).map { case (attribute, unscaled) =>
          AggSpec("sum", FactColumn(attribute), unscaled, sumDataType)
        }
      case average: Average =>
        resolveMeasure(average.child, child).map { case (attribute, unscaled) =>
          AggSpec("avg", FactColumn(attribute), unscaled, sumDataType)
        }
      case other => Left(s"unsupported aggregate function ${other.prettyName}")
    }
  }

  private def isNonNullLiteral(expression: Expression): Boolean = expression match {
    case literal: Literal => literal.value != null
    case _ => false
  }

  private def resolveMeasure(expression: Expression, child: SparkPlan): Either[String, (Attribute, Boolean)] = {
    extractSurfaceAttribute(expression).flatMap { case (surfaceAttribute, unscaled) =>
      resolve(surfaceAttribute.exprId, child).flatMap {
        case OnFact(resolved) if unscaled =>
          resolved.dataType match {
            case decimal: DecimalType if decimal.precision <= 9 => Right((resolved, true))
            case other => Left(s"UnscaledValue measure column is not an INT32-backed decimal: $other")
          }
        case OnFact(resolved) =>
          if (resolved.dataType == IntegerType) Right((resolved, false))
          else Left(s"non-int32 measure column: ${resolved.dataType.simpleString}")
        case OnDimension(_) =>
          Left("aggregate input resolves to the dimension side, not the fact scan")
      }
    }
  }

  private def extractSurfaceAttribute(expression: Expression): Either[String, (AttributeReference, Boolean)] =
    expression match {
      case attribute: AttributeReference => Right((attribute, false))
      case UnscaledValue(attribute: AttributeReference) => Right((attribute, true))
      case _ => Left("aggregate input is not a bare column or UnscaledValue(column)")
    }

  /**
   * Resolves `exprId` -- an attribute referenced somewhere at or above
   * `plan` -- down to the leaf it structurally originates from: a
   * [[FileSourceScanExec]] reached only through streamed (non-broadcast)
   * join sides ([[OnFact]]), or an attribute that showed up in some join's
   * broadcast-side output ([[OnDimension]], resolved no further).
   *
   * Only attribute-preserving nodes are walked: [[ProjectExec]] (bare
   * attributes or simple aliases of one), [[FilterExec]], [[ColumnarToRowExec]],
   * and [[BroadcastHashJoinExec]]. Anything else -- a computed projection, an
   * unsupported node shape -- fails resolution with a reason.
   */
  private def resolve(exprId: ExprId, plan: SparkPlan): Either[String, Lineage] = plan match {
    // Whole-stage codegen wraps fused operators in WholeStageCodegenExec /
    // InputAdapter without changing attributes -- transparent pass-through.
    case codegen: WholeStageCodegenExec => resolve(exprId, codegen.child)
    case adapter: InputAdapter => resolve(exprId, adapter.child)
    case project: ProjectExec =>
      project.projectList.find(_.exprId == exprId) match {
        case Some(attribute: AttributeReference) => resolve(attribute.exprId, project.child)
        case Some(Alias(inner: AttributeReference, _)) => resolve(inner.exprId, project.child)
        case Some(_) => Left("group key or measure resolves through a computed projection")
        case None => Left(s"attribute $exprId missing from project output")
      }
    case filter: FilterExec =>
      if (isOnlyNotNullPredicate(filter.condition)) resolve(exprId, filter.child)
      else Left("filter beyond IsNotNull")
    case columnarToRow: ColumnarToRowExec => resolve(exprId, columnarToRow.child)
    case join: BroadcastHashJoinExec =>
      val (buildPlan, streamedPlan) =
        if (join.buildSide == BuildLeft) (join.left, join.right) else (join.right, join.left)
      buildPlan.output.find(_.exprId == exprId) match {
        case Some(attribute) => Right(OnDimension(attribute))
        case None =>
          if (streamedPlan.output.exists(_.exprId == exprId)) resolve(exprId, streamedPlan)
          else Left("attribute not found on either side of the broadcast join")
      }
    case scan: FileSourceScanExec =>
      scan.output.find(_.exprId == exprId).map(OnFact).toRight(s"attribute $exprId missing from scan output")
    case other => Left(s"unsupported node while resolving attribute lineage: ${other.nodeName}")
  }

  /**
   * Walks from `plan` (the aggregate's child) down to the fact scan,
   * collecting the join chain outermost-first and validating each join and
   * the terminal scan against the v1 shape.
   */
  private def walkFactSide(plan: SparkPlan): Either[String, FactWalk] = plan match {
    case codegen: WholeStageCodegenExec => walkFactSide(codegen.child)
    case adapter: InputAdapter => walkFactSide(adapter.child)
    case project: ProjectExec => walkFactSide(project.child)
    case filter: FilterExec =>
      if (isOnlyNotNullPredicate(filter.condition)) walkFactSide(filter.child)
      else Left("filter beyond IsNotNull")
    case columnarToRow: ColumnarToRowExec => walkFactSide(columnarToRow.child)
    case join: BroadcastHashJoinExec =>
      if (join.joinType != org.apache.spark.sql.catalyst.plans.Inner) {
        Left(s"non-inner join type ${join.joinType}")
      } else if (join.condition.isDefined) {
        Left("join has a residual condition beyond the equi-key")
      } else if (join.leftKeys.length != 1 || join.rightKeys.length != 1) {
        Left("join is not a single-column equi-key")
      } else {
        val (streamedPlan, streamedKey) =
          if (join.buildSide == BuildLeft) (join.right, join.rightKeys.head) else (join.left, join.leftKeys.head)
        streamedKey match {
          case factAttribute: AttributeReference if factAttribute.dataType == IntegerType =>
            walkFactSide(streamedPlan).map { walk =>
              walk.copy(joins = join +: walk.joins, factKeys = factAttribute +: walk.factKeys)
            }
          case _ => Left("join equi-key is not a bare int32 attribute on the fact (streamed) side")
        }
      }
    case scan: FileSourceScanExec =>
      val relation = scan.relation
      if (!relation.fileFormat.isInstanceOf[ParquetFileFormat]) {
        Left("scan is not parquet")
      } else if (relation.partitionSchema.nonEmpty) {
        Left("scan has partition columns")
      } else if (relation.bucketSpec.nonEmpty) {
        Left("scan has bucket columns")
      } else if (!scan.dataFilters.forall(isOnlyNotNullPredicate)) {
        Left("scan data filter beyond IsNotNull")
      } else {
        Right(FactWalk(scan, Seq.empty, Seq.empty))
      }
    case other => Left(s"unsupported node between aggregate and scan: ${other.nodeName}")
  }

  private def isOnlyNotNullPredicate(expression: Expression): Boolean = expression match {
    case IsNotNull(_: AttributeReference) => true
    case And(left, right) => isOnlyNotNullPredicate(left) && isOnlyNotNullPredicate(right)
    case _ => false
  }

  // ---------------------------------------------------------------------
  // CLI: classify every TPC-DS query's plan against matchRegion and report
  // ELIGIBLE/INELIGIBLE. Always exits 0 -- this is a report, not a gate.
  // ---------------------------------------------------------------------

  def main(args: Array[String]): Unit = {
    val dataDir = args.lift(0).getOrElse("benchmark-data/tpcds-sf10-parquet")
    val queriesDir = args.lift(1).getOrElse(".tools/spark-assets/sql/core/src/test/resources/tpcds")
    val selection = args.lift(2).getOrElse("all")

    val spark = SparkSession.builder().appName("spark-metal-grouped-aggregate-shape-probe").getOrCreate()
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

      var eligibleCount = 0
      chosen.foreach { file =>
        val name = file.getName.stripSuffix(".sql")
        val outcomes = evaluateQuery(spark, file)
        outcomes.collectFirst { case Right(region) => region } match {
          case Some(region) =>
            eligibleCount += 1
            println(s"ELIGIBLE $name ${describe(region)}")
          case None =>
            val reason = outcomes.collectFirst { case Left(reason) => reason }
              .getOrElse("no partial hash aggregate found in the plan")
            println(s"INELIGIBLE $name $reason")
        }
      }
      println(s"# $eligibleCount/${chosen.length} queries eligible")
    } finally {
      spark.stop()
    }
  }

  private def evaluateQuery(spark: SparkSession, file: File): Seq[Either[String, Region]] = {
    val statements = readFile(file).split(";").map(_.trim).filter(_.nonEmpty)
    statements.toSeq.flatMap { statement =>
      try {
        val plan = spark.sql(statement).queryExecution.executedPlan
        plan.collect {
          case aggregate: HashAggregateExec
              if aggregate.aggregateExpressions.nonEmpty &&
                aggregate.aggregateExpressions.forall(_.mode == Partial) =>
            matchRegion(aggregate)
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
    val aggsText = region.aggs.map(describeAgg).mkString(",")
    s"joins=${region.joins.length} groups=${region.groupKeys.length} aggs=[$aggsText]"
  }

  private def describeAgg(spec: AggSpec): String = {
    val suffix = if (spec.unscaled) "~unscaled" else ""
    s"${spec.function}:${spec.sumDataType.simpleString}$suffix"
  }
}
