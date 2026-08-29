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
  private[sparkmetal] sealed trait Lineage
  private[sparkmetal] case class OnFact(attribute: Attribute) extends Lineage
  private[sparkmetal] case class OnDimension(attribute: Attribute) extends Lineage

  private[sparkmetal] case class FactWalk(
      scan: FileSourceScanExec,
      joins: Seq[BroadcastHashJoinExec],
      factKeys: Seq[Attribute],
      // Every attribute referenced by an IsNotNull conjunct encountered while
      // walking from the aggregate down to the scan (FilterExec conditions
      // and the terminal scan's dataFilters) -- never a broadcast (build)
      // side subtree, since walkFactSide only ever descends the streamed
      // side of a join. Collected here, not in isOnlyNotNullPredicate (which
      // only judges predicate SHAPE and is also used, unrelated, by the
      // membership operators via SparkMetalColumnarRule's own copy), so that
      // the fix in matchAggregate below is scoped to the grouped matcher and
      // never touches membership-operator matching.
      notNullTargets: Seq[AttributeReference])

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
      _ <- validateNotNullTargets(walk)
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

  /**
   * Grouped-matcher-only restriction (final-review fix wave): the operator
   * enforces nullness ONLY for the fact-side join key columns it walks in
   * `walkFactSide` (an inner broadcast join already drops null keys, so an
   * explicit IsNotNull on a factKey is a harmless no-op to accept) -- it
   * never re-checks an IsNotNull found on some other column, whether that is
   * a measure (`WHERE q IS NOT NULL`) or a dimension attribute referenced
   * after a join (`WHERE d.name IS NOT NULL`). Before this check,
   * `isOnlyNotNullPredicate` accepted IsNotNull over ANY attribute purely by
   * predicate SHAPE, so such a filter matched the region and the operator
   * would silently count rows the CPU plan filters out.
   *
   * `walk.notNullTargets` already excludes anything found only inside a
   * broadcast (dimension) build-side subtree -- walkFactSide never descends
   * there -- so a legitimate dimension-side filter (already applied by
   * ordinary Spark execution, since the GPU takeover never touches that
   * subtree) is never even a candidate here. What remains is either a
   * genuine factKey (accept) or something else entirely -- a fact-side
   * measure, or a dimension attribute pulled in by a post-join filter
   * (reject either way).
   *
   * The comparison is a direct exprId membership test against
   * `walk.factKeys`, NOT a second call to `resolve` from the aggregate's
   * root: `resolve`'s ProjectExec case only looks a target attribute up in
   * that Project's OWN output list, which is correct for a group key or
   * measure (guaranteed to still be referenced above every Project on its
   * way up to the aggregate) but wrong here -- a join key is routinely
   * consumed by its join and then dropped by the next Project, so it is
   * simply absent from that Project's output, and re-resolving it from the
   * top would misreport "attribute missing from project output" for a
   * perfectly ordinary factKey filter (observed on TPC-DS q3's
   * ss_sold_date_sk/ss_item_sk IsNotNull filters, both dropped from the
   * outer Project once their joins are done). factKeys and notNullTargets
   * are both collected by the SAME walkFactSide traversal without renaming
   * either one, so their exprIds already line up whenever they name the
   * same underlying column -- no re-resolution needed.
   */
  private[sparkmetal] def validateNotNullTargets(walk: FactWalk): Either[String, Unit] = {
    val factKeyIds = walk.factKeys.map(_.exprId).toSet
    walk.notNullTargets.find(attribute => !factKeyIds.contains(attribute.exprId)) match {
      case Some(attribute) => Left(s"IsNotNull filter on non-key column ${attribute.name}#${attribute.exprId.id}")
      case None => Right(())
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
  private[sparkmetal] def resolve(exprId: ExprId, plan: SparkPlan): Either[String, Lineage] = plan match {
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
  private[sparkmetal] def walkFactSide(plan: SparkPlan): Either[String, FactWalk] = plan match {
    case codegen: WholeStageCodegenExec => walkFactSide(codegen.child)
    case adapter: InputAdapter => walkFactSide(adapter.child)
    case project: ProjectExec => walkFactSide(project.child)
    case filter: FilterExec =>
      if (isOnlyNotNullPredicate(filter.condition)) {
        walkFactSide(filter.child).map { walk =>
          walk.copy(notNullTargets = collectNotNullAttributes(filter.condition) ++ walk.notNullTargets)
        }
      } else Left("filter beyond IsNotNull")
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
        Right(FactWalk(scan, Seq.empty, Seq.empty, scan.dataFilters.flatMap(collectNotNullAttributes)))
      }
    case other => Left(s"unsupported node between aggregate and scan: ${other.nodeName}")
  }

  private def isOnlyNotNullPredicate(expression: Expression): Boolean = expression match {
    case IsNotNull(_: AttributeReference) => true
    case And(left, right) => isOnlyNotNullPredicate(left) && isOnlyNotNullPredicate(right)
    case _ => false
  }

  /**
   * Extracts every IsNotNull target out of a predicate tree already known
   * (by `isOnlyNotNullPredicate`) to be built from nothing but IsNotNull and
   * And -- used only to figure out WHICH column each IsNotNull names, for
   * `validateNotNullTargets`'s key-only restriction. This never changes
   * whether a predicate is accepted, only which attributes get reported for
   * a predicate already accepted.
   */
  private def collectNotNullAttributes(expression: Expression): Seq[AttributeReference] = expression match {
    case IsNotNull(attribute: AttributeReference) => Seq(attribute)
    case And(left, right) => collectNotNullAttributes(left) ++ collectNotNullAttributes(right)
    case _ => Seq.empty
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
