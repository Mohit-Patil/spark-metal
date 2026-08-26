package io.github.mohitpatil.sparkmetal

import scala.collection.mutable
import scala.util.control.NonFatal

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Count, Partial, Sum}
import org.apache.spark.sql.catalyst.optimizer.{BuildLeft, BuildRight}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ColumnarRule, FileSourceScanExec, FilterExec, ProjectExec, SparkPlan}
import org.apache.spark.sql.execution.adaptive.BroadcastQueryStageExec
import org.apache.spark.sql.execution.aggregate.HashAggregateExec
import org.apache.spark.sql.execution.datasources.parquet.ParquetFileFormat
import org.apache.spark.sql.execution.exchange.{BroadcastExchangeExec, ReusedExchangeExec}
import org.apache.spark.sql.execution.joins.BroadcastHashJoinExec
import org.apache.spark.sql.types.{IntegerType, LongType}

private case class FusedSumSpec(
    attribute: AttributeReference,
    threshold: Int,
    multiplier: Int,
    addend: Int)

private case class MembershipJoinSpec(
    factAttribute: AttributeReference,
    keyPlan: SparkPlan)

private case class MembershipTreeSpec(
    factPlan: SparkPlan,
    joins: Seq[MembershipJoinSpec])

final class SparkMetalColumnarRule(
    nativeLibrary: String,
    metalLibrary: String,
    ansiEnabled: Boolean,
    adaptiveEnabled: Boolean,
    parquetScanEnabled: Boolean,
    parquetAggregateEnabled: Boolean,
    parquetAggregateMaxRegions: Int)
    extends ColumnarRule with Logging {

  override def preColumnarTransitions: Rule[SparkPlan] = new Rule[SparkPlan] {
    override def apply(plan: SparkPlan): SparkPlan = {
      // Controller ruling (Task 7b): the grouped-aggregate operator's
      // per-region fixed cost (dimension collection, group-space build,
      // split planning, row-at-a-time emission) multiplies with the number
      // of regions in a query, while Spark's CPU plan shares work across
      // them -- every query with >=3 regions in the full-suite run lost
      // badly (0.09x-0.50x) and every single-region query won or tied. So
      // the grouped branch is only allowed to fire at all for this plan
      // when the number of regions it would match does not exceed
      // `spark.metal.parquetAggregate.maxRegions` (values <= 0 mean no
      // limit). This is computed ONCE for the whole plan, here, rather than
      // per node -- `groupedAggregateAllowed` is then just a captured local
      // read by every `replaceGroupedAggregate` call below.
      val groupedAggregateAllowed = isGroupedAggregateAllowed(plan)
      plan.transformUp {
        case aggregate: HashAggregateExec =>
          (if (groupedAggregateAllowed) replaceGroupedAggregate(aggregate) else None)
            .orElse(replaceMembershipCount(aggregate))
            .orElse(replace(aggregate))
            .getOrElse(aggregate)
      }
    }
  }

  /**
   * Whether the grouped-aggregate branch may fire anywhere in `plan`: false
   * outright when the branch's own flags (`parquetAggregateEnabled`,
   * `ansiEnabled`, `adaptiveEnabled`) already disable it -- skipping the
   * plan walk entirely in that (common, flag-off) case -- otherwise
   * `countMatchingRegions(plan) <= parquetAggregateMaxRegions` (a
   * `maxRegions <= 0` config means unlimited).
   *
   * Safe-fallback: `countMatchingRegions` calls the same pure, filesystem-
   * free `GroupedAggregateShape.matchRegion` that `replaceGroupedAggregate`
   * already wraps in `NonFatal`, but a thrown exception here (unlike there)
   * would abort planning of the whole query rather than falling back to a
   * single region -- so any such surprise is caught here too, and treated
   * as "the grouped branch is not allowed for this plan" (the same
   * conservative choice `replaceGroupedAggregate` makes per-region) rather
   * than as unlimited.
   */
  private def isGroupedAggregateAllowed(plan: SparkPlan): Boolean = {
    if (!parquetAggregateEnabled || ansiEnabled || adaptiveEnabled) {
      return false
    }
    try {
      parquetAggregateMaxRegions <= 0 || countMatchingRegions(plan) <= parquetAggregateMaxRegions
    } catch {
      case NonFatal(e) =>
        logWarning(
          "Grouped-aggregate region count failed; disabling the grouped-aggregate branch " +
            "for this plan", e)
        false
    }
  }

  /**
   * Counts the query plan's distinct, non-nested `matchRegion` matches:
   * walks `plan` top-down, and whenever a [[HashAggregateExec]] node
   * matches, counts it as one region WITHOUT recursing into that node's own
   * children -- a matched region's subtree, down to its fact scan, is
   * already consumed by that match, so any nested `HashAggregateExec`
   * within it cannot be an independent region. A non-matching aggregate (or
   * any other node) recurses into its children, so independent regions
   * elsewhere in the plan (a second star-join island, a subquery) are still
   * found and counted.
   */
  private def countMatchingRegions(plan: SparkPlan): Int = plan match {
    case aggregate: HashAggregateExec =>
      GroupedAggregateShape.matchRegion(aggregate) match {
        case scala.util.Right(_) => 1
        case scala.util.Left(_) => aggregate.children.map(countMatchingRegions).sum
      }
    case other => other.children.map(countMatchingRegions).sum
  }

  /**
   * Task 6: plans [[MetalParquetGroupedAggregateExec]] in place of a partial
   * Sum/Count/Average [[HashAggregateExec]] grouping on star-schema
   * dimension attributes -- [[GroupedAggregateShape.matchRegion]] does the
   * structural matching (Task 1); this method is the eligibility gate on
   * top of a match: the feature flag, the kernel's join-count and internal-
   * aggregate-slot caps, [[GroupSpace]]'s supported group-key attribute
   * types, and [[ParquetEligibility]] over both the join keys and the
   * measure columns -- neither is required to be dictionary-encoded (Task
   * 6b relaxed the join-key requirement to match the measure columns': PLAIN
   * or dictionary, per chunk -- see `ParquetEligibility.checkKeys`/
   * `checkMeasures`). A PLAIN join-key column carries one further
   * obligation this planning-time gate cannot check: its dimension's
   * join-key domain must fit the GPU decoder's value-space code table, a
   * RUNTIME condition enforced by `MetalParquetGroupedAggregateExec.
   * doExecute` once the dimension's rows are actually collected (violating
   * it routes the whole operator through its CPU hash-join fallback, not a
   * planning-time rejection).
   *
   * Ordered BEFORE [[replaceMembershipCount]]: `matchRegion` itself already
   * refuses a count-only, zero-group-key region (q96/q88/q90's shape), so
   * the two rules never compete for the same region -- this method returns
   * `None` for those regions (by way of `matchRegion` returning `Left`)
   * before doing any filesystem work, and the membership branches proceed
   * exactly as before.
   *
   * Safe-fallback: mirrors `parquetMembershipCountExec` -- the whole
   * evaluation (shape re-derivation, footer reads via `ParquetEligibility`)
   * is wrapped in `NonFatal` so a planner-time surprise (a corrupt footer, an
   * unexpected plan shape `matchRegion` didn't anticipate) falls back to the
   * untouched `HashAggregateExec` rather than aborting planning of the whole
   * query.
   */
  private def replaceGroupedAggregate(aggregate: HashAggregateExec): Option[SparkPlan] = {
    // ansiEnabled: this operator's GPU sum accumulators are wrapping int64 --
    // ANSI SQL's overflow-throwing Sum semantics diverge from that, exactly
    // why `replace` (the fused-sum branch) also gates on ansiEnabled.
    // adaptiveEnabled: this branch sits ahead of (preempts) the only other
    // adaptiveEnabled check (in replaceMembershipCount) for every region it
    // matches, and the keyPlans it builds re-execute a stripped
    // BroadcastQueryStageExec child directly, outside AQE's own replan
    // machinery -- disabled under AQE for the same reason the membership
    // branch is.
    if (!parquetAggregateEnabled || ansiEnabled || adaptiveEnabled) {
      return None
    }
    try {
      GroupedAggregateShape.matchRegion(aggregate) match {
        case scala.util.Left(_) => None
        case scala.util.Right(region) => buildGroupedAggregateExec(region)
      }
    } catch {
      case NonFatal(e) =>
        logWarning(
          "Grouped-aggregate eligibility check failed; falling back to the next candidate " +
            "operator for this region", e)
        None
    }
  }

  private def buildGroupedAggregateExec(region: GroupedAggregateShape.Region): Option[SparkPlan] = {
    // Kernel key cap: at most 4 joins (dimensions), and this operator always
    // needs at least one to have a join key column to decode at all.
    if (region.joins.isEmpty || region.joins.length > 4) {
      return None
    }
    // Internal aggregate slot cap: 2 slots per sum/avg (sum + paired count),
    // 1 per count, plus 1 occupancy slot -- computed conservatively (no
    // dedup across repeated aggregates on the same column), matching
    // MetalParquetGroupedAggregateExec.buildInternalAggPlan's actual layout.
    val internalSlotCount = region.aggs.map(spec => if (spec.function == "count") 1 else 2).sum + 1
    if (internalSlotCount > 8) {
      return None
    }
    // Pre-reject any group-key attribute type GroupSpace cannot represent
    // (Double/Float in particular) here, at planning time, rather than
    // letting the exec's driver-side throw be the first place this surfaces.
    // Reuses GroupSpace's own predicate rather than a second, independently
    // drifting copy of the supported-type list.
    if (!region.groupKeys.forall(attribute => GroupSpace.isSupportedAttributeType(attribute.dataType))) {
      return None
    }

    val buildPlanOptions = region.joins.map(joinBuildPlan)
    if (buildPlanOptions.exists(_.isEmpty)) {
      return None
    }
    val buildPlans = buildPlanOptions.map(_.get)
    val buildKeyExprs = region.joins.map(joinBuildKey)
    // The build-side join key must itself be int32 -- it is compared for
    // equality against the fact-side int32 key (matchRegion already
    // validated the fact side), and MetalParquetGroupedAggregateExec's
    // dimension collector always reads ordinal 0 of a keyPlan row as an Int.
    if (buildKeyExprs.exists(_.dataType != IntegerType)) {
      return None
    }

    // Every group key resolved (by matchRegion) to some join's build-side
    // output; re-derive WHICH join here so it can be projected out of that
    // join's own keyPlan, in the canonical per-dimension order it is first
    // encountered.
    val exprIdToDimension: Map[ExprId, Int] =
      buildPlans.zipWithIndex.flatMap { case (plan, index) => plan.output.map(_.exprId -> index) }.toMap
    if (!region.groupKeys.forall(attribute => exprIdToDimension.contains(attribute.exprId))) {
      return None
    }

    val dimensionAttributes = Array.fill(region.joins.length)(mutable.ArrayBuffer.empty[Attribute])
    val groupKeyDimensionIndex = region.groupKeys.map { attribute =>
      val dimensionIndex = exprIdToDimension(attribute.exprId)
      val attributes = dimensionAttributes(dimensionIndex)
      val existingPosition = attributes.indexWhere(_.exprId == attribute.exprId)
      if (existingPosition >= 0) {
        (dimensionIndex, existingPosition)
      } else {
        attributes += attribute
        (dimensionIndex, attributes.length - 1)
      }
    }

    val keyPlans = region.joins.indices.map { index =>
      val keyExpression: NamedExpression = buildKeyExprs(index) match {
        case named: NamedExpression => named
        case other => Alias(other, "spark_metal_group_key")()
      }
      ProjectExec(keyExpression +: dimensionAttributes(index).toSeq, buildPlans(index))
    }

    val keyColumnNames = region.factKeys.map(_.name)
    val measureColumnNames = region.measureColumns.map(_.name)
    // MetalParquetGroupedAggregateExec.require caps this at 4; pre-check it
    // here so an over-wide region declines quietly (falls through to the
    // next candidate operator) instead of reaching the exec's constructor
    // require and surfacing as a logWarning stack trace on a plan that
    // should simply not have been rewritten.
    if (measureColumnNames.length > 4) {
      return None
    }
    val files = region.scan.relation.location.inputFiles.toSeq

    if (ParquetEligibility.checkKeys(files, keyColumnNames).isLeft) {
      return None
    }
    if (measureColumnNames.nonEmpty && ParquetEligibility.checkMeasures(files, measureColumnNames).isLeft) {
      return None
    }

    Some(MetalParquetGroupedAggregateExec(
      region.aggregate.output,
      files,
      keyColumnNames,
      measureColumnNames,
      region.aggs,
      groupKeyDimensionIndex,
      keyPlans,
      nativeLibrary,
      metalLibrary))
  }

  /**
   * The join's build-side plan, with its [[BroadcastExchangeExec]] (or an
   * AQE [[ReusedExchangeExec]]/[[BroadcastQueryStageExec]] wrapping one)
   * stripped off -- mirrors `stripBroadcast`'s use in
   * `extractMembershipTree`. A raw `join.left`/`join.right` here is always
   * headed by one of these exchange nodes (EnsureRequirements guarantees a
   * broadcast exchange sits directly above a `BroadcastHashJoinExec`'s build
   * side), and `BroadcastExchangeExec.doExecute()` throws
   * `UnsupportedOperationException` -- it only supports
   * `doExecuteBroadcast()` -- so a `ProjectExec` wrapping the unstripped
   * node would compile and plan fine but blow up the first time
   * `executeCollect()` actually ran it. `None` when the build side is not
   * headed by a recognized exchange node, an unexpected shape that should
   * fall back to the next candidate operator rather than risk that crash.
   */
  private def joinBuildPlan(join: BroadcastHashJoinExec): Option[SparkPlan] = {
    val raw = if (join.buildSide == BuildLeft) join.left else join.right
    stripBroadcast(raw)
  }

  private def joinBuildKey(join: BroadcastHashJoinExec): Expression =
    if (join.buildSide == BuildLeft) join.leftKeys.head else join.rightKeys.head

  private def replaceMembershipCount(aggregate: HashAggregateExec): Option[SparkPlan] = {
    if (adaptiveEnabled ||
        aggregate.groupingExpressions.nonEmpty ||
        aggregate.aggregateExpressions.length != 1 ||
        aggregate.output.length != 1 ||
        aggregate.output.head.dataType != LongType ||
        !isPartialCountOne(aggregate.aggregateExpressions.head)) {
      return None
    }
    extractMembershipTree(aggregate.child).flatMap { tree =>
      if (tree.joins.length != 3 || tree.joins.map(_.factAttribute.exprId).distinct.length != 3) {
        None
      } else {
        val ordinals = tree.joins.map { join =>
          tree.factPlan.output.indexWhere(_.exprId == join.factAttribute.exprId)
        }
        if (ordinals.contains(-1)) {
          None
        } else {
          val keyPlans = tree.joins.map(_.keyPlan)
          parquetMembershipCountExec(aggregate.output.head, tree.factPlan, ordinals, keyPlans)
            .orElse(Some(MetalFusedMembershipCountExec(
              aggregate.output.head,
              ordinals,
              keyPlans,
              nativeLibrary,
              metalLibrary,
              tree.factPlan)))
        }
      }
    }
  }

  /**
   * When the fact-side plan is a bare Parquet [[FileSourceScanExec]] eligible
   * for the GPU Parquet decode path (Tasks 1-4), emit
   * [[MetalParquetMembershipCountExec]] instead, which decodes pages
   * straight off disk on the GPU rather than routing through the CPU
   * columnar scan + row conversion the fused path relies on. The scan's own
   * `IsNotNull` data filters over the three key columns are dropped: q96's
   * joins are inner joins on these keys, and both Metal operators already
   * drop null-key rows, so the filters are subsumed. Falls back to `None`
   * (letting the fused path proceed) whenever the region does not match
   * these conditions exactly, `spark.metal.parquetScan.enabled` is false, or
   * `ParquetEligibility.check` rejects the files.
   *
   * Safe-fallback: the shape check and `ParquetEligibility.check` both touch
   * the filesystem (file listing, footer reads) on the planner's hot path.
   * `ParquetEligibility.checkFile` does not itself catch I/O failures (a
   * corrupt footer, a transient read error, a permissions problem), so any
   * such throwable is caught here and treated as "not eligible" rather than
   * aborting planning of the whole query -- the fused path is always a safe
   * fallback for a file this GPU path cannot even inspect.
   */
  private def parquetMembershipCountExec(
      outputAttribute: Attribute,
      factPlan: SparkPlan,
      ordinals: Seq[Int],
      keyPlans: Seq[SparkPlan]): Option[SparkPlan] = {
    if (!parquetScanEnabled) {
      return None
    }
    try {
      factPlan match {
        case scan: FileSourceScanExec =>
          val relation = scan.relation
          val isEligibleShape =
            relation.fileFormat.isInstanceOf[ParquetFileFormat] &&
              relation.partitionSchema.isEmpty &&
              relation.bucketSpec.isEmpty &&
              scan.output.length == 3 &&
              scan.output.forall(_.dataType == IntegerType) &&
              ordinals.toSet == scan.output.indices.toSet &&
              scan.dataFilters.forall(isOnlyNotNullPredicate)
          if (!isEligibleShape) {
            None
          } else {
            val columnNames = ordinals.map(scan.output(_).name)
            val files = relation.location.inputFiles.toSeq
            if (ParquetEligibility.check(files, columnNames).isRight) {
              Some(MetalParquetMembershipCountExec(
                outputAttribute, files, columnNames, keyPlans, nativeLibrary, metalLibrary))
            } else {
              None
            }
          }
        case _ => None
      }
    } catch {
      case NonFatal(e) =>
        logWarning(
          "Parquet eligibility check for the GPU scan failed; falling back to " +
            "MetalFusedMembershipCountExec for this region", e)
        None
    }
  }

  private def isPartialCountOne(expression: AggregateExpression): Boolean = {
    if (expression.mode != Partial || expression.isDistinct || expression.filter.nonEmpty) {
      return false
    }
    expression.aggregateFunction match {
      case count: Count => count.children match {
        case Seq(literal: Literal) => literal.value != null
        case _ => false
      }
      case _ => false
    }
  }

  private def extractMembershipTree(plan: SparkPlan): Option[MembershipTreeSpec] = plan match {
    case ProjectExec(projectList, child)
        if projectList.forall(_.isInstanceOf[AttributeReference]) =>
      extractMembershipTree(child)
    case FilterExec(condition, child) if isOnlyNotNullPredicate(condition) =>
      extractMembershipTree(child)
    case join: BroadcastHashJoinExec
        if join.joinType == Inner && join.buildSide == BuildRight && join.condition.isEmpty &&
          join.leftKeys.length == 1 && join.rightKeys.length == 1 =>
      (join.leftKeys.head, join.rightKeys.head) match {
        case (factAttribute: AttributeReference, _: AttributeReference) =>
          for {
            tree <- extractMembershipTree(join.left)
            keyPlan <- stripBroadcast(join.right)
            if keyPlan.output.length == 1 && keyPlan.output.head.dataType == IntegerType
          } yield tree.copy(joins = tree.joins :+ MembershipJoinSpec(factAttribute, keyPlan))
        case _ => None
      }
    case scan: FileSourceScanExec => Some(MembershipTreeSpec(scan, Seq.empty))
    case _ => None
  }

  private def stripBroadcast(plan: SparkPlan): Option[SparkPlan] = plan match {
    case exchange: BroadcastExchangeExec => Some(exchange.child)
    case reused: ReusedExchangeExec => stripBroadcast(reused.child)
    case stage: BroadcastQueryStageExec => stripBroadcast(stage.plan)
    case _ => None
  }

  private def isOnlyNotNullPredicate(expression: Expression): Boolean = expression match {
    case IsNotNull(_: AttributeReference) => true
    case And(left, right) => isOnlyNotNullPredicate(left) && isOnlyNotNullPredicate(right)
    case _ => false
  }

  private def replace(aggregate: HashAggregateExec): Option[SparkPlan] = {
    if (ansiEnabled ||
        aggregate.groupingExpressions.nonEmpty ||
        aggregate.aggregateExpressions.length != 1 ||
        aggregate.output.length != 1 ||
        aggregate.output.head.dataType != LongType) {
      return None
    }
    extract(aggregate.aggregateExpressions.head).flatMap { spec =>
      val ordinal = aggregate.child.output.indexWhere(_.exprId == spec.attribute.exprId)
      if (ordinal < 0 || aggregate.child.output(ordinal).dataType != IntegerType) {
        None
      } else {
        Some(MetalFusedSumExec(
          aggregate.output.head,
          ordinal,
          spec.threshold,
          spec.multiplier,
          spec.addend,
          nativeLibrary,
          metalLibrary,
          aggregate.child))
      }
    }
  }

  private def extract(expression: AggregateExpression): Option[FusedSumSpec] = {
    if (expression.mode != Partial || expression.isDistinct || expression.filter.nonEmpty) {
      return None
    }
    expression.aggregateFunction match {
      case sum: Sum => extractCase(sum.child)
      case _ => None
    }
  }

  private def extractCase(expression: Expression): Option[FusedSumSpec] = expression match {
    case CaseWhen(Seq((condition, projection)), Some(otherwise)) if literalInt(otherwise).contains(0) =>
      for {
        (attribute, threshold) <- greaterThanAttributeLiteral(condition)
        (projectedAttribute, multiplier, addend) <- multipliedThenAdded(projection)
        if projectedAttribute.exprId == attribute.exprId
      } yield FusedSumSpec(attribute, threshold, multiplier, addend)
    case _ => None
  }

  private def greaterThanAttributeLiteral(
      expression: Expression): Option[(AttributeReference, Int)] = expression match {
    case GreaterThan(attribute: AttributeReference, literal: Literal) =>
      literalInt(literal).map(attribute -> _)
    case _ => None
  }

  private def multipliedThenAdded(
      expression: Expression): Option[(AttributeReference, Int, Int)] = expression match {
    case add: Add =>
      multiplied(add.left).flatMap { case (attribute, multiplier) =>
        literalInt(add.right).map(addend => (attribute, multiplier, addend))
      }.orElse {
        multiplied(add.right).flatMap { case (attribute, multiplier) =>
          literalInt(add.left).map(addend => (attribute, multiplier, addend))
        }
      }
    case _ => None
  }

  private def multiplied(expression: Expression): Option[(AttributeReference, Int)] = expression match {
    case multiply: Multiply =>
      (multiply.left, multiply.right) match {
        case (attribute: AttributeReference, literal: Literal) =>
          literalInt(literal).map(attribute -> _)
        case (literal: Literal, attribute: AttributeReference) =>
          literalInt(literal).map(attribute -> _)
        case _ => None
      }
    case _ => None
  }

  private def literalInt(expression: Expression): Option[Int] = expression match {
    case literal: Literal if literal.dataType == IntegerType => Option(literal.value).map(_.asInstanceOf[Int])
    case _ => None
  }
}
