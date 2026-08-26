package io.github.mohitpatil.sparkmetal

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Count, Partial, Sum}
import org.apache.spark.sql.catalyst.optimizer.BuildRight
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ColumnarRule, FileSourceScanExec, FilterExec, ProjectExec, SparkPlan}
import org.apache.spark.sql.execution.adaptive.BroadcastQueryStageExec
import org.apache.spark.sql.execution.aggregate.HashAggregateExec
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
    adaptiveEnabled: Boolean)
    extends ColumnarRule {

  override def preColumnarTransitions: Rule[SparkPlan] = new Rule[SparkPlan] {
    override def apply(plan: SparkPlan): SparkPlan = plan.transformUp {
      case aggregate: HashAggregateExec =>
        replaceMembershipCount(aggregate).orElse(replace(aggregate)).getOrElse(aggregate)
    }
  }

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
          Some(MetalFusedMembershipCountExec(
            aggregate.output.head,
            ordinals,
            tree.joins.map(_.keyPlan),
            nativeLibrary,
            metalLibrary,
            tree.factPlan))
        }
      }
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
