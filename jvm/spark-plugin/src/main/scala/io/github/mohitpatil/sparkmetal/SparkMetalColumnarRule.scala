package io.github.mohitpatil.sparkmetal

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Partial, Sum}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ColumnarRule, SparkPlan}
import org.apache.spark.sql.execution.aggregate.HashAggregateExec
import org.apache.spark.sql.types.{IntegerType, LongType}

private case class FusedSumSpec(
    attribute: AttributeReference,
    threshold: Int,
    multiplier: Int,
    addend: Int)

final class SparkMetalColumnarRule(
    nativeLibrary: String,
    metalLibrary: String,
    ansiEnabled: Boolean)
    extends ColumnarRule {

  override def preColumnarTransitions: Rule[SparkPlan] = new Rule[SparkPlan] {
    override def apply(plan: SparkPlan): SparkPlan = plan.transformUp {
      case aggregate: HashAggregateExec => replace(aggregate).getOrElse(aggregate)
    }
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
