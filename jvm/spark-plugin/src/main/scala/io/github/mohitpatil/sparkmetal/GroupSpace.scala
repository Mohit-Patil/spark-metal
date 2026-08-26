package io.github.mohitpatil.sparkmetal

import scala.collection.mutable

import org.apache.spark.sql.catalyst.InternalRow

/**
 * Driver-side (pure JVM, no GPU/native calls) builder for the dense
 * mixed-radix group space a grouped aggregate's dimensions define.
 *
 * Each [[Dimension]] is one join-key -> attribute-tuple relation (the
 * "build side" of what would otherwise be a broadcast hash join): distinct
 * attribute tuples become that dimension's components 0..Di-1 (an
 * attribute-free dimension -- attributeCount == 0 -- has exactly one
 * component, the empty tuple, and contributes nothing to the group id;
 * see below). Dimension i's radix -- the multiplier applied to its
 * component index when forming a group id -- is the product of every
 * PRECEDING dimension's component count (radix_0 = 1, radix_1 = D_0,
 * radix_2 = D_0*D_1, ...), so a row's group id is simply the sum of its
 * per-dimension premultiplied codes, and that sum decomposes back into a
 * unique tuple of component indices.
 *
 * The maps in [[Built]] are keyed by the dimension's join-key VALUE, not by
 * Parquet dictionary id: this builder is entirely value-space and knows
 * nothing about Parquet or dictionaries. Translating a value-keyed map into
 * a dictionary-id-indexed array for [[NativeBridge.parquetRowGroupAggregate]]
 * (`table(dictId) = valueKeyedMap.getOrElse(dictionaryValues(dictId), -1)`
 * for codes, default `1` for factors) is the caller's job -- see the
 * `aggregate` mode of ParquetDecodeSmoke for a worked example, and Task 5
 * for where that translation becomes a reusable piece of this pipeline.
 */
private[sparkmetal] object GroupSpace {

  /**
   * One dimension's collected build-side rows: join key -> its group-key
   * attribute values (an empty [[InternalRow]] when the dimension
   * contributes no group keys, i.e. attributeCount == 0). Every attribute
   * field is assumed to be a nullable INT32 -- the only attribute/measure
   * type this codebase's GPU tier supports anywhere (see
   * GroupedAggregateShape's INT32-only measure/group-key restrictions) --
   * so tuple equality below reads fields with `isNullAt`/`getInt` rather
   * than a general schema-driven getter.
   */
  case class Dimension(rows: Array[(Int, InternalRow)], attributeCount: Int)

  case class Built(
      groupCount: Int,
      // per dimension: joinKey -> premultiplied component code (unique keys).
      codesByKey: Seq[Map[Int, Int]],
      // per dimension: joinKey -> multiplicity factor. Populated only for
      // attribute-free dimensions (where duplicate join keys are allowed and
      // fold into a count); empty for attributed dimensions, where duplicate
      // join keys are rejected by `build` and the factor is always 1.
      factorsByKey: Seq[Map[Int, Int]],
      // groupId -> the per-dimension attribute row for that group, i.e.
      // groupTuples(g)(i) is dimension i's attribute InternalRow for group g
      // (the empty row for an attribute-free dimension). These are COPIES:
      // `rows` may hold buffers a collector reuses across calls, so every
      // InternalRow retained here has had `.copy()` called on it.
      groupTuples: Array[Array[InternalRow]])

  private val EmptyTuple: InternalRow = InternalRow()

  // Sentinel distinguishing a null attribute field from every possible boxed
  // Int value, so two tuples that are both-null in some field compare equal
  // to each other and unequal to any int (including 0).
  private object NullField

  private def attributeKey(row: InternalRow, attributeCount: Int): Seq[Any] =
    (0 until attributeCount).map { i =>
      if (row.isNullAt(i)) NullField else row.getInt(i)
    }

  private case class DimensionSpace(
      componentCount: Int,
      componentByKey: Map[Int, Int],
      factorByKey: Map[Int, Int],
      tupleByComponent: Array[InternalRow])

  /**
   * Left(reason) when: a dimension with attributes has duplicate join keys,
   * the mixed-radix product exceeds maxGroups, or any dimension is empty.
   */
  def build(dimensions: Seq[Dimension], maxGroups: Int): Either[String, Built] = {
    val spaces = mutable.ArrayBuffer.empty[DimensionSpace]
    for ((dimension, index) <- dimensions.zipWithIndex) {
      buildDimensionSpace(dimension, index) match {
        case Left(reason) => return Left(reason)
        case Right(space) => spaces += space
      }
    }

    // radix_i = product of D_j for j < i (dimension 0 is least-significant).
    // Long arithmetic guards the running product against Int overflow while
    // still comparing the final count against maxGroups.
    val radices = new Array[Long](spaces.length)
    var groupCount = 1L
    for (i <- spaces.indices) {
      radices(i) = groupCount
      groupCount *= spaces(i).componentCount
    }
    if (groupCount > maxGroups) {
      return Left(s"group space size $groupCount exceeds maxGroups $maxGroups")
    }
    val groupCountInt = groupCount.toInt

    val codesByKey = spaces.indices.map { i =>
      spaces(i).componentByKey.view.mapValues(component => (component * radices(i)).toInt).toMap
    }
    val factorsByKey = spaces.map(_.factorByKey).toSeq

    val groupTuples = Array.tabulate(groupCountInt) { group =>
      Array.tabulate(spaces.length) { i =>
        val component = ((group / radices(i)) % spaces(i).componentCount).toInt
        spaces(i).tupleByComponent(component)
      }
    }

    Right(Built(groupCountInt, codesByKey, factorsByKey, groupTuples))
  }

  private def buildDimensionSpace(dimension: Dimension, index: Int): Either[String, DimensionSpace] = {
    if (dimension.rows.isEmpty) {
      return Left(s"dimension $index is empty")
    }

    if (dimension.attributeCount == 0) {
      // Attribute-free: every distinct join key maps to the SAME (only)
      // component, 0 -- this dimension contributes nothing to the group id,
      // only a membership gate and a multiplicity factor. Duplicate join
      // keys are expected here and fold into that factor.
      val factorByKey = mutable.LinkedHashMap.empty[Int, Int]
      for ((key, _) <- dimension.rows) {
        factorByKey(key) = factorByKey.getOrElse(key, 0) + 1
      }
      val componentByKey = factorByKey.keys.map(key => key -> 0).toMap
      Right(DimensionSpace(1, componentByKey, factorByKey.toMap, Array(EmptyTuple)))
    } else {
      val seenKeys = mutable.HashSet.empty[Int]
      for ((key, _) <- dimension.rows) {
        if (!seenKeys.add(key)) {
          return Left(s"dimension $index: duplicate join key $key in an attributed dimension")
        }
      }
      val componentIndexByTuple = mutable.LinkedHashMap.empty[Seq[Any], Int]
      val tuples = mutable.ArrayBuffer.empty[InternalRow]
      val componentByKey = mutable.HashMap.empty[Int, Int]
      for ((key, row) <- dimension.rows) {
        val tupleKey = attributeKey(row, dimension.attributeCount)
        val component = componentIndexByTuple.getOrElseUpdate(tupleKey, {
          tuples += row.copy()
          tuples.length - 1
        })
        componentByKey(key) = component
      }
      Right(DimensionSpace(tuples.length, componentByKey.toMap, Map.empty, tuples.toArray))
    }
  }
}
