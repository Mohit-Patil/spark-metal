package io.github.mohitpatil.sparkmetal

import scala.collection.mutable

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.types.{
  BooleanType, ByteType, DataType, DateType, DecimalType, IntegerType, LongType, ShortType, StringType}

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
   * attribute values (an empty [[InternalRow]], with `attributeCount == 0`
   * and `attributeTypes` empty, when the dimension contributes no group
   * keys). `attributeTypes` names every attribute field's type, in row
   * order, and must have length `attributeCount`; see
   * [[GroupSpace.SupportedAttributeTypes]] for what `build` accepts --
   * anything else is rejected with `Left`, naming the offending type.
   */
  case class Dimension(
      rows: Array[(Int, InternalRow)],
      attributeCount: Int,
      attributeTypes: Seq[DataType] = Seq.empty)

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

  /**
   * Attribute types `build` knows how to read a value out of and compare by
   * value: every fixed-width integral/boolean type, int-backed [[DateType]],
   * [[StringType]] (compared via [[org.apache.spark.unsafe.types.UTF8String]]
   * equality, which IS value-based), and [[DecimalType]] (any
   * precision/scale). Anything else -- DoubleType/FloatType, arrays, maps,
   * structs, ... -- is rejected by `build` before any row is ever read.
   */
  private def isSupportedAttributeType(dataType: DataType): Boolean = dataType match {
    case IntegerType | LongType | ShortType | ByteType | BooleanType | DateType | StringType => true
    case _: DecimalType => true
    case _ => false
  }

  /**
   * Extracts field `ordinal`'s value as a plain, independently-comparable
   * Scala value: `null` for a null field (Scala's `==`/collection equality
   * treats `null` elements of a `Seq` correctly, so no sentinel is needed);
   * a UTF8String is `.copy()`d since `row` may be a buffer a collector
   * reuses. Only called for a `dataType` that `isSupportedAttributeType`
   * has already accepted -- `build` validates every dimension's
   * `attributeTypes` before any row is read.
   */
  private def attributeValue(row: InternalRow, ordinal: Int, dataType: DataType): Any = {
    if (row.isNullAt(ordinal)) {
      null
    } else {
      dataType match {
        case IntegerType => row.getInt(ordinal)
        case LongType => row.getLong(ordinal)
        case ShortType => row.getShort(ordinal)
        case ByteType => row.getByte(ordinal)
        case BooleanType => row.getBoolean(ordinal)
        case DateType => row.getInt(ordinal) // int-backed: days since the epoch
        case StringType => row.getUTF8String(ordinal).copy()
        case decimalType: DecimalType => row.getDecimal(ordinal, decimalType.precision, decimalType.scale)
        case other =>
          // Unreachable: build() rejects any dimension carrying this type
          // before attributeValue is ever called.
          throw new IllegalStateException(s"unsupported attribute type reached attributeValue: $other")
      }
    }
  }

  private def attributeKey(row: InternalRow, attributeTypes: Seq[DataType]): Seq[Any] =
    attributeTypes.zipWithIndex.map { case (dataType, ordinal) => attributeValue(row, ordinal, dataType) }

  private case class DimensionSpace(
      componentCount: Int,
      componentByKey: Map[Int, Int],
      factorByKey: Map[Int, Int],
      tupleByComponent: Array[InternalRow])

  /**
   * Left(reason) when: a dimension with attributes has duplicate join keys,
   * a dimension's attribute type is not one `build` supports, the
   * mixed-radix product exceeds maxGroups, or any dimension is empty.
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
    if (dimension.attributeTypes.length != dimension.attributeCount) {
      return Left(
        s"dimension $index: attributeTypes has length ${dimension.attributeTypes.length}, " +
          s"expected attributeCount ${dimension.attributeCount}")
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
      dimension.attributeTypes.zipWithIndex.find { case (dataType, _) => !isSupportedAttributeType(dataType) } match {
        case Some((dataType, ordinal)) =>
          return Left(s"dimension $index: unsupported attribute type $dataType at attribute position $ordinal")
        case None => ()
      }

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
        val tupleKey = attributeKey(row, dimension.attributeTypes)
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
