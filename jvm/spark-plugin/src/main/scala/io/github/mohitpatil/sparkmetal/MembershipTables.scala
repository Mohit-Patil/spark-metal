package io.github.mohitpatil.sparkmetal

/**
 * Builds the per-dictionary membership lookup tables shared by every exec
 * that streams Parquet dictionary ids (or off-heap dictionary-encoded
 * columns) straight into the GPU membership kernel instead of materializing
 * decoded values first.
 */
private[sparkmetal] object MembershipTables {
  /** presence(id) = 1 iff dictionary value id is in the key map (unique keys). */
  def presence(dictionary: Array[Int], keys: Map[Int, Long]): Array[Byte] = {
    val table = new Array[Byte](dictionary.length)
    var id = 0
    while (id < dictionary.length) {
      if (keys.contains(dictionary(id))) {
        table(id) = 1
      }
      id += 1
    }
    table
  }

  /** multiplicity(id) = key multiplicity of dictionary value id (duplicate keys). */
  def multiplicity(dictionary: Array[Int], keys: Map[Int, Long]): Array[Int] = {
    val table = new Array[Int](dictionary.length)
    var id = 0
    while (id < dictionary.length) {
      table(id) = Math.toIntExact(keys.getOrElse(dictionary(id), 0L))
      id += 1
    }
    table
  }
}
