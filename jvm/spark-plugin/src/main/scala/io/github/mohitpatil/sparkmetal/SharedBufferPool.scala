package io.github.mohitpatil.sparkmetal

import java.nio.{ByteBuffer, ByteOrder}
import java.util.concurrent.ConcurrentHashMap

private[sparkmetal] final case class SharedBuffers(
    values: ByteBuffer,
    nulls: ByteBuffer,
    capacity: Int)

private[sparkmetal] object SharedBufferPool {
  private val buffers = new ConcurrentHashMap[Long, SharedBuffers]()

  def acquire(required: Int): SharedBuffers = {
    val threadId = Thread.currentThread().threadId()
    val current = buffers.get(threadId)
    if (current != null && current.capacity >= required) {
      current
    } else {
      if (current != null) {
        NativeBridge.releaseShared(current.nulls)
        NativeBridge.releaseShared(current.values)
      }
      val capacity = nextPowerOfTwo(required)
      val created = SharedBuffers(
        NativeBridge.allocateSharedInt32(capacity).order(ByteOrder.nativeOrder()),
        NativeBridge.allocateSharedBytes(capacity),
        capacity)
      buffers.put(threadId, created)
      created
    }
  }

  private def nextPowerOfTwo(value: Int): Int = {
    val highest = Integer.highestOneBit(value)
    if (highest == value) value else Math.multiplyExact(highest, 2)
  }
}
