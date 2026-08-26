package io.github.mohitpatil.sparkmetal

private[sparkmetal] object SparkMetalNative {
  @volatile private var initialized = false

  def ensureInitialized(nativeLibrary: String, metalLibrary: String): Unit = synchronized {
    if (!initialized) {
      System.load(nativeLibrary)
      NativeBridge.initialize(metalLibrary)
      initialized = true
    }
  }
}
