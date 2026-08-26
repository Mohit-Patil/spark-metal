# First benchmark host

Captured on 2026-08-26.

| Item | Detected value |
|---|---|
| Machine | MacBook Air |
| Model identifier | Mac17,3 |
| Chip | Apple M5 |
| CPU configuration | 10 cores: 4 performance, 6 efficiency |
| Unified memory | 16 GB |
| Architecture | arm64 |
| macOS | 26.6.2 (25G83) |
| Python | 3.13.14 |
| Xcode | 26.6 (17F113) |
| Metal compiler | Available through the Xcode default toolchain |
| Java | Homebrew OpenJDK 21.0.12.1 |
| Apache Spark | 4.2.0, Scala 2.13.18, revision 32f7299601108917fb01920a54e084595b7b3bf8 |
| Free disk at capture | Approximately 203 GiB |

## Consequences

- The first benchmark target is Apple M5, although code should avoid M5-only assumptions unless guarded by capability detection.
- M4 remains a compatibility target, but claims must be based on hardware actually tested.
- The 16 GB unified-memory limit requires conservative Spark heap sizing and explicit swap monitoring.
- The Homebrew JDK is kept project-local through `scripts/project-env.sh`; no system-wide JDK symlink is required.
- A local ARM64 Spark example completed successfully with Java 21 before benchmark work began.
