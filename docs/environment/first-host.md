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
| Java | Not installed/detected |
| Apache Spark | Not installed/detected |
| Free disk at capture | Approximately 203 GiB |

## Consequences

- The first benchmark target is Apple M5, although code should avoid M5-only assumptions unless guarded by capability detection.
- M4 remains a compatibility target, but claims must be based on hardware actually tested.
- The 16 GB unified-memory limit requires conservative Spark heap sizing and explicit swap monitoring.
- Java and Spark installation is the first environment task.

