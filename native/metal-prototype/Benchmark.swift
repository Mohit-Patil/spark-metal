import Foundation
import Metal

private let threadsPerGroup = 256

private struct FusedParameters {
    var count: UInt32
    var threshold: Int32
    var multiplier: Int32
    var addend: Int32
    var hasNulls: UInt32
}

private struct TimedResult {
    let sum: Int64
    let wallSeconds: Double
    let deviceSeconds: Double
}

private struct ConfigurationResult: Codable {
    let rows: Int
    let expectedSum: Int64
    let cpuSeconds: [Double]
    let gpuDispatchSeconds: [Double]
    let gpuCopyInclusiveSeconds: [Double]
    let gpuDeviceSeconds: [Double]
    let cpuMedianSeconds: Double
    let gpuDispatchMedianSeconds: Double
    let gpuCopyInclusiveMedianSeconds: Double
    let dispatchSpeedup: Double
    let copyInclusiveSpeedup: Double
}

private struct Report: Codable {
    let createdAt: String
    let deviceName: String
    let operatingSystem: String
    let threshold: Int32
    let multiplier: Int32
    let addend: Int32
    let warmups: Int
    let runs: Int
    let configurations: [ConfigurationResult]
}

private struct Arguments {
    var libraryPath: String = ""
    var sizes = [65_536, 1_048_576, 8_388_608]
    var warmups = 2
    var runs = 7
    var outputPath: String?

    init() throws {
        var iterator = CommandLine.arguments.dropFirst().makeIterator()
        guard let first = iterator.next() else {
            throw BenchmarkError.usage("Missing metallib path")
        }
        libraryPath = first
        while let argument = iterator.next() {
            switch argument {
            case "--sizes":
                guard let value = iterator.next() else { throw BenchmarkError.usage("--sizes needs a value") }
                sizes = try value.split(separator: ",").map {
                    guard let parsed = Int($0), parsed > 0 else {
                        throw BenchmarkError.usage("Invalid row count: \($0)")
                    }
                    return parsed
                }
            case "--warmups":
                guard let value = iterator.next(), let parsed = Int(value), parsed >= 0 else {
                    throw BenchmarkError.usage("--warmups needs a non-negative integer")
                }
                warmups = parsed
            case "--runs":
                guard let value = iterator.next(), let parsed = Int(value), parsed > 0 else {
                    throw BenchmarkError.usage("--runs needs a positive integer")
                }
                runs = parsed
            case "--output":
                guard let value = iterator.next() else { throw BenchmarkError.usage("--output needs a path") }
                outputPath = value
            default:
                throw BenchmarkError.usage("Unknown argument: \(argument)")
            }
        }
    }
}

private enum BenchmarkError: Error, CustomStringConvertible {
    case usage(String)
    case runtime(String)

    var description: String {
        switch self {
        case .usage(let message):
            return "\(message)\nUsage: metal-benchmark KERNELS.metallib [--sizes N,N] [--warmups N] [--runs N] [--output FILE]"
        case .runtime(let message):
            return message
        }
    }
}

private final class MetalFusedRunner {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let inputBuffer: MTLBuffer
    private let validityBuffer: MTLBuffer
    private let partialBuffer: MTLBuffer
    private let groupCount: Int
    private var parameters: FusedParameters

    init(device: MTLDevice, libraryPath: String, count: Int, threshold: Int32, multiplier: Int32, addend: Int32) throws {
        guard count <= Int(UInt32.max) else { throw BenchmarkError.runtime("Row count exceeds UInt32") }
        guard let queue = device.makeCommandQueue() else { throw BenchmarkError.runtime("Cannot create Metal command queue") }
        let library = try device.makeLibrary(URL: URL(fileURLWithPath: libraryPath))
        guard let function = library.makeFunction(name: "fused_filter_project_sum") else {
            throw BenchmarkError.runtime("Kernel fused_filter_project_sum was not found")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        guard pipeline.maxTotalThreadsPerThreadgroup >= threadsPerGroup else {
            throw BenchmarkError.runtime("Metal device does not support \(threadsPerGroup) threads per group")
        }
        groupCount = (count + threadsPerGroup - 1) / threadsPerGroup
        guard let inputBuffer = device.makeBuffer(length: count * MemoryLayout<Int32>.stride, options: .storageModeShared),
              let validityBuffer = device.makeBuffer(length: count, options: .storageModeShared),
              let partialBuffer = device.makeBuffer(length: groupCount * MemoryLayout<Int64>.stride, options: .storageModeShared) else {
            throw BenchmarkError.runtime("Cannot allocate Metal shared buffers")
        }
        self.device = device
        self.queue = queue
        self.pipeline = pipeline
        self.inputBuffer = inputBuffer
        self.validityBuffer = validityBuffer
        self.partialBuffer = partialBuffer
        self.parameters = FusedParameters(
            count: UInt32(count), threshold: threshold, multiplier: multiplier, addend: addend,
            hasNulls: 0
        )
        memset(validityBuffer.contents(), 0, count)
    }

    func copyInput(_ input: [Int32]) {
        _ = input.withUnsafeBytes { bytes in
            memcpy(inputBuffer.contents(), bytes.baseAddress!, bytes.count)
        }
    }

    func execute(input: [Int32], includeCopy: Bool) throws -> TimedResult {
        let started = DispatchTime.now().uptimeNanoseconds
        if includeCopy { copyInput(input) }
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw BenchmarkError.runtime("Cannot create Metal command encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(validityBuffer, offset: 0, index: 1)
        encoder.setBuffer(partialBuffer, offset: 0, index: 2)
        encoder.setBytes(&parameters, length: MemoryLayout<FusedParameters>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: groupCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status == .error {
            throw BenchmarkError.runtime("Metal command failed: \(commandBuffer.error?.localizedDescription ?? "unknown error")")
        }
        let partials = partialBuffer.contents().bindMemory(to: Int64.self, capacity: groupCount)
        var sum: Int64 = 0
        for index in 0..<groupCount { sum += partials[index] }
        let ended = DispatchTime.now().uptimeNanoseconds
        return TimedResult(
            sum: sum,
            wallSeconds: Double(ended - started) / 1_000_000_000,
            deviceSeconds: commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        )
    }
}

@inline(never)
private func cpuReference(_ input: [Int32], threshold: Int32, multiplier: Int32, addend: Int32) -> TimedResult {
    let started = DispatchTime.now().uptimeNanoseconds
    var sum: Int64 = 0
    for value in input where value > threshold {
        sum += Int64(value &* multiplier &+ addend)
    }
    let ended = DispatchTime.now().uptimeNanoseconds
    return TimedResult(sum: sum, wallSeconds: Double(ended - started) / 1_000_000_000, deviceSeconds: 0)
}

private func makeInput(count: Int) -> [Int32] {
    var state: UInt32 = 0x5eed1234
    return (0..<count).map { _ in
        state = state &* 1_664_525 &+ 1_013_904_223
        return Int32(state % 2_001) - 1_000
    }
}

private func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    let middle = sorted.count / 2
    return sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
}

private func run() throws {
    let arguments = try Arguments()
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw BenchmarkError.runtime("No Metal device is available")
    }
    let threshold: Int32 = 100
    let multiplier: Int32 = 3
    let addend: Int32 = 7
    var configurations: [ConfigurationResult] = []

    for rowCount in arguments.sizes {
        print("Benchmarking \(rowCount) rows on \(device.name)")
        let input = makeInput(count: rowCount)
        let runner = try MetalFusedRunner(
            device: device,
            libraryPath: arguments.libraryPath,
            count: rowCount,
            threshold: threshold,
            multiplier: multiplier,
            addend: addend
        )
        runner.copyInput(input)
        let expected = cpuReference(input, threshold: threshold, multiplier: multiplier, addend: addend).sum

        for _ in 0..<arguments.warmups {
            guard try runner.execute(input: input, includeCopy: false).sum == expected else {
                throw BenchmarkError.runtime("GPU result mismatch during warm-up")
            }
            _ = cpuReference(input, threshold: threshold, multiplier: multiplier, addend: addend)
        }

        var cpuTimes: [Double] = []
        var dispatchTimes: [Double] = []
        var copyTimes: [Double] = []
        var deviceTimes: [Double] = []
        for _ in 0..<arguments.runs {
            let cpu = cpuReference(input, threshold: threshold, multiplier: multiplier, addend: addend)
            let dispatch = try runner.execute(input: input, includeCopy: false)
            let copyInclusive = try runner.execute(input: input, includeCopy: true)
            guard cpu.sum == expected, dispatch.sum == expected, copyInclusive.sum == expected else {
                throw BenchmarkError.runtime("CPU/GPU correctness mismatch at \(rowCount) rows")
            }
            cpuTimes.append(cpu.wallSeconds)
            dispatchTimes.append(dispatch.wallSeconds)
            copyTimes.append(copyInclusive.wallSeconds)
            deviceTimes.append(dispatch.deviceSeconds)
        }
        let cpuMedian = median(cpuTimes)
        let dispatchMedian = median(dispatchTimes)
        let copyMedian = median(copyTimes)
        configurations.append(ConfigurationResult(
            rows: rowCount,
            expectedSum: expected,
            cpuSeconds: cpuTimes,
            gpuDispatchSeconds: dispatchTimes,
            gpuCopyInclusiveSeconds: copyTimes,
            gpuDeviceSeconds: deviceTimes,
            cpuMedianSeconds: cpuMedian,
            gpuDispatchMedianSeconds: dispatchMedian,
            gpuCopyInclusiveMedianSeconds: copyMedian,
            dispatchSpeedup: cpuMedian / dispatchMedian,
            copyInclusiveSpeedup: cpuMedian / copyMedian
        ))
    }

    let report = Report(
        createdAt: ISO8601DateFormatter().string(from: Date()),
        deviceName: device.name,
        operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
        threshold: threshold,
        multiplier: multiplier,
        addend: addend,
        warmups: arguments.warmups,
        runs: arguments.runs,
        configurations: configurations
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(report)
    if let outputPath = arguments.outputPath {
        try data.write(to: URL(fileURLWithPath: outputPath), options: .atomic)
        print("Wrote \(outputPath)")
    } else {
        print(String(decoding: data, as: UTF8.self))
    }
}

do {
    try run()
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
