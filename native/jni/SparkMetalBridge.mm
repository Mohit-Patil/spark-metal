#include <jni.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ParquetPageRuns.h"

namespace {

id<MTLDevice> device;
id<MTLCommandQueue> commandQueue;
id<MTLComputePipelineState> fusedPipeline;
id<MTLComputePipelineState> membershipCountUniquePipeline;
id<MTLComputePipelineState> membershipCountMultiplicityPipeline;
id<MTLComputePipelineState> expandValueRunsPipeline;
id<MTLComputePipelineState> scatterSegmentsPipeline;

struct FusedParameters {
    uint32_t count;
    int32_t threshold;
    int32_t multiplier;
    int32_t addend;
    uint32_t hasNulls;
};

struct MembershipCountParameters {
    uint32_t count;
    uint32_t nullMask;
    int32_t keyMin0;
    int32_t keyMin1;
    int32_t keyMin2;
    uint32_t keySpan0;
    uint32_t keySpan1;
    uint32_t keySpan2;
};

struct PreparedMembershipCount3 {
    id<MTLBuffer> keyBuffers[3];
    id<MTLBuffer> nullPlaceholder;
    id<MTLBuffer> partialBuffer;
    NSUInteger partialCapacity = 0;
    int32_t keyMinimums[3];
    uint32_t keySpans[3];
    bool allKeysUnique = true;
    std::atomic<uint64_t> copyFallbacks{0};
};

struct ParquetRowGroup;  // forward declaration: MembershipStream tracks row groups by pointer.

// One in-flight partition worth of asynchronously committed command buffers.
// Spark reuses each partition's off-heap vectors for the next batch, so every
// submit copies its inputs into stream-owned staging buffers; a staging buffer
// returns to the free pool once its command buffer completes. The prepared
// handle may be shared by concurrent tasks, so a stream also owns its own
// partial-count buffers.
struct MembershipStream {
    PreparedMembershipCount3 *prepared = nullptr;
    std::vector<id<MTLCommandBuffer>> commandBuffers;
    std::vector<std::pair<id<MTLBuffer>, id<MTLCommandBuffer>>> pendingStaging;
    // Free buffers bucketed by their exact allocated length (see
    // stagingBucketSize). A single flat free list served first-fit degenerates
    // badly on the Parquet path: each page acquires a ~32KB page buffer and a
    // ~1KB work-item buffer, and first fit hands the 1KB request the recycled
    // 32KB buffer, leaving only 1KB buffers behind, so every page buffer had to
    // be allocated fresh (~17us of the 37us each page cost).
    std::unordered_map<size_t, std::vector<id<MTLBuffer>>> freeStaging;
    std::vector<id<MTLBuffer>> partialBuffers;
    std::vector<NSUInteger> partialGroupCounts;
    // Parquet row groups allocated from this stream that have not yet been
    // released. Registered at parquetRowGroupBegin, unregistered at
    // parquetRowGroupRelease; any left behind when the stream is torn down
    // (Finish/Abort) are deleted there so releasing a row group after its
    // stream is gone can never dereference a dangling stream pointer.
    std::vector<ParquetRowGroup *> rowGroups;
};

struct ExpandParams {
    uint32_t itemCount;
    uint32_t bitWidth;
    uint32_t valueBytesOffset;
    uint32_t outputBase;
};

struct ScatterParams {
    uint32_t segmentCount;
    uint32_t rowBase;
};

// Holds the decoded ids/validity planes for one Parquet row group while its
// pages are being expanded on the GPU. The planes are NOT staging buffers
// returned via pendingStaging after each submit -- the row group outlives
// several command buffers, so they are only recycled by
// parquetRowGroupRelease, once the caller is done reading them.
//
// A row group also owns ONE open command buffer and ONE open compute encoder
// that every one of its pages encodes into. Committing per page cost ~120us of
// CPU per page (5040 pages -> 605ms of the 663ms decode budget on the 33.5M-row
// synthetic benchmark), so the whole row group is encoded once and committed
// once, at parquetRowGroupCount / parquetRowGroupRead / parquetRowGroupRelease.
// Ordering inside the row group is preserved because a compute encoder created
// with -computeCommandEncoder uses MTLDispatchTypeSerial: every dispatch, page
// N's expand and page N's scatter included, completes before the next starts.
// Ordering across row groups is preserved because the one shared command queue
// executes committed command buffers in order.
//
// pendingStaging holds the buffers this row group's dispatches read from. They
// cannot be keyed to a command buffer at push time (the command buffer is not
// committed yet, and acquireStagingBuffer only recycles buffers whose keyed
// command buffer *completed* -- an uncommitted one never completes and would
// wedge the pool), so they are held here and handed to the stream, keyed to the
// row group's command buffer, at commit time.
struct ParquetRowGroup {
    uint32_t rowCount = 0;
    id<MTLBuffer> ids[3];
    id<MTLBuffer> validity[3];
    bool columnHasNulls[3] = {false, false, false};
    MembershipStream *stream = nullptr;
    id<MTLCommandBuffer> commandBuffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    id<MTLCommandBuffer> lastCommandBuffer = nil;
    std::vector<id<MTLBuffer>> pendingStaging;
    uint32_t pagesSinceCommit = 0;
};

// Pages encoded into one command buffer before it is committed and a fresh one
// opened. One commit for the whole row group is not the optimum: a row group of
// the 33.5M-row synthetic benchmark holds ~630 pages, and while its command
// buffer stays open none of its page staging can be recycled (the pool only
// reclaims buffers whose command buffer completed), so every page paid a fresh
// newBufferWithLength. Committing in chunks lets the pool turn over and lets the
// GPU chew on chunk N while the CPU encodes chunk N+1, at the cost of ~20 extra
// commits per row group. Correctness is unaffected: the one shared command queue
// runs committed command buffers in order, so chunk N still lands before N+1 and
// all of them before parquetRowGroupCount's dispatch.
constexpr uint32_t kPagesPerCommit = 32;

// Every staging buffer is allocated at a 4KB-rounded size so buffers of
// near-identical request sizes (the pages of one column chunk, say) share a
// bucket and recycle into each other. 4KB granularity, rather than powers of
// two, keeps the row-group planes -- tens of megabytes each -- from rounding up
// to twice their size.
size_t stagingBucketSize(size_t length) {
    constexpr size_t granularity = 4096;
    return ((length + granularity - 1) / granularity) * granularity;
}

id<MTLBuffer> acquireStagingBuffer(MembershipStream *stream, size_t length) {
    // Reclaim only the completed *prefix* rather than rescanning the whole
    // list. Entries are appended in command-buffer commit order and the one
    // shared queue completes command buffers in that order, so in the common
    // case the first incomplete entry is followed only by incomplete ones and
    // stopping there reclaims everything a full scan would. (Entries pushed by
    // parquetRowGroupRelease can be keyed to an older command buffer than the
    // ones already listed; the only effect is that they wait one more call to
    // be reclaimed.) Reclaiming early is impossible either way: every entry is
    // checked for MTLCommandBufferStatusCompleted before it is freed. This
    // matters because -[MTLCommandBuffer status] is a real property read and
    // the Parquet path calls this twice per page, so a full rescan cost
    // O(pages^2) status reads per row group.
    size_t reclaimed = 0;
    while (reclaimed < stream->pendingStaging.size() &&
           stream->pendingStaging[reclaimed].second.status == MTLCommandBufferStatusCompleted) {
        id<MTLBuffer> buffer = stream->pendingStaging[reclaimed].first;
        stream->freeStaging[buffer.length].push_back(buffer);
        ++reclaimed;
    }
    if (reclaimed > 0) {
        stream->pendingStaging.erase(
            stream->pendingStaging.begin(), stream->pendingStaging.begin() + reclaimed);
    }
    size_t bucket = stagingBucketSize(length);
    auto found = stream->freeStaging.find(bucket);
    if (found != stream->freeStaging.end() && !found->second.empty()) {
        id<MTLBuffer> buffer = found->second.back();
        found->second.pop_back();
        return buffer;
    }
    return [device newBufferWithLength:bucket options:MTLResourceStorageModeShared];
}

// Returns the row group's open compute encoder, opening a command buffer and/or
// an encoder first if this is the first dispatch since the last commit. Returns
// nil only if the command buffer or encoder could not be created.
id<MTLComputeCommandEncoder> rowGroupEncoder(ParquetRowGroup *rowGroup) {
    if (rowGroup->commandBuffer == nil) {
        rowGroup->commandBuffer = [commandQueue commandBuffer];
        if (rowGroup->commandBuffer == nil) return nil;
    }
    if (rowGroup->encoder == nil) {
        rowGroup->encoder = [rowGroup->commandBuffer computeCommandEncoder];
    }
    return rowGroup->encoder;
}

// Closes the row group's encoder and commits its command buffer, registering it
// with the stream and keying every staging buffer it read to it so the pool can
// only recycle them once the GPU is finished. Safe to call when nothing is open.
void commitRowGroup(ParquetRowGroup *rowGroup) {
    if (rowGroup->encoder != nil) {
        [rowGroup->encoder endEncoding];
        rowGroup->encoder = nil;
    }
    id<MTLCommandBuffer> commandBuffer = rowGroup->commandBuffer;
    if (commandBuffer == nil) return;
    rowGroup->commandBuffer = nil;
    [commandBuffer commit];
    rowGroup->lastCommandBuffer = commandBuffer;
    MembershipStream *stream = rowGroup->stream;
    if (stream != nullptr) {
        stream->commandBuffers.push_back(commandBuffer);
        for (id<MTLBuffer> buffer : rowGroup->pendingStaging) {
            stream->pendingStaging.push_back({buffer, commandBuffer});
        }
    }
    // Cleared even without a stream to hand them to (ARC then frees them),
    // so the row group never carries a stale list into its next command buffer.
    rowGroup->pendingStaging.clear();
    rowGroup->pagesSinceCommit = 0;
}

void throwRuntime(JNIEnv *environment, NSString *message) {
    jclass exceptionClass = environment->FindClass("java/lang/RuntimeException");
    environment->ThrowNew(exceptionClass, message.UTF8String);
}

size_t roundedAllocationSize(size_t requestedBytes) {
    size_t pageSize = static_cast<size_t>(getpagesize());
    return ((requestedBytes + pageSize - 1) / pageSize) * pageSize;
}

jlong executeFused(
    JNIEnv *environment,
    id<MTLBuffer> inputBuffer,
    id<MTLBuffer> validityBuffer,
    FusedParameters parameters) {
    constexpr NSUInteger threadsPerGroup = 256;
    NSUInteger groupCount =
        (static_cast<NSUInteger>(parameters.count) + threadsPerGroup - 1) / threadsPerGroup;
    id<MTLBuffer> partialBuffer = [device
        newBufferWithLength:groupCount * sizeof(int64_t)
        options:MTLResourceStorageModeShared];
    if (partialBuffer == nil) {
        throwRuntime(environment, @"Cannot allocate Metal partial-sum buffer");
        return 0;
    }
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:fusedPipeline];
    [encoder setBuffer:inputBuffer offset:0 atIndex:0];
    [encoder setBuffer:validityBuffer offset:0 atIndex:1];
    [encoder setBuffer:partialBuffer offset:0 atIndex:2];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusError) {
        throwRuntime(environment, [NSString stringWithFormat:@"Metal command failed: %@", commandBuffer.error]);
        return 0;
    }
    int64_t *partials = static_cast<int64_t *>(partialBuffer.contents);
    int64_t result = 0;
    for (NSUInteger index = 0; index < groupCount; ++index) {
        result += partials[index];
    }
    return static_cast<jlong>(result);
}

struct MetalBufferSlice {
    id<MTLBuffer> buffer;
    NSUInteger offset;
    bool copied;
};

MetalBufferSlice bufferFromAddress(
    JNIEnv *environment,
    jlong address,
    size_t length,
    NSString *description) {
    if (address == 0 || length == 0) {
        throwRuntime(environment, [NSString stringWithFormat:@"Invalid %@ address or length", description]);
        return {nil, 0, false};
    }
    void *pointer = reinterpret_cast<void *>(static_cast<uintptr_t>(address));
    size_t pageSize = static_cast<size_t>(getpagesize());
    uintptr_t pointerValue = reinterpret_cast<uintptr_t>(pointer);
    uintptr_t mappedAddress = pointerValue - pointerValue % pageSize;
    NSUInteger offset = static_cast<NSUInteger>(pointerValue - mappedAddress);
    size_t mappedLength = roundedAllocationSize(offset + length);
    id<MTLBuffer> buffer = [device
        newBufferWithBytesNoCopy:reinterpret_cast<void *>(mappedAddress)
        length:mappedLength
        options:MTLResourceStorageModeShared
        deallocator:nil];
    bool copied = false;
    if (buffer == nil) {
        buffer = [device newBufferWithBytes:pointer
            length:length
            options:MTLResourceStorageModeShared];
        offset = 0;
        copied = buffer != nil;
    }
    if (buffer == nil) {
        throwRuntime(environment, [NSString stringWithFormat:@"Cannot create %@ Metal buffer", description]);
    }
    return {buffer, offset, copied};
}

}  // namespace

extern "C" JNIEXPORT void JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_initialize(
    JNIEnv *environment,
    jclass,
    jstring libraryPath) {
    @autoreleasepool {
        const char *characters = environment->GetStringUTFChars(libraryPath, nullptr);
        NSString *path = [NSString stringWithUTF8String:characters];
        environment->ReleaseStringUTFChars(libraryPath, characters);

        device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            throwRuntime(environment, @"No Metal device is available");
            return;
        }
        commandQueue = [device newCommandQueue];
        if (commandQueue == nil) {
            throwRuntime(environment, @"Cannot create Metal command queue");
            return;
        }
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
        if (library == nil) {
            throwRuntime(environment, [NSString stringWithFormat:@"Cannot load metallib: %@", error]);
            return;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"fused_filter_project_sum"];
        if (function == nil) {
            throwRuntime(environment, @"Kernel fused_filter_project_sum was not found");
            return;
        }
        fusedPipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (fusedPipeline == nil) {
            throwRuntime(environment, [NSString stringWithFormat:@"Cannot build Metal pipeline: %@", error]);
            return;
        }
        id<MTLFunction> membershipUniqueFunction =
            [library newFunctionWithName:@"fused_membership_count_3_unique"];
        if (membershipUniqueFunction == nil) {
            throwRuntime(environment, @"Kernel fused_membership_count_3_unique was not found");
            return;
        }
        membershipCountUniquePipeline =
            [device newComputePipelineStateWithFunction:membershipUniqueFunction error:&error];
        if (membershipCountUniquePipeline == nil) {
            throwRuntime(environment, [NSString stringWithFormat:@"Cannot build unique membership-count pipeline: %@", error]);
            return;
        }
        id<MTLFunction> membershipMultiplicityFunction =
            [library newFunctionWithName:@"fused_membership_count_3_multiplicity"];
        if (membershipMultiplicityFunction == nil) {
            throwRuntime(environment, @"Kernel fused_membership_count_3_multiplicity was not found");
            return;
        }
        membershipCountMultiplicityPipeline =
            [device newComputePipelineStateWithFunction:membershipMultiplicityFunction error:&error];
        if (membershipCountMultiplicityPipeline == nil) {
            throwRuntime(environment, [NSString stringWithFormat:@"Cannot build multiplicity membership-count pipeline: %@", error]);
            return;
        }
        id<MTLFunction> expandValueRunsFunction = [library newFunctionWithName:@"expand_value_runs"];
        if (expandValueRunsFunction == nil) {
            throwRuntime(environment, @"Kernel expand_value_runs was not found");
            return;
        }
        expandValueRunsPipeline =
            [device newComputePipelineStateWithFunction:expandValueRunsFunction error:&error];
        if (expandValueRunsPipeline == nil) {
            throwRuntime(environment, [NSString stringWithFormat:@"Cannot build expand-value-runs pipeline: %@", error]);
            return;
        }
        id<MTLFunction> scatterSegmentsFunction = [library newFunctionWithName:@"scatter_segments"];
        if (scatterSegmentsFunction == nil) {
            throwRuntime(environment, @"Kernel scatter_segments was not found");
            return;
        }
        scatterSegmentsPipeline =
            [device newComputePipelineStateWithFunction:scatterSegmentsFunction error:&error];
        if (scatterSegmentsPipeline == nil) {
            throwRuntime(environment, [NSString stringWithFormat:@"Cannot build scatter-segments pipeline: %@", error]);
        }
    }
}

extern "C" JNIEXPORT jobject JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_allocateSharedInt32(
    JNIEnv *environment,
    jclass,
    jint count) {
    if (count <= 0) {
        throwRuntime(environment, @"Buffer element count must be positive");
        return nullptr;
    }
    size_t requestedBytes = static_cast<size_t>(count) * sizeof(int32_t);
    size_t allocatedBytes = roundedAllocationSize(requestedBytes);
    void *pointer = nullptr;
    int status = posix_memalign(&pointer, static_cast<size_t>(getpagesize()), allocatedBytes);
    if (status != 0 || pointer == nullptr) {
        throwRuntime(environment, @"Cannot allocate page-aligned shared memory");
        return nullptr;
    }
    memset(pointer, 0, allocatedBytes);
    return environment->NewDirectByteBuffer(pointer, static_cast<jlong>(requestedBytes));
}

extern "C" JNIEXPORT jobject JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_allocateSharedBytes(
    JNIEnv *environment,
    jclass,
    jint count) {
    if (count <= 0) {
        throwRuntime(environment, @"Buffer element count must be positive");
        return nullptr;
    }
    size_t requestedBytes = static_cast<size_t>(count);
    size_t allocatedBytes = roundedAllocationSize(requestedBytes);
    void *pointer = nullptr;
    int status = posix_memalign(&pointer, static_cast<size_t>(getpagesize()), allocatedBytes);
    if (status != 0 || pointer == nullptr) {
        throwRuntime(environment, @"Cannot allocate page-aligned shared memory");
        return nullptr;
    }
    memset(pointer, 0, allocatedBytes);
    return environment->NewDirectByteBuffer(pointer, static_cast<jlong>(requestedBytes));
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_releaseShared(
    JNIEnv *environment,
    jclass,
    jobject buffer) {
    void *pointer = environment->GetDirectBufferAddress(buffer);
    if (pointer == nullptr) {
        throwRuntime(environment, @"Expected a direct buffer allocated by Spark Metal");
        return;
    }
    free(pointer);
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_clearSharedBytes(
    JNIEnv *environment,
    jclass,
    jobject buffer,
    jint count) {
    void *pointer = environment->GetDirectBufferAddress(buffer);
    jlong capacity = environment->GetDirectBufferCapacity(buffer);
    if (pointer == nullptr || count < 0 || capacity < count) {
        throwRuntime(environment, @"Expected a sufficiently large direct byte buffer");
        return;
    }
    memset(pointer, 0, static_cast<size_t>(count));
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_fusedFilterProjectSum(
    JNIEnv *environment,
    jclass,
    jobject input,
    jobject validity,
    jboolean hasNulls,
    jint count,
    jint threshold,
    jint multiplier,
    jint addend) {
    @autoreleasepool {
        if (fusedPipeline == nil || commandQueue == nil) {
            throwRuntime(environment, @"NativeBridge.initialize must be called first");
            return 0;
        }
        void *inputPointer = environment->GetDirectBufferAddress(input);
        void *validityPointer = environment->GetDirectBufferAddress(validity);
        jlong inputCapacity = environment->GetDirectBufferCapacity(input);
        jlong validityCapacity = environment->GetDirectBufferCapacity(validity);
        size_t requestedBytes = static_cast<size_t>(count) * sizeof(int32_t);
        if (inputPointer == nullptr || validityPointer == nullptr || count <= 0 ||
            inputCapacity < static_cast<jlong>(requestedBytes) || validityCapacity < count) {
            throwRuntime(environment, @"Inputs must be sufficiently large direct buffers");
            return 0;
        }

        size_t allocatedBytes = roundedAllocationSize(requestedBytes);
        id<MTLBuffer> inputBuffer = [device
            newBufferWithBytesNoCopy:inputPointer
            length:allocatedBytes
            options:MTLResourceStorageModeShared
            deallocator:nil];
        if (inputBuffer == nil) {
            throwRuntime(environment, @"Cannot wrap the JVM direct buffer as a Metal buffer");
            return 0;
        }
        id<MTLBuffer> validityBuffer = [device
            newBufferWithBytesNoCopy:validityPointer
            length:roundedAllocationSize(static_cast<size_t>(count))
            options:MTLResourceStorageModeShared
            deallocator:nil];
        if (validityBuffer == nil) {
            throwRuntime(environment, @"Cannot wrap the validity buffer as a Metal buffer");
            return 0;
        }
        FusedParameters parameters = {
            static_cast<uint32_t>(count), threshold, multiplier, addend,
            hasNulls == JNI_TRUE ? 1u : 0u
        };
        return executeFused(environment, inputBuffer, validityBuffer, parameters);
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_fusedFilterProjectSumAddress(
    JNIEnv *environment,
    jclass,
    jlong inputAddress,
    jlong nullAddress,
    jboolean hasNulls,
    jint count,
    jint threshold,
    jint multiplier,
    jint addend) {
    @autoreleasepool {
        if (fusedPipeline == nil || commandQueue == nil) {
            throwRuntime(environment, @"NativeBridge.initialize must be called first");
            return 0;
        }
        if (inputAddress == 0 || count <= 0 || (hasNulls == JNI_TRUE && nullAddress == 0)) {
            throwRuntime(environment, @"Invalid Spark off-heap column address");
            return 0;
        }
        void *inputPointer = reinterpret_cast<void *>(
            static_cast<uintptr_t>(inputAddress));
        size_t inputBytes = static_cast<size_t>(count) * sizeof(int32_t);
        size_t pageSize = static_cast<size_t>(getpagesize());
        bool inputIsPageAligned =
            reinterpret_cast<uintptr_t>(inputPointer) % pageSize == 0 && inputBytes % pageSize == 0;
        id<MTLBuffer> inputBuffer = inputIsPageAligned
            ? [device newBufferWithBytesNoCopy:inputPointer
                    length:inputBytes
                    options:MTLResourceStorageModeShared
                    deallocator:nil]
            : [device newBufferWithBytes:inputPointer
                    length:inputBytes
                    options:MTLResourceStorageModeShared];
        if (inputBuffer == nil) {
            throwRuntime(environment, @"Cannot copy the Spark off-heap column to Metal");
            return 0;
        }
        id<MTLBuffer> validityBuffer = nil;
        if (hasNulls == JNI_TRUE) {
            void *nullPointer = reinterpret_cast<void *>(
                static_cast<uintptr_t>(nullAddress));
            size_t nullBytes = static_cast<size_t>(count);
            bool nullsArePageAligned =
                reinterpret_cast<uintptr_t>(nullPointer) % pageSize == 0 && nullBytes % pageSize == 0;
            validityBuffer = nullsArePageAligned
                ? [device newBufferWithBytesNoCopy:nullPointer
                        length:nullBytes
                        options:MTLResourceStorageModeShared
                        deallocator:nil]
                : [device newBufferWithBytes:nullPointer
                        length:nullBytes
                        options:MTLResourceStorageModeShared];
        } else {
            validityBuffer = [device newBufferWithLength:1 options:MTLResourceStorageModeShared];
        }
        if (validityBuffer == nil) {
            throwRuntime(environment, @"Cannot create the Metal null mask buffer");
            return 0;
        }
        FusedParameters parameters = {
            static_cast<uint32_t>(count), threshold, multiplier, addend,
            hasNulls == JNI_TRUE ? 1u : 0u
        };
        return executeFused(environment, inputBuffer, validityBuffer, parameters);
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_prepareMembershipCount3(
    JNIEnv *environment,
    jclass,
    jintArray keys0,
    jintArray keys1,
    jintArray keys2) {
    @autoreleasepool {
        if (membershipCountUniquePipeline == nil ||
            membershipCountMultiplicityPipeline == nil || commandQueue == nil) {
            throwRuntime(environment, @"NativeBridge.initialize must be called first");
            return 0;
        }
        if (keys0 == nullptr || keys1 == nullptr || keys2 == nullptr) {
            throwRuntime(environment, @"Membership preparation requires three key arrays");
            return 0;
        }
        jsize keyCount0 = environment->GetArrayLength(keys0);
        jsize keyCount1 = environment->GetArrayLength(keys1);
        jsize keyCount2 = environment->GetArrayLength(keys2);
        if (keyCount0 <= 0 || keyCount1 <= 0 || keyCount2 <= 0) {
            throwRuntime(environment, @"Membership preparation requires non-empty key arrays");
            return 0;
        }

        jint *keyPointers[3] = {
            environment->GetIntArrayElements(keys0, nullptr),
            environment->GetIntArrayElements(keys1, nullptr),
            environment->GetIntArrayElements(keys2, nullptr)
        };
        if (keyPointers[0] == nullptr || keyPointers[1] == nullptr || keyPointers[2] == nullptr) {
            if (keyPointers[0] != nullptr) {
                environment->ReleaseIntArrayElements(keys0, keyPointers[0], JNI_ABORT);
            }
            if (keyPointers[1] != nullptr) {
                environment->ReleaseIntArrayElements(keys1, keyPointers[1], JNI_ABORT);
            }
            if (keyPointers[2] != nullptr) {
                environment->ReleaseIntArrayElements(keys2, keyPointers[2], JNI_ABORT);
            }
            if (!environment->ExceptionCheck()) {
                throwRuntime(environment, @"Cannot access membership-key arrays");
            }
            return 0;
        }
        jsize keyCounts[3] = {keyCount0, keyCount1, keyCount2};
        auto *prepared = new PreparedMembershipCount3();
        bool allKeysUnique = true;
        std::vector<uint8_t> uniqueDenseMaps[3];
        for (NSUInteger index = 0; index < 3; ++index) {
            int32_t minimum = std::numeric_limits<int32_t>::max();
            int32_t maximum = std::numeric_limits<int32_t>::min();
            for (jsize keyIndex = 0; keyIndex < keyCounts[index]; ++keyIndex) {
                minimum = std::min(minimum, static_cast<int32_t>(keyPointers[index][keyIndex]));
                maximum = std::max(maximum, static_cast<int32_t>(keyPointers[index][keyIndex]));
            }
            int64_t span64 = static_cast<int64_t>(maximum) - static_cast<int64_t>(minimum) + 1;
            if (span64 <= 0 || span64 > 16 * 1024 * 1024) {
                environment->ReleaseIntArrayElements(keys0, keyPointers[0], JNI_ABORT);
                environment->ReleaseIntArrayElements(keys1, keyPointers[1], JNI_ABORT);
                environment->ReleaseIntArrayElements(keys2, keyPointers[2], JNI_ABORT);
                delete prepared;
                throwRuntime(environment, @"Membership-key domain is too large for a dense map");
                return 0;
            }
            uniqueDenseMaps[index].resize(static_cast<size_t>(span64), 0);
            for (jsize keyIndex = 0; keyIndex < keyCounts[index]; ++keyIndex) {
                uint8_t &present = uniqueDenseMaps[index][
                    static_cast<size_t>(keyPointers[index][keyIndex] - minimum)];
                if (present != 0) allKeysUnique = false;
                present = 1;
            }
            prepared->keyMinimums[index] = minimum;
            prepared->keySpans[index] = static_cast<uint32_t>(span64);
        }
        prepared->allKeysUnique = allKeysUnique;
        if (allKeysUnique) {
            for (NSUInteger index = 0; index < 3; ++index) {
                prepared->keyBuffers[index] = [device newBufferWithBytes:uniqueDenseMaps[index].data()
                    length:uniqueDenseMaps[index].size() * sizeof(uint8_t)
                    options:MTLResourceStorageModeShared];
            }
        } else {
            for (NSUInteger index = 0; index < 3; ++index) {
                std::vector<uint32_t> dense(static_cast<size_t>(prepared->keySpans[index]), 0);
                for (jsize keyIndex = 0; keyIndex < keyCounts[index]; ++keyIndex) {
                    dense[static_cast<size_t>(
                        keyPointers[index][keyIndex] - prepared->keyMinimums[index])] += 1;
                }
                prepared->keyBuffers[index] = [device newBufferWithBytes:dense.data()
                    length:dense.size() * sizeof(uint32_t)
                    options:MTLResourceStorageModeShared];
            }
        }
        environment->ReleaseIntArrayElements(keys0, keyPointers[0], JNI_ABORT);
        environment->ReleaseIntArrayElements(keys1, keyPointers[1], JNI_ABORT);
        environment->ReleaseIntArrayElements(keys2, keyPointers[2], JNI_ABORT);
        if (prepared->keyBuffers[0] == nil || prepared->keyBuffers[1] == nil ||
            prepared->keyBuffers[2] == nil) {
            delete prepared;
            throwRuntime(environment, @"Cannot create Metal membership-key buffers");
            return 0;
        }
        prepared->nullPlaceholder =
            [device newBufferWithLength:1 options:MTLResourceStorageModeShared];
        if (prepared->nullPlaceholder == nil) {
            delete prepared;
            throwRuntime(environment, @"Cannot create Metal null placeholder");
            return 0;
        }
        return static_cast<jlong>(reinterpret_cast<uintptr_t>(prepared));
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_releaseMembershipCount3(
    JNIEnv *environment,
    jclass,
    jlong preparedHandle) {
    @autoreleasepool {
        if (preparedHandle == 0) {
            throwRuntime(environment, @"Invalid prepared membership handle");
            return;
        }
        auto *prepared = reinterpret_cast<PreparedMembershipCount3 *>(
            static_cast<uintptr_t>(preparedHandle));
        delete prepared;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_membershipCount3CopyFallbacks(
    JNIEnv *environment,
    jclass,
    jlong preparedHandle) {
    if (preparedHandle == 0) {
        throwRuntime(environment, @"Invalid prepared membership handle");
        return 0;
    }
    auto *prepared = reinterpret_cast<PreparedMembershipCount3 *>(
        static_cast<uintptr_t>(preparedHandle));
    return static_cast<jlong>(prepared->copyFallbacks);
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_membershipCount3PreparedAddress(
    JNIEnv *environment,
    jclass,
    jlong input0Address,
    jlong null0Address,
    jboolean hasNull0,
    jlong input1Address,
    jlong null1Address,
    jboolean hasNull1,
    jlong input2Address,
    jlong null2Address,
    jboolean hasNull2,
    jint count,
    jlong preparedHandle) {
    @autoreleasepool {
        if (membershipCountUniquePipeline == nil ||
            membershipCountMultiplicityPipeline == nil || commandQueue == nil) {
            throwRuntime(environment, @"NativeBridge.initialize must be called first");
            return 0;
        }
        if (count <= 0 || preparedHandle == 0) {
            throwRuntime(environment, @"Membership count requires rows and a prepared handle");
            return 0;
        }
        auto *prepared = reinterpret_cast<PreparedMembershipCount3 *>(
            static_cast<uintptr_t>(preparedHandle));
        size_t inputBytes = static_cast<size_t>(count) * sizeof(int32_t);
        MetalBufferSlice inputs[3] = {
            bufferFromAddress(environment, input0Address, inputBytes, @"input 0"),
            bufferFromAddress(environment, input1Address, inputBytes, @"input 1"),
            bufferFromAddress(environment, input2Address, inputBytes, @"input 2")
        };
        if (environment->ExceptionCheck()) return 0;
        jlong nullAddresses[3] = {null0Address, null1Address, null2Address};
        jboolean hasNulls[3] = {hasNull0, hasNull1, hasNull2};
        MetalBufferSlice nullBuffers[3];
        uint32_t nullMask = 0;
        for (NSUInteger index = 0; index < 3; ++index) {
            if (hasNulls[index] == JNI_TRUE) {
                nullMask |= 1u << index;
                nullBuffers[index] = bufferFromAddress(
                    environment, nullAddresses[index], static_cast<size_t>(count), @"null mask");
            } else {
                nullBuffers[index] = {prepared->nullPlaceholder, 0, false};
            }
            if (nullBuffers[index].buffer == nil || environment->ExceptionCheck()) return 0;
        }
        for (NSUInteger index = 0; index < 3; ++index) {
            if (inputs[index].copied) prepared->copyFallbacks += 1;
            if (nullBuffers[index].copied) prepared->copyFallbacks += 1;
        }

        constexpr NSUInteger threadsPerGroup = 256;
        NSUInteger groupCount =
            (static_cast<NSUInteger>(count) + threadsPerGroup - 1) / threadsPerGroup;
        if (prepared->partialCapacity < groupCount) {
            prepared->partialBuffer = [device
                newBufferWithLength:groupCount * sizeof(int64_t)
                options:MTLResourceStorageModeShared];
            prepared->partialCapacity = prepared->partialBuffer == nil ? 0 : groupCount;
        }
        if (prepared->partialBuffer == nil) {
            throwRuntime(environment, @"Cannot allocate Metal partial-count buffer");
            return 0;
        }
        MembershipCountParameters parameters = {
            static_cast<uint32_t>(count), nullMask,
            prepared->keyMinimums[0], prepared->keyMinimums[1], prepared->keyMinimums[2],
            prepared->keySpans[0], prepared->keySpans[1], prepared->keySpans[2]
        };
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:prepared->allKeysUnique
            ? membershipCountUniquePipeline
            : membershipCountMultiplicityPipeline];
        for (NSUInteger index = 0; index < 3; ++index) {
            [encoder setBuffer:inputs[index].buffer offset:inputs[index].offset atIndex:index];
            [encoder setBuffer:nullBuffers[index].buffer
                    offset:nullBuffers[index].offset
                    atIndex:index + 3];
            [encoder setBuffer:prepared->keyBuffers[index] offset:0 atIndex:index + 6];
        }
        [encoder setBuffer:prepared->partialBuffer offset:0 atIndex:9];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:10];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status == MTLCommandBufferStatusError) {
            throwRuntime(environment, [NSString stringWithFormat:@"Metal command failed: %@", commandBuffer.error]);
            return 0;
        }
        int64_t result = 0;
        if (prepared->allKeysUnique) {
            uint32_t *partials = static_cast<uint32_t *>(prepared->partialBuffer.contents);
            for (NSUInteger index = 0; index < groupCount; ++index) result += partials[index];
        } else {
            int64_t *partials = static_cast<int64_t *>(prepared->partialBuffer.contents);
            for (NSUInteger index = 0; index < groupCount; ++index) result += partials[index];
        }
        return static_cast<jlong>(result);
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_membershipCount3PreparedBatches(
    JNIEnv *environment,
    jclass,
    jlongArray input0Addresses,
    jlongArray null0Addresses,
    jbooleanArray hasNull0,
    jlongArray input1Addresses,
    jlongArray null1Addresses,
    jbooleanArray hasNull1,
    jlongArray input2Addresses,
    jlongArray null2Addresses,
    jbooleanArray hasNull2,
    jintArray counts,
    jlong preparedHandle) {
    @autoreleasepool {
        if (membershipCountUniquePipeline == nil ||
            membershipCountMultiplicityPipeline == nil || commandQueue == nil) {
            throwRuntime(environment, @"NativeBridge.initialize must be called first");
            return 0;
        }
        if (preparedHandle == 0 || counts == nullptr) {
            throwRuntime(environment, @"Batched membership count requires a prepared handle");
            return 0;
        }
        jsize batchCount = environment->GetArrayLength(counts);
        if (batchCount <= 0 ||
            environment->GetArrayLength(input0Addresses) != batchCount ||
            environment->GetArrayLength(null0Addresses) != batchCount ||
            environment->GetArrayLength(hasNull0) != batchCount ||
            environment->GetArrayLength(input1Addresses) != batchCount ||
            environment->GetArrayLength(null1Addresses) != batchCount ||
            environment->GetArrayLength(hasNull1) != batchCount ||
            environment->GetArrayLength(input2Addresses) != batchCount ||
            environment->GetArrayLength(null2Addresses) != batchCount ||
            environment->GetArrayLength(hasNull2) != batchCount) {
            throwRuntime(environment, @"Batched membership arrays must have equal, non-zero lengths");
            return 0;
        }

        std::vector<jint> rowCounts(static_cast<size_t>(batchCount));
        std::vector<jlong> inputAddresses[3];
        std::vector<jlong> nullAddresses[3];
        std::vector<jboolean> nullFlags[3];
        for (NSUInteger column = 0; column < 3; ++column) {
            inputAddresses[column].resize(static_cast<size_t>(batchCount));
            nullAddresses[column].resize(static_cast<size_t>(batchCount));
            nullFlags[column].resize(static_cast<size_t>(batchCount));
        }
        environment->GetIntArrayRegion(counts, 0, batchCount, rowCounts.data());
        jlongArray inputArrays[3] = {input0Addresses, input1Addresses, input2Addresses};
        jlongArray nullArrays[3] = {null0Addresses, null1Addresses, null2Addresses};
        jbooleanArray flagArrays[3] = {hasNull0, hasNull1, hasNull2};
        for (NSUInteger column = 0; column < 3; ++column) {
            environment->GetLongArrayRegion(
                inputArrays[column], 0, batchCount, inputAddresses[column].data());
            environment->GetLongArrayRegion(
                nullArrays[column], 0, batchCount, nullAddresses[column].data());
            environment->GetBooleanArrayRegion(
                flagArrays[column], 0, batchCount, nullFlags[column].data());
        }
        if (environment->ExceptionCheck()) return 0;

        constexpr NSUInteger threadsPerGroup = 256;
        std::vector<NSUInteger> groupCounts(static_cast<size_t>(batchCount));
        NSUInteger totalGroups = 0;
        for (jsize batch = 0; batch < batchCount; ++batch) {
            if (rowCounts[batch] <= 0) {
                throwRuntime(environment, @"Batched membership count requires non-empty batches");
                return 0;
            }
            NSUInteger groups =
                (static_cast<NSUInteger>(rowCounts[batch]) + threadsPerGroup - 1) / threadsPerGroup;
            groupCounts[static_cast<size_t>(batch)] = groups;
            totalGroups += groups;
        }

        auto *prepared = reinterpret_cast<PreparedMembershipCount3 *>(
            static_cast<uintptr_t>(preparedHandle));
        if (prepared->partialCapacity < totalGroups) {
            prepared->partialBuffer = [device
                newBufferWithLength:totalGroups * sizeof(int64_t)
                options:MTLResourceStorageModeShared];
            prepared->partialCapacity = prepared->partialBuffer == nil ? 0 : totalGroups;
        }
        if (prepared->partialBuffer == nil) {
            throwRuntime(environment, @"Cannot allocate batched Metal partial-count buffer");
            return 0;
        }

        std::vector<MetalBufferSlice> inputs[3];
        std::vector<MetalBufferSlice> nullBuffers[3];
        for (NSUInteger column = 0; column < 3; ++column) {
            inputs[column].reserve(static_cast<size_t>(batchCount));
            nullBuffers[column].reserve(static_cast<size_t>(batchCount));
        }
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:prepared->allKeysUnique
            ? membershipCountUniquePipeline
            : membershipCountMultiplicityPipeline];
        NSUInteger partialOffset = 0;
        NSUInteger partialElementBytes = prepared->allKeysUnique
            ? sizeof(uint32_t) : sizeof(int64_t);
        for (jsize batch = 0; batch < batchCount; ++batch) {
            size_t inputBytes = static_cast<size_t>(rowCounts[batch]) * sizeof(int32_t);
            uint32_t nullMask = 0;
            for (NSUInteger column = 0; column < 3; ++column) {
                MetalBufferSlice input = bufferFromAddress(
                    environment, inputAddresses[column][batch], inputBytes, @"batched input");
                if (input.buffer == nil || environment->ExceptionCheck()) return 0;
                inputs[column].push_back(input);
                MetalBufferSlice nullBuffer;
                if (nullFlags[column][batch] == JNI_TRUE) {
                    nullMask |= 1u << column;
                    nullBuffer = bufferFromAddress(
                        environment, nullAddresses[column][batch],
                        static_cast<size_t>(rowCounts[batch]), @"batched null mask");
                } else {
                    nullBuffer = {prepared->nullPlaceholder, 0, false};
                }
                if (nullBuffer.buffer == nil || environment->ExceptionCheck()) return 0;
                nullBuffers[column].push_back(nullBuffer);
                if (input.copied) prepared->copyFallbacks += 1;
                if (nullBuffer.copied) prepared->copyFallbacks += 1;
                [encoder setBuffer:input.buffer offset:input.offset atIndex:column];
                [encoder setBuffer:nullBuffer.buffer offset:nullBuffer.offset atIndex:column + 3];
                [encoder setBuffer:prepared->keyBuffers[column] offset:0 atIndex:column + 6];
            }
            MembershipCountParameters parameters = {
                static_cast<uint32_t>(rowCounts[batch]), nullMask,
                prepared->keyMinimums[0], prepared->keyMinimums[1], prepared->keyMinimums[2],
                prepared->keySpans[0], prepared->keySpans[1], prepared->keySpans[2]
            };
            [encoder setBuffer:prepared->partialBuffer
                    offset:partialOffset * partialElementBytes atIndex:9];
            [encoder setBytes:&parameters length:sizeof(parameters) atIndex:10];
            [encoder dispatchThreadgroups:MTLSizeMake(groupCounts[batch], 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
            partialOffset += groupCounts[batch];
        }
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status == MTLCommandBufferStatusError) {
            throwRuntime(environment, [NSString stringWithFormat:@"Batched Metal command failed: %@", commandBuffer.error]);
            return 0;
        }
        int64_t result = 0;
        if (prepared->allKeysUnique) {
            uint32_t *partials = static_cast<uint32_t *>(prepared->partialBuffer.contents);
            for (NSUInteger index = 0; index < totalGroups; ++index) result += partials[index];
        } else {
            int64_t *partials = static_cast<int64_t *>(prepared->partialBuffer.contents);
            for (NSUInteger index = 0; index < totalGroups; ++index) result += partials[index];
        }
        return static_cast<jlong>(result);
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_membershipCount3StreamBegin(
    JNIEnv *environment,
    jclass,
    jlong preparedHandle) {
    if (membershipCountUniquePipeline == nil ||
        membershipCountMultiplicityPipeline == nil || commandQueue == nil) {
        throwRuntime(environment, @"NativeBridge.initialize must be called first");
        return 0;
    }
    if (preparedHandle == 0) {
        throwRuntime(environment, @"Streamed membership count requires a prepared handle");
        return 0;
    }
    auto *stream = new MembershipStream();
    stream->prepared = reinterpret_cast<PreparedMembershipCount3 *>(
        static_cast<uintptr_t>(preparedHandle));
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(stream));
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_membershipCount3StreamSubmit(
    JNIEnv *environment,
    jclass,
    jlong streamHandle,
    jlong input0Address,
    jlong null0Address,
    jboolean hasNull0,
    jbyteArray dictPresence0,
    jintArray dictMultiplicity0,
    jlong input1Address,
    jlong null1Address,
    jboolean hasNull1,
    jbyteArray dictPresence1,
    jintArray dictMultiplicity1,
    jlong input2Address,
    jlong null2Address,
    jboolean hasNull2,
    jbyteArray dictPresence2,
    jintArray dictMultiplicity2,
    jint count) {
    @autoreleasepool {
        if (streamHandle == 0 || count <= 0) {
            throwRuntime(environment, @"Streamed membership submit requires a stream and rows");
            return;
        }
        auto *stream = reinterpret_cast<MembershipStream *>(
            static_cast<uintptr_t>(streamHandle));
        PreparedMembershipCount3 *prepared = stream->prepared;
        size_t inputBytes = static_cast<size_t>(count) * sizeof(int32_t);
        jlong inputAddresses[3] = {input0Address, input1Address, input2Address};
        jlong nullAddresses[3] = {null0Address, null1Address, null2Address};
        jboolean hasNulls[3] = {hasNull0, hasNull1, hasNull2};
        jbyteArray presenceArrays[3] = {dictPresence0, dictPresence1, dictPresence2};
        jintArray multiplicityArrays[3] = {dictMultiplicity0, dictMultiplicity1, dictMultiplicity2};

        // Spark reuses the off-heap vectors for the following batch, so the
        // inputs must be copied out before this call returns. Columns carrying
        // dictionary ids test membership against a per-dictionary table
        // (index space 0..maxId) instead of the prepared key-value map; the
        // kernels are unchanged because both are dense maps.
        std::vector<id<MTLBuffer>> usedStaging;
        id<MTLBuffer> inputBuffers[3];
        id<MTLBuffer> nullBuffers[3];
        id<MTLBuffer> keyBuffers[3];
        int32_t keyMinimums[3];
        uint32_t keySpans[3];
        uint32_t nullMask = 0;
        for (NSUInteger index = 0; index < 3; ++index) {
            if (inputAddresses[index] == 0) {
                throwRuntime(environment, @"Invalid streamed input address");
                return;
            }
            id<MTLBuffer> input = acquireStagingBuffer(stream, inputBytes);
            if (input == nil) {
                throwRuntime(environment, @"Cannot allocate streamed input staging buffer");
                return;
            }
            memcpy(input.contents,
                reinterpret_cast<void *>(static_cast<uintptr_t>(inputAddresses[index])),
                inputBytes);
            usedStaging.push_back(input);
            inputBuffers[index] = input;

            if (hasNulls[index] == JNI_TRUE) {
                if (nullAddresses[index] == 0) {
                    throwRuntime(environment, @"Invalid streamed null-mask address");
                    return;
                }
                nullMask |= 1u << index;
                id<MTLBuffer> nulls = acquireStagingBuffer(stream, static_cast<size_t>(count));
                if (nulls == nil) {
                    throwRuntime(environment, @"Cannot allocate streamed null-mask staging buffer");
                    return;
                }
                memcpy(nulls.contents,
                    reinterpret_cast<void *>(static_cast<uintptr_t>(nullAddresses[index])),
                    static_cast<size_t>(count));
                usedStaging.push_back(nulls);
                nullBuffers[index] = nulls;
            } else {
                nullBuffers[index] = prepared->nullPlaceholder;
            }

            if (presenceArrays[index] != nullptr || multiplicityArrays[index] != nullptr) {
                bool presence = presenceArrays[index] != nullptr;
                if (presence != prepared->allKeysUnique) {
                    throwRuntime(environment, @"Dictionary table type does not match the prepared kernel");
                    return;
                }
                jsize length = presence
                    ? environment->GetArrayLength(presenceArrays[index])
                    : environment->GetArrayLength(multiplicityArrays[index]);
                if (length <= 0) {
                    throwRuntime(environment, @"Dictionary membership table must not be empty");
                    return;
                }
                size_t elementBytes = presence ? sizeof(uint8_t) : sizeof(uint32_t);
                id<MTLBuffer> table = acquireStagingBuffer(
                    stream, static_cast<size_t>(length) * elementBytes);
                if (table == nil) {
                    throwRuntime(environment, @"Cannot allocate dictionary membership table");
                    return;
                }
                if (presence) {
                    environment->GetByteArrayRegion(
                        presenceArrays[index], 0, length,
                        static_cast<jbyte *>(table.contents));
                } else {
                    environment->GetIntArrayRegion(
                        multiplicityArrays[index], 0, length,
                        static_cast<jint *>(table.contents));
                }
                if (environment->ExceptionCheck()) return;
                usedStaging.push_back(table);
                keyBuffers[index] = table;
                keyMinimums[index] = 0;
                keySpans[index] = static_cast<uint32_t>(length);
            } else {
                keyBuffers[index] = prepared->keyBuffers[index];
                keyMinimums[index] = prepared->keyMinimums[index];
                keySpans[index] = prepared->keySpans[index];
            }
        }

        constexpr NSUInteger threadsPerGroup = 256;
        NSUInteger groupCount =
            (static_cast<NSUInteger>(count) + threadsPerGroup - 1) / threadsPerGroup;
        NSUInteger partialElementBytes = prepared->allKeysUnique
            ? sizeof(uint32_t) : sizeof(int64_t);
        id<MTLBuffer> partialBuffer = [device
            newBufferWithLength:groupCount * partialElementBytes
            options:MTLResourceStorageModeShared];
        if (partialBuffer == nil) {
            throwRuntime(environment, @"Cannot allocate streamed Metal partial-count buffer");
            return;
        }
        MembershipCountParameters parameters = {
            static_cast<uint32_t>(count), nullMask,
            keyMinimums[0], keyMinimums[1], keyMinimums[2],
            keySpans[0], keySpans[1], keySpans[2]
        };
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:prepared->allKeysUnique
            ? membershipCountUniquePipeline
            : membershipCountMultiplicityPipeline];
        for (NSUInteger index = 0; index < 3; ++index) {
            [encoder setBuffer:inputBuffers[index] offset:0 atIndex:index];
            [encoder setBuffer:nullBuffers[index] offset:0 atIndex:index + 3];
            [encoder setBuffer:keyBuffers[index] offset:0 atIndex:index + 6];
        }
        [encoder setBuffer:partialBuffer offset:0 atIndex:9];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:10];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];

        stream->commandBuffers.push_back(commandBuffer);
        for (id<MTLBuffer> buffer : usedStaging) {
            stream->pendingStaging.push_back({buffer, commandBuffer});
        }
        stream->partialBuffers.push_back(partialBuffer);
        stream->partialGroupCounts.push_back(groupCount);
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_membershipCount3StreamFinish(
    JNIEnv *environment,
    jclass,
    jlong streamHandle) {
    @autoreleasepool {
        if (streamHandle == 0) {
            throwRuntime(environment, @"Invalid membership stream handle");
            return 0;
        }
        auto *stream = reinterpret_cast<MembershipStream *>(
            static_cast<uintptr_t>(streamHandle));
        // Any row group the caller never Counted or Released still owns an open
        // command buffer and encoder. Close and commit them before the wait so
        // nothing is left half-encoded when the row groups are deleted below --
        // and so the wait covers them.
        for (ParquetRowGroup *rowGroup : stream->rowGroups) {
            commitRowGroup(rowGroup);
        }
        NSError *failure = nil;
        for (id<MTLCommandBuffer> commandBuffer : stream->commandBuffers) {
            [commandBuffer waitUntilCompleted];
            if (commandBuffer.status == MTLCommandBufferStatusError && failure == nil) {
                failure = commandBuffer.error;
            }
        }
        int64_t result = 0;
        if (failure == nil) {
            bool allKeysUnique = stream->prepared->allKeysUnique;
            for (size_t batch = 0; batch < stream->partialBuffers.size(); ++batch) {
                NSUInteger groups = stream->partialGroupCounts[batch];
                if (allKeysUnique) {
                    uint32_t *partials =
                        static_cast<uint32_t *>(stream->partialBuffers[batch].contents);
                    for (NSUInteger index = 0; index < groups; ++index) result += partials[index];
                } else {
                    int64_t *partials =
                        static_cast<int64_t *>(stream->partialBuffers[batch].contents);
                    for (NSUInteger index = 0; index < groups; ++index) result += partials[index];
                }
            }
        }
        // Reclaim any Parquet row groups the caller never released. Their
        // planes are staging buffers owned by the row group itself (never
        // pendingStaging), and the stream -- along with its freeStaging pool
        // -- is being destroyed right below, so there's nothing to return
        // them to; just delete the row groups (their id<MTLBuffer> fields
        // release themselves via ARC).
        for (ParquetRowGroup *rowGroup : stream->rowGroups) {
            delete rowGroup;
        }
        delete stream;
        if (failure != nil) {
            throwRuntime(environment, [NSString stringWithFormat:@"Streamed Metal command failed: %@", failure]);
            return 0;
        }
        return static_cast<jlong>(result);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_membershipCount3StreamAbort(
    JNIEnv *environment,
    jclass,
    jlong streamHandle) {
    @autoreleasepool {
        if (streamHandle == 0) return;
        auto *stream = reinterpret_cast<MembershipStream *>(
            static_cast<uintptr_t>(streamHandle));
        // See membershipCount3StreamFinish: close and commit any row group left
        // mid-encode so nothing is deleted with an open encoder.
        for (ParquetRowGroup *rowGroup : stream->rowGroups) {
            commitRowGroup(rowGroup);
        }
        for (id<MTLCommandBuffer> commandBuffer : stream->commandBuffers) {
            [commandBuffer waitUntilCompleted];
        }
        // See membershipCount3StreamFinish: reclaim any row groups the
        // caller never released before the stream (and its freeStaging
        // pool) goes away.
        for (ParquetRowGroup *rowGroup : stream->rowGroups) {
            delete rowGroup;
        }
        delete stream;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_parquetRowGroupBegin(
    JNIEnv *environment,
    jclass,
    jlong streamHandle,
    jint rowCount) {
    @autoreleasepool {
        if (commandQueue == nil) {
            throwRuntime(environment, @"NativeBridge.initialize must be called first");
            return 0;
        }
        if (streamHandle == 0 || rowCount <= 0) {
            throwRuntime(environment, @"Row-group begin requires a stream and a positive row count");
            return 0;
        }
        auto *stream = reinterpret_cast<MembershipStream *>(
            static_cast<uintptr_t>(streamHandle));
        size_t idsBytes = static_cast<size_t>(rowCount) * sizeof(int32_t);
        size_t validityBytes = static_cast<size_t>(rowCount);

        auto *rowGroup = new ParquetRowGroup();
        rowGroup->rowCount = static_cast<uint32_t>(rowCount);
        rowGroup->stream = stream;

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        bool allocationFailed = false;
        for (int column = 0; column < 3; ++column) {
            id<MTLBuffer> ids = acquireStagingBuffer(stream, idsBytes);
            id<MTLBuffer> validity = acquireStagingBuffer(stream, validityBytes);
            if (ids == nil || validity == nil) {
                allocationFailed = true;
                break;
            }
            rowGroup->ids[column] = ids;
            rowGroup->validity[column] = validity;
            rowGroup->columnHasNulls[column] = false;
            // Zero both planes: validity so unwritten rows read as "not
            // null" (correct for allValid pages that never touch it), and
            // ids so a null row -- whose id is otherwise never written --
            // never carries a recycled staging-buffer id. Task 4 indexes a
            // dictionary presence table by id, so a stale id would be an
            // out-of-bounds read there.
            [blit fillBuffer:validity range:NSMakeRange(0, validityBytes) value:0];
            [blit fillBuffer:ids range:NSMakeRange(0, idsBytes) value:0];
        }
        [blit endEncoding];
        if (allocationFailed) {
            // Nothing was committed, so the command buffer is simply dropped
            // (ARC releases it) along with whatever planes were acquired.
            delete rowGroup;
            throwRuntime(environment, @"Cannot allocate Parquet row-group planes");
            return 0;
        }
        // NOT committed here: this command buffer stays open so every page of
        // this row group encodes into it. Encoders within one command buffer
        // execute in the order they were created, so the zero-fill blit above
        // is guaranteed to precede the compute encoder's dispatches. It is
        // deliberately not pushed onto stream->commandBuffers yet either --
        // Finish/Abort/Read call waitUntilCompleted on everything in there, and
        // waiting on an uncommitted command buffer would hang.
        rowGroup->commandBuffer = commandBuffer;
        stream->rowGroups.push_back(rowGroup);
        return static_cast<jlong>(reinterpret_cast<uintptr_t>(rowGroup));
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_parquetDecodePage(
    JNIEnv *environment,
    jclass,
    jlong streamHandle,
    jlong rowGroupHandle,
    jint column,
    jbyteArray pageBytes,
    jint pageLength,
    jint valueCount,
    jint rowOffset,
    jboolean hasDefLevels) {
    @autoreleasepool {
        if (expandValueRunsPipeline == nil || scatterSegmentsPipeline == nil || commandQueue == nil) {
            throwRuntime(environment, @"NativeBridge.initialize must be called first");
            return;
        }
        if (streamHandle == 0 || rowGroupHandle == 0 || column < 0 || column >= 3 ||
            pageBytes == nullptr || pageLength <= 0 || valueCount < 0 || rowOffset < 0) {
            throwRuntime(environment, @"Invalid Parquet page decode arguments");
            return;
        }
        auto *stream = reinterpret_cast<MembershipStream *>(
            static_cast<uintptr_t>(streamHandle));
        auto *rowGroup = reinterpret_cast<ParquetRowGroup *>(
            static_cast<uintptr_t>(rowGroupHandle));
        if (rowGroup->stream != stream) {
            throwRuntime(environment, @"Row group does not belong to the given stream");
            return;
        }

        // Guard against writing past the ids/validity planes: a corrupt
        // file, an unexpected writer, or a rowOffset accounting slip on the
        // JVM side could otherwise turn into an unbounded GPU device-memory
        // write in expand_value_runs/scatter_segments. int64 avoids overflow
        // from the two int32 operands.
        if (static_cast<int64_t>(rowOffset) + static_cast<int64_t>(valueCount) >
            static_cast<int64_t>(rowGroup->rowCount)) {
            throwRuntime(environment, @"Parquet page rowOffset + valueCount exceeds the row-group row count");
            return;
        }

        // pageLength + 4: expand_value_runs's 4-byte bit window may read up to
        // 3 bytes past the payload; the tail is zeroed so that read is inert.
        id<MTLBuffer> pageStaging = acquireStagingBuffer(stream, static_cast<size_t>(pageLength) + 4);
        if (pageStaging == nil) {
            throwRuntime(environment, @"Cannot allocate Parquet page staging buffer");
            return;
        }
        uint8_t *pageContents = static_cast<uint8_t *>(pageStaging.contents);
        environment->GetByteArrayRegion(
            pageBytes, 0, pageLength, reinterpret_cast<jbyte *>(pageContents));
        if (environment->ExceptionCheck()) return;
        memset(pageContents + pageLength, 0, 4);

        // One PageRuns per decoding thread, reused across pages: see
        // parseDataPageV1, which resets it in place and so keeps its capacity.
        static thread_local sparkmetal::PageRuns runs;
        bool parsed = sparkmetal::parseDataPageV1(
            pageContents, static_cast<size_t>(pageLength), static_cast<uint32_t>(valueCount),
            hasDefLevels == JNI_TRUE, runs);
        if (!parsed) {
            throwRuntime(environment, @"Unsupported Parquet page");
            return;
        }
        if (runs.allValid && runs.items.empty()) {
            // Zero-value page (valueCount == 0); nothing to expand.
            return;
        }
        if (!runs.allValid && runs.segments.empty()) {
            // Defensive: a non-empty page always yields at least one segment
            // once its definition levels are folded, but guard against
            // dispatching an empty scatter regardless.
            return;
        }

        // items can be empty here only when the whole page is null
        // (nonNullCount == 0): scatter_segments still must run so the
        // validity plane records those rows as null.
        id<MTLBuffer> itemsBuffer = nil;
        if (!runs.items.empty()) {
            itemsBuffer = acquireStagingBuffer(
                stream, runs.items.size() * sizeof(sparkmetal::ValueWorkItem));
            if (itemsBuffer == nil) {
                throwRuntime(environment, @"Cannot allocate Parquet work-item staging buffer");
                return;
            }
            memcpy(itemsBuffer.contents, runs.items.data(),
                runs.items.size() * sizeof(sparkmetal::ValueWorkItem));
        }

        std::vector<id<MTLBuffer>> usedStaging = {pageStaging};
        if (itemsBuffer != nil) usedStaging.push_back(itemsBuffer);

        // Encode into the row group's shared, still-open command buffer rather
        // than creating and committing one per page.
        id<MTLComputeCommandEncoder> encoder = rowGroupEncoder(rowGroup);
        if (encoder == nil) {
            throwRuntime(environment, @"Cannot open a Parquet row-group compute encoder");
            return;
        }

        if (runs.allValid) {
            // allValid implies nonNullCount == valueCount > 0, so items is
            // guaranteed non-empty by the guard above.
            ExpandParams params = {
                static_cast<uint32_t>(runs.items.size()),
                runs.bitWidth,
                runs.valueBytesOffset,
                static_cast<uint32_t>(rowOffset)
            };
            [encoder setComputePipelineState:expandValueRunsPipeline];
            [encoder setBuffer:pageStaging offset:0 atIndex:0];
            [encoder setBuffer:itemsBuffer offset:0 atIndex:1];
            [encoder setBuffer:rowGroup->ids[column] offset:0 atIndex:2];
            [encoder setBytes:&params length:sizeof(params) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(runs.items.size(), 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(
                         sparkmetal::decodeThreadgroupWidth(runs.maxItemCount), 1, 1)];
        } else {
            rowGroup->columnHasNulls[column] = true;
            size_t valuesBytes = std::max<size_t>(
                4, static_cast<size_t>(runs.nonNullCount) * sizeof(int32_t));
            id<MTLBuffer> valuesBuffer = acquireStagingBuffer(stream, valuesBytes);
            if (valuesBuffer == nil) {
                // Leave the encoder open: it belongs to the row group, and the
                // caller's failure path (parquetRowGroupRelease, or
                // Finish/Abort) closes and commits it. Hand over the staging
                // already referenced by this page's dispatches so it is keyed
                // to that commit rather than recycled early.
                rowGroup->pendingStaging.insert(
                    rowGroup->pendingStaging.end(), usedStaging.begin(), usedStaging.end());
                throwRuntime(environment, @"Cannot allocate Parquet value-scratch staging buffer");
                return;
            }
            usedStaging.push_back(valuesBuffer);

            if (itemsBuffer != nil) {
                ExpandParams expandParams = {
                    static_cast<uint32_t>(runs.items.size()),
                    runs.bitWidth,
                    runs.valueBytesOffset,
                    0u
                };
                [encoder setComputePipelineState:expandValueRunsPipeline];
                [encoder setBuffer:pageStaging offset:0 atIndex:0];
                [encoder setBuffer:itemsBuffer offset:0 atIndex:1];
                [encoder setBuffer:valuesBuffer offset:0 atIndex:2];
                [encoder setBytes:&expandParams length:sizeof(expandParams) atIndex:3];
                [encoder dispatchThreadgroups:MTLSizeMake(runs.items.size(), 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(
                             sparkmetal::decodeThreadgroupWidth(runs.maxItemCount), 1, 1)];
            }

            id<MTLBuffer> segmentsBuffer = acquireStagingBuffer(
                stream, runs.segments.size() * sizeof(sparkmetal::RowSegment));
            if (segmentsBuffer == nil) {
                rowGroup->pendingStaging.insert(
                    rowGroup->pendingStaging.end(), usedStaging.begin(), usedStaging.end());
                throwRuntime(environment, @"Cannot allocate Parquet segment staging buffer");
                return;
            }
            memcpy(segmentsBuffer.contents, runs.segments.data(),
                runs.segments.size() * sizeof(sparkmetal::RowSegment));
            usedStaging.push_back(segmentsBuffer);

            ScatterParams scatterParams = {
                static_cast<uint32_t>(runs.segments.size()),
                static_cast<uint32_t>(rowOffset)
            };
            [encoder setComputePipelineState:scatterSegmentsPipeline];
            [encoder setBuffer:valuesBuffer offset:0 atIndex:0];
            [encoder setBuffer:segmentsBuffer offset:0 atIndex:1];
            [encoder setBuffer:rowGroup->ids[column] offset:0 atIndex:2];
            [encoder setBuffer:rowGroup->validity[column] offset:0 atIndex:3];
            [encoder setBytes:&scatterParams length:sizeof(scatterParams) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(runs.segments.size(), 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(
                         sparkmetal::decodeThreadgroupWidth(runs.maxSegmentCount), 1, 1)];
        }

        // No endEncoding / commit here: the encoder stays open for the next
        // page. The staging this page's dispatches read is parked on the row
        // group and keyed to its command buffer when commitRowGroup runs.
        rowGroup->pendingStaging.insert(
            rowGroup->pendingStaging.end(), usedStaging.begin(), usedStaging.end());
        if (++rowGroup->pagesSinceCommit >= kPagesPerCommit) {
            commitRowGroup(rowGroup);
        }
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_parquetRowGroupRead(
    JNIEnv *environment,
    jclass,
    jlong streamHandle,
    jlong rowGroupHandle,
    jint column,
    jintArray idsOut,
    jbyteArray validityOut) {
    @autoreleasepool {
        if (streamHandle == 0 || rowGroupHandle == 0 || column < 0 || column >= 3 ||
            idsOut == nullptr || validityOut == nullptr) {
            throwRuntime(environment, @"Invalid Parquet row-group read arguments");
            return;
        }
        auto *stream = reinterpret_cast<MembershipStream *>(
            static_cast<uintptr_t>(streamHandle));
        auto *rowGroup = reinterpret_cast<ParquetRowGroup *>(
            static_cast<uintptr_t>(rowGroupHandle));
        if (rowGroup->stream != stream) {
            throwRuntime(environment, @"Row group does not belong to the given stream");
            return;
        }
        // The row group's own command buffer is still open at this point (see
        // ParquetRowGroup): close and commit it, or the wait below would return
        // before any of this row group's pages had run. A later
        // parquetDecodePage on the same row group simply opens a fresh command
        // buffer, which the queue still runs after this one.
        commitRowGroup(rowGroup);
        for (id<MTLCommandBuffer> commandBuffer : stream->commandBuffers) {
            [commandBuffer waitUntilCompleted];
            if (commandBuffer.status == MTLCommandBufferStatusError) {
                throwRuntime(environment, [NSString stringWithFormat:@"Metal command failed: %@", commandBuffer.error]);
                return;
            }
        }
        jsize idsLength = environment->GetArrayLength(idsOut);
        jsize validityLength = environment->GetArrayLength(validityOut);
        if (static_cast<uint32_t>(idsLength) != rowGroup->rowCount ||
            static_cast<uint32_t>(validityLength) != rowGroup->rowCount) {
            throwRuntime(environment, @"Output arrays must match the row-group row count");
            return;
        }
        environment->SetIntArrayRegion(
            idsOut, 0, idsLength, static_cast<jint *>(rowGroup->ids[column].contents));
        environment->SetByteArrayRegion(
            validityOut, 0, validityLength, static_cast<jbyte *>(rowGroup->validity[column].contents));
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_parquetRowGroupRelease(
    JNIEnv *environment,
    jclass,
    jlong rowGroupHandle) {
    @autoreleasepool {
        if (rowGroupHandle == 0) {
            throwRuntime(environment, @"Invalid Parquet row-group handle");
            return;
        }
        auto *rowGroup = reinterpret_cast<ParquetRowGroup *>(
            static_cast<uintptr_t>(rowGroupHandle));
        MembershipStream *stream = rowGroup->stream;
        // Close and commit whatever this row group had encoded but not yet
        // submitted (a partial encode from a page that threw, or a complete one
        // the caller chose not to Count). That both frees the encoder and gives
        // the planes below a command buffer to be keyed to.
        commitRowGroup(rowGroup);
        id<MTLCommandBuffer> lastCommandBuffer = rowGroup->lastCommandBuffer;
        if (lastCommandBuffer == nil && stream != nullptr && !stream->commandBuffers.empty()) {
            lastCommandBuffer = stream->commandBuffers.back();
        }
        for (int column = 0; column < 3; ++column) {
            if (lastCommandBuffer != nil) {
                stream->pendingStaging.push_back({rowGroup->ids[column], lastCommandBuffer});
                stream->pendingStaging.push_back({rowGroup->validity[column], lastCommandBuffer});
            } else if (stream != nullptr) {
                stream->freeStaging[rowGroup->ids[column].length].push_back(rowGroup->ids[column]);
                stream->freeStaging[rowGroup->validity[column].length]
                    .push_back(rowGroup->validity[column]);
            }
        }
        if (stream != nullptr) {
            auto &rowGroups = stream->rowGroups;
            rowGroups.erase(std::remove(rowGroups.begin(), rowGroups.end(), rowGroup), rowGroups.end());
        }
        delete rowGroup;
    }
}

// Runs the membership-count kernel over one row group's decoded planes
// (Task 3's ids/validity planes, fused with the dictionary-table logic from
// membershipCount3StreamSubmit) and releases the row-group handle. Commits
// without waiting; the result is folded into stream->partialBuffers /
// partialGroupCounts for the existing membershipCount3StreamFinish to
// accumulate.
extern "C" JNIEXPORT void JNICALL
Java_io_github_mohitpatil_sparkmetal_NativeBridge_parquetRowGroupCount(
    JNIEnv *environment,
    jclass,
    jlong streamHandle,
    jlong rowGroupHandle,
    jbyteArray dictPresence0,
    jintArray dictMultiplicity0,
    jbyteArray dictPresence1,
    jintArray dictMultiplicity1,
    jbyteArray dictPresence2,
    jintArray dictMultiplicity2) {
    @autoreleasepool {
        if (membershipCountUniquePipeline == nil ||
            membershipCountMultiplicityPipeline == nil || commandQueue == nil) {
            throwRuntime(environment, @"NativeBridge.initialize must be called first");
            return;
        }
        if (streamHandle == 0 || rowGroupHandle == 0) {
            throwRuntime(environment, @"Row-group count requires a stream and a row group");
            return;
        }
        auto *stream = reinterpret_cast<MembershipStream *>(
            static_cast<uintptr_t>(streamHandle));
        auto *rowGroup = reinterpret_cast<ParquetRowGroup *>(
            static_cast<uintptr_t>(rowGroupHandle));
        if (rowGroup->stream != stream) {
            throwRuntime(environment, @"Row group does not belong to the given stream");
            return;
        }
        PreparedMembershipCount3 *prepared = stream->prepared;

        jbyteArray presenceArrays[3] = {dictPresence0, dictPresence1, dictPresence2};
        jintArray multiplicityArrays[3] = {dictMultiplicity0, dictMultiplicity1, dictMultiplicity2};

        std::vector<id<MTLBuffer>> usedStaging;
        id<MTLBuffer> keyBuffers[3];
        int32_t keyMinimums[3];
        uint32_t keySpans[3];
        uint32_t nullMask = 0;
        for (NSUInteger index = 0; index < 3; ++index) {
            if (rowGroup->columnHasNulls[index]) {
                nullMask |= 1u << index;
            }

            if (presenceArrays[index] == nullptr && multiplicityArrays[index] == nullptr) {
                throwRuntime(environment, @"Row-group count requires a dictionary table per column");
                return;
            }
            bool presence = presenceArrays[index] != nullptr;
            if (presence != prepared->allKeysUnique) {
                throwRuntime(environment, @"Dictionary table type does not match the prepared kernel");
                return;
            }
            jsize length = presence
                ? environment->GetArrayLength(presenceArrays[index])
                : environment->GetArrayLength(multiplicityArrays[index]);
            if (length <= 0) {
                throwRuntime(environment, @"Dictionary membership table must not be empty");
                return;
            }
            size_t elementBytes = presence ? sizeof(uint8_t) : sizeof(uint32_t);
            id<MTLBuffer> table = acquireStagingBuffer(
                stream, static_cast<size_t>(length) * elementBytes);
            if (table == nil) {
                throwRuntime(environment, @"Cannot allocate dictionary membership table");
                return;
            }
            if (presence) {
                environment->GetByteArrayRegion(
                    presenceArrays[index], 0, length,
                    static_cast<jbyte *>(table.contents));
            } else {
                environment->GetIntArrayRegion(
                    multiplicityArrays[index], 0, length,
                    static_cast<jint *>(table.contents));
            }
            if (environment->ExceptionCheck()) return;
            usedStaging.push_back(table);
            keyBuffers[index] = table;
            keyMinimums[index] = 0;
            keySpans[index] = static_cast<uint32_t>(length);
        }

        constexpr NSUInteger threadsPerGroup = 256;
        NSUInteger groupCount =
            (static_cast<NSUInteger>(rowGroup->rowCount) + threadsPerGroup - 1) / threadsPerGroup;
        NSUInteger partialElementBytes = prepared->allKeysUnique
            ? sizeof(uint32_t) : sizeof(int64_t);
        id<MTLBuffer> partialBuffer = [device
            newBufferWithLength:groupCount * partialElementBytes
            options:MTLResourceStorageModeShared];
        if (partialBuffer == nil) {
            throwRuntime(environment, @"Cannot allocate row-group Metal partial-count buffer");
            return;
        }
        MembershipCountParameters parameters = {
            rowGroup->rowCount, nullMask,
            keyMinimums[0], keyMinimums[1], keyMinimums[2],
            keySpans[0], keySpans[1], keySpans[2]
        };
        // The count dispatch goes into the same encoder as this row group's
        // page expansions. MTLDispatchTypeSerial guarantees every expand and
        // scatter has completed before the count kernel reads the planes, so no
        // extra command buffer, commit, or barrier is needed.
        id<MTLComputeCommandEncoder> encoder = rowGroupEncoder(rowGroup);
        if (encoder == nil) {
            throwRuntime(environment, @"Cannot open a Parquet row-group compute encoder");
            return;
        }
        [encoder setComputePipelineState:prepared->allKeysUnique
            ? membershipCountUniquePipeline
            : membershipCountMultiplicityPipeline];
        for (NSUInteger index = 0; index < 3; ++index) {
            [encoder setBuffer:rowGroup->ids[index] offset:0 atIndex:index];
            [encoder setBuffer:rowGroup->validity[index] offset:0 atIndex:index + 3];
            [encoder setBuffer:keyBuffers[index] offset:0 atIndex:index + 6];
        }
        [encoder setBuffer:partialBuffer offset:0 atIndex:9];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:10];
        [encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];

        // The dictionary-table staging is scoped to this call, and the row
        // group's own ids/validity planes are never pendingStaging entries (see
        // the ParquetRowGroup comment) -- both must be keyed to this row group's
        // command buffer, which commitRowGroup does for everything parked here.
        rowGroup->pendingStaging.insert(
            rowGroup->pendingStaging.end(), usedStaging.begin(), usedStaging.end());
        for (int column = 0; column < 3; ++column) {
            rowGroup->pendingStaging.push_back(rowGroup->ids[column]);
            rowGroup->pendingStaging.push_back(rowGroup->validity[column]);
        }
        commitRowGroup(rowGroup);

        stream->partialBuffers.push_back(partialBuffer);
        stream->partialGroupCounts.push_back(groupCount);

        auto &rowGroups = stream->rowGroups;
        rowGroups.erase(std::remove(rowGroups.begin(), rowGroups.end(), rowGroup), rowGroups.end());
        delete rowGroup;
    }
}
