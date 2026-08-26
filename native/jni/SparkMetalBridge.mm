#include <jni.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace {

id<MTLDevice> device;
id<MTLCommandQueue> commandQueue;
id<MTLComputePipelineState> fusedPipeline;
id<MTLComputePipelineState> membershipCountUniquePipeline;
id<MTLComputePipelineState> membershipCountMultiplicityPipeline;

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

id<MTLBuffer> bufferFromAddress(
    JNIEnv *environment,
    jlong address,
    size_t length,
    NSString *description) {
    if (address == 0 || length == 0) {
        throwRuntime(environment, [NSString stringWithFormat:@"Invalid %@ address or length", description]);
        return nil;
    }
    void *pointer = reinterpret_cast<void *>(static_cast<uintptr_t>(address));
    size_t pageSize = static_cast<size_t>(getpagesize());
    bool pageAligned =
        reinterpret_cast<uintptr_t>(pointer) % pageSize == 0 && length % pageSize == 0;
    id<MTLBuffer> buffer = pageAligned
        ? [device newBufferWithBytesNoCopy:pointer
                length:length
                options:MTLResourceStorageModeShared
                deallocator:nil]
        : [device newBufferWithBytes:pointer
                length:length
                options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        throwRuntime(environment, [NSString stringWithFormat:@"Cannot create %@ Metal buffer", description]);
    }
    return buffer;
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
Java_io_github_mohitpatil_sparkmetal_NativeBridge_membershipCount3Address(
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
    jintArray keys0,
    jintArray keys1,
    jintArray keys2) {
    @autoreleasepool {
        if (membershipCountUniquePipeline == nil ||
            membershipCountMultiplicityPipeline == nil || commandQueue == nil) {
            throwRuntime(environment, @"NativeBridge.initialize must be called first");
            return 0;
        }
        if (count <= 0 || keys0 == nullptr || keys1 == nullptr || keys2 == nullptr) {
            throwRuntime(environment, @"Membership count requires rows and three key arrays");
            return 0;
        }
        jsize keyCount0 = environment->GetArrayLength(keys0);
        jsize keyCount1 = environment->GetArrayLength(keys1);
        jsize keyCount2 = environment->GetArrayLength(keys2);
        if (keyCount0 <= 0 || keyCount1 <= 0 || keyCount2 <= 0) {
            return 0;
        }
        size_t inputBytes = static_cast<size_t>(count) * sizeof(int32_t);
        id<MTLBuffer> inputs[3] = {
            bufferFromAddress(environment, input0Address, inputBytes, @"input 0"),
            bufferFromAddress(environment, input1Address, inputBytes, @"input 1"),
            bufferFromAddress(environment, input2Address, inputBytes, @"input 2")
        };
        if (environment->ExceptionCheck()) return 0;
        jlong nullAddresses[3] = {null0Address, null1Address, null2Address};
        jboolean hasNulls[3] = {hasNull0, hasNull1, hasNull2};
        id<MTLBuffer> nullBuffers[3];
        uint32_t nullMask = 0;
        for (NSUInteger index = 0; index < 3; ++index) {
            if (hasNulls[index] == JNI_TRUE) {
                nullMask |= 1u << index;
                nullBuffers[index] = bufferFromAddress(
                    environment, nullAddresses[index], static_cast<size_t>(count), @"null mask");
            } else {
                nullBuffers[index] = [device newBufferWithLength:1 options:MTLResourceStorageModeShared];
            }
            if (nullBuffers[index] == nil || environment->ExceptionCheck()) return 0;
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
        id<MTLBuffer> keyBuffers[3];
        int32_t keyMinimums[3];
        uint32_t keySpans[3];
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
            keyMinimums[index] = minimum;
            keySpans[index] = static_cast<uint32_t>(span64);
        }
        if (allKeysUnique) {
            for (NSUInteger index = 0; index < 3; ++index) {
                keyBuffers[index] = [device newBufferWithBytes:uniqueDenseMaps[index].data()
                    length:uniqueDenseMaps[index].size() * sizeof(uint8_t)
                    options:MTLResourceStorageModeShared];
            }
        } else {
            for (NSUInteger index = 0; index < 3; ++index) {
                std::vector<uint32_t> dense(static_cast<size_t>(keySpans[index]), 0);
                for (jsize keyIndex = 0; keyIndex < keyCounts[index]; ++keyIndex) {
                    dense[static_cast<size_t>(
                        keyPointers[index][keyIndex] - keyMinimums[index])] += 1;
                }
                keyBuffers[index] = [device newBufferWithBytes:dense.data()
                    length:dense.size() * sizeof(uint32_t)
                    options:MTLResourceStorageModeShared];
            }
        }
        environment->ReleaseIntArrayElements(keys0, keyPointers[0], JNI_ABORT);
        environment->ReleaseIntArrayElements(keys1, keyPointers[1], JNI_ABORT);
        environment->ReleaseIntArrayElements(keys2, keyPointers[2], JNI_ABORT);
        if (keyBuffers[0] == nil || keyBuffers[1] == nil || keyBuffers[2] == nil) {
            throwRuntime(environment, @"Cannot create Metal membership-key buffers");
            return 0;
        }

        constexpr NSUInteger threadsPerGroup = 256;
        NSUInteger groupCount =
            (static_cast<NSUInteger>(count) + threadsPerGroup - 1) / threadsPerGroup;
        id<MTLBuffer> partialBuffer = [device
            newBufferWithLength:groupCount * sizeof(int64_t)
            options:MTLResourceStorageModeShared];
        if (partialBuffer == nil) {
            throwRuntime(environment, @"Cannot allocate Metal partial-count buffer");
            return 0;
        }
        MembershipCountParameters parameters = {
            static_cast<uint32_t>(count), nullMask,
            keyMinimums[0], keyMinimums[1], keyMinimums[2],
            keySpans[0], keySpans[1], keySpans[2]
        };
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:allKeysUnique
            ? membershipCountUniquePipeline
            : membershipCountMultiplicityPipeline];
        for (NSUInteger index = 0; index < 3; ++index) {
            [encoder setBuffer:inputs[index] offset:0 atIndex:index];
            [encoder setBuffer:nullBuffers[index] offset:0 atIndex:index + 3];
            [encoder setBuffer:keyBuffers[index] offset:0 atIndex:index + 6];
        }
        [encoder setBuffer:partialBuffer offset:0 atIndex:9];
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
        int64_t *partials = static_cast<int64_t *>(partialBuffer.contents);
        int64_t result = 0;
        for (NSUInteger index = 0; index < groupCount; ++index) result += partials[index];
        return static_cast<jlong>(result);
    }
}
