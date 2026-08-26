#include <jni.h>
#include <cstdint>
#include <sys/mman.h>
#include <unistd.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace {

id<MTLDevice> device;
id<MTLCommandQueue> commandQueue;
id<MTLComputePipelineState> fusedPipeline;

struct FusedParameters {
    uint32_t count;
    int32_t threshold;
    int32_t multiplier;
    int32_t addend;
    uint32_t hasNulls;
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
