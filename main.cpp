#include <hip/hip_runtime.h>
#include <iostream>

const int numGPUs = 8; // Total number of GPUs
const size_t bufferSize = 1024; // Size of buffer in each GPU

// Kernel to initialize the data in the buffer
__global__ void initDataKernel(char* data, size_t dataSize, char value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dataSize) {
        data[idx] = value;
    }
}

int main() {
    hipStream_t streams[numGPUs];
    char* gpuBuffers[numGPUs];

    // Initialize GPUs, allocate memory, and create streams
    for (int i = 0; i < numGPUs; ++i) {
        hipSetDevice(i);
        hipMalloc(&gpuBuffers[i], bufferSize);
        hipStreamCreate(&streams[i]);

        // Launch kernel to initialize data
        dim3 threadsPerBlock(256);
        dim3 numBlocks((bufferSize + threadsPerBlock.x - 1) / threadsPerBlock.x);
        initDataKernel<<<numBlocks, threadsPerBlock, 0, streams[i]>>>(gpuBuffers[i], bufferSize, static_cast<char>(i)); // Initialize with a unique value for each GPU
    }

    for (int targetGpu = 0; targetGpu < numGPUs; ++targetGpu) {
        hipSetDevice(targetGpu);

        for (int sourceGpu = 0; sourceGpu < numGPUs; ++sourceGpu) {
            if (sourceGpu != targetGpu) {
                // Asynchronously copy data from gpuBuffers[sourceGpu] to gpuBuffers[targetGpu]
                hipMemcpyPeerAsync(gpuBuffers[targetGpu], targetGpu, gpuBuffers[sourceGpu], sourceGpu, bufferSize, streams[targetGpu]);
            }
        }
    }

    // Synchronize all streams
    for (int i = 0; i < numGPUs; ++i) {
        hipSetDevice(i);
        hipStreamSynchronize(streams[i]);
    }

    // Cleanup
    for (int i = 0; i < numGPUs; ++i) {
        hipSetDevice(i);
        hipFree(gpuBuffers[i]);
        hipStreamDestroy(streams[i]);
    }

    return 0;
}
