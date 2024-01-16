#include <hip/hip_runtime.h>
#include <iostream>

const int numGPUs = 8; // Total number of GPUs
const size_t bufferSize = 1024; // Size of buffer in each GPU
const size_t sharedDataSize = 256; // Size of the shared data portion

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
    char* sharedDataBuffers[numGPUs];

    // Initialize GPUs, allocate memory, and create streams
    for (int i = 0; i < numGPUs; ++i) {
        hipSetDevice(i);
        hipMalloc(&gpuBuffers[i], bufferSize);
        hipMalloc(&sharedDataBuffers[i], sharedDataSize);
        hipStreamCreate(&streams[i]);

        // Launch kernel to initialize data
        dim3 threadsPerBlock(256);
        dim3 numBlocks((bufferSize + threadsPerBlock.x - 1) / threadsPerBlock.x);
        initDataKernel<<<numBlocks, threadsPerBlock, 0, streams[i]>>>(gpuBuffers[i], bufferSize, static_cast<char>(i)); // Initialize with a unique value for each GPU
    }

    for (int sharedDataGpu = 0; sharedDataGpu < numGPUs; ++sharedDataGpu) {
        for (int targetGpu = 0; targetGpu < numGPUs; ++targetGpu) {
            if (targetGpu != sharedDataGpu) {
                hipSetDevice(targetGpu);
                // Asynchronously copy shared data portion from sharedDataBuffers[sharedDataGpu]
                hipMemcpyPeerAsync(gpuBuffers[targetGpu], targetGpu, sharedDataBuffers[sharedDataGpu], sharedDataGpu, sharedDataSize, streams[targetGpu]);
            }
        }
      // Synchronize all streams
      for (int i = 0; i < numGPUs; ++i) {
        if (i != sharedDataGpu) {
          hipSetDevice(i);
          hipStreamSynchronize(streams[i]);
        } 
      }
    }

    // Cleanup
    for (int i = 0; i < numGPUs; ++i) {
        hipSetDevice(i);
        hipFree(gpuBuffers[i]);
        hipFree(sharedDataBuffers[i]);
        hipStreamDestroy(streams[i]);
    }

    return 0;
}
