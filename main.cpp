#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>

const int numGPUs = 8;
const size_t bufferSize = 16L * 1024 * 1024 * 1024; // Buffer size of 16 GB to accommodate the largest data size
const std::vector<size_t> dataSizes = {
    8,                  // 8 bytes
    1024,               // 1 KB
    1024 * 1024,        // 1 MB
    1024 * 1024 * 1024, // 1 GB
    16L * 1024 * 1024 * 1024 // 16 GB
};

__global__ void initDataKernel(char* data, size_t dataSize, char value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dataSize) {
        data[idx] = value;
    }
}

#define HIP_CHECK(call) \
do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error in " << #call << " at " << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

int main() {
    std::ofstream logFile("data_transfer_log.csv");
    logFile << "DataSize(Bytes),TimeTaken(s),Bandwidth(GB/s)" << std::endl;

    hipStream_t streams[numGPUs];
    char* gpuBuffers[numGPUs];
    char* sharedDataBuffers[numGPUs];

    // Initialize GPUs, allocate memory, and create streams
    for (int i = 0; i < numGPUs; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipMalloc(&gpuBuffers[i], bufferSize));
        HIP_CHECK(hipMalloc(&sharedDataBuffers[i], bufferSize)); // Allocate buffer for the largest size
        HIP_CHECK(hipStreamCreate(&streams[i]));
    }

    for (size_t sharedDataSize : dataSizes) {
        std::cout << "Running data transfer with size: " << sharedDataSize << " bytes" << std::endl;

        // Launch kernel to initialize data for each size
        for (int i = 0; i < numGPUs; ++i) {
            HIP_CHECK(hipSetDevice(i));
            dim3 threadsPerBlock(256);
            dim3 numBlocks((sharedDataSize + threadsPerBlock.x - 1) / threadsPerBlock.x);
            initDataKernel<<<numBlocks, threadsPerBlock, 0, streams[i]>>>(sharedDataBuffers[i], sharedDataSize, static_cast<char>(i));
        }

        // Perform data transfer operations
        auto start = std::chrono::high_resolution_clock::now();
        for (int sharedDataGpu = 0; sharedDataGpu < numGPUs; ++sharedDataGpu) {
            for (int targetGpu = 0; targetGpu < numGPUs; ++targetGpu) {
                if (targetGpu != sharedDataGpu) {
                    HIP_CHECK(hipSetDevice(targetGpu));
                    HIP_CHECK(hipMemcpyPeerAsync(gpuBuffers[targetGpu], targetGpu, sharedDataBuffers[sharedDataGpu], sharedDataGpu, sharedDataSize, streams[targetGpu]));
                }
            }
            // Synchronize all streams
            for (int i = 0; i < numGPUs; ++i) {
                if (i != sharedDataGpu) {
                    HIP_CHECK(hipSetDevice(i));
                    HIP_CHECK(hipStreamSynchronize(streams[i]));
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        double timeTaken = diff.count();
        double bandwidthGBps = (sharedDataSize / (1024.0 * 1024.0 * 1024.0)) / timeTaken; // Convert bytes to GB and divide by seconds
        std::cout << "Completed data transfer for size: " << sharedDataSize << " bytes in " << timeTaken << " s, Bandwidth: " << bandwidthGBps << " GB/s" << std::endl;
        logFile << sharedDataSize << "," << std::setprecision(9) << timeTaken << "," << std::setprecision(9) << bandwidthGBps << std::endl;
        // std::cout << "Completed data transfer for size: " << sharedDataSize << " bytes in " << diff.count() << " s" << std::endl;
        // logFile << sharedDataSize << "," << std::setprecision(9) << diff.count() << std::endl;
    }

    // Cleanup
    for (int i = 0; i < numGPUs; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipFree(gpuBuffers[i]));
        HIP_CHECK(hipFree(sharedDataBuffers[i]));
        HIP_CHECK(hipStreamDestroy(streams[i]));
    }

    logFile.close();
    std::cout << "All data transfers completed successfully. Results logged to data_transfer_log.csv" << std::endl;

    return 0;
}
