#include <cstdio>

void printDeviceProp(const cudaDeviceProp &prop) {
    printf("Device Name: %s.\n", prop.name);
    printf("totalGlobalMem: %lu.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock: %lu.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock: %d.\n", prop.regsPerBlock);
    printf("warpSize: %d.\n", prop.warpSize);
    printf("memPitch: %lu.\n", prop.memPitch);
    printf("maxThreadsPerBlock: %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2]: %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem: %lu.\n", prop.totalConstMem);
    printf("Compute capability: %d.%d.\n", prop.major, prop.minor);
    printf("clockRate: %d.\n", prop.clockRate);
    printf("textureAlignment: %lu.\n", prop.textureAlignment);
    printf("deviceOverlap: %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount: %d.\n", prop.multiProcessorCount);
}


int main () {
    int count;
    cudaGetDeviceCount(&count);
    printf("Count of device: %d\n", count);
    
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) {
                printDeviceProp(prop);
                break;
            }
        }
    }

    return 0;
}