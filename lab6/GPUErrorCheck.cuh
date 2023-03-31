#ifndef GPU_ERROR_CHECK
#define GPU_ERROR_CHECK

#include <iostream>

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(code) << ". " << file << ':' << line << '\n';
        if (abort) {
            exit(0);
        }
    }
}

#endif