#ifndef GPU_ERROR_CHECK
#define GPU_ERROR_CHECK

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(0);
        }
    }
}

#endif