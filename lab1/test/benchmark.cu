#include <cstdio>
#include <cstdlib>
#include <chrono>

using duration_t = std::chrono::microseconds;
const int COUNT_OF_ARRAYS = 2;

void kernel_cpu (double *arrays, int count_of_arrays, int n) {
    int idx = 0;
    int offset = 1;
    double tmp;
    while (idx < n) {
        tmp = arrays[idx];
        for (int i = 1; i < count_of_arrays; ++i) {
            tmp = max(tmp, arrays[idx + i * n]);
        }
        arrays[idx] = tmp;
        idx += offset;
    }
}

__global__ void kernel_gpu (double *arrays, int count_of_arrays, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    double tmp;
    while (idx < n) {
        tmp = arrays[idx];
        for (int i = 1; i < count_of_arrays; ++i) {
            tmp = max(tmp, arrays[idx + i * n]);
        }
        arrays[idx] = tmp;
        idx += offset;
    }
}


int main () {
    int n;
    scanf("%d", &n);

    double *arrays[COUNT_OF_ARRAYS + 1], el;
    for (int i = 0; i < COUNT_OF_ARRAYS + 1; ++i) {
        arrays[i] = (double *)malloc(sizeof(double) * n);
        if (arrays[i] == NULL) {
            fprintf(stderr, "ERROR: out of RAM\n");
            return 0;
        }
        if (i < COUNT_OF_ARRAYS) {
            for (int j = 0; j < n; ++j) {
                scanf("%lf", &el);
                arrays[i][j] = el;
            }
        }
    }

    double *dev_arrays1 = (double *)malloc(sizeof(double) * n * COUNT_OF_ARRAYS);
    for (int i = 0; i < COUNT_OF_ARRAYS; ++i) {
        memcpy(dev_arrays1 + i * n, arrays[i], sizeof(double) * n);
    }

    double *dev_arrays2;
    cudaMalloc(&dev_arrays2, sizeof(double) * n * COUNT_OF_ARRAYS);
    if (dev_arrays2 == NULL) {
        fprintf(stderr, "ERROR: out of VRAM\n");
        return 0;
    }
    for (int i = 0; i < COUNT_OF_ARRAYS; ++i) {
        cudaMemcpy(dev_arrays2 + i * n, arrays[i], sizeof(double) * n, cudaMemcpyHostToDevice);
    }
    
    std::chrono::time_point <std::chrono::system_clock> startt, endt;
    startt = std::chrono::system_clock::now();
    kernel_cpu(dev_arrays1, COUNT_OF_ARRAYS, n);
    endt = std::chrono::system_clock::now();
    printf("CPU time: %lumcs\n", std::chrono::duration_cast <duration_t>(endt - startt).count());


    cudaEvent_t e_start, e_stop;
	cudaEventCreate(&e_start);
	cudaEventCreate(&e_stop);
    cudaEventRecord(e_start, 0);
    kernel_gpu<<<1, 1>>>(dev_arrays2, COUNT_OF_ARRAYS, n);
    cudaEventRecord(e_stop, 0);
    cudaEventSynchronize(e_stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    printf("GPU time: %fmcs\n", elapsedTime);

    // for (int i = 0; i < n; ++i) {
    //     printf("%.10e", dev_arrays[i]);
    //     if (i < n - 1) {
    //         printf(" ");
    //     }
    // }

    for (int i = 0; i < COUNT_OF_ARRAYS + 1; ++i) {
        free(arrays[i]);
    }
    free(dev_arrays1);
    cudaFree(dev_arrays2);

    return 0;
}