#include <cstdio>
#include <cstdlib>

const int COUNT_OF_ARRAYS = 2;

__global__ void kernel (double *arrays, int count_of_arrays, int n) {
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

    double *dev_arrays;
    cudaMalloc(&dev_arrays, sizeof(double) * n * COUNT_OF_ARRAYS);
    if (dev_arrays == NULL) {
        fprintf(stderr, "ERROR: out of VRAM\n");
        return 0;
    }
    for (int i = 0; i < COUNT_OF_ARRAYS; ++i) {
        cudaMemcpy(dev_arrays + i * n, arrays[i], sizeof(double) * n, cudaMemcpyHostToDevice);
    }

    kernel<<<1024, 1024>>>(dev_arrays, COUNT_OF_ARRAYS, n);

    cudaMemcpy(arrays[COUNT_OF_ARRAYS], dev_arrays, sizeof(double) * n, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        printf("%.10e", arrays[COUNT_OF_ARRAYS][i]);
        if (i < n - 1) {
            printf(" ");
        }
    }

    for (int i = 0; i < COUNT_OF_ARRAYS + 1; ++i) {
        free(arrays[i]);
    }
    cudaFree(dev_arrays);

    return 0;
}