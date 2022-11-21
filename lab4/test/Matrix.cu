#include "Matrix.cuh"
#include <cuda_profiler_api.h>

struct Comparator {
    __host__ __device__ bool operator()(double num1, double num2) {
        return fabs(num1) < fabs(num2);
    }
};

uint64_t findMax(double *data, uint64_t n, uint64_t i) {
    double max = 0;
    uint64_t ans = i;
    for (uint64_t j = i; j < n; ++j) {
        if (std::abs(data[j]) > std::abs(max)) {
            max = data[j];
            ans = j;
        }
    }
    return ans;
}

__global__ void swapRows (double *data, uint64_t n, uint64_t i, uint64_t j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    double tmp;
    for (uint64_t k = idx; k < 2 * n; k += offsetX) {
        tmp = data[k * n + j];
        data[k * n + j] = data[k * n + i];
        data[k * n + i] = tmp;
    }
}

void swapRowsCPU (double *data, uint64_t n, uint64_t i, uint64_t j) {
    int idx = 0;
    int offsetX = 1;

    double tmp;
    for (uint64_t k = idx; k < 2 * n; k += offsetX) {
        tmp = data[k * n + j];
        data[k * n + j] = data[k * n + i];
        data[k * n + i] = tmp;
    }
}

__global__ void normalisation (double *data, uint64_t n, uint64_t i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    for (uint64_t k = idx + i + 1; k < 2 * n; k += offsetX) {
        data[k * n + i] /= data[i * n + i];;
    }
}

 void normalisationCPU (double *data, uint64_t n, uint64_t i) {
    int idx = 0;
    int offsetX = 1;

    for (uint64_t k = idx + i + 1; k < 2 * n; k += offsetX) {
        data[k * n + i] /= data[i * n + i];;
    }
}

__global__ void iteration (double *data, uint64_t n, uint64_t id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = gridDim.x * blockDim.x;
    int offsetY = gridDim.y * blockDim.y;

    for (uint64_t i = idy + id + 1; i < 2 * n; i += offsetY) {
        for (uint64_t j = idx + id + 1; j < n; j += offsetX) {
            data[i * n + j] -= data[i * n + id] * data[id * n + j];
        }
    }
}

void iterationCPU (double *data, uint64_t n, uint64_t id) {
    int idx = 0;
    int idy = 0;
    int offsetX = 1;
    int offsetY = 1;

    for (uint64_t i = idy + id + 1; i < 2 * n; i += offsetY) {
        for (uint64_t j = idx + id + 1; j < n; j += offsetX) {
            data[i * n + j] -= data[i * n + id] * data[id * n + j];
        }
    }
}

__global__ void backIteration (double *data, uint64_t n, uint64_t id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = gridDim.x * blockDim.x;
    int offsetY = gridDim.y * blockDim.y;

    for (uint64_t i = idy + id + 1; i < 2 * n; i += offsetY) {
        for (uint64_t j = idx; j <= id - 1; j += offsetX) {
            data[i * n + j] -= data[i * n + id] * data[id * n + j];
        }
    }
}

void backIterationCPU (double *data, uint64_t n, uint64_t id) {
    int idx = 0;
    int idy = 0;
    int offsetX = 1;
    int offsetY = 1;

    for (uint64_t i = idy + id + 1; i < 2 * n; i += offsetY) {
        for (uint64_t j = idx; j <= id - 1; j += offsetX) {
            data[i * n + j] -= data[i * n + id] * data[id * n + j];
        }
    }
}

Matrix::Matrix () : n(0), m(0) {}

Matrix::Matrix (uint64_t size) : n(size), m(size), data(n * m, 0) {
    for (uint64_t i = 0; i < size; ++i) {
        data[i * size + i] = 1;
    }
}

Matrix::Matrix (Matrix const &matrix) {
    n = matrix.n;
    m = matrix.m;
    data = matrix.data;
}

Matrix::Matrix (Matrix &&matrix) {
    n = matrix.n;
    m = matrix.m;
    data = std::move(matrix.data);
}

Matrix::Matrix (uint64_t n, uint64_t m, const std::vector<double> &buff) : n(n), m(m), data(buff) {}

Matrix::Matrix (uint64_t n, uint64_t m, double *buff) : n(n), m(m) {
    data.resize(n * m);
    data.assign(buff, buff + n * m);
}

Matrix::Matrix (uint64_t n, uint64_t m) : n(n), m(m), data(n * m, 0) {}

Matrix::~Matrix () {}

double &Matrix::operator() (uint64_t i, uint64_t j) {
    return data[i * m + j];
}

double Matrix::operator() (uint64_t i, uint64_t j) const {
    return data[i * m + j];
}

Matrix &Matrix::operator= (const Matrix &matrix) {
    n = matrix.n;
    m = matrix.m;
    data = matrix.data;
    return *this;
}

Matrix &Matrix::operator= (Matrix &&matrix) {
    n = matrix.n;
    m = matrix.m;
    data = std::move(matrix.data);
    return *this;
}

std::pair<uint64_t, Matrix> Matrix::reverseGPU () const {
    uint64_t time = 0;
    float elapsedTime;
    Matrix ans(n), tmp(2 * n, n);
    double *old_data;

    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            tmp(j, i) = data[i * m + j];
        }
    }
    for (uint64_t i = 0; i < n; ++i) {
        tmp(i + n, i) = 1;
    }
    cudaEvent_t e_start, e_stop;
	cudaEventCreate(&e_start);
	cudaEventCreate(&e_stop);

    gpuErrorCheck(cudaMalloc(&old_data, sizeof(double) * n * n * 2));
    gpuErrorCheck(cudaMemcpy(old_data, tmp.data.data(), sizeof(double) * n * n * 2, cudaMemcpyHostToDevice));

    Comparator check;


    for (uint64_t i = 0; i < n - 1; ++i) {
        thrust::device_ptr<double> device_data = thrust::device_pointer_cast(old_data + i * n);
        thrust::device_ptr<double> max = thrust::max_element(device_data + i, device_data + n, check);
        uint64_t idx = max - device_data;

        if (i != idx) {
            cudaEventRecord(e_start, 0);
            swapRows<<<256, 256>>>(old_data, n, i, idx);
            gpuErrorCheck(cudaGetLastError());
            cudaEventRecord(e_stop, 0);
            cudaEventSynchronize(e_stop);
            cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
            time += (uint64_t)(elapsedTime * 1000);
        }

        cudaEventRecord(e_start, 0);
        normalisation<<<256, 256>>>(old_data, n, i);
        gpuErrorCheck(cudaGetLastError());
        cudaEventRecord(e_stop, 0);
        cudaEventSynchronize(e_stop);
        cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
        time += (uint64_t)(elapsedTime * 1000);

        cudaEventRecord(e_start, 0);
        iteration<<<dim3(32, 32), dim3(32, 32)>>>(old_data, n, i);
        gpuErrorCheck(cudaGetLastError());
        cudaEventRecord(e_stop, 0);
        cudaEventSynchronize(e_stop);
        cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
        time += (uint64_t)(elapsedTime * 1000);
    }
    cudaEventRecord(e_start, 0);
    cudaProfilerStart();
    normalisation<<<256, 256>>>(old_data, n, n - 1);
    cudaProfilerStop();
    gpuErrorCheck(cudaGetLastError());
    cudaEventRecord(e_stop, 0);
    cudaEventSynchronize(e_stop);
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    time += (uint64_t)(elapsedTime * 1000);

    for (uint64_t i = n - 1; i > 0; --i) {
        cudaEventRecord(e_start, 0);
        backIteration<<<dim3(32, 32), dim3(32, 32)>>>(old_data, n, i);
        gpuErrorCheck(cudaGetLastError());
        cudaEventRecord(e_stop, 0);
        cudaEventSynchronize(e_stop);
        cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
        time += (uint64_t)(elapsedTime * 1000);
    }

    gpuErrorCheck(cudaMemcpy(&tmp.data[0], old_data, sizeof(double) * n * n * 2, cudaMemcpyDeviceToHost));
    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            ans(i, j) = tmp(j + n, i);
        }
    }

    gpuErrorCheck(cudaFree(old_data));
    return std::make_pair(time, ans);
}

std::pair<uint64_t, Matrix> Matrix::reverseCPU () const {
    uint64_t time = 0;
    Matrix ans(n), tmp(2 * n, n);
    double *old_data;
    std::chrono::time_point <std::chrono::system_clock> startt, endt;

    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            tmp(j, i) = data[i * m + j];
        }
    }
    for (uint64_t i = 0; i < n; ++i) {
        tmp(i + n, i) = 1;
    }

    old_data = (double*)malloc(sizeof(double) * n * n * 2);
    memcpy(old_data, tmp.data.data(), sizeof(double) * n * n * 2);
    //gpuErrorCheck(cudaMalloc(&old_data, sizeof(double) * n * n * 2));
    //gpuErrorCheck(cudaMemcpy(old_data, tmp.data.data(), sizeof(double) * n * n * 2, cudaMemcpyHostToDevice));

    startt = std::chrono::system_clock::now();
    for (uint64_t i = 0; i < n - 1; ++i) {
        //double *device_data = old_data + i * n;
        //double *max = 
        //thrust::device_ptr<double> device_data = thrust::device_pointer_cast(old_data + i * n);
        //thrust::device_ptr<double> max = thrust::max_element(device_data + i, device_data + n, check);
        uint64_t idx = findMax(old_data, i * n + n, i * n + i) - i * n;//max - device_data;

        if (i != idx) {
            swapRowsCPU(old_data, n, i, idx);
        }
        normalisationCPU(old_data, n, i);
        iterationCPU(old_data, n, i);
    }
    normalisationCPU(old_data, n, n - 1);

    for (uint64_t i = n - 1; i > 0; --i) {
        backIterationCPU(old_data, n, i);
    }
    endt = std::chrono::system_clock::now();
    time += std::chrono::duration_cast <duration_t>(endt - startt).count();

    memcpy(&tmp.data[0], old_data, sizeof(double) * n * n * 2);
    //gpuErrorCheck(cudaMemcpy(&tmp.data[0], old_data, sizeof(double) * n * n * 2, cudaMemcpyDeviceToHost));
    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            ans(i, j) = tmp(j + n, i);
        }
    }

    free(old_data);
    //gpuErrorCheck(cudaFree(old_data));
    return std::make_pair(time, ans);
}

const Matrix operator* (const Matrix &m1, const Matrix &m2) {
    Matrix ans(m1.n);
    for (uint64_t i = 0; i < m1.n; ++i) {
        for (uint64_t j = 0; j < m1.m; ++j) {
            double tmp = 0;
            for (uint64_t k = 0; k < m1.m; ++k) {
                tmp += m1(i, k) * m2(k, j);
            }
            ans(i, j) =  tmp;
        }
    }
    return ans;
}

std::istream &operator>> (std::istream &input, Matrix &matrix) {
    for (uint64_t i = 0; i < matrix.n; ++i) {
        for (uint64_t j = 0; j < matrix.m; ++j) {
            input >> matrix.data[i * matrix.m + j];
        }
    }
    return input;
}

std::ostream &operator<< (std::ostream &output, const Matrix &matrix) {
    for (uint64_t i = 0; i < matrix.n; ++i) {
        for (uint64_t j = 0; j < matrix.m - 1; ++j) {
            output << matrix.data[i * matrix.m + j] << ' ';
        }
        output << matrix.data[(i + 1) * matrix.m - 1] << '\n';
    }
    return output;
}