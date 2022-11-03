#include "Matrix.cuh"

struct Comparator {
    __host__ __device__ bool operator()(double num1, double num2) {
        return fabs(num1) < fabs(num2);
    }
};

__global__ void swapRows (double *data, uint64_t n, uint64_t i, uint64_t j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    double tmp;
    for (uint64_t k = idx; k < 2 * n; k += offsetX) {
        tmp = data[k * n + i];
        data[k * n + i] = data[k * n + j];
        data[k * n + j] = tmp;
    }
}

__global__ void normalisation (double *data, uint64_t n, uint64_t i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    //double coeff = data[n * i + i];
    for (uint64_t k = idx + i + 1; k < 2 * n; k += offsetX) {
        data[k * n + i] /= data[i * n + i];;
    }
}

__global__ void iteration (double *data, uint64_t n, uint64_t id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = gridDim.x * blockDim.x;
    int offsetY = gridDim.y * blockDim.y;

    for (uint64_t i = idx + id + 1; i < 2 * n; i += offsetX) {
        //double coeff = data[id * n + i];
        for (uint64_t j = idy + id + 1; j < n; j += offsetY) {
            data[i * n + j] -= data[i * n + id] * data[id * n + j];
        }
    }
}

// __global__ void backIteration (double *data, uint64_t n, uint64_t id) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;
//     int offsetX = gridDim.x * blockDim.x;
//     int offsetY = gridDim.y * blockDim.y;

//     for (uint64_t i = idx + id; i < n; i += offsetX) {
//         double coeff = data[n * (n - id + 1) - i - 1] / data[n * (n - id + 1) - id];
//         for (uint64_t j = idy; j < 2 * n; j += offsetY) {
//             data[n * (j + 1) - i - 1] -= data[n * (j + 1) - id] * coeff;
//         }
//     }
// }

__global__ void backIteration (double *data, uint64_t n, uint64_t id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = gridDim.x * blockDim.x;
    int offsetY = gridDim.y * blockDim.y;

    for (uint64_t i = idx + id + 1; i < 2 * n; i += offsetX) {
        //double coeff = data[n * (n - id + 1) - i - 1] / data[n * (n - id + 1) - id];
        //double coeff = 1;
        for (uint64_t j = idy; j <= id - 1; j += offsetY) {
            //data[n * (j + 1) - i - 1] -= data[n * (j + 1) - id] * coeff;
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

Matrix Matrix::reverse () const {
    Matrix ans(n), tmp(2 * n, n);
    double *data;

    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            tmp(j, i) = this->operator()(i, j);
        }
    }
    for (uint64_t i = 0; i < n; ++i) {
        tmp(i + n, i) = 1;
    }

    gpuErrorCheck(cudaMalloc(&data, sizeof(double) * n * n * 2));
    gpuErrorCheck(cudaMemcpy(data, tmp.data.data(), sizeof(double) * n * n * 2, cudaMemcpyHostToDevice));

    Comparator check;

    for (uint64_t i = 0; i < n - 1; ++i) {
        thrust::device_ptr<double> device_data = thrust::device_pointer_cast(data + i * n);
        thrust::device_ptr<double> max = thrust::max_element(device_data + i, device_data + n, check);
        uint64_t idx = max - device_data;

        if (i != idx) {
            swapRows<<<256, 256>>>(data, n, i, idx);
            gpuErrorCheck(cudaGetLastError());
            gpuErrorCheck(cudaDeviceSynchronize());
        }

        normalisation<<<256, 256>>>(data, n, i);
        gpuErrorCheck(cudaGetLastError());
        gpuErrorCheck(cudaDeviceSynchronize());

        iteration<<<256, 256>>>(data, n, i);
        gpuErrorCheck(cudaGetLastError());
        gpuErrorCheck(cudaDeviceSynchronize());
    }
    normalisation<<<256, 256>>>(data, n, n - 1);
    gpuErrorCheck(cudaGetLastError());
    gpuErrorCheck(cudaDeviceSynchronize());

    for (uint64_t i = n - 1; i > 0; --i) {
        backIteration<<<256, 256>>>(data, n, i);
        gpuErrorCheck(cudaGetLastError());
        gpuErrorCheck(cudaDeviceSynchronize());
    }

    gpuErrorCheck(cudaMemcpy(&tmp.data[0], data, sizeof(double) * n * n * 2, cudaMemcpyDeviceToHost));
    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            ans(i, j) = tmp(j + n, i);
        }
    }

    gpuErrorCheck(cudaFree(data));
    return ans;
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