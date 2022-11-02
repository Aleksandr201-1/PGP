#include "Matrix.cuh"

struct Comparator {
    __host__ __device__ bool operator()(double num1, double num2) {
        return fabs(num1) < fabs(num2);
    }
};

__global__ void swapRows (double *old_data, double *new_data, uint64_t n, uint64_t i, uint64_t j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    double tmp;
    for (uint64_t k = idx; k < n; k += offsetX) {
        tmp = old_data[k * n + i];
        old_data[k * n + i] = old_data[k * n + j];
        old_data[k * n + j] = tmp;

        tmp = new_data[k * n + i];
        new_data[k * n + i] = new_data[k * n + j];
        new_data[k * n + j] = tmp;
    }
}

__global__ void normalisation (double *old_data, double *new_data, uint64_t n, uint64_t i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    double coeff = old_data[n * i + i];
    for (uint64_t k = idx; k < n; k += offsetX) {
        old_data[k * n + i] /= coeff;
        new_data[k * n + i] /= coeff;
    }
}

__global__ void iteration (double *old_data, double *new_data, uint64_t n, uint64_t id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = gridDim.x * blockDim.x;
    int offsetY = gridDim.y * blockDim.y;

    for (uint64_t i = idx + id + 1; i < n; i += offsetX) {
        double coeff = old_data[i + n * id] / old_data[id + n * id];
        for (uint64_t j = idy; j < n; j += offsetY) {
            old_data[i + n * j] -= old_data[id + n * j] * coeff;//old_data[i + n * id];
            new_data[i + n * j] -= new_data[id + n * j] * coeff;//old_data[i + n * id];
        }
    }
    // for (uint64_t i = idx; i < n; i += offsetX) {
    //     double coeff = old_data[i + n * id] / old_data[id + n * id];
    //     for (uint64_t j = idy + id + 1; j < n; j += offsetY) {
    //         old_data[i * n + j] -= old_data[i * n + id] * coeff;//old_data[id * n + j];
    //         new_data[i * n + j] -= new_data[i * n + id] * coeff;//old_data[id * n + j];
    //     }
    // }
}

__global__ void backIteration (double *old_data, double *new_data, uint64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = gridDim.x * blockDim.x;
    int offsetY = gridDim.y * blockDim.y;

    for (uint64_t i = idx + 1; i < n; i += offsetX) {
        for (uint64_t j = idy + i; j < n; j += offsetY) {
            double coeff = old_data[n * (n - i + 1) - j - 1] / old_data[n * (n - i + 1) - i];
            for (uint64_t k = 0; k < n; ++k) {
                old_data[n * (k + 1) - j - 1] -= old_data[n * (k + 1) - i] * coeff;
                new_data[n * (k + 1) - j - 1] -= new_data[n * (k + 1) - i] * coeff;
            }
        }
    }
}

__global__ void backIteration (double *old_data, double *new_data, uint64_t n, uint64_t id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = gridDim.x * blockDim.x;
    int offsetY = gridDim.y * blockDim.y;

    //for (uint64_t i = idx + 1; i < n; i += offsetX) {
    for (uint64_t i = idx + id; i < n; i += offsetX) {
        double coeff = old_data[n * (n - id + 1) - i - 1] / old_data[n * (n - id + 1) - id];
        for (uint64_t j = idy; j < n; j += offsetY) {
            old_data[n * (j + 1) - i - 1] -= old_data[n * (j + 1) - id] * coeff;
            new_data[n * (j + 1) - i - 1] -= new_data[n * (j + 1) - id] * coeff;
        }
    }
    //}
}

// __global__ void backIteration (double *old_data, double *new_data, uint64_t n, uint64_t id) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;
//     int offsetX = gridDim.x * blockDim.x;
//     int offsetY = gridDim.y * blockDim.y;

//     for (uint64_t i = idx + id + 1; i < n; i += offsetY) {
// 		for (uint64_t j = idy; j <= id - 1; j += offsetX) {
// 			old_data[i * n + j] -= old_data[i * n + id] * old_data[id * n + j];
//             new_data[i * n + j] -= new_data[i * n + id] * old_data[id * n + j];
// 		}
// 	}
// }

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

Matrix::~Matrix () {}

double &Matrix::operator() (uint64_t i, uint64_t j) {
    return data[i * n + j];
}

double Matrix::operator() (uint64_t i, uint64_t j) const {
    return data[i * n + j];
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
    Matrix ans(n);
    double *old_data, *new_data;

    gpuErrorCheck(cudaMalloc(&old_data, sizeof(double) * n * m));
    gpuErrorCheck(cudaMalloc(&new_data, sizeof(double) * n * m));
    gpuErrorCheck(cudaMemcpy(old_data, data.data(), sizeof(double) * n * m, cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(new_data, ans.data.data(), sizeof(double) * n * m, cudaMemcpyHostToDevice));

    Comparator check;

    for (uint64_t i = 0; i < n - 1; ++i) {
        thrust::device_ptr<double> device_data = thrust::device_pointer_cast(old_data + i * n);
        thrust::device_ptr<double> max = thrust::max_element(device_data + i, device_data + n, check);
        uint64_t idx = max - device_data;
        if (i != idx) {
            swapRows<<<1024, 1024>>>(old_data, new_data, n, i, idx);
            gpuErrorCheck(cudaGetLastError());
            gpuErrorCheck(cudaThreadSynchronize());
        }

        normalisation<<<1024, 1024>>>(old_data, new_data, n, i);
        gpuErrorCheck(cudaGetLastError());
        gpuErrorCheck(cudaThreadSynchronize());

        iteration<<<1024, 1024>>>(old_data, new_data, n, i);
        gpuErrorCheck(cudaGetLastError());
        gpuErrorCheck(cudaThreadSynchronize());
    }
    // for (uint64_t i = 0; i < n; ++i) {
    //     normalisation<<<1024, 1024>>>(old_data, new_data, n, i);
    //     gpuErrorCheck(cudaGetLastError());
    //     gpuErrorCheck(cudaThreadSynchronize());
    // }
    normalisation<<<1024, 1024>>>(old_data, new_data, n, n - 1);
    gpuErrorCheck(cudaGetLastError());
    gpuErrorCheck(cudaThreadSynchronize());

//backIteration<<<1024, 1024>>>(old_data, new_data, n);
    for (uint64_t i = n - 1; i > 0; --i) {
        backIteration<<<1024, 1024>>>(old_data, new_data, n, i);
        gpuErrorCheck(cudaGetLastError());
        gpuErrorCheck(cudaThreadSynchronize());
    }

    gpuErrorCheck(cudaMemcpy(&ans.data[0], new_data, sizeof(double) * n * m, cudaMemcpyDeviceToHost));

    gpuErrorCheck(cudaFree(old_data));
    gpuErrorCheck(cudaFree(new_data));
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

void Matrix::printMatrix () const {
    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < m - 1; ++j) {
            printf("%.10e ", data[i + m * j]);
        }
        printf("%.10e\n", data[(m - 1) * n + i]);
    }
}

std::istream &operator>> (std::istream &input, Matrix &matrix) {
    for (uint64_t i = 0; i < matrix.n; ++i) {
        for (uint64_t j = 0; j < matrix.m; ++j) {
            input >> matrix.data[i + j * matrix.n];
        }
    }
    return input;
}

std::ostream &operator<< (std::ostream &output, const Matrix &matrix) {
    for (uint64_t i = 0; i < matrix.n; ++i) {
        for (uint64_t j = 0; j < matrix.m - 1; ++j) {
            output << matrix.data[i + j * matrix.n] << ' ';
        }
        output << matrix.data[(matrix.m - 1) * matrix.n + i] << '\n';
    }
    return output;
}