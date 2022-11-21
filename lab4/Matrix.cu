#include "Matrix.cuh"

struct Comparator {
    __host__ __device__ bool operator()(double num1, double num2) {
        return fabs(num1) < fabs(num2);
    }
};

__global__ void colInit (double *data, double *col, uint64_t n, uint64_t i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    for (uint64_t k = idx + i; k < n; k += offsetX) {
        col[k] = data[k * n + i];
    }
}

__global__ void swapRows (double *data, double *new_data, uint64_t n, uint64_t i, uint64_t j, uint64_t border) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    double tmp;
    for (uint64_t k = idx + i; k < n; k += offsetX) {
        tmp = data[i * n + k];
        data[i * n + k] = data[j * n + k];
        data[j * n + k] = tmp;
    }
    for (uint64_t k = idx; k < border + 1; k += offsetX) {
        tmp = new_data[i * n + k];
        new_data[i * n + k] = new_data[j * n + k];
        new_data[j * n + k] = tmp;
    }
}

__global__ void normalisation (double *data, double *new_data, uint64_t n, uint64_t id, uint64_t border) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    double coeff;
    coeff = data[id * n + id];
    for (uint64_t k = idx + id + 1; k < n; k += offsetX) {
        data[id * n + k] /= coeff;
    }
    for (uint64_t k = idx; k < border + 1; k += offsetX) {
        new_data[id * n + k] /= coeff;
    }
}

__global__ void iteration (double *data, double *new_data, uint64_t n, uint64_t id, uint64_t border) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = gridDim.x * blockDim.x;
    int offsetY = gridDim.y * blockDim.y;

    double coeff;

    for (uint64_t i = idy + id + 1; i < n; i += offsetY) {
        coeff = data[i * n + id];
        for (uint64_t j = idx + id + 1; j < n; j += offsetX) {
            data[i * n + j] -= coeff *  data[id * n + j];
        }
        for (uint64_t j = idx; j < border + 1; j += offsetX) {
            new_data[i * n + j] -= coeff * new_data[id * n + j];
        }
    }
}

__global__ void backIteration (double *data, double *new_data, uint64_t n, uint64_t id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = gridDim.x * blockDim.x;
    int offsetY = gridDim.y * blockDim.y;

    double coeff;
    for (uint64_t i = idy; i <= id - 1; i += offsetY) {
        coeff = data[i * n + id];
        for (uint64_t j = idx; j < n; j += offsetX) {
            new_data[i * n + j] -= coeff * new_data[id * n + j];
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
    Matrix ans(n);
    double *old_data, *new_data, *col;

    gpuErrorCheck(cudaMalloc(&old_data, sizeof(double) * n * n));
    gpuErrorCheck(cudaMalloc(&new_data, sizeof(double) * n * n));
    gpuErrorCheck(cudaMalloc(&col, sizeof(double) * n));
    gpuErrorCheck(cudaMemcpy(old_data, data.data(), sizeof(double) * n * n, cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(new_data, ans.data.data(), sizeof(double) * n * n, cudaMemcpyHostToDevice));

    Comparator check;

    uint64_t border = 0;
    for (uint64_t i = 0; i < n - 1; ++i) {
        colInit<<<256, 256>>>(old_data, col, n, i);
        thrust::device_ptr<double> device_data = thrust::device_pointer_cast(col);
        uint64_t idx = thrust::max_element(device_data + i, device_data + n, check) - device_data;
        border = std::max(border, idx);
        if (i != idx) {
            swapRows<<<256, 256>>>(old_data, new_data, n, i, idx, border);
        }
        normalisation<<<256, 256>>>(old_data, new_data, n, i, border);
        iteration<<<dim3(32, 32), dim3(32, 32)>>>(old_data, new_data, n, i, border);
    }
    normalisation<<<256, 256>>>(old_data, new_data, n, n - 1, n - 1);
    for (uint64_t i = n - 1; i > 0; --i) {
        backIteration<<<dim3(32, 32), dim3(32, 32)>>>(old_data, new_data, n, i);
    }

    gpuErrorCheck(cudaMemcpy(&ans.data[0], new_data, sizeof(double) * n * n, cudaMemcpyDeviceToHost));

    gpuErrorCheck(cudaFree(old_data));
    gpuErrorCheck(cudaFree(new_data));
    gpuErrorCheck(cudaFree(col));
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