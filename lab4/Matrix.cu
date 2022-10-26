#include "Matrix.cuh"

__global__ void swapRows (double *old_data, double *new_data, uint64_t n, uint64_t i, uint64_t j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    for (uint64_t k = idx; k < n; k += offsetX) {
        thrust::swap(old_data[k * n + i], old_data[k * n + j]);
        thrust::swap(new_data[k * n + i], new_data[k * n + j]);
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
        double coeff = old_data[i + n * id];
        for (uint64_t j = idy; j < n; j += offsetY) {
            old_data[i + n * j] -= old_data[id + n * j] * coeff;
            new_data[i + n * j] -= new_data[id + n * j] * coeff;
        }
    }
}

__global__ void backIteration (double *old_data, double *new_data, uint64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = gridDim.x * blockDim.x;
    int offsetY = gridDim.y * blockDim.y;

    for (uint64_t i = idx + 1; i < n; i += offsetX) {
        for (uint64_t j = idy + i; j < n; j += offsetY) {
            double coeff = old_data[n * (n - i + 1) - j - 1];
            for (uint64_t k = 0; k < n; ++k) {
                old_data[n * (k + 1) - j - 1] -= old_data[n * (k + 1) - i] * coeff;
                new_data[n * (k + 1) - j - 1] -= new_data[n * (k + 1) - i] * coeff;
            }
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

    auto check = [] __host__ __device__ (double num1, double num2) -> bool {
        return fabs(num1) < fabs(num2);
    };
// 3
// 3 4 5
// 9 8 2
// 0 4 5
    for (uint64_t i = 0; i < n; ++i) {
        Matrix tmp1(n), tmp2(n);
        thrust::device_ptr<double> data_ptr = thrust::device_pointer_cast(old_data + i * n);
        thrust::device_ptr<double> max_ptr = thrust::max_element(data_ptr + i, data_ptr + n, check);
        uint64_t idx = max_ptr - data_ptr;
        swapRows<<<1024, 1024>>>(old_data, new_data, n, i, idx);
        gpuErrorCheck(cudaGetLastError());
        gpuErrorCheck(cudaThreadSynchronize());
        gpuErrorCheck(cudaMemcpy(&tmp1.data[0], old_data, sizeof(double) * n * m, cudaMemcpyDeviceToHost));
        gpuErrorCheck(cudaMemcpy(&tmp2.data[0], new_data, sizeof(double) * n * m, cudaMemcpyDeviceToHost));
        std::cout << "After swap:\n" << tmp1 << "\n" << tmp2 << "\n";

        normalisation<<<1024, 1024>>>(old_data, new_data, n, i);
        gpuErrorCheck(cudaGetLastError());
        gpuErrorCheck(cudaThreadSynchronize());
        gpuErrorCheck(cudaMemcpy(&tmp1.data[0], old_data, sizeof(double) * n * m, cudaMemcpyDeviceToHost));
        gpuErrorCheck(cudaMemcpy(&tmp2.data[0], new_data, sizeof(double) * n * m, cudaMemcpyDeviceToHost));
        std::cout << "After normalise:\n" << tmp1 << "\n" << tmp2 << "\n";

        iteration<<<1024, 1024>>>(old_data, new_data, n, i);
        gpuErrorCheck(cudaGetLastError());
        gpuErrorCheck(cudaThreadSynchronize());
        gpuErrorCheck(cudaMemcpy(&tmp1.data[0], old_data, sizeof(double) * n * m, cudaMemcpyDeviceToHost));
        gpuErrorCheck(cudaMemcpy(&tmp2.data[0], new_data, sizeof(double) * n * m, cudaMemcpyDeviceToHost));
        std::cout << "After iter:\n" << tmp1 << "\n" << tmp2 << "\n";
    }
    backIteration<<<1024, 1024>>>(old_data, new_data, n);
    gpuErrorCheck(cudaGetLastError());
    gpuErrorCheck(cudaThreadSynchronize());

    //ans.data.resize(n * m);
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
            //input >> matrix.data[i * matrix.m + j];
        }
    }
    return input;
}

std::ostream &operator<< (std::ostream &output, const Matrix &matrix) {
    for (uint64_t i = 0; i < matrix.n; ++i) {
        for (uint64_t j = 0; j < matrix.m - 1; ++j) {
            output << matrix.data[i + j * matrix.n] << ' ';
            //output << matrix.data[i * matrix.m + j] << ' ';
        }
        output << matrix.data[(matrix.m - 1) * matrix.n + i] << '\n';
    }
    return output;
}