#include "Matrix.cuh"

//__device__ double max_val[1];

struct Comparator {
    __host__ __device__ bool operator()(double num1, double num2) {
        return fabs(num1) < fabs(num2);
    }
};

__global__ void colInit (double *data, double *col, uint64_t n, uint64_t i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    //__shared__ double t[256];
    //t[threadIdx.x] = data[threadIdx.x * n * 2 + i];
    //unsigned int row_idx= threadIdx.y * blockDim.x + threadIdx.x;
    //extern __shared__ double tmp[];
    //tmp[threadIdx.x] = data[idx];
    //__syncthreads();
    for (uint64_t k = idx + i; k < n; k += offsetX) {
        col[k] = data[k * n + i];
    }
}

__global__ void swapRows (double *data, double *new_data, uint64_t n, uint64_t i, uint64_t j, uint64_t border) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    //__shared__ double t[512];
    //t[threadIdx.x] = data[i * n * 2 + threadIdx.x];
    double tmp;
    //for (uint64_t k = idx + i; k < n + border + 1; k += offsetX) {
    for (uint64_t k = idx + i; k < n; k += offsetX) {
        tmp = data[i * n + k];
        data[i * n + k] = data[j * n + k];
        data[j * n + k] = tmp;//t[threadIdx.x];
    }
    for (uint64_t k = idx; k < border + 1; k += offsetX) {
        tmp = new_data[i * n + k];
        new_data[i * n + k] = new_data[j * n + k];
        new_data[j * n + k] = tmp;//t[threadIdx.x];
    }
}

__global__ void swapRowsO (double *data, uint64_t n, uint64_t i, uint64_t j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    double tmp;
    for (uint64_t k = idx + i; k < n; k += offsetX) {
        tmp = data[i * n + k];
        data[i * n + k] = data[j * n + k];
        data[j * n + k] = tmp;
    }
}

__global__ void swapRowsN (double *new_data, uint64_t n, uint64_t i, uint64_t j, uint64_t border) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    double tmp;
    for (uint64_t k = idx; k < border + 1; k += offsetX) {
        tmp = new_data[i * n + k];
        new_data[i * n + k] = new_data[j * n + k];
        new_data[j * n + k] = tmp;//t[threadIdx.x];
    }
}

__global__ void normalisation (double *data, double *new_data, uint64_t n, uint64_t id, uint64_t border) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    double coeff;
    coeff = data[id * n + id];
    //for (uint64_t k = idx + id + 1; k < n + border + 1; k += offsetX) {
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
    //__shared__ double coef[256];
    //coef[threadIdx.x] = data[id * n * 2 + threadIdx.x];
    //__syncthreads();

    //for (uint64_t i = idx + id + 1; i < n; i += offsetX) {
    for (uint64_t i = idx + id + 1; i < n; i += offsetX) {
        coeff = data[i * n + id];
        //for (uint64_t j = idy + id + 1; j < n + border + 1; j += offsetY) {
        for (uint64_t j = idy + id + 1; j < n; j += offsetY) {
            data[i * n + j] -= coeff *  data[id * n + j];
        }
        for (uint64_t j = idy; j < border + 1; j += offsetY) {
            new_data[i * n + j] -= coeff * new_data[id * n + j];
        }
    }
}

__global__ void unstable (double *data, double *new_data, uint64_t n, uint64_t id, uint64_t border) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = gridDim.x * blockDim.x;
    int offsetY = gridDim.y * blockDim.y;

    double coeff;
    double div = data[id * n + id];

    //for (uint64_t i = idx + id + 1; i < n; i += offsetX) {
    for (uint64_t i = idx + id + 1; i < n; i += offsetX) {
        coeff = data[i * n + id];
        //for (uint64_t j = idy + id + 1; j < n + border + 1; j += offsetY) {
        for (uint64_t j = idy + id + 1; j < n; j += offsetY) {
            data[id * n + j] /= div;
            data[i * n + j] -= coeff * data[id * n + j];
        }
        for (uint64_t j = idy; j < border + 1; j += offsetY) {
            new_data[id * n + j] /= div;
            new_data[i * n + j] -= coeff * new_data[id * n + j];
        }
    }
}

__global__ void unstable_with_swap (double *data, uint64_t n, uint64_t id1, uint64_t id2, uint64_t border) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = gridDim.x * blockDim.x;
    int offsetY = gridDim.y * blockDim.y;

    double tmp;
    // if (idx == 0) {
    //     tmp = data[id1 * n + id1];
    //     data[id1 * n + id1] = data[id1 * n + id2];
    //     data[id1 * n + id2] = tmp;
    // }
    for (uint64_t i = idx + id1 + 1; i < n + border + 1; i += offsetX) {
        //swap
        tmp = data[i * n + id1];
        data[i * n + id1] = data[i * n + id2];
        data[i * n + id2] = tmp;
        //normalisation
        data[i * n + id1] /= data[id1 * n + id1];
        //iteration
        for (uint64_t j = idy + id1 + 1; j < n; j += offsetY) {
            data[i * n + j] -= data[i * n + id1] * data[id1 * n + j];
        }
    }
}

__global__ void backIteration (double *data, double *new_data, uint64_t n, uint64_t id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = gridDim.x * blockDim.x;
    int offsetY = gridDim.y * blockDim.y;

    double coeff;
    //for (uint64_t i = idx; i <= id - 1; i += offsetX) {
    for (uint64_t i = idx; i <= id - 1; i += offsetX) {
        coeff = data[i * n + id];
        //for (uint64_t j = idy + n; j < 2 * n; j += offsetY) {
        for (uint64_t j = idy; j < n; j += offsetY) {
            new_data[i * n + j] -= coeff * new_data[id * n + j];
        }
    }
}

__global__ void backIterationU (double *data, double *new_data, uint64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    //int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int offsetX = gridDim.x * blockDim.x;
    int offsetY = gridDim.y * blockDim.y;
    //int offsetZ = gridDim.z * blockDim.z;

    double coeff;
    for (uint64_t id = n - 1; id > 0; --id) {
        for (uint64_t i = idx; i <= id - 1; i += offsetX) {
            coeff = data[i * n + id];
            //for (uint64_t j = idy + n; j < 2 * n; j += offsetY) {
            for (uint64_t j = idy; j < n; j += offsetY) {
                new_data[i * n + j] -= coeff * new_data[id * n + j];
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
    Matrix ans(n);//, tmp(n, 2 * n);
    double *old_data, *new_data, *col;


    float *buffer;
    cudaMalloc(&buffer, n * n * sizeof(double));
    // create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = buffer;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = n * n * sizeof(double);
    //
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    // for (uint64_t i = 0; i < n; ++i) {
    //     memcpy(tmp.data.data() + i * n * 2, data.data() + i * n, n * sizeof(double));
    // }
    // for (uint64_t i = 0; i < n; ++i) {
    //     tmp(i, i + n) = 1;
    // }

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
        //std::cout << "max idx: " << idx << '\n';
        //double max = *(device_data + idx);
        //gpuErrorCheck(cudaMemcpyToSymbol(max_val, &max, sizeof(double)));
        //std::cout << *(device_data + idx) << '\n';

        border = std::max(border, idx);
        if (i != idx) {
            swapRows<<<256, 256>>>(old_data, new_data, n, i, idx, border);
            //gpuErrorCheck(cudaGetLastError());
            //swapRowsO<<<256, 256>>>(old_data, n, i, idx);
            //gpuErrorCheck(cudaGetLastError());
            //swapRowsN<<<256, 256>>>(new_data, n, i, idx, border);
            //gpuErrorCheck(cudaGetLastError());
            // thrust::device_ptr<double> val1 = thrust::device_pointer_cast(old_data + i * n + i);
            // thrust::device_ptr<double> val2 = thrust::device_pointer_cast(old_data + i * n + idx);
            // thrust::swap(*val1, *val2);
            //unstable_with_swap<<<256, 256>>>(old_data, n, i, idx, border);
            //gpuErrorCheck(cudaGetLastError());
        }// else {
            //unstable<<<256, 256>>>(old_data, n, i, border);
            //gpuErrorCheck(cudaGetLastError());
        //}
        //unstable<<<256, 256>>>(old_data, n, i, border);
        //gpuErrorCheck(cudaGetLastError());
        // gpuErrorCheck(cudaMemcpy(&tmp.data[0], old_data, sizeof(double) * n * n * 2, cudaMemcpyDeviceToHost));
        // std::cout << "after swap\n" << tmp << '\n';

        normalisation<<<256, 256>>>(old_data, new_data, n, i, border);
        //gpuErrorCheck(cudaGetLastError());
        // gpuErrorCheck(cudaMemcpy(&tmp.data[0], old_data, sizeof(double) * n * n * 2, cudaMemcpyDeviceToHost));
        // std::cout << "after norm\n" << tmp << '\n';

        iteration<<<256, 256>>>(old_data, new_data, n, i, border);
        //unstable<<<256, 256>>>(old_data, new_data, n, i, border);
        //gpuErrorCheck(cudaGetLastError());
        // gpuErrorCheck(cudaMemcpy(&tmp.data[0], old_data, sizeof(double) * n * n * 2, cudaMemcpyDeviceToHost));
        // std::cout << "after iter\n" << tmp << '\n';

        // unstable<<<256, 256>>>(old_data, n, i, border);
        // gpuErrorCheck(cudaGetLastError());
    }
    normalisation<<<256, 256>>>(old_data, new_data, n, n - 1, n - 1);
    //gpuErrorCheck(cudaGetLastError());

    // for (uint64_t i = n - 1; i > 0; --i) {
    //     backIteration<<<256, 256>>>(old_data, new_data, n, i);
    //     gpuErrorCheck(cudaGetLastError());
    // }
    backIterationU<<<256, 256>>>(old_data, new_data, n);
    // gpuErrorCheck(cudaGetLastError());

    gpuErrorCheck(cudaMemcpy(&ans.data[0], new_data, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
    // for (uint64_t i = 0; i < n; ++i) {
    //     memcpy(ans.data.data() + i * n, tmp.data.data() + i * n * 2 + n, n * sizeof(double));
    // }

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

void Matrix::printMatrix () const {
    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < m - 1; ++j) {
            printf("%.10e ", data[i * m + j]);
        }
        printf("%.10e\n", data[(i + 1) * m - 1]);
    }
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