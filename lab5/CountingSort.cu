#include "CountingSort.cuh"

//__constant__ uint32_t count[DATA_MAX_EL + 1];

template <class T>
void printVector (const std::vector<T> &v) {
    for (uint64_t i = 0; i < v.size(); ++i) {
        std::cout << v[i] << ' ';
    }
    std::cout << '\n';
}

__global__ void CountingSortKernel (DATA_TYPE *data, uint32_t n) {
    uint64_t pos = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = 1;//gridDim.x * blockDim.x;
    uint8_t el = data[idx];

    for (uint64_t i = 0; i < n; i += offsetX) {
        //uint8_t el = data[i];
        //uint64_t pos = 0;
        pos += ((data[i] - el) & (1 << 7)) >> 7;
        // if (el > data[i]) {
        //     ++pos;
        // }
    }
    __syncthreads();
    data[pos] = el;
}

std::vector<uint32_t> Count (const std::vector<uint8_t> &data) {
    std::vector<uint32_t> ans(DATA_MAX_EL + 1, 0);
    for (uint32_t i = 0; i < data.size(); ++i) {
        //std::cout << "el: " << data[i] << '\n';
        ++ans[data[i]];
    }
    ans[DATA_MAX_EL] = data.size();
    for (uint32_t i = 1; i < data.size(); ++i) {
        //std::cout << "el: " << data[i] << '\n';
        ans[i] += ans[i - 1];
    }
    return ans;
}

__global__ void histogram (uint64_t *count, uint64_t n) {}

__global__ void scan (uint64_t *count, uint64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;
    uint64_t step = 2;
    while (step <= n) {
        for (uint64_t i = idx; i < n / step; i += offsetX) {
            count[(i + 1) * step - 1] += count[i * step  + step / 2 - 1];
        }
        step *= 2;
        __syncthreads();
    }
    while (step >= 2) {
        uint64_t tmp;
        for (uint64_t i = idx; i < n / step; i += offsetX) {
            tmp = count[(i + 1) * step - 1] + count[i * step  + step / 2 - 1];
            count[i * step  + step / 2 - 1] = count[(i + 1) * step - 1];
            count[(i + 1) * step - 1] = tmp;
        }
        step /= 2;
        __syncthreads();
    }
}

__global__ void CountingSortWrite (DATA_TYPE *data, uint64_t *count, uint64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetY = gridDim.y * blockDim.y;

    for (uint64_t i = idx; i < DATA_MAX_EL; i += offsetX) {
        for (uint64_t j = idy + count[i]; j < count[i + 1]; j += offsetY) {
            data[j] = i;
        }
    }
}

Array::Array () : n(0) {}

Array::Array (Array const &array) {
    n = array.n;
    data = array.data;
}

Array::Array (Array &&array) {
    n = array.n;
    data = std::move(array.data);
}

Array::Array (uint32_t n, const std::vector<DATA_TYPE> &data) : n(n), data(data) {}

Array::~Array () {}

void Array::CountingSort () {
    std::cout << "Sorting\n";
    std::cout << "size: " << (DATA_TYPE)(-1) << '\n';
    auto v = Count(data);
    printVector(data);
    //printVector(v);
    DATA_TYPE *old_data;

    gpuErrorCheck(cudaMalloc(&old_data, sizeof(DATA_TYPE) * n));
    gpuErrorCheck(cudaMemcpy(old_data, data.data(), sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));

    CountingSortKernel<<<256, 256>>>(old_data, n);
    gpuErrorCheck(cudaGetLastError());

    gpuErrorCheck(cudaMemcpy(&data[0], old_data, sizeof(DATA_TYPE) * n, cudaMemcpyDeviceToHost));

    gpuErrorCheck(cudaFree(old_data));
}

std::istream &operator>> (std::istream &input, Array &array) {
    input >> array.n;
    uint32_t &n = array.n;
    //uint32_t v1 = n & 255, v2 = (n >> 8) & 255, v3 = (n >> 16) & 255, v4 = (n >> 24) & 255;
    //n = (v1 << 24) + (v2 << 16) + (v3 << 8) + v4;
    //std::cout << n << "\n";
    //exit(0);
    array.data.resize(array.n);
    uint f;
    for (uint32_t i = 0; i < array.n; ++i) {
        input >> array.data[i];
    }
    return input;
}

std::ostream &operator<< (std::ostream &output, const Array &array) {
    for (uint32_t i = 0; i < array.n - 1; ++i) {
        output << array.data[i] << ' ';
    }
    output << array.data.back() << '\n';
    return output;
}

Array& Array::operator= (const Array &array) {
    if (&array == this) {
        return *this;
    }
    n = array.n;
    data = array.data;
    return *this;
}

Array& Array::operator= (Array &&array) {
    if (&array == this) {
        return *this;
    }
    n = array.n;
    data = std::move(array.data);
    return *this;
}