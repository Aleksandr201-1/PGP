#include "CountingSort.cuh"

const uint32_t BLOCK_SIZE = 1024;

__host__ __device__ uint32_t index (uint32_t num) {
    return num + (num >> 5);
}

__global__ void countInit (uint32_t *count) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t offsetX = gridDim.x * blockDim.x;

    for (uint32_t i = idx; i < DATA_MAX_EL + 1; i += offsetX) {
        count[i] = 0;
    }
}

__global__ void histogram (uint32_t *count, DATA_TYPE *data, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t offsetX = gridDim.x * blockDim.x;

    for (uint32_t i = idx; i < n; i += offsetX) {
        atomicAdd(count + data[i], 1);
    }
}

__global__ void addKernel(uint32_t* data, uint32_t* sums, uint32_t sums_size) {
    uint32_t blockId = blockIdx.x + 1;
    while (blockId < sums_size) {
        uint32_t idx = blockId * blockDim.x + threadIdx.x;
        if (blockId != 0) {
            data[idx] += sums[blockId];
        }
        blockId += gridDim.x;
    }
}

__global__ void scanKernel(uint32_t* data, uint32_t* sums, uint32_t sums_size) {
    uint32_t blockId = blockIdx.x;
    //uint32_t idx = blockId * blockDim.x + threadIdx.x;
    while (blockId < sums_size) {
        __shared__ uint32_t tmp[1024];
        uint32_t step = 1, a, b, tmp_el;
        tmp[index(threadIdx.x)] = data[blockId * blockDim.x + threadIdx.x];
        __syncthreads();
        while (step < BLOCK_SIZE) {
            if ((threadIdx.x + 1) % (step << 1) == 0) {
                a = index(threadIdx.x);
                b = index(threadIdx.x - step);
                tmp[a] += tmp[b];
            }
            step <<= 1;
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            sums[blockId] = tmp[index(BLOCK_SIZE - 1)];
            tmp[index(BLOCK_SIZE - 1)] = 0;
        }
        __syncthreads();
        step = 1 << 10 - 1;
        while (step >= 1) {
            if ((threadIdx.x + 1) % (step << 1) == 0) {
                a = index(threadIdx.x);
                b = index(threadIdx.x - step);
                tmp_el = tmp[a];
                tmp[a] += tmp[b];
                tmp[b] = tmp_el;
            }
            step >>= 1;
            __syncthreads();
        }
        data[blockId * blockDim.x + threadIdx.x] = tmp[index(threadIdx.x)];
        blockId += gridDim.x;
    }
}

__global__ void CountingSortWrite (DATA_TYPE *data, uint32_t *count, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t offsetX = gridDim.x * blockDim.x;
    uint32_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t offsetY = gridDim.y * blockDim.y;

    for (uint32_t i = idx; i < DATA_MAX_EL; i += offsetX) {
        uint32_t k = count[i];
        for (uint32_t j = idy + k; j < count[i + 1]; j += offsetY) {
            data[j] = i;
        }
    }
    for (uint32_t i = idx + count[DATA_MAX_EL]; i < size; i += offsetX) {
        data[i] = DATA_MAX_EL;
    }
}

void Array::scan(uint32_t* data, uint32_t n) {
    if (n % BLOCK_SIZE != 0) {
        n += BLOCK_SIZE - (n % BLOCK_SIZE);
    }
    uint32_t sums_size = n / BLOCK_SIZE;
    uint32_t* sums;
    gpuErrorCheck(cudaMalloc(&sums, (sums_size * sizeof(uint32_t))));
    scanKernel<<<1024, 1024, sizeof(uint32_t) * index(BLOCK_SIZE)>>> (data, sums, sums_size);
    gpuErrorCheck(cudaPeekAtLastError());
    gpuErrorCheck(cudaDeviceSynchronize());

    if (n <= BLOCK_SIZE) {
        return;
    }
    scan(sums, sums_size);

    addKernel<<<1024, 1024>>> (data, sums, sums_size);
    gpuErrorCheck(cudaPeekAtLastError());
    gpuErrorCheck(cudaDeviceSynchronize());

    cudaFree(sums);
}

Array::Array () {}

Array::Array (Array const &array) {
    data = array.data;
}

Array::Array (Array &&array) {
    data = std::move(array.data);
}

Array::Array (const std::vector<DATA_TYPE> &data) : data(data) {}

Array::~Array () {}

void Array::CountingSort () {
    uint32_t n = data.size();
    DATA_TYPE *old_data;
    uint32_t *count;

    if (n == 0) {
        return;
    }

    gpuErrorCheck(cudaMalloc(&old_data, sizeof(DATA_TYPE) * n));
    gpuErrorCheck(cudaMalloc(&count, sizeof(uint32_t) * (DATA_MAX_EL + 1)));
    gpuErrorCheck(cudaMemcpy(old_data, data.data(), sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));

    countInit<<<1024, 1024>>>(count);
    gpuErrorCheck(cudaPeekAtLastError());
    gpuErrorCheck(cudaDeviceSynchronize());

    histogram<<<1024, 1024>>>(count, old_data, n);
    gpuErrorCheck(cudaPeekAtLastError());
    gpuErrorCheck(cudaDeviceSynchronize());

    scan(count, DATA_MAX_EL);

    CountingSortWrite<<<dim3(32, 32), dim3(32, 32)>>>(old_data, count, n);
    gpuErrorCheck(cudaPeekAtLastError());
    gpuErrorCheck(cudaDeviceSynchronize());

    gpuErrorCheck(cudaMemcpy(&data[0], old_data, sizeof(DATA_TYPE) * n, cudaMemcpyDeviceToHost));

    gpuErrorCheck(cudaFree(old_data));
    gpuErrorCheck(cudaFree(count));
}

std::istream &operator>> (std::istream &input, Array &array) {
    uint32_t n;
    input.read(reinterpret_cast<char *>(&n), sizeof(n));
    //input >> n;
    array.data.resize(n);
    // for (uint32_t i = 0; i < n; ++i) {
    //     input >> array.data[i];
    // }
    input.read(reinterpret_cast<char *>(&array.data[0]), sizeof(DATA_TYPE) * n);
    return input;
}

std::ostream &operator<< (std::ostream &output, const Array &array) {
    if (array.data.size() != 0) {
        output.write(reinterpret_cast<const char *>(&array.data[0]), sizeof(DATA_TYPE) * array.data.size());
        // for (uint32_t i = 0; i < array.data.size() - 1; ++i) {
        //     output << array.data[i] << ' ';
        // }
        // output << array.data.back() << '\n';
    }
    return output;
}

Array& Array::operator= (const Array &array) {
    if (&array == this) {
        return *this;
    }
    data = array.data;
    return *this;
}

Array& Array::operator= (Array &&array) {
    if (&array == this) {
        return *this;
    }
    data = std::move(array.data);
    return *this;
}