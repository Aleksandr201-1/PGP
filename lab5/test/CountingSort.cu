#include "CountingSort.cuh"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
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

__global__ void histogram (uint32_t *count, uint32_t *data, uint32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    for (uint32_t i = idx; i < n; i += offsetX) {
        atomicAdd(&count[data[i]], 1);
    }
}

__global__ void scanRec (uint32_t *count, uint32_t start, uint32_t prev_sum_idx) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t offsetX = blockDim.x * gridDim.x;
    uint32_t step = 1, a, b, tmp_el;
    extern __shared__ uint32_t tmp[];
    __shared__ uint32_t first;
    if (idx == 0) {
        first = count[start + BLOCK_SIZE - 1];
        if (prev_sum_idx != (uint32_t)-1) {
            count[start] += count[prev_sum_idx];
        }
    }
    for (uint32_t i = idx; i < BLOCK_SIZE; i += offsetX) {
        tmp[i] = count[start + i];
    }
    __syncthreads();
    while (step < BLOCK_SIZE) {
    // for (uint32_t n = BLOCK_SIZE >> 1; n > 0; n >>= 1) {
        for (uint32_t i = idx; i < BLOCK_SIZE / (step * 2); i += offsetX) {
            b = step * (2 * i + 1) - 1;
            a = step * (2 * i + 2) - 1;
            tmp[a] += tmp[b];
        }
        step <<= 1;
        __syncthreads();
    // }
    }
    if (idx == 0) {
        tmp[BLOCK_SIZE - 1] = 0;
    }
    step >>= 1;
    while (step >= 1) {
        for (uint32_t i = idx; i < BLOCK_SIZE / (step * 2); i += offsetX) {
            b = step * (2 * i + 1) - 1;
            a = step * (2 * i + 2) - 1;
            // a = (i + 1) * step - 1;
            // b = i * step  + step / 2 - 1;
            tmp_el = tmp[a] + tmp[b];
            tmp[b] = tmp[a];
            tmp[a] = tmp_el;
        }
        step >>= 1;
        __syncthreads();
    }
    for (uint32_t i = idx; i < BLOCK_SIZE - 1; i += offsetX) {
        count[start + i] = tmp[i + 1];
    }
    //if (idx == 0 && prev_sum_idx != DATA_MAX_EL) {
    if (idx == 0) {
        //count[start] = count[prev_sum_idx];
        count[start + BLOCK_SIZE - 1] = first + count[start + BLOCK_SIZE - 2];
    }
    __syncthreads();
}


__global__ void CountingSortWrite (DATA_TYPE *data, uint32_t *count, uint32_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;
    int idy = 0;
    int offsetY = 1;

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

void Array::scan(uint32_t* data, uint32_t n) {
    if (n % BLOCK_SIZE != 0) {
        n += BLOCK_SIZE - (n % BLOCK_SIZE);
    }
    uint32_t sums_size = n / BLOCK_SIZE;
    uint32_t* sums;
    gpuErrorCheck(cudaMalloc(&sums, (sums_size * sizeof(uint32_t))));
    scanKernel<<<1, 32, sizeof(uint32_t) * index(BLOCK_SIZE)>>> (data, sums, sums_size);
    gpuErrorCheck(cudaPeekAtLastError());
    gpuErrorCheck(cudaDeviceSynchronize());

    if (n <= BLOCK_SIZE) {
        return;
    }
    scan(sums, sums_size);

    addKernel<<<1, 32>>> (data, sums, sums_size);
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

void printRes (uint32_t *count, uint32_t size) {
    std::vector<uint32_t> c;
    c.resize(size);
    gpuErrorCheck(cudaMemcpy(&c[0], count, sizeof(uint32_t) * (DATA_MAX_EL + 1), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < std::min(10lu, c.size()); ++i) {
        std::cout << i << ": " << c[i] << "\n";
    }
    std::cout << "\n";
    for (uint32_t i = BLOCK_SIZE - 3; i < BLOCK_SIZE + 10; ++i) {
        std::cout << i << ": " << c[i] << "\n";
    }
    std::cout << "\n";
}

uint64_t Array::CountingSortGPU () {
    uint32_t n = data.size();
    DATA_TYPE *old_data;
    uint32_t *count;

    uint64_t time = 0;
    float elapsedTime;
    cudaEvent_t e_start, e_stop;

	cudaEventCreate(&e_start);
	cudaEventCreate(&e_stop);

    gpuErrorCheck(cudaMalloc(&old_data, sizeof(DATA_TYPE) * n));
    gpuErrorCheck(cudaMalloc(&count, sizeof(uint32_t) * (DATA_MAX_EL + 1)));
    gpuErrorCheck(cudaMemcpy(old_data, data.data(), sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));

    cudaEventRecord(e_start, 0);
    countInit<<<1, 32>>>(count);
    gpuErrorCheck(cudaPeekAtLastError());
    gpuErrorCheck(cudaDeviceSynchronize());
    cudaEventRecord(e_stop, 0);
    cudaEventSynchronize(e_stop);
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    time += (uint64_t)(elapsedTime * 1000);

    cudaEventRecord(e_start, 0);
    histogram<<<1, 32>>>(count, old_data, n);
    gpuErrorCheck(cudaGetLastError());
    cudaEventRecord(e_stop, 0);
    cudaEventSynchronize(e_stop);
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    time += (uint64_t)(elapsedTime * 1000);
    //printRes(count, DATA_MAX_EL + 1);

    //scan(count, 0, BLOCK_SIZE, (uint32_t)-1);
    cudaEventRecord(e_start, 0);
    scan(count, DATA_MAX_EL);
    cudaEventRecord(e_stop, 0);
    cudaEventSynchronize(e_stop);
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    time += (uint64_t)(elapsedTime * 1000);
    //s(count);
    //scan(count, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE - 1);
    //printRes(count, DATA_MAX_EL + 1);

    cudaEventRecord(e_start, 0);
    CountingSortWrite<<<1, 32>>>(old_data, count, n);
    gpuErrorCheck(cudaGetLastError());
    cudaEventRecord(e_stop, 0);
    cudaEventSynchronize(e_stop);
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    time += (uint64_t)(elapsedTime * 1000);

    gpuErrorCheck(cudaMemcpy(&data[0], old_data, sizeof(DATA_TYPE) * n, cudaMemcpyDeviceToHost));

    gpuErrorCheck(cudaFree(old_data));
    gpuErrorCheck(cudaFree(count));
    return time;
}

uint64_t Array::CountingSortCPU () {
    data = {};//CountSort(Count(data), data.size());
    return 0;
}

std::istream &operator>> (std::istream &input, Array &array) {
    uint32_t n;
    input >> n;
    array.data.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        input >> array.data[i];
    }
    return input;
}

std::ostream &operator<< (std::ostream &output, const Array &array) {
    for (uint32_t i = 0; i < array.data.size() - 1; ++i) {
        output << array.data[i] << ' ';
    }
    output << array.data.back() << '\n';
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